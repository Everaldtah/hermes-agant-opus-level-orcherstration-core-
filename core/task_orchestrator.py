"""
Task Orchestrator
==================

Multi-step task execution for long complex workflows.

The single-shot process() pipeline (classify → allocate → LLM → return)
breaks down on complex tasks that require:
- Decomposing a big request into subtasks
- Executing subtasks sequentially or in parallel
- Maintaining intermediate state across steps
- Handling context window overflow mid-task
- Aggregating partial results into a final output

This orchestrator sits between the HermesAgentCore and the LLM provider
to manage that complexity.

Designed specifically to handle the constraints of:
- GLM 5.1: 128k context, slower on long outputs, custom rate limits
- Kimi K2: 128k+ context, 3RPM free tier, partial responses on timeout
"""

import asyncio
import logging
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStrategy(Enum):
    """How to execute subtask steps."""
    SEQUENTIAL = "sequential"       # One after another, each sees prior results
    PARALLEL = "parallel"           # All at once, independent
    PIPELINE = "pipeline"           # Sequential but output of N feeds input of N+1
    FAN_OUT_FAN_IN = "fan_out_in"   # Parallel then aggregate


@dataclass
class TaskStep:
    """A single step in a multi-step task."""
    id: str
    prompt: str
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    max_retries: int = 2
    
    # Context control
    include_prior_results: bool = True
    max_context_tokens: int = 0  # 0 = use default
    system_prompt_override: str = ""


@dataclass
class TaskPlan:
    """Complete plan for executing a complex task."""
    task_id: str
    description: str
    steps: List[TaskStep]
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    created_at: float = field(default_factory=time.time)
    
    # Aggregation
    aggregation_prompt: str = ""  # Prompt to combine step results
    
    # Budget
    total_token_budget: int = 0  # 0 = unlimited
    
    def pending_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]
    
    def completed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]
    
    def failed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]
    
    @property
    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
            for s in self.steps
        )
    
    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.steps)
    
    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)


@dataclass
class OrchestratorResult:
    """Final result of orchestrated task execution."""
    success: bool
    final_response: str = ""
    steps_completed: int = 0
    steps_failed: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class TaskDecomposer:
    """
    Decomposes a complex task into subtask steps.
    
    Uses heuristic rules first (fast, no LLM call), then optionally
    asks the LLM itself to plan if the task is truly complex.
    """
    
    # Patterns that indicate multi-step work
    MULTI_STEP_INDICATORS = [
        r'\b(and then|after that|next|finally|first|second|third)\b',
        r'\b(step \d|phase \d|part \d)\b',
        r'\b(analyze|write|create|build|implement).+\b(and|then|also)\b.+\b(analyze|write|create|build|implement)\b',
        r'\b(comprehensive|thorough|detailed|complete|full).+\b(report|analysis|review|audit)\b',
    ]
    
    # Domain-specific decomposition templates
    TEMPLATES = {
        "code_review": [
            "Identify the main components and architecture of the code",
            "Check for bugs, error handling gaps, and edge cases",
            "Evaluate code quality, naming, and patterns",
            "Suggest specific improvements with code examples",
        ],
        "security_audit": [
            "Map the attack surface: inputs, endpoints, authentication",
            "Check for common vulnerabilities (injection, XSS, CSRF, auth bypass)",
            "Review cryptographic usage and data handling",
            "Produce a prioritized findings report with remediation steps",
        ],
        "research_report": [
            "Define scope and key questions to answer",
            "Gather and analyze relevant information",
            "Identify patterns, contradictions, and key insights",
            "Synthesize findings into a structured report",
        ],
        "data_analysis": [
            "Understand the data structure and quality",
            "Perform exploratory analysis and identify patterns",
            "Run specific analyses relevant to the question",
            "Summarize findings with actionable recommendations",
        ],
    }
    
    def needs_decomposition(self, message: str, profile: Any = None) -> bool:
        """Check if a message represents a complex multi-step task."""
        # Length heuristic: very long prompts usually need decomposition
        if len(message.split()) > 150:
            return True
        
        # Pattern matching
        for pattern in self.MULTI_STEP_INDICATORS:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        
        # Complexity from classifier
        if profile and hasattr(profile, 'complexity'):
            if profile.complexity.value >= 3:  # COMPLEX
                return True
        
        return False
    
    def decompose(
        self,
        message: str,
        profile: Any = None,
        max_context_per_step: int = 32000,
    ) -> TaskPlan:
        """
        Decompose a complex task into a TaskPlan.
        
        Uses heuristic template matching first. Falls back to a
        generic decomposition if no template matches.
        """
        task_id = f"task-{int(time.time())}"
        
        # Try to match a domain template
        template = self._match_template(message, profile)
        
        if template:
            steps = [
                TaskStep(
                    id=f"{task_id}-step-{i}",
                    prompt=step_prompt,
                    depends_on=[f"{task_id}-step-{i-1}"] if i > 0 else [],
                    include_prior_results=True,
                    max_context_tokens=max_context_per_step,
                )
                for i, step_prompt in enumerate(template)
            ]
            strategy = ExecutionStrategy.PIPELINE
        else:
            # Generic decomposition for unrecognized complex tasks
            steps = self._generic_decompose(message, task_id, max_context_per_step)
            strategy = ExecutionStrategy.SEQUENTIAL
        
        return TaskPlan(
            task_id=task_id,
            description=message[:200],
            steps=steps,
            strategy=strategy,
            aggregation_prompt=(
                "You are finalizing a multi-step analysis. Combine the results "
                "from all previous steps into a single coherent, well-structured response. "
                "Remove any redundancy. Ensure the final output directly addresses "
                f"the original request: {message[:500]}"
            ),
        )
    
    def _match_template(self, message: str, profile: Any) -> Optional[List[str]]:
        """Match message to a decomposition template."""
        msg_lower = message.lower()
        
        if any(kw in msg_lower for kw in ("review code", "code review", "audit code", "review this code")):
            return self.TEMPLATES["code_review"]
        if any(kw in msg_lower for kw in ("security", "vulnerability", "pentest", "exploit", "cve")):
            return self.TEMPLATES["security_audit"]
        if any(kw in msg_lower for kw in ("research", "report on", "investigate", "deep dive")):
            return self.TEMPLATES["research_report"]
        if any(kw in msg_lower for kw in ("analyze data", "data analysis", "dataset", "csv", "metrics")):
            return self.TEMPLATES["data_analysis"]
        
        return None
    
    def _generic_decompose(
        self, message: str, task_id: str, max_ctx: int
    ) -> List[TaskStep]:
        """Generic 3-step decomposition: plan → execute → synthesize."""
        return [
            TaskStep(
                id=f"{task_id}-plan",
                prompt=(
                    f"Break down the following task into concrete steps. "
                    f"List each step clearly. Task: {message}"
                ),
                max_context_tokens=max_ctx,
                include_prior_results=False,
            ),
            TaskStep(
                id=f"{task_id}-execute",
                prompt=(
                    f"Execute the plan from the previous step. "
                    f"Work through each item thoroughly. Original task: {message}"
                ),
                depends_on=[f"{task_id}-plan"],
                max_context_tokens=max_ctx,
                include_prior_results=True,
            ),
            TaskStep(
                id=f"{task_id}-synthesize",
                prompt=(
                    f"Review and synthesize all work done so far into a final, "
                    f"polished response for the original task: {message[:300]}"
                ),
                depends_on=[f"{task_id}-execute"],
                max_context_tokens=max_ctx,
                include_prior_results=True,
            ),
        ]


class TaskOrchestrator:
    """
    Executes TaskPlans against an LLM provider, handling:
    - Sequential/parallel step execution
    - Context window management between steps
    - Rate limit backoff (critical for Kimi's 3RPM free tier)
    - Token budget enforcement
    - Partial failure recovery
    - Result aggregation
    """
    
    def __init__(
        self,
        provider_manager,
        event_bus=None,
        max_context_tokens: int = 120000,
        rate_limit_delay: float = 1.0,
    ):
        self.provider_manager = provider_manager
        self.events = event_bus
        self.max_context_tokens = max_context_tokens
        self.rate_limit_delay = rate_limit_delay
        self._decomposer = TaskDecomposer()
    
    @property
    def decomposer(self) -> TaskDecomposer:
        return self._decomposer
    
    async def execute_plan(self, plan: TaskPlan) -> OrchestratorResult:
        """
        Execute a complete task plan.
        
        Returns aggregated result from all steps.
        """
        start_time = time.time()
        logger.info(
            f"Orchestrator starting plan {plan.task_id} "
            f"({len(plan.steps)} steps, strategy={plan.strategy.value})"
        )
        
        if self.events:
            self.events.publish("orchestrator.plan_started", {
                "task_id": plan.task_id,
                "steps": len(plan.steps),
                "strategy": plan.strategy.value,
            }, source="orchestrator")
        
        try:
            if plan.strategy in (ExecutionStrategy.SEQUENTIAL, ExecutionStrategy.PIPELINE):
                await self._execute_sequential(plan)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(plan)
            elif plan.strategy == ExecutionStrategy.FAN_OUT_FAN_IN:
                await self._execute_fan_out_in(plan)
            else:
                await self._execute_sequential(plan)
            
            # Aggregate results if there's an aggregation prompt and
            # we completed more than one step
            final_response = ""
            if plan.aggregation_prompt and len(plan.completed_steps()) > 1:
                final_response = await self._aggregate(plan)
            elif plan.completed_steps():
                # Just use the last completed step's result
                final_response = plan.completed_steps()[-1].result or ""
            
            result = OrchestratorResult(
                success=len(plan.failed_steps()) == 0,
                final_response=final_response,
                steps_completed=len(plan.completed_steps()),
                steps_failed=len(plan.failed_steps()),
                total_tokens=plan.total_tokens,
                total_latency_ms=(time.time() - start_time) * 1000,
                step_results=[
                    {
                        "id": s.id,
                        "status": s.status.value,
                        "tokens": s.tokens_used,
                        "latency_ms": round(s.latency_ms, 1),
                        "result_preview": (s.result or "")[:200],
                    }
                    for s in plan.steps
                ],
            )
            
        except Exception as e:
            logger.error(f"Orchestrator plan {plan.task_id} failed: {e}")
            result = OrchestratorResult(
                success=False,
                error=str(e),
                steps_completed=len(plan.completed_steps()),
                steps_failed=len(plan.failed_steps()) + len(plan.pending_steps()),
                total_tokens=plan.total_tokens,
                total_latency_ms=(time.time() - start_time) * 1000,
            )
        
        if self.events:
            self.events.publish("orchestrator.plan_completed", {
                "task_id": plan.task_id,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "total_tokens": result.total_tokens,
            }, source="orchestrator")
        
        return result
    
    async def _execute_sequential(self, plan: TaskPlan):
        """Execute steps one after another, feeding results forward."""
        accumulated_context = []
        
        for step in plan.steps:
            # Check dependencies
            if step.depends_on:
                unmet = [
                    dep for dep in step.depends_on
                    if not any(
                        s.id == dep and s.status == StepStatus.COMPLETED
                        for s in plan.steps
                    )
                ]
                if unmet:
                    step.status = StepStatus.SKIPPED
                    step.error = f"Unmet dependencies: {unmet}"
                    continue
            
            # Check token budget
            if plan.total_token_budget and plan.total_tokens >= plan.total_token_budget:
                step.status = StepStatus.SKIPPED
                step.error = "Token budget exhausted"
                logger.warning(f"Step {step.id} skipped: budget exhausted")
                continue
            
            # Build context from prior steps
            messages = []
            if step.include_prior_results and accumulated_context:
                context_text = self._build_context_summary(
                    accumulated_context, step.max_context_tokens or self.max_context_tokens
                )
                messages.append({
                    "role": "assistant",
                    "content": f"[Prior work completed]\n{context_text}"
                })
            
            messages.append({"role": "user", "content": step.prompt})
            
            # Execute with retry
            await self._execute_step(step, messages)
            
            if step.status == StepStatus.COMPLETED and step.result:
                accumulated_context.append({
                    "step_id": step.id,
                    "result": step.result,
                })
            
            # Rate limit courtesy delay (important for Kimi free tier)
            if plan.steps.index(step) < len(plan.steps) - 1:
                await asyncio.sleep(self.rate_limit_delay)
    
    async def _execute_parallel(self, plan: TaskPlan):
        """Execute independent steps in parallel."""
        tasks = []
        for step in plan.steps:
            messages = [{"role": "user", "content": step.prompt}]
            tasks.append(self._execute_step(step, messages))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_fan_out_in(self, plan: TaskPlan):
        """Fan out parallel steps, then aggregate."""
        # All steps except the last run in parallel
        parallel_steps = plan.steps[:-1] if len(plan.steps) > 1 else plan.steps
        agg_step = plan.steps[-1] if len(plan.steps) > 1 else None
        
        tasks = []
        for step in parallel_steps:
            messages = [{"role": "user", "content": step.prompt}]
            tasks.append(self._execute_step(step, messages))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Fan-in: aggregation step
        if agg_step:
            context = [
                {"step_id": s.id, "result": s.result or "(failed)"}
                for s in parallel_steps
            ]
            context_text = self._build_context_summary(context, self.max_context_tokens)
            messages = [
                {"role": "assistant", "content": f"[Results from parallel analysis]\n{context_text}"},
                {"role": "user", "content": agg_step.prompt},
            ]
            await self._execute_step(agg_step, messages)
    
    async def _execute_step(
        self,
        step: TaskStep,
        messages: List[Dict[str, str]],
    ):
        """Execute a single step with retry logic."""
        from .llm_provider import LLMRequest
        
        step.status = StepStatus.RUNNING
        
        for attempt in range(step.max_retries + 1):
            try:
                request = LLMRequest(
                    messages=messages,
                    max_tokens=min(4096, step.max_context_tokens or 4096),
                    system_prompt=step.system_prompt_override or (
                        "You are a precise, thorough AI assistant working on one step "
                        "of a larger task. Be detailed and complete in your response."
                    ),
                )
                
                start = time.time()
                response = await self.provider_manager.complete(request)
                
                step.result = response.content
                step.tokens_used = response.total_tokens
                step.latency_ms = (time.time() - start) * 1000
                step.status = StepStatus.COMPLETED
                step.retries = attempt
                
                logger.info(
                    f"Step {step.id} completed: {response.total_tokens} tokens, "
                    f"{step.latency_ms:.0f}ms"
                )
                return
                
            except ConnectionError as e:
                # Rate limit — back off
                step.retries = attempt + 1
                wait = self.rate_limit_delay * (2 ** attempt)
                logger.warning(
                    f"Step {step.id} rate limited (attempt {attempt + 1}), "
                    f"waiting {wait:.1f}s: {e}"
                )
                await asyncio.sleep(wait)
                
            except ValueError as e:
                # Context overflow — try to truncate and retry
                if "context" in str(e).lower() and messages:
                    logger.warning(f"Step {step.id} context overflow, truncating")
                    # Remove oldest assistant message to free space
                    messages = [m for m in messages if m["role"] != "assistant"][-3:]
                    step.retries = attempt + 1
                else:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    return
                    
            except Exception as e:
                step.retries = attempt + 1
                logger.warning(f"Step {step.id} attempt {attempt + 1} failed: {e}")
                if attempt >= step.max_retries:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    return
                await asyncio.sleep(self.rate_limit_delay)
    
    async def _aggregate(self, plan: TaskPlan) -> str:
        """Aggregate results from completed steps into a final response."""
        from .llm_provider import LLMRequest
        
        context_parts = []
        for step in plan.completed_steps():
            context_parts.append(
                f"## Step: {step.id}\n{step.result}\n"
            )
        
        context_text = self._build_context_summary(
            [{"result": "\n".join(context_parts)}],
            self.max_context_tokens // 2,  # Leave room for aggregation output
        )
        
        request = LLMRequest(
            messages=[
                {"role": "assistant", "content": context_text},
                {"role": "user", "content": plan.aggregation_prompt},
            ],
            max_tokens=4096,
            system_prompt=(
                "You are synthesizing the results of a multi-step analysis. "
                "Create a polished, comprehensive response."
            ),
        )
        
        try:
            response = await self.provider_manager.complete(request)
            return response.content
        except Exception as e:
            logger.error(f"Aggregation failed: {e}, using concatenated results")
            return "\n\n".join(
                f"**{s.id}:**\n{s.result}" for s in plan.completed_steps()
            )
    
    def _build_context_summary(
        self,
        items: List[Dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """
        Build a context string that fits within token budget.
        
        Truncates from the middle (keeps first and last items)
        if the total would exceed the budget.
        """
        parts = [item.get("result", "") for item in items]
        full_text = "\n---\n".join(parts)
        
        # Rough token estimate
        estimated = len(full_text.split()) * 1.3
        if estimated <= max_tokens:
            return full_text
        
        # Truncate: keep first and last, summarize middle
        if len(parts) <= 2:
            # Just hard-truncate
            char_budget = int(max_tokens * 3)  # ~3 chars per token
            return full_text[:char_budget] + "\n...[truncated]"
        
        first = parts[0]
        last = parts[-1]
        middle_count = len(parts) - 2
        
        return (
            f"{first}\n---\n"
            f"[...{middle_count} intermediate steps summarized for space...]\n---\n"
            f"{last}"
        )
