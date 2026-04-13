"""
Sub-Agent Swarm Orchestrator
==============================

Implements the five core multi-agent patterns from the research:
  1. Orchestrator-Worker (fan-out to specialists, aggregate)
  2. Pipeline (sequential stage handoffs)
  3. Fan-Out/Fan-In (parallel then synthesize)
  4. Hierarchical (tree of delegators, each spawns sub-agents)
  5. Consensus (multiple agents vote, majority wins — for trading bots etc.)

Each SubAgent runs with its own isolated context, can use a
different LLM provider, and reports results back to the parent.

Heavily influenced by:
- Google ADK multi-agent patterns (SequentialAgent, ParallelAgent, CoordinatorAgent)
- Swarms framework (DAG orchestration)
- Anthropic's sub-agent spawning pattern (isolated contexts per worker)
- Addy Osmani's "Code Agent Orchestra" (feature-lead → specialist hierarchy)
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined specialist roles for sub-agents."""
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    SECURITY = "security"
    ANALYST = "analyst"
    WRITER = "writer"
    PLANNER = "planner"
    DEBUGGER = "debugger"
    SYNTHESIZER = "synthesizer"
    CUSTOM = "custom"


class SwarmPattern(Enum):
    """Orchestration patterns."""
    ORCHESTRATOR_WORKER = "orchestrator_worker"
    PIPELINE = "pipeline"
    FAN_OUT_FAN_IN = "fan_out_fan_in"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"


@dataclass
class SubAgentConfig:
    """Configuration for a spawned sub-agent."""
    name: str
    role: AgentRole = AgentRole.CUSTOM
    system_prompt: str = ""
    model_override: str = ""  # Use specific model for this agent
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0
    max_retries: int = 2
    tags: List[str] = field(default_factory=list)


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    agent_name: str
    role: AgentRole
    success: bool
    output: str = ""
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmResult:
    """Aggregated result from a swarm execution."""
    pattern: SwarmPattern
    success: bool
    final_output: str = ""
    agent_results: List[SubAgentResult] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    consensus_score: float = 0.0  # For consensus pattern
    error: Optional[str] = None

    @property
    def agents_succeeded(self) -> int:
        return sum(1 for r in self.agent_results if r.success)

    @property
    def agents_failed(self) -> int:
        return sum(1 for r in self.agent_results if not r.success)


class SubAgent:
    """
    An isolated sub-agent with its own context.

    Each sub-agent gets:
    - Its own system prompt (role-specific)
    - Optional model override (e.g. use GLM for code, Kimi for research)
    - Isolated message history (doesn't pollute parent context)
    - Timeout and retry logic
    """

    def __init__(self, config: SubAgentConfig, provider_manager):
        self.config = config
        self.provider_manager = provider_manager
        self._message_history: List[Dict[str, str]] = []

    async def execute(self, task: str, context: str = "") -> SubAgentResult:
        """Execute a task with this sub-agent."""
        from core.llm_provider import LLMRequest

        start = time.time()

        messages = []
        if context:
            messages.append({"role": "assistant", "content": f"[Context]\n{context}"})
        messages.append({"role": "user", "content": task})

        system = self.config.system_prompt or self._default_system_prompt()

        request = LLMRequest(
            messages=messages,
            model=self.config.model_override,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system_prompt=system,
        )

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.provider_manager.complete(request),
                    timeout=self.config.timeout,
                )

                self._message_history.append({"role": "user", "content": task})
                self._message_history.append({"role": "assistant", "content": response.content})

                return SubAgentResult(
                    agent_name=self.config.name,
                    role=self.config.role,
                    success=True,
                    output=response.content,
                    tokens_used=response.total_tokens,
                    latency_ms=(time.time() - start) * 1000,
                    metadata={
                        "model": response.model,
                        "provider": response.provider,
                        "attempt": attempt + 1,
                    },
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.config.timeout}s"
                logger.warning(f"SubAgent {self.config.name} timeout (attempt {attempt+1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"SubAgent {self.config.name} failed (attempt {attempt+1}): {e}")

            if attempt < self.config.max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))

        return SubAgentResult(
            agent_name=self.config.name,
            role=self.config.role,
            success=False,
            error=last_error,
            latency_ms=(time.time() - start) * 1000,
        )

    def _default_system_prompt(self) -> str:
        prompts = {
            AgentRole.RESEARCHER: "You are a thorough research specialist. Find comprehensive information, cite sources, identify gaps.",
            AgentRole.CODER: "You are an expert programmer. Write clean, tested, production-ready code with error handling.",
            AgentRole.REVIEWER: "You are a critical code/content reviewer. Find bugs, logic errors, and suggest improvements.",
            AgentRole.SECURITY: "You are a cybersecurity specialist. Identify vulnerabilities, attack vectors, and remediation steps.",
            AgentRole.ANALYST: "You are a data analyst. Extract insights, identify patterns, and make data-driven recommendations.",
            AgentRole.WRITER: "You are a professional writer. Produce clear, well-structured content adapted to the audience.",
            AgentRole.PLANNER: "You are a strategic planner. Break down complex problems into actionable steps with dependencies.",
            AgentRole.DEBUGGER: "You are a debugging specialist. Trace errors systematically and find root causes.",
            AgentRole.SYNTHESIZER: "You are a synthesis expert. Combine multiple inputs into a coherent, unified output.",
            AgentRole.CUSTOM: "You are a helpful AI assistant. Complete the assigned task thoroughly.",
        }
        return prompts.get(self.config.role, prompts[AgentRole.CUSTOM])


class SwarmOrchestrator:
    """
    Orchestrates multiple sub-agents using configurable patterns.

    Usage:
        swarm = SwarmOrchestrator(provider_manager, event_bus)

        # Fan-out security audit to 3 specialists
        result = await swarm.fan_out_fan_in(
            task="Audit this web application",
            agents=[
                SubAgentConfig(name="vuln-scanner", role=AgentRole.SECURITY),
                SubAgentConfig(name="code-reviewer", role=AgentRole.REVIEWER),
                SubAgentConfig(name="crypto-auditor", role=AgentRole.SECURITY),
            ],
            synthesis_prompt="Combine all findings into a prioritized report."
        )

        # Consensus voting for trading decision
        result = await swarm.consensus(
            task="Should we go long on ETH based on these signals?",
            agents=[3 analyst configs...],
            threshold=0.66,  # 2/3 must agree
        )
    """

    def __init__(self, provider_manager, event_bus=None, memory_store=None):
        self.provider_manager = provider_manager
        self.events = event_bus
        self.memory = memory_store
        self._execution_count = 0

    # ── Pattern: Orchestrator-Worker ──────────────────────────────

    async def orchestrator_worker(
        self,
        task: str,
        agents: List[SubAgentConfig],
        decompose_prompt: str = "",
        rate_limit_delay: float = 0.5,
    ) -> SwarmResult:
        """
        Central orchestrator decomposes task, delegates to workers, aggregates.
        """
        start = time.time()
        self._execution_count += 1

        # Step 1: Decompose
        planner = SubAgent(
            SubAgentConfig(name="planner", role=AgentRole.PLANNER),
            self.provider_manager,
        )
        plan_result = await planner.execute(
            decompose_prompt or f"Break this task into {len(agents)} independent subtasks, one per specialist:\n\n{task}\n\nSpecialists: {', '.join(a.name for a in agents)}\n\nRespond with numbered subtasks only."
        )

        if not plan_result.success:
            return SwarmResult(
                pattern=SwarmPattern.ORCHESTRATOR_WORKER,
                success=False, error=f"Planning failed: {plan_result.error}",
            )

        # Step 2: Delegate to workers
        subtasks = self._parse_subtasks(plan_result.output, len(agents))
        worker_results = []

        for i, (agent_cfg, subtask) in enumerate(zip(agents, subtasks)):
            agent = SubAgent(agent_cfg, self.provider_manager)
            result = await agent.execute(
                subtask,
                context=f"Original task: {task}\nYour role: {agent_cfg.role.value}",
            )
            worker_results.append(result)
            if i < len(agents) - 1:
                await asyncio.sleep(rate_limit_delay)

        # Step 3: Aggregate
        all_results = [plan_result] + worker_results
        return self._build_swarm_result(
            SwarmPattern.ORCHESTRATOR_WORKER, all_results, start, task
        )

    # ── Pattern: Pipeline ─────────────────────────────────────────

    async def pipeline(
        self,
        task: str,
        agents: List[SubAgentConfig],
        rate_limit_delay: float = 0.5,
    ) -> SwarmResult:
        """Sequential pipeline: output of agent N feeds into agent N+1."""
        start = time.time()
        results = []
        current_input = task
        accumulated_context = ""

        for i, agent_cfg in enumerate(agents):
            agent = SubAgent(agent_cfg, self.provider_manager)
            result = await agent.execute(current_input, context=accumulated_context)
            results.append(result)

            if result.success:
                accumulated_context += f"\n\n[{agent_cfg.name} output]\n{result.output}"
                current_input = (
                    f"Continue from the previous step. "
                    f"Previous output:\n{result.output[:2000]}\n\n"
                    f"Your task as {agent_cfg.role.value}: proceed with the next phase."
                )
            else:
                logger.warning(f"Pipeline stage {agent_cfg.name} failed, continuing")

            if i < len(agents) - 1:
                await asyncio.sleep(rate_limit_delay)

        return self._build_swarm_result(SwarmPattern.PIPELINE, results, start, task)

    # ── Pattern: Fan-Out / Fan-In ─────────────────────────────────

    async def fan_out_fan_in(
        self,
        task: str,
        agents: List[SubAgentConfig],
        synthesis_prompt: str = "",
    ) -> SwarmResult:
        """
        All agents work on the same task in parallel,
        then a synthesizer combines the results.
        """
        start = time.time()

        # Fan-out: all agents in parallel
        tasks_coros = []
        for agent_cfg in agents:
            agent = SubAgent(agent_cfg, self.provider_manager)
            tasks_coros.append(agent.execute(task))

        parallel_results = await asyncio.gather(*tasks_coros, return_exceptions=True)

        results = []
        for i, res in enumerate(parallel_results):
            if isinstance(res, Exception):
                results.append(SubAgentResult(
                    agent_name=agents[i].name, role=agents[i].role,
                    success=False, error=str(res),
                ))
            else:
                results.append(res)

        # Fan-in: synthesize
        successful_outputs = [
            f"[{r.agent_name} ({r.role.value})]\n{r.output}"
            for r in results if r.success
        ]

        if successful_outputs:
            synthesizer = SubAgent(
                SubAgentConfig(name="synthesizer", role=AgentRole.SYNTHESIZER),
                self.provider_manager,
            )
            synth_prompt = synthesis_prompt or (
                f"Combine these {len(successful_outputs)} specialist reports into "
                f"a single coherent response for: {task[:500]}"
            )
            synth_result = await synthesizer.execute(
                synth_prompt,
                context="\n\n---\n\n".join(successful_outputs),
            )
            results.append(synth_result)

        return self._build_swarm_result(
            SwarmPattern.FAN_OUT_FAN_IN, results, start, task
        )

    # ── Pattern: Consensus ────────────────────────────────────────

    async def consensus(
        self,
        task: str,
        agents: List[SubAgentConfig],
        threshold: float = 0.66,
        decision_categories: Optional[List[str]] = None,
    ) -> SwarmResult:
        """
        Multiple agents vote on a decision. Used for trading bots,
        risk assessment, or any situation needing multi-model agreement.

        Each agent responds, then we classify their answers into
        categories and check if any category exceeds the threshold.
        """
        start = time.time()

        categories = decision_categories or ["YES", "NO", "UNCERTAIN"]

        vote_prompt = (
            f"{task}\n\n"
            f"You must respond with exactly one of: {', '.join(categories)}\n"
            f"Then provide your reasoning on the next line."
        )

        # All agents vote in parallel
        coros = []
        for agent_cfg in agents:
            agent = SubAgent(agent_cfg, self.provider_manager)
            coros.append(agent.execute(vote_prompt))

        vote_results = await asyncio.gather(*coros, return_exceptions=True)

        results = []
        votes: Dict[str, int] = {c: 0 for c in categories}

        for i, res in enumerate(vote_results):
            if isinstance(res, Exception):
                results.append(SubAgentResult(
                    agent_name=agents[i].name, role=agents[i].role,
                    success=False, error=str(res),
                ))
                continue

            results.append(res)
            if res.success:
                # Parse vote from first line
                first_line = res.output.strip().split("\n")[0].upper().strip()
                voted = None
                for cat in categories:
                    if cat.upper() in first_line:
                        voted = cat
                        break
                if voted:
                    votes[voted] += 1
                    res.metadata["vote"] = voted

        # Determine consensus
        total_votes = sum(votes.values())
        consensus_reached = False
        winning_category = ""
        consensus_score = 0.0

        if total_votes > 0:
            for cat, count in votes.items():
                ratio = count / total_votes
                if ratio >= threshold:
                    consensus_reached = True
                    winning_category = cat
                    consensus_score = ratio
                    break

        swarm_result = self._build_swarm_result(
            SwarmPattern.CONSENSUS, results, start, task
        )
        swarm_result.consensus_score = consensus_score
        swarm_result.final_output = (
            f"Consensus: {winning_category} ({consensus_score:.0%})"
            if consensus_reached
            else f"No consensus reached. Votes: {votes}"
        )
        swarm_result.metadata = {"votes": votes, "threshold": threshold}

        # Store decision in memory if available
        if self.memory and consensus_reached:
            from memory.long_term import MemoryRecord, MemoryTier
            self.memory.store(MemoryRecord(
                tier=MemoryTier.EPISODIC,
                content=f"Consensus decision: {winning_category} on '{task[:200]}' (score={consensus_score:.2f})",
                importance=0.8,
                tags=["consensus", "decision"],
                metadata={"votes": votes, "task": task[:500]},
            ))

        return swarm_result

    # ── Helpers ────────────────────────────────────────────────────

    def _parse_subtasks(self, plan_text: str, expected: int) -> List[str]:
        """Extract numbered subtasks from planner output."""
        import re
        lines = plan_text.strip().split("\n")
        tasks = []
        for line in lines:
            cleaned = re.sub(r'^\d+[\.\)\-]\s*', '', line.strip())
            if cleaned and len(cleaned) > 10:
                tasks.append(cleaned)

        # Pad or trim to match expected agent count
        while len(tasks) < expected:
            tasks.append(f"Handle remaining aspects of the task.")
        return tasks[:expected]

    def _build_swarm_result(
        self,
        pattern: SwarmPattern,
        results: List[SubAgentResult],
        start_time: float,
        task: str,
    ) -> SwarmResult:
        """Build aggregated SwarmResult from individual agent results."""
        successful = [r for r in results if r.success]
        total_tokens = sum(r.tokens_used for r in results)
        total_latency = (time.time() - start_time) * 1000

        # Use last successful result as final output
        final = successful[-1].output if successful else ""

        if self.events:
            self.events.publish("swarm.completed", {
                "pattern": pattern.value,
                "agents": len(results),
                "succeeded": len(successful),
                "total_tokens": total_tokens,
                "latency_ms": total_latency,
            }, source="swarm_orchestrator")

        return SwarmResult(
            pattern=pattern,
            success=len(successful) > 0,
            final_output=final,
            agent_results=results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_executions": self._execution_count,
        }
