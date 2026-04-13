"""
Hermes Agent Core - Main Orchestrator
======================================

Unified core that integrates all 4 phases with enhanced processing capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, AsyncIterator, Union
from collections import deque
import threading
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import enhanced modules
from .resilience_enhanced import EnhancedResilienceEngine, ProtectionResult
from .context_enhanced import EnhancedTaskClassifier, EnhancedTokenAllocator, TaskProfile
from .persistence_enhanced import EnhancedDeltaPersistence
from .llm_provider import (
    LLMProvider, MockProvider, ProviderManager, LLMRequest, LLMResponse, ProviderType
)
from .event_bus import EventBus, get_event_bus
from .task_orchestrator import TaskOrchestrator, TaskDecomposer, TaskPlan

# Import memory and swarm modules
from memory.long_term import MemoryStore, MemoryRecord, MemoryTier
from swarm.orchestrator import SwarmOrchestrator, SubAgentConfig, AgentRole, SwarmPattern

# Import processing power modules
from processing.async_engine import AsyncProcessingEngine
from processing.worker_pool import IntelligentWorkerPool
from processing.batch_processor import BatchProcessor
from efficiency.cache_manager import MultiLevelCache
from efficiency.load_balancer import SmartLoadBalancer
from monitoring.metrics_collector import MetricsCollector
from recovery.self_healing import SelfHealingRecovery

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode for the agent."""
    SYNC = auto()           # Synchronous processing
    ASYNC = auto()          # Asynchronous processing
    BATCH = auto()          # Batch processing
    PARALLEL = auto()       # Parallel processing
    SWARM = auto()          # Distributed swarm processing


class AgentState(Enum):
    """Agent operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"


@dataclass
class AgentConfig:
    """Configuration for Hermes Agent Core."""
    # Resilience settings
    tokens_per_minute: int = 3000
    token_burst: int = 100
    circuit_failure_threshold: int = 5
    circuit_timeout: int = 60
    health_check_interval: int = 30
    
    # Context settings
    max_context: int = 16000
    enable_compression: bool = True
    min_confidence: float = 0.6
    
    # Persistence settings
    db_path: str = "~/.hermes/session_deltas.db"
    checkpoint_interval: int = 300
    auto_checkpoint: bool = True
    
    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.ASYNC
    max_workers: int = 4
    batch_size: int = 10
    batch_timeout: float = 0.1
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl: int = 300
    enable_predictive_cache: bool = True
    
    # Swarm settings
    enable_swarm_mode: bool = False
    swarm_node_id: Optional[str] = None
    coordinator_url: Optional[str] = None
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 8080
    
    # Recovery settings
    enable_self_healing: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # LLM provider settings
    llm_provider: ProviderType = ProviderType.AUTO
    llm_model: str = ""
    llm_api_key: str = ""


@dataclass
class ProcessingResult:
    """Result from processing a message."""
    success: bool
    response: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    task_profile: Optional[TaskProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HermesAgentCore:
    """
    Enhanced Hermes Agent Core with full 4-phase integration.
    
    Features:
    - Resilience Engine (Phase 1): Token bucket, circuit breaker, health tracking
    - Context Intelligence (Phase 2): Task classification, adaptive token allocation
    - Delta Persistence (Phase 3): Session continuity, checkpoint recovery
    - Processing Power: Async engine, worker pool, batch processing
    - Efficiency: Multi-level caching, smart load balancing
    - Monitoring: Real-time metrics collection
    - Recovery: Self-healing error recovery
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = AgentState.INITIALIZING
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Initialize all components
        self._init_event_bus()
        self._init_llm_provider()
        self._init_resilience()
        self._init_context()
        self._init_persistence()
        self._init_processing()
        self._init_efficiency()
        self._init_monitoring()
        self._init_recovery()
        
        # Processing statistics
        self._processed_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        self._start_time = time.time()
        
        self.state = AgentState.READY
        logger.info(f"HermesAgentCore v2.0 initialized in {self.config.processing_mode.name} mode")
    
    def _init_event_bus(self):
        """Initialize event bus for inter-component communication."""
        self.events = get_event_bus()
        logger.info("Event bus initialized")
    
    def _init_llm_provider(self):
        """
        Initialize LLM provider manager with failover chain.
        
        Routing priority (AUTO mode matches your Hermes setup):
          1. GLM-5.1 (Zhipu AI) — primary preference
          2. Kimi K2.5 (via OpenRouter) — current session model
          3. Claude (Anthropic) — fallback when API limits hit
          4. LM Studio (local) — offline/air-gapped
          5. Mock — last resort so agent never crashes
        """
        self.provider_manager = ProviderManager()
        
        provider_type = self.config.llm_provider
        
        if provider_type == ProviderType.AUTO:
            # Full chain matching your actual setup
            from .llm_provider import (
                GLMProvider, KimiOpenRouterProvider, AnthropicProvider,
                LMStudioProvider,
            )
            
            # 1. GLM-5.1 (primary)
            glm_key = self.config.llm_api_key or os.environ.get("GLM_API_KEY", "")
            if glm_key:
                self.provider_manager.add_provider(
                    GLMProvider(model="glm-4-plus", api_key=glm_key)
                )
            
            # 2. Kimi K2.5 via OpenRouter
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
            if or_key:
                self.provider_manager.add_provider(
                    KimiOpenRouterProvider(
                        model="moonshotai/kimi-k2",
                        api_key=or_key,
                    )
                )
            
            # 3. Kimi direct API (fallback if OpenRouter is down)
            kimi_key = os.environ.get("KIMI_API_KEY", "")
            if kimi_key:
                from .llm_provider import KimiProvider
                self.provider_manager.add_provider(
                    KimiProvider(model="moonshot-v1-128k", api_key=kimi_key)
                )
            
            # 4. Claude (Anthropic)
            ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if ant_key:
                self.provider_manager.add_provider(
                    AnthropicProvider(api_key=ant_key)
                )
            
            # 5. LM Studio local
            self.provider_manager.add_provider(
                LMStudioProvider(base_url="http://127.0.0.1:1234")
            )
            
            # 6. Mock (always present as last resort)
            self.provider_manager.add_provider(MockProvider())
        
        elif provider_type == ProviderType.GLM:
            from .llm_provider import GLMProvider
            self.provider_manager.add_provider(
                GLMProvider(
                    model=self.config.llm_model or "glm-4-plus",
                    api_key=self.config.llm_api_key,
                )
            )
            self.provider_manager.add_provider(MockProvider())
        
        elif provider_type == ProviderType.KIMI:
            from .llm_provider import KimiProvider
            self.provider_manager.add_provider(
                KimiProvider(
                    model=self.config.llm_model or "moonshot-v1-128k",
                    api_key=self.config.llm_api_key,
                )
            )
            self.provider_manager.add_provider(MockProvider())
        
        elif provider_type == ProviderType.KIMI_OPENROUTER:
            from .llm_provider import KimiOpenRouterProvider
            self.provider_manager.add_provider(
                KimiOpenRouterProvider(
                    model=self.config.llm_model or "moonshotai/kimi-k2",
                    api_key=self.config.llm_api_key,
                )
            )
            self.provider_manager.add_provider(MockProvider())
        
        elif provider_type == ProviderType.ANTHROPIC:
            from .llm_provider import AnthropicProvider
            self.provider_manager.add_provider(
                AnthropicProvider(
                    model=self.config.llm_model or "claude-sonnet-4-20250514",
                    api_key=self.config.llm_api_key,
                )
            )
            self.provider_manager.add_provider(MockProvider())
        
        elif provider_type == ProviderType.LOCAL:
            from .llm_provider import LMStudioProvider
            self.provider_manager.add_provider(
                LMStudioProvider(
                    model=self.config.llm_model or "local-model",
                    base_url="http://127.0.0.1:1234",
                )
            )
            self.provider_manager.add_provider(MockProvider())
        
        elif provider_type == ProviderType.OPENROUTER:
            from .llm_provider import OpenRouterProvider
            self.provider_manager.add_provider(
                OpenRouterProvider(
                    model=self.config.llm_model,
                    api_key=self.config.llm_api_key,
                )
            )
            self.provider_manager.add_provider(MockProvider())
        
        else:
            # MOCK or unknown
            self.provider_manager.add_provider(MockProvider())
        
        logger.info(
            f"LLM providers initialized: {provider_type.value} "
            f"({len(self.provider_manager._providers)} in chain)"
        )
    
    def _init_resilience(self):
        """Initialize enhanced resilience engine."""
        self.resilience = EnhancedResilienceEngine(
            tokens_per_minute=self.config.tokens_per_minute,
            token_burst=self.config.token_burst,
            circuit_failure_threshold=self.config.circuit_failure_threshold,
            circuit_timeout=self.config.circuit_timeout,
            health_check_interval=self.config.health_check_interval
        )
        logger.info("Resilience engine initialized")
    
    def _init_context(self):
        """Initialize enhanced context intelligence."""
        self.classifier = EnhancedTaskClassifier(
            min_confidence=self.config.min_confidence
        )
        self.allocator = EnhancedTokenAllocator(
            max_context=self.config.max_context,
            enable_compression=self.config.enable_compression
        )
        logger.info("Context intelligence initialized")
    
    def _init_persistence(self):
        """Initialize enhanced delta persistence."""
        self.persistence = EnhancedDeltaPersistence(
            db_path=self.config.db_path,
            checkpoint_interval=self.config.checkpoint_interval,
            auto_checkpoint=self.config.auto_checkpoint
        )
        logger.info("Delta persistence initialized")
    
    def _init_processing(self):
        """Initialize processing power modules."""
        # Async processing engine
        self.async_engine = AsyncProcessingEngine(
            max_concurrent=self.config.max_workers * 2
        )
        
        # Intelligent worker pool
        self.worker_pool = IntelligentWorkerPool(
            min_workers=2,
            max_workers=self.config.max_workers,
            scaling_threshold=0.8
        )
        
        # Batch processor
        self.batch_processor = BatchProcessor(
            batch_size=self.config.batch_size,
            timeout=self.config.batch_timeout
        )
        
        logger.info(f"Processing modules initialized (workers={self.config.max_workers})")
    
    def _init_efficiency(self):
        """Initialize efficiency modules."""
        # Multi-level cache
        self.cache = MultiLevelCache(
            max_size=self.config.cache_size,
            ttl=self.config.cache_ttl,
            enable_predictive=self.config.enable_predictive_cache
        )
        
        # Smart load balancer
        self.load_balancer = SmartLoadBalancer()
        
        logger.info("Efficiency modules initialized")
    
    def _init_monitoring(self):
        """Initialize monitoring and metrics."""
        if self.config.enable_metrics:
            self.metrics = MetricsCollector()
            logger.info("Metrics collector initialized")
        else:
            self.metrics = None
    
    def _init_recovery(self):
        """Initialize self-healing recovery."""
        if self.config.enable_self_healing:
            self.recovery = SelfHealingRecovery(
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            logger.info("Self-healing recovery initialized")
        else:
            self.recovery = None
        
        # Task orchestrator for complex multi-step tasks
        self.orchestrator = TaskOrchestrator(
            provider_manager=self.provider_manager,
            event_bus=self.events,
            max_context_tokens=self.config.max_context,
            rate_limit_delay=self.config.retry_delay,
        )
        logger.info("Task orchestrator initialized")
        
        # Long-term memory system
        self.memory_store = MemoryStore(
            db_path=self.config.db_path.replace("session_deltas", "memory")
        )
        logger.info("Long-term memory initialized")
        
        # Swarm orchestrator for sub-agent delegation
        self.swarm = SwarmOrchestrator(
            provider_manager=self.provider_manager,
            event_bus=self.events,
            memory_store=self.memory_store,
        )
        logger.info("Swarm orchestrator initialized")
    
    # ==================== Main Processing Methods ====================
    
    def process(self, message: str, context: Optional[Dict] = None) -> ProcessingResult:
        """
        Process a message synchronously.
        
        Automatically detects complex tasks and routes them through
        the multi-step orchestrator when needed.
        
        Args:
            message: User message to process
            context: Optional conversation context
            
        Returns:
            ProcessingResult with response and metadata
        """
        start_time = time.time()
        
        try:
            # Check resilience protection
            protection = self.resilience.check("process", tokens=1)
            if not protection.allowed:
                return ProcessingResult(
                    success=False,
                    error=f"Blocked by resilience: {protection.reason}",
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            # Check cache first
            cache_key = f"msg:{hash(message)}"
            cached = self.cache.get(cache_key)
            if cached:
                return ProcessingResult(
                    success=True,
                    response=cached,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"cache_hit": True}
                )
            
            # Classify task
            profile = self.classifier.classify(message, context)
            
            # Allocate tokens
            allocation = self.allocator.allocate(
                message=message,
                profile=profile
            )
            
            # Process with resilience protection
            result = self._execute_with_protection(
                self._process_internal,
                message,
                context,
                profile,
                allocation
            )
            
            # Cache result
            if result.success and result.response:
                self.cache.set(cache_key, result.response)
            
            # Update metrics
            self._update_metrics(result, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self._error_count += 1
            return ProcessingResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def process_async(self, message: str, context: Optional[Dict] = None) -> ProcessingResult:
        """
        Process a message asynchronously.
        
        Args:
            message: User message to process
            context: Optional conversation context
            
        Returns:
            ProcessingResult with response and metadata
        """
        start_time = time.time()
        
        try:
            # Check resilience protection
            protection = self.resilience.check("process", tokens=1)
            if not protection.allowed:
                return ProcessingResult(
                    success=False,
                    error=f"Blocked by resilience: {protection.reason}",
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            # Check cache first
            cache_key = f"msg:{hash(message)}"
            cached = self.cache.get(cache_key)
            if cached:
                return ProcessingResult(
                    success=True,
                    response=cached,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"cache_hit": True}
                )
            
            # Classify task
            profile = self.classifier.classify(message, context)
            
            # Process asynchronously
            result = await self.async_engine.run(
                self._process_internal_async,
                message,
                context,
                profile
            )
            
            # Cache result
            if result.success and result.response:
                self.cache.set(cache_key, result.response)
            
            # Update metrics
            self._update_metrics(result, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Async processing error: {e}")
            self._error_count += 1
            return ProcessingResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def process_batch(self, messages: List[str], context: Optional[Dict] = None) -> List[ProcessingResult]:
        """
        Process multiple messages in batch.
        
        Args:
            messages: List of messages to process
            context: Optional conversation context
            
        Returns:
            List of ProcessingResult objects
        """
        start_time = time.time()
        
        # Use batch processor for efficiency
        results = await self.batch_processor.process_batch(
            messages,
            lambda msg: self.process_async(msg, context)
        )
        
        logger.info(f"Batch processed {len(messages)} messages in {(time.time() - start_time)*1000:.1f}ms")
        return results
    
    def process_parallel(self, messages: List[str], context: Optional[Dict] = None) -> List[ProcessingResult]:
        """
        Process multiple messages in parallel using worker pool.
        
        Args:
            messages: List of messages to process
            context: Optional conversation context
            
        Returns:
            List of ProcessingResult objects
        """
        start_time = time.time()
        
        # Submit tasks to worker pool
        futures = [
            self.worker_pool.submit(self.process, msg, context)
            for msg in messages
        ]
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                results.append(ProcessingResult(
                    success=False,
                    error=str(e)
                ))
        
        logger.info(f"Parallel processed {len(messages)} messages in {(time.time() - start_time)*1000:.1f}ms")
        return results
    
    # ==================== Complex Task Processing ====================
    
    async def process_complex(
        self,
        message: str,
        context: Optional[Dict] = None,
        force_decompose: bool = False,
    ) -> ProcessingResult:
        """
        Process a complex multi-step task using the orchestrator.
        
        Automatically decomposes the task, executes steps against the
        LLM provider (GLM, Kimi, etc.), and aggregates results.
        
        Args:
            message: The complex task description
            context: Optional conversation context
            force_decompose: If True, always decompose even if heuristics say no
            
        Returns:
            ProcessingResult with the aggregated response
        """
        start_time = time.time()
        
        try:
            # Classify first
            profile = self.classifier.classify(message, context)
            decomposer = self.orchestrator.decomposer
            
            # Check if this actually needs multi-step processing
            if not force_decompose and not decomposer.needs_decomposition(message, profile):
                # Simple enough for single-shot — use normal async path
                return await self.process_async(message, context)
            
            # Decompose into plan
            plan = decomposer.decompose(
                message,
                profile=profile,
                max_context_per_step=self.config.max_context,
            )
            
            logger.info(
                f"Complex task decomposed: {len(plan.steps)} steps, "
                f"strategy={plan.strategy.value}"
            )
            
            # Execute the plan
            orch_result = await self.orchestrator.execute_plan(plan)
            
            latency = (time.time() - start_time) * 1000
            
            result = ProcessingResult(
                success=orch_result.success,
                response=orch_result.final_response,
                tokens_used=orch_result.total_tokens,
                latency_ms=latency,
                task_profile=profile,
                error=orch_result.error,
                metadata={
                    "task_type": profile.task_type.value,
                    "complexity": profile.complexity.name,
                    "orchestrated": True,
                    "steps_completed": orch_result.steps_completed,
                    "steps_failed": orch_result.steps_failed,
                    "step_details": orch_result.step_results,
                }
            )
            
            self._update_metrics(result, start_time)
            return result
            
        except Exception as e:
            logger.error(f"Complex processing error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    # ==================== Internal Processing ====================
    
    def _process_internal(
        self,
        message: str,
        context: Optional[Dict],
        profile: TaskProfile,
        allocation
    ) -> ProcessingResult:
        """Internal processing logic using LLM provider (sync wrapper)."""
        start_time = time.time()
        
        # Build LLM request
        messages = [{"role": "user", "content": message}]
        if context and "history" in context:
            messages = context["history"] + messages
        
        request = LLMRequest(
            messages=messages,
            max_tokens=allocation.allocated if allocation else 2048,
            system_prompt=(
                f"You are Hermes Agent. Task type: {profile.task_type.value}, "
                f"complexity: {profile.complexity.name}."
            ),
        )
        
        # Run async provider in a new event loop if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're inside an event loop already — use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                llm_response = pool.submit(
                    asyncio.run, self.provider_manager.complete(request)
                ).result(timeout=30)
        else:
            llm_response = asyncio.run(self.provider_manager.complete(request))
        
        latency = (time.time() - start_time) * 1000
        
        # Emit event
        self.events.publish("task.completed", {
            "task_type": profile.task_type.value,
            "latency_ms": latency,
            "tokens": llm_response.total_tokens,
        }, source="hermes_core")
        
        return ProcessingResult(
            success=True,
            response=llm_response.content,
            tokens_used=llm_response.total_tokens,
            latency_ms=latency,
            task_profile=profile,
            metadata={
                "task_type": profile.task_type.value,
                "complexity": profile.complexity.name,
                "provider": llm_response.provider,
                "model": llm_response.model,
            }
        )
    
    async def _process_internal_async(
        self,
        message: str,
        context: Optional[Dict],
        profile: TaskProfile
    ) -> ProcessingResult:
        """Internal async processing logic using LLM provider."""
        start_time = time.time()
        
        # Build LLM request
        messages = [{"role": "user", "content": message}]
        if context and "history" in context:
            messages = context["history"] + messages
        
        request = LLMRequest(
            messages=messages,
            max_tokens=profile.estimated_tokens if profile else 2048,
            system_prompt=(
                f"You are Hermes Agent. Task type: {profile.task_type.value}, "
                f"complexity: {profile.complexity.name}."
            ),
        )
        
        llm_response = await self.provider_manager.complete(request)
        
        latency = (time.time() - start_time) * 1000
        
        # Emit event
        self.events.publish("task.completed", {
            "task_type": profile.task_type.value,
            "latency_ms": latency,
            "tokens": llm_response.total_tokens,
            "mode": "async",
        }, source="hermes_core")
        
        return ProcessingResult(
            success=True,
            response=llm_response.content,
            tokens_used=llm_response.total_tokens,
            latency_ms=latency,
            task_profile=profile,
            metadata={
                "task_type": profile.task_type.value,
                "mode": "async",
                "provider": llm_response.provider,
                "model": llm_response.model,
            }
        )
    
    def _execute_with_protection(self, func: Callable, *args, **kwargs):
        """Execute function with resilience protection and self-healing."""
        if self.recovery:
            recovery_result = self.recovery.execute_with_retry(
                lambda: func(*args, **kwargs),
                on_failure=self._on_execution_failure
            )
            if recovery_result.success:
                return recovery_result.result
            else:
                raise recovery_result.error or Exception("Recovery failed after all retries")
        else:
            return func(*args, **kwargs)
    
    def _on_execution_failure(self, error: Exception, attempt: int):
        """Handle execution failure."""
        logger.warning(f"Execution failed (attempt {attempt}): {error}")
        self.resilience.record_failure("process")
    
    def _update_metrics(self, result: ProcessingResult, start_time: float):
        """Update processing metrics."""
        self._processed_count += 1
        if not result.success:
            self._error_count += 1
        self._total_latency += (time.time() - start_time) * 1000
        
        if self.metrics:
            self.metrics.record_processing(
                success=result.success,
                latency_ms=result.latency_ms,
                tokens_used=result.tokens_used
            )
    
    # ==================== Session Management ====================
    
    def init_session(self, session_id: str, initial_state: Optional[Dict] = None):
        """Initialize a new session with persistence."""
        self.persistence.init_session(session_id, initial_state)
        logger.info(f"Session initialized: {session_id}")
    
    def save_checkpoint(self):
        """Manually save a checkpoint."""
        return self.persistence.create_checkpoint()
    
    def recover_session(self, session_id: str) -> bool:
        """Recover a session from checkpoint."""
        success, state, replayed = self.persistence.recover(session_id)
        if success:
            logger.info(f"Session recovered: {session_id} ({replayed} deltas)")
        return success
    
    # ==================== Health & Metrics ====================
    
    def get_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "state": self.state.value,
            "resilience": self.resilience.get_metrics(),
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._processed_count),
            "avg_latency_ms": self._total_latency / max(1, self._processed_count),
            "uptime_seconds": time.time() - self._start_time,
            "cache_hit_rate": self.cache.get_hit_rate() if self.cache else 0,
            "worker_pool": self.worker_pool.get_status() if self.worker_pool else None
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics."""
        metrics = {
            "processing": {
                "total": self._processed_count,
                "errors": self._error_count,
                "avg_latency_ms": round(self._total_latency / max(1, self._processed_count), 2)
            },
            "resilience": self.resilience.get_metrics(),
            "context": self.classifier.get_stats(),
            "persistence": self.persistence.get_stats(),
            "cache": self.cache.get_stats() if self.cache else None
        }
        
        if self.metrics:
            metrics["realtime"] = self.metrics.get_summary()
        
        return metrics
    
    # ==================== Lifecycle ====================
    
    def pause(self):
        """Pause processing."""
        self.state = AgentState.PAUSED
        logger.info("Agent paused")
    
    def resume(self):
        """Resume processing."""
        self.state = AgentState.READY
        logger.info("Agent resumed")
    
    def shutdown(self):
        """Gracefully shutdown the agent."""
        self.state = AgentState.SHUTDOWN
        self._shutdown = True
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Shutdown components
        self.worker_pool.shutdown()
        self.async_engine.shutdown()
        self.persistence.shutdown()
        
        logger.info("HermesAgentCore shutdown complete")


# Singleton instance for global access
_global_core: Optional[HermesAgentCore] = None


def get_core() -> Optional[HermesAgentCore]:
    """Get global HermesAgentCore instance."""
    return _global_core


def init_core(config: Optional[AgentConfig] = None) -> HermesAgentCore:
    """Initialize global HermesAgentCore instance."""
    global _global_core
    _global_core = HermesAgentCore(config)
    return _global_core
