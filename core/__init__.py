"""
Hermes Agent Orchestration & Brain Core - UPGRADED VERSION
===========================================================

A comprehensive 4-phase implementation with enhanced processing power 
and efficiency modules for distributed agent swarms.

Upgrades Included:
- Async/Parallel Processing Engine
- Intelligent Worker Pool with Dynamic Scaling
- Multi-Level Caching System (LRU + Time-based + Predictive)
- Smart Load Balancer with Predictive Allocation
- Real-time Metrics & Monitoring Dashboard
- Self-Healing Error Recovery
- Distributed Coordination for Swarm Mode
- Batch Processing Optimization
- Connection Pool Management

Usage:
    from hermes_agent_upgraded import HermesAgentCore
    
    agent = HermesAgentCore(
        tokens_per_minute=3000,
        max_context=16000,
        enable_swarm_mode=True
    )
    
    # Process with full resilience protection
    result = await agent.process_async(message, context)
"""

__version__ = "2.0.0-upgraded"
__author__ = "Hermes Agent Team - Enhanced by AI"

from .hermes_core import (
    HermesAgentCore,
    AgentConfig,
    ProcessingResult,
    AgentState,
    ProcessingMode
)

from .resilience_enhanced import (
    EnhancedResilienceEngine,
    ProtectionStrategy,
    CircuitBreakerV2,
    AdaptiveTokenBucketV2
)

from .context_enhanced import (
    EnhancedTaskClassifier,
    EnhancedTokenAllocator,
    HierarchicalContextV2,
    TaskProfile
)

from .persistence_enhanced import (
    EnhancedDeltaPersistence,
    SessionManager,
    CheckpointManager
)

from .llm_provider import (
    LLMProvider,
    MockProvider,
    OpenRouterProvider,
    ProviderManager,
    LLMRequest,
    LLMResponse,
    ProviderType,
)

from .event_bus import EventBus, get_event_bus

from .task_orchestrator import (
    TaskOrchestrator,
    TaskDecomposer,
    TaskPlan,
    TaskStep,
    OrchestratorResult,
    ExecutionStrategy,
)

__all__ = [
    # Core
    'HermesAgentCore',
    'AgentConfig',
    'ProcessingResult',
    'AgentState',
    'ProcessingMode',
    
    # Resilience
    'EnhancedResilienceEngine',
    'ProtectionStrategy',
    'CircuitBreakerV2',
    'AdaptiveTokenBucketV2',
    
    # Context
    'EnhancedTaskClassifier',
    'EnhancedTokenAllocator',
    'HierarchicalContextV2',
    'TaskProfile',
    
    # Persistence
    'EnhancedDeltaPersistence',
    'SessionManager',
    'CheckpointManager',
    
    # LLM Providers
    'LLMProvider',
    'MockProvider',
    'OpenRouterProvider',
    'ProviderManager',
    'LLMRequest',
    'LLMResponse',
    'ProviderType',
    
    # Event Bus
    'EventBus',
    'get_event_bus',
]
