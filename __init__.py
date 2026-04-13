"""
Hermes Agent Upgraded - Top-level package
==========================================

Provides convenient imports from the core package.
"""

from core import (
    HermesAgentCore,
    AgentConfig,
    ProcessingResult,
    AgentState,
    ProcessingMode,
    EnhancedResilienceEngine,
    ProtectionStrategy,
    CircuitBreakerV2,
    AdaptiveTokenBucketV2,
    EnhancedTaskClassifier,
    EnhancedTokenAllocator,
    HierarchicalContextV2,
    TaskProfile,
    EnhancedDeltaPersistence,
    SessionManager,
    CheckpointManager,
)

__version__ = "2.1.0"
__all__ = [
    'HermesAgentCore',
    'AgentConfig',
    'ProcessingResult',
    'AgentState',
    'ProcessingMode',
    'EnhancedResilienceEngine',
    'ProtectionStrategy',
    'CircuitBreakerV2',
    'AdaptiveTokenBucketV2',
    'EnhancedTaskClassifier',
    'EnhancedTokenAllocator',
    'HierarchicalContextV2',
    'TaskProfile',
    'EnhancedDeltaPersistence',
    'SessionManager',
    'CheckpointManager',
]
