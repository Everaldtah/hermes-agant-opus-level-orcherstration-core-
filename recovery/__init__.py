"""
Recovery Modules
================

Self-healing and error recovery systems.

Modules:
- self_healing: Automatic error recovery with retry logic
"""

from .self_healing import SelfHealingRecovery, RecoveryResult

__all__ = [
    'SelfHealingRecovery',
    'RecoveryResult',
]
