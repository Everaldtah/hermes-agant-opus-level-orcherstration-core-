"""
Self-Healing Recovery System
============================

Automatic error recovery with intelligent retry logic.
"""

import time
import logging
import random
from dataclasses import dataclass
from typing import Callable, Any, Optional, List, Dict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    IMMEDIATE = "immediate"           # Retry immediately
    FIXED_DELAY = "fixed_delay"       # Fixed delay between retries
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    LINEAR_BACKOFF = "linear_backoff" # Linear backoff
    JITTERED = "jittered"             # Jittered backoff


@dataclass
class RecoveryResult:
    """Result of recovery attempt."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_duration_ms: float = 0.0
    strategy_used: RecoveryStrategy = RecoveryStrategy.IMMEDIATE


class SelfHealingRecovery:
    """
    Self-healing recovery system with intelligent retry logic.
    
    Features:
    - Multiple retry strategies
    - Circuit breaker integration
    - Error classification
    - Recovery metrics
    - Custom recovery actions
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF,
        max_delay: float = 60.0,
        retryable_errors: Optional[List[type]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = retry_delay
        self.strategy = strategy
        self.max_delay = max_delay
        
        # Error classification
        self._retryable_errors = retryable_errors or [
            ConnectionError,
            TimeoutError,
            Exception  # Fallback
        ]
        
        # Recovery actions
        self._recovery_actions: Dict[type, Callable] = {}
        
        # Metrics
        self._metrics = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "total_retries": 0
        }
        
        self._lock = threading.Lock()
    
    def register_recovery_action(self, error_type: type, action: Callable):
        """Register a custom recovery action for an error type."""
        self._recovery_actions[error_type] = action
    
    def execute_with_retry(
        self,
        fn: Callable,
        on_failure: Optional[Callable[[Exception, int], None]] = None,
        on_success: Optional[Callable[[Any], None]] = None
    ) -> RecoveryResult:
        """
        Execute function with automatic retry.
        
        Args:
            fn: Function to execute
            on_failure: Callback on each failure (error, attempt)
            on_success: Callback on success (result)
            
        Returns:
            RecoveryResult
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            with self._lock:
                self._metrics["total_attempts"] += 1
            
            try:
                result = fn()
                
                # Success
                duration = (time.time() - start_time) * 1000
                
                with self._lock:
                    self._metrics["successful_recoveries"] += 1
                
                if on_success:
                    on_success(result)
                
                return RecoveryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_duration_ms=duration,
                    strategy_used=self.strategy
                )
                
            except Exception as e:
                last_error = e
                
                # Check if error is retryable
                if not self._is_retryable(e):
                    break
                
                # Try custom recovery action
                recovery_action = self._get_recovery_action(e)
                if recovery_action:
                    try:
                        recovery_action(e)
                    except Exception as recovery_error:
                        logger.warning(f"Recovery action failed: {recovery_error}")
                
                # Call failure callback
                if on_failure:
                    on_failure(e, attempt + 1)
                
                # Don't retry on last attempt
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    
                    with self._lock:
                        self._metrics["total_retries"] += 1
        
        # All retries exhausted
        duration = (time.time() - start_time) * 1000
        
        with self._lock:
            self._metrics["failed_recoveries"] += 1
        
        logger.error(f"All {self.max_retries + 1} attempts failed. Last error: {last_error}")
        
        return RecoveryResult(
            success=False,
            error=last_error,
            attempts=self.max_retries + 1,
            total_duration_ms=duration,
            strategy_used=self.strategy
        )
    
    async def execute_with_retry_async(
        self,
        fn: Callable,
        on_failure: Optional[Callable[[Exception, int], None]] = None,
        on_success: Optional[Callable[[Any], None]] = None
    ) -> RecoveryResult:
        """Async version of execute_with_retry."""
        import asyncio
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            with self._lock:
                self._metrics["total_attempts"] += 1
            
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn()
                else:
                    result = fn()
                
                duration = (time.time() - start_time) * 1000
                
                with self._lock:
                    self._metrics["successful_recoveries"] += 1
                
                if on_success:
                    on_success(result)
                
                return RecoveryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_duration_ms=duration,
                    strategy_used=self.strategy
                )
                
            except Exception as e:
                last_error = e
                
                if not self._is_retryable(e):
                    break
                
                if on_failure:
                    on_failure(e, attempt + 1)
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    
                    with self._lock:
                        self._metrics["total_retries"] += 1
        
        duration = (time.time() - start_time) * 1000
        
        with self._lock:
            self._metrics["failed_recoveries"] += 1
        
        return RecoveryResult(
            success=False,
            error=last_error,
            attempts=self.max_retries + 1,
            total_duration_ms=duration,
            strategy_used=self.strategy
        )
    
    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        for error_type in self._retryable_errors:
            if isinstance(error, error_type):
                return True
        return False
    
    def _get_recovery_action(self, error: Exception) -> Optional[Callable]:
        """Get recovery action for error type."""
        for error_type, action in self._recovery_actions.items():
            if isinstance(error, error_type):
                return action
        return None
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.strategy == RecoveryStrategy.IMMEDIATE:
            return 0.0
        
        elif self.strategy == RecoveryStrategy.FIXED_DELAY:
            return self.base_delay
        
        elif self.strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)
        
        elif self.strategy == RecoveryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
            return min(delay, self.max_delay)
        
        elif self.strategy == RecoveryStrategy.JITTERED:
            base = self.base_delay * (2 ** attempt)
            jitter = random.uniform(0, base * 0.5)
            return min(base + jitter, self.max_delay)
        
        return self.base_delay
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            
            total = metrics["successful_recoveries"] + metrics["failed_recoveries"]
            metrics["recovery_rate"] = (
                metrics["successful_recoveries"] / total if total > 0 else 0.0
            )
            
            return metrics


class CircuitBreakerRecovery:
    """
    Recovery system integrated with circuit breaker.
    
    Pauses retries when circuit is open.
    """
    
    def __init__(
        self,
        recovery: SelfHealingRecovery,
        circuit_breaker
    ):
        self.recovery = recovery
        self.circuit_breaker = circuit_breaker
    
    def execute(self, fn: Callable) -> RecoveryResult:
        """Execute with circuit breaker awareness."""
        # Check circuit state
        if hasattr(self.circuit_breaker, 'can_execute'):
            if not self.circuit_breaker.can_execute():
                return RecoveryResult(
                    success=False,
                    error=Exception("Circuit breaker is OPEN"),
                    attempts=0
                )
        
        # Execute with retry
        result = self.recovery.execute_with_retry(fn)
        
        # Update circuit breaker
        if result.success:
            if hasattr(self.circuit_breaker, 'record_success'):
                self.circuit_breaker.record_success()
        else:
            if hasattr(self.circuit_breaker, 'record_failure'):
                self.circuit_breaker.record_failure()
        
        return result
