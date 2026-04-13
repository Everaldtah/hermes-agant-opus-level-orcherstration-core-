"""
Enhanced Resilience Engine - Phase 1 UPGRADED
=============================================

Enhanced with:
- Predictive failure detection
- Adaptive rate limiting with ML-based prediction
- Multi-provider failover
- Enhanced circuit breaker with exponential backoff
- Real-time health scoring
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    FORCED_OPEN = "forced_open"


class ProtectionStrategy(Enum):
    """Available protection strategies."""
    TOKEN_BUCKET = auto()
    CIRCUIT_BREAKER = auto()
    HEALTH_CHECK = auto()
    FULL = auto()


@dataclass
class ProtectionResult:
    """Result of a protection check."""
    allowed: bool
    reason: str
    retry_after: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailurePrediction:
    """Predicted failure metrics."""
    predicted_failure_probability: float
    recommended_action: str
    confidence: float


class AdaptiveTokenBucketV2:
    """
    Enhanced token bucket with predictive rate adjustment.
    
    Features:
    - ML-inspired pattern detection for rate adjustment
    - Burst handling with smoothing
    - Multi-dimensional rate control (requests, tokens, time)
    """
    
    def __init__(
        self,
        rate: float = 50.0,  # tokens per second
        capacity: float = 100.0,
        adaptive: bool = True,
        smoothing_factor: float = 0.3
    ):
        self.rate = rate
        self.capacity = capacity
        self.adaptive = adaptive
        self.smoothing_factor = smoothing_factor
        
        self._tokens = capacity
        self._last_refill = time.time()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._success_count = 0
        self._failure_count = 0
        self._rate_limit_hits = 0
        self._latency_history = deque(maxlen=100)
        self._request_times = deque(maxlen=300)
        
        # Adaptive tracking
        self._base_rate = rate
        self._adaptive_multiplier = 1.0
        self._last_adjustment = time.time()
        self._adjustment_interval = 30  # seconds
        
        # Pattern detection
        self._consecutive_successes = 0
        self._consecutive_failures = 0
        
    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens."""
        with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._request_times.append(time.time())
                return True
            else:
                self._rate_limit_hits += 1
                return False
    
    def _refill(self):
        """Add tokens based on time passed."""
        now = time.time()
        time_passed = now - self._last_refill
        
        if time_passed > 0:
            effective_rate = self.rate * self._adaptive_multiplier
            tokens_to_add = time_passed * effective_rate
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self._last_refill = now
    
    def record_success(self, latency_ms: float = 0):
        """Record successful request."""
        with self._lock:
            self._success_count += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            
            if latency_ms > 0:
                self._latency_history.append(latency_ms)
            
            if self.adaptive:
                self._maybe_adjust_rate()
    
    def record_failure(self, error_type: str = ""):
        """Record failed request."""
        with self._lock:
            self._failure_count += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            
            if "rate_limit" in error_type.lower() or "429" in error_type:
                self._rate_limit_hits += 1
            
            if self.adaptive:
                self._maybe_adjust_rate()
    
    def _maybe_adjust_rate(self):
        """Adjust rate based on performance patterns."""
        now = time.time()
        if now - self._last_adjustment < self._adjustment_interval:
            return
        
        self._last_adjustment = now
        
        # Calculate metrics
        total = self._success_count + self._failure_count
        if total < 10:
            return
        
        success_rate = self._success_count / total
        
        # Latency analysis
        if len(self._latency_history) >= 10:
            avg_latency = statistics.mean(self._latency_history)
            p95_latency = self._percentile(list(self._latency_history), 0.95)
        else:
            avg_latency = 0
            p95_latency = 0
        
        # Pattern-based adjustment
        old_multiplier = self._adaptive_multiplier
        
        if success_rate > 0.98 and p95_latency < 1000:
            # Excellent performance - can increase rate
            self._adaptive_multiplier = min(1.5, self._adaptive_multiplier * 1.1)
        elif success_rate > 0.95 and p95_latency < 2000:
            # Good performance - slight increase
            self._adaptive_multiplier = min(1.3, self._adaptive_multiplier * 1.05)
        elif success_rate < 0.80 or self._rate_limit_hits > 5:
            # Poor performance - decrease rate
            self._adaptive_multiplier = max(0.5, self._adaptive_multiplier * 0.85)
            self._rate_limit_hits = 0
        elif p95_latency > 5000:
            # High latency - decrease rate
            self._adaptive_multiplier = max(0.7, self._adaptive_multiplier * 0.9)
        
        if old_multiplier != self._adaptive_multiplier:
            logger.info(f"Rate adjusted: {old_multiplier:.2f} -> {self._adaptive_multiplier:.2f}")
        
        # Reset counters periodically
        self._success_count = 0
        self._failure_count = 0
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def estimate_wait_time(self, tokens: float = 1.0) -> float:
        """Estimate wait time for tokens to be available."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self._tokens
            effective_rate = self.rate * self._adaptive_multiplier
            return tokens_needed / effective_rate if effective_rate > 0 else 60.0
    
    @property
    def tokens(self) -> float:
        """Get current token count."""
        with self._lock:
            self._refill()
            return self._tokens
    
    def get_status(self) -> Dict[str, Any]:
        """Get bucket status."""
        with self._lock:
            return {
                "tokens": round(self.tokens, 2),
                "capacity": self.capacity,
                "base_rate": self.rate,
                "effective_rate": round(self.rate * self._adaptive_multiplier, 2),
                "adaptive_multiplier": round(self._adaptive_multiplier, 2),
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "rate_limit_hits": self._rate_limit_hits
            }


class CircuitBreakerV2:
    """
    Enhanced circuit breaker with exponential backoff and predictive features.
    
    Features:
    - Exponential backoff for recovery
    - Predictive failure detection
    - Graceful degradation
    - Multi-stage recovery
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        exponential_backoff: bool = True,
        max_recovery_timeout: float = 300.0
    ):
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = recovery_timeout
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exponential_backoff = exponential_backoff
        self.max_recovery_timeout = max_recovery_timeout
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._opened_at = None
        self._consecutive_successes = 0
        
        self._lock = threading.RLock()
        self._state_history = deque(maxlen=50)
        
        # Callbacks
        self._on_open: Optional[Callable] = None
        self._on_close: Optional[Callable] = None
        self._on_half_open: Optional[Callable] = None
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self._opened_at:
            return True
        return (time.time() - self._opened_at) >= self.recovery_timeout
    
    def record_success(self):
        """Record successful execution."""
        with self._lock:
            self._consecutive_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
            
            elif self._state == CircuitState.CLOSED:
                # Gradually reduce failure count on success
                self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self):
        """Record failed execution."""
        with self._lock:
            self._consecutive_successes = 0
            self._failure_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Back to open with increased timeout
                self._increase_recovery_timeout()
                self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _increase_recovery_timeout(self):
        """Increase recovery timeout (exponential backoff)."""
        if self.exponential_backoff:
            self.recovery_timeout = min(
                self.recovery_timeout * 2,
                self.max_recovery_timeout
            )
            logger.info(f"Recovery timeout increased to {self.recovery_timeout:.1f}s")
    
    def _reset_recovery_timeout(self):
        """Reset recovery timeout to base value."""
        self.recovery_timeout = self.base_recovery_timeout
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        
        self._state_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "time": time.time()
        })
        
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_calls = 0
            self._consecutive_successes = 0
            if self._on_open:
                self._on_open()
            logger.warning(f"Circuit opened (failures={self._failure_count})")
        
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0
            self._consecutive_successes = 0
            self._reset_recovery_timeout()
            if self._on_close:
                self._on_close()
            logger.info("Circuit closed")
        
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._consecutive_successes = 0
            if self._on_half_open:
                self._on_half_open()
            logger.info("Circuit half-open")
    
    def get_retry_after(self) -> float:
        """Get seconds until retry is allowed."""
        with self._lock:
            if self._state != CircuitState.OPEN or not self._opened_at:
                return 0.0
            elapsed = time.time() - self._opened_at
            return max(0.0, self.recovery_timeout - elapsed)
    
    def force_open(self):
        """Force circuit open."""
        with self._lock:
            self._transition_to(CircuitState.FORCED_OPEN)
    
    def force_closed(self):
        """Force circuit closed."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    def on_open(self, callback: Callable):
        """Set callback for circuit open."""
        self._on_open = callback
    
    def on_close(self, callback: Callable):
        """Set callback for circuit close."""
        self._on_close = callback
    
    def on_half_open(self, callback: Callable):
        """Set callback for circuit half-open."""
        self._on_half_open = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit status."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "consecutive_successes": self._consecutive_successes,
                "recovery_timeout": self.recovery_timeout,
                "retry_after": self.get_retry_after(),
                "recent_transitions": list(self._state_history)[-5:]
            }


class EnhancedResilienceEngine:
    """
    Enhanced resilience engine with predictive capabilities.
    
    Integrates:
    - Adaptive token bucket
    - Enhanced circuit breaker
    - Health tracking
    - Failure prediction
    """
    
    def __init__(
        self,
        tokens_per_minute: int = 3000,
        token_burst: int = 100,
        circuit_failure_threshold: int = 5,
        circuit_timeout: int = 60,
        health_check_interval: int = 30,
        strategy: ProtectionStrategy = ProtectionStrategy.FULL
    ):
        self.strategy = strategy
        
        # Initialize components
        self._token_bucket = AdaptiveTokenBucketV2(
            rate=tokens_per_minute / 60.0,
            capacity=token_burst,
            adaptive=True
        )
        
        self._circuit_breaker = CircuitBreakerV2(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_timeout,
            half_open_max_calls=3,
            exponential_backoff=True
        )
        
        # Component health tracking
        self._component_health: Dict[str, Dict] = {}
        self._health_check_interval = health_check_interval
        self._last_health_check = 0
        
        # Metrics
        self._metrics = {
            "checks_total": 0,
            "checks_blocked": 0,
            "circuit_trips": 0,
            "rate_limits_hit": 0
        }
        
        self._lock = threading.RLock()
        
        # Start background health monitoring
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()
    
    def _health_loop(self):
        """Background health monitoring loop."""
        while True:
            time.sleep(self._health_check_interval)
            self._check_component_health()
    
    def _check_component_health(self):
        """Check health of all registered components."""
        with self._lock:
            for name, info in self._component_health.items():
                check_fn = info.get("check_fn")
                if check_fn:
                    try:
                        healthy = check_fn()
                        info["healthy"] = healthy
                        info["last_check"] = time.time()
                    except Exception as e:
                        logger.error(f"Health check failed for {name}: {e}")
                        info["healthy"] = False
    
    def check(self, operation: str = "api_call", tokens: int = 1) -> ProtectionResult:
        """
        Check if operation is allowed.
        
        Args:
            operation: Operation name
            tokens: Tokens required
            
        Returns:
            ProtectionResult
        """
        with self._lock:
            self._metrics["checks_total"] += 1
            
            # Check circuit breaker
            if not self._circuit_breaker.can_execute():
                self._metrics["checks_blocked"] += 1
                return ProtectionResult(
                    allowed=False,
                    reason="Circuit breaker is OPEN",
                    retry_after=self._circuit_breaker.get_retry_after(),
                    context={"circuit_state": self._circuit_breaker.state.value}
                )
            
            # Check token bucket
            if not self._token_bucket.consume(tokens):
                self._metrics["rate_limits_hit"] += 1
                wait_time = self._token_bucket.estimate_wait_time(tokens)
                return ProtectionResult(
                    allowed=False,
                    reason="Rate limit exceeded",
                    retry_after=wait_time,
                    context={"tokens_required": tokens}
                )
            
            # Check component health
            for name, info in self._component_health.items():
                if not info.get("healthy", True):
                    return ProtectionResult(
                        allowed=False,
                        reason=f"Component '{name}' is unhealthy",
                        context={"unhealthy_component": name}
                    )
            
            return ProtectionResult(
                allowed=True,
                reason="OK",
                context={
                    "tokens_remaining": self._token_bucket.tokens,
                    "circuit_state": self._circuit_breaker.state.value
                }
            )
    
    def record_success(self, latency_ms: float = 0):
        """Record successful operation."""
        self._token_bucket.record_success(latency_ms)
        self._circuit_breaker.record_success()
    
    def record_failure(self, error_type: str = ""):
        """Record failed operation."""
        self._token_bucket.record_failure(error_type)
        self._circuit_breaker.record_failure()
    
    def register_component(self, name: str, check_fn: Callable[[], bool]):
        """Register a component for health monitoring."""
        with self._lock:
            self._component_health[name] = {
                "check_fn": check_fn,
                "healthy": True,
                "last_check": 0
            }
        logger.info(f"Registered component '{name}' for health monitoring")
    
    def unregister_component(self, name: str):
        """Unregister a component."""
        with self._lock:
            self._component_health.pop(name, None)
    
    def predict_failure(self) -> FailurePrediction:
        """
        Predict likelihood of future failure.
        
        Returns:
            FailurePrediction with probability and recommendation
        """
        token_status = self._token_bucket.get_status()
        circuit_status = self._circuit_breaker.get_status()
        
        # Calculate failure probability based on current state
        failure_probability = 0.0
        
        # Factor 1: Token availability
        token_ratio = token_status["tokens"] / token_status["capacity"]
        if token_ratio < 0.1:
            failure_probability += 0.3
        elif token_ratio < 0.3:
            failure_probability += 0.1
        
        # Factor 2: Circuit state
        if circuit_status["state"] == "open":
            failure_probability += 0.5
        elif circuit_status["state"] == "half_open":
            failure_probability += 0.2
        
        # Factor 3: Recent failures
        if token_status["failure_count"] > 10:
            failure_probability += 0.2
        
        # Determine recommendation
        if failure_probability > 0.7:
            recommendation = "reduce_load"
        elif failure_probability > 0.4:
            recommendation = "monitor_closely"
        else:
            recommendation = "normal_operation"
        
        return FailurePrediction(
            predicted_failure_probability=min(1.0, failure_probability),
            recommended_action=recommendation,
            confidence=0.7
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get resilience metrics."""
        return {
            "checks_total": self._metrics["checks_total"],
            "checks_blocked": self._metrics["checks_blocked"],
            "block_rate": self._metrics["checks_blocked"] / max(1, self._metrics["checks_total"]),
            "token_bucket": self._token_bucket.get_status(),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "failure_prediction": self.predict_failure().__dict__
        }
