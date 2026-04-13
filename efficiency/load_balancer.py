"""
Smart Load Balancer
===================

Intelligent load distribution with health-aware routing.
"""

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)


class LoadStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    HEALTH_BASED = "health_based"
    PREDICTIVE = "predictive"


@dataclass
class LoadMetrics:
    """Metrics for a load-balanced resource."""
    resource_id: str
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_used: float = field(default_factory=time.time)
    healthy: bool = True
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)."""
        if not self.healthy:
            return float('inf')
        
        # Factor in connections, latency, and success rate
        connection_factor = self.active_connections / 10.0
        latency_factor = self.avg_latency_ms / 1000.0
        success_factor = 1.0 - self.success_rate
        
        return (connection_factor + latency_factor + success_factor) / self.weight


class SmartLoadBalancer:
    """
    Smart load balancer with multiple strategies.
    
    Features:
    - Multiple load balancing strategies
    - Health-aware routing
    - Dynamic weight adjustment
    - Performance tracking
    - Automatic failover
    """
    
    def __init__(
        self,
        strategy: LoadStrategy = LoadStrategy.HEALTH_BASED,
        health_check_interval: float = 30.0
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        self._resources: Dict[str, LoadMetrics] = {}
        self._round_robin_index = 0
        
        # Health check callbacks
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        
        # Metrics history
        self._metrics_history: Dict[str, deque] = {}
        
        self._lock = threading.RLock()
        
        # Start health check thread
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()
    
    def _health_loop(self):
        """Periodic health check loop."""
        while True:
            time.sleep(self.health_check_interval)
            self._run_health_checks()
    
    def _run_health_checks(self):
        """Run health checks for all resources."""
        with self._lock:
            for resource_id, check_fn in self._health_checks.items():
                if resource_id in self._resources:
                    try:
                        healthy = check_fn()
                        self._resources[resource_id].healthy = healthy
                    except Exception as e:
                        logger.error(f"Health check failed for {resource_id}: {e}")
                        self._resources[resource_id].healthy = False
    
    def register(
        self,
        resource_id: str,
        weight: float = 1.0,
        health_check: Optional[Callable[[], bool]] = None
    ):
        """
        Register a resource for load balancing.
        
        Args:
            resource_id: Unique resource identifier
            weight: Resource weight for weighted routing
            health_check: Optional health check function
        """
        with self._lock:
            self._resources[resource_id] = LoadMetrics(
                resource_id=resource_id,
                weight=weight
            )
            
            if health_check:
                self._health_checks[resource_id] = health_check
            
            self._metrics_history[resource_id] = deque(maxlen=100)
        
        logger.info(f"Registered resource: {resource_id} (weight={weight})")
    
    def unregister(self, resource_id: str):
        """Unregister a resource."""
        with self._lock:
            self._resources.pop(resource_id, None)
            self._health_checks.pop(resource_id, None)
            self._metrics_history.pop(resource_id, None)
    
    def select(self) -> Optional[str]:
        """
        Select a resource based on current strategy.
        
        Returns:
            Selected resource ID or None if no healthy resources
        """
        with self._lock:
            healthy_resources = {
                rid: metrics for rid, metrics in self._resources.items()
                if metrics.healthy
            }
            
            if not healthy_resources:
                return None
            
            if self.strategy == LoadStrategy.ROUND_ROBIN:
                return self._round_robin_select(healthy_resources)
            elif self.strategy == LoadStrategy.RANDOM:
                return self._random_select(healthy_resources)
            elif self.strategy == LoadStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_resources)
            elif self.strategy == LoadStrategy.WEIGHTED:
                return self._weighted_select(healthy_resources)
            elif self.strategy == LoadStrategy.HEALTH_BASED:
                return self._health_based_select(healthy_resources)
            elif self.strategy == LoadStrategy.PREDICTIVE:
                return self._predictive_select(healthy_resources)
            else:
                return self._round_robin_select(healthy_resources)
    
    def _round_robin_select(self, resources: Dict[str, LoadMetrics]) -> str:
        """Round-robin selection."""
        resource_ids = list(resources.keys())
        
        # Find next healthy resource
        for _ in range(len(resource_ids)):
            self._round_robin_index = (self._round_robin_index + 1) % len(resource_ids)
            selected = resource_ids[self._round_robin_index]
            if selected in resources:
                resources[selected].last_used = time.time()
                return selected
        
        return resource_ids[0] if resource_ids else None
    
    def _random_select(self, resources: Dict[str, LoadMetrics]) -> str:
        """Random selection."""
        selected = random.choice(list(resources.keys()))
        resources[selected].last_used = time.time()
        return selected
    
    def _least_connections_select(self, resources: Dict[str, LoadMetrics]) -> str:
        """Select resource with least active connections."""
        selected = min(resources.items(), key=lambda x: x[1].active_connections)[0]
        resources[selected].last_used = time.time()
        return selected
    
    def _weighted_select(self, resources: Dict[str, LoadMetrics]) -> str:
        """Weighted random selection."""
        total_weight = sum(m.weight for m in resources.values())
        rand = random.uniform(0, total_weight)
        
        cumulative = 0
        for rid, metrics in resources.items():
            cumulative += metrics.weight
            if rand <= cumulative:
                resources[rid].last_used = time.time()
                return rid
        
        return list(resources.keys())[-1] if resources else None
    
    def _health_based_select(self, resources: Dict[str, LoadMetrics]) -> str:
        """Select based on health score."""
        # Sort by load score (lower is better)
        sorted_resources = sorted(
            resources.items(),
            key=lambda x: x[1].load_score
        )
        
        # Select best, with some randomization for load distribution
        if len(sorted_resources) > 1 and random.random() < 0.2:
            # 20% chance to select second best
            selected = sorted_resources[1][0]
        else:
            selected = sorted_resources[0][0]
        
        resources[selected].last_used = time.time()
        return selected
    
    def _predictive_select(self, resources: Dict[str, LoadMetrics]) -> str:
        """Predictive selection based on historical performance."""
        # Use health-based selection with predictive adjustments
        best_score = float('inf')
        best_resource = None
        
        for rid, metrics in resources.items():
            score = metrics.load_score
            
            # Adjust based on historical performance
            if rid in self._metrics_history:
                history = self._metrics_history[rid]
                if len(history) >= 5:
                    recent_avg = sum(h["latency"] for h in history[-5:]) / 5
                    score *= (1 + recent_avg / 1000)  # Penalize high recent latency
            
            if score < best_score:
                best_score = score
                best_resource = rid
        
        if best_resource:
            resources[best_resource].last_used = time.time()
        
        return best_resource
    
    def record_request_start(self, resource_id: str):
        """Record start of request to resource."""
        with self._lock:
            if resource_id in self._resources:
                self._resources[resource_id].active_connections += 1
                self._resources[resource_id].total_requests += 1
    
    def record_request_end(
        self,
        resource_id: str,
        success: bool = True,
        latency_ms: float = 0.0
    ):
        """Record end of request to resource."""
        with self._lock:
            if resource_id in self._resources:
                metrics = self._resources[resource_id]
                metrics.active_connections = max(0, metrics.active_connections - 1)
                
                if not success:
                    metrics.failed_requests += 1
                
                # Update average latency
                if metrics.avg_latency_ms == 0:
                    metrics.avg_latency_ms = latency_ms
                else:
                    metrics.avg_latency_ms = (
                        0.9 * metrics.avg_latency_ms + 0.1 * latency_ms
                    )
                
                # Record in history
                if resource_id in self._metrics_history:
                    self._metrics_history[resource_id].append({
                        "success": success,
                        "latency": latency_ms,
                        "time": time.time()
                    })
    
    def set_resource_weight(self, resource_id: str, weight: float):
        """Update resource weight."""
        with self._lock:
            if resource_id in self._resources:
                self._resources[resource_id].weight = weight
    
    def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific resource."""
        with self._lock:
            if resource_id in self._resources:
                metrics = self._resources[resource_id]
                return {
                    "resource_id": resource_id,
                    "healthy": metrics.healthy,
                    "weight": metrics.weight,
                    "active_connections": metrics.active_connections,
                    "total_requests": metrics.total_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": round(metrics.success_rate, 4),
                    "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                    "load_score": round(metrics.load_score, 4)
                }
            return None
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all resources."""
        with self._lock:
            return {
                rid: self.get_resource_status(rid)
                for rid in self._resources.keys()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_requests = sum(
                m.total_requests for m in self._resources.values()
            )
            total_failed = sum(
                m.failed_requests for m in self._resources.values()
            )
            
            return {
                "strategy": self.strategy.value,
                "total_resources": len(self._resources),
                "healthy_resources": sum(
                    1 for m in self._resources.values() if m.healthy
                ),
                "total_requests": total_requests,
                "total_failed": total_failed,
                "overall_success_rate": (
                    (total_requests - total_failed) / total_requests
                    if total_requests > 0 else 1.0
                )
            }


class AdaptiveLoadBalancer(SmartLoadBalancer):
    """
    Load balancer with adaptive strategy selection.
    
    Automatically selects best strategy based on conditions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._adaptive_enabled = True
        self._strategy_performance: Dict[LoadStrategy, List[float]] = {
            strategy: [] for strategy in LoadStrategy
        }
    
    def select(self) -> Optional[str]:
        """Select resource with adaptive strategy."""
        if self._adaptive_enabled:
            self._adapt_strategy()
        
        return super().select()
    
    def _adapt_strategy(self):
        """Adapt strategy based on current conditions."""
        with self._lock:
            healthy_count = sum(
                1 for m in self._resources.values() if m.healthy
            )
            
            if healthy_count == 0:
                return
            
            # Select strategy based on conditions
            if healthy_count == 1:
                # Only one resource - use round robin (trivial)
                self.strategy = LoadStrategy.ROUND_ROBIN
            elif healthy_count <= 3:
                # Few resources - use health-based
                self.strategy = LoadStrategy.HEALTH_BASED
            else:
                # Many resources - use predictive
                self.strategy = LoadStrategy.PREDICTIVE
