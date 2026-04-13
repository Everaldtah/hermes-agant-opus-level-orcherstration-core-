"""
Metrics Collector
=================

Real-time metrics collection and aggregation.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import threading
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: float
    processing_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    tokens_used: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.processing_count > 0:
            return self.total_latency_ms / self.processing_count
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processing_count > 0:
            return self.success_count / self.processing_count
        return 1.0


class MetricsCollector:
    """
    Real-time metrics collector.
    
    Features:
    - Real-time metric recording
    - Historical data retention
    - Statistical analysis
    - Alert thresholds
    - Export capabilities
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        snapshot_interval: float = 60.0,
        enable_alerts: bool = True
    ):
        self.history_size = history_size
        self.snapshot_interval = snapshot_interval
        self.enable_alerts = enable_alerts
        
        # Current metrics
        self._current = MetricsSnapshot(timestamp=time.time())
        
        # Historical data
        self._history: deque = deque(maxlen=history_size)
        self._snapshots: deque = deque(maxlen=100)
        
        # Latency distribution
        self._latencies: deque = deque(maxlen=1000)
        
        # Alert thresholds
        self._thresholds = {
            "error_rate": 0.1,
            "avg_latency_ms": 5000,
            "token_usage_rate": 0.9
        }
        
        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
        
        self._lock = threading.RLock()
        
        # Start snapshot thread
        self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._snapshot_thread.start()
    
    def _snapshot_loop(self):
        """Periodic snapshot creation loop."""
        while True:
            time.sleep(self.snapshot_interval)
            self._create_snapshot()
    
    def _create_snapshot(self):
        """Create a metrics snapshot."""
        with self._lock:
            snapshot = MetricsSnapshot(
                timestamp=time.time(),
                processing_count=self._current.processing_count,
                success_count=self._current.success_count,
                error_count=self._current.error_count,
                total_latency_ms=self._current.total_latency_ms,
                tokens_used=self._current.tokens_used
            )
            
            self._snapshots.append(snapshot)
            
            # Check alerts
            if self.enable_alerts:
                self._check_alerts(snapshot)
    
    def _check_alerts(self, snapshot: MetricsSnapshot):
        """Check if any alert thresholds are breached."""
        alerts = []
        
        if snapshot.success_rate < (1 - self._thresholds["error_rate"]):
            alerts.append(f"High error rate: {(1 - snapshot.success_rate) * 100:.1f}%")
        
        if snapshot.avg_latency_ms > self._thresholds["avg_latency_ms"]:
            alerts.append(f"High latency: {snapshot.avg_latency_ms:.0f}ms")
        
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def record_processing(
        self,
        success: bool,
        latency_ms: float,
        tokens_used: int = 0
    ):
        """Record a processing event."""
        with self._lock:
            self._current.processing_count += 1
            
            if success:
                self._current.success_count += 1
            else:
                self._current.error_count += 1
            
            self._current.total_latency_ms += latency_ms
            self._current.tokens_used += tokens_used
            
            self._latencies.append(latency_ms)
    
    def record_custom(self, metric_name: str, value: Any):
        """Record a custom metric."""
        with self._lock:
            if not hasattr(self._current, metric_name):
                setattr(self._current, metric_name, [])
            
            attr = getattr(self._current, metric_name)
            if isinstance(attr, list):
                attr.append(value)
            else:
                setattr(self._current, metric_name, value)
    
    def get_current(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                "timestamp": self._current.timestamp,
                "processing_count": self._current.processing_count,
                "success_count": self._current.success_count,
                "error_count": self._current.error_count,
                "avg_latency_ms": self._current.avg_latency_ms,
                "success_rate": self._current.success_rate,
                "tokens_used": self._current.tokens_used
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            current = self.get_current()
            
            # Latency statistics
            if self._latencies:
                latencies_list = list(self._latencies)
                latency_stats = {
                    "min_ms": min(latencies_list),
                    "max_ms": max(latencies_list),
                    "avg_ms": statistics.mean(latencies_list),
                    "p50_ms": self._percentile(latencies_list, 0.5),
                    "p95_ms": self._percentile(latencies_list, 0.95),
                    "p99_ms": self._percentile(latencies_list, 0.99)
                }
            else:
                latency_stats = {}
            
            return {
                "current": current,
                "latency_stats": latency_stats,
                "snapshot_count": len(self._snapshots)
            }
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def get_historical(
        self,
        metric_name: str,
        duration_seconds: float = 3600
    ) -> List[Any]:
        """Get historical values for a metric."""
        with self._lock:
            cutoff = time.time() - duration_seconds
            
            values = []
            for snapshot in self._snapshots:
                if snapshot.timestamp >= cutoff:
                    if hasattr(snapshot, metric_name):
                        values.append(getattr(snapshot, metric_name))
            
            return values
    
    def set_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold."""
        self._thresholds[metric_name] = threshold
    
    def on_alert(self, callback: Callable[[str], None]):
        """Register alert callback."""
        self._alert_callbacks.append(callback)
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external systems."""
        with self._lock:
            return {
                "current": self.get_current(),
                "summary": self.get_summary(),
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "processing_count": s.processing_count,
                        "success_rate": s.success_rate,
                        "avg_latency_ms": s.avg_latency_ms
                    }
                    for s in list(self._snapshots)[-10:]
                ]
            }


class PrometheusExporter:
    """
    Prometheus-compatible metrics exporter.
    
    Formats metrics for Prometheus scraping.
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def export(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.collector.get_current()
        
        lines = []
        
        # Processing count
        lines.append(f"# HELP hermes_processing_total Total processing count")
        lines.append(f"# TYPE hermes_processing_total counter")
        lines.append(f"hermes_processing_total {metrics['processing_count']}")
        
        # Success rate
        lines.append(f"# HELP hermes_success_rate Success rate")
        lines.append(f"# TYPE hermes_success_rate gauge")
        lines.append(f"hermes_success_rate {metrics['success_rate']}")
        
        # Latency
        lines.append(f"# HELP hermes_avg_latency_ms Average latency in ms")
        lines.append(f"# TYPE hermes_avg_latency_ms gauge")
        lines.append(f"hermes_avg_latency_ms {metrics['avg_latency_ms']}")
        
        return "\n".join(lines)
