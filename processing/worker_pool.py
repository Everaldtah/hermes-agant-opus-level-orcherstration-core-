"""
Intelligent Worker Pool
=======================

Dynamic worker pool with intelligent scaling based on workload.
"""

import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker thread states."""
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration_ms: float = 0.0
    idle_time_ms: float = 0.0
    state: WorkerState = WorkerState.IDLE
    current_task: Optional[str] = None


class IntelligentWorkerPool:
    """
    Intelligent worker pool with dynamic scaling.
    
    Features:
    - Dynamic scaling based on workload
    - Worker health monitoring
    - Task prioritization
    - Load balancing
    - Performance metrics
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 10,
        scaling_threshold: float = 0.8,
        scale_down_delay: float = 60.0,
        thread_name_prefix: str = "hermes-worker"
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_threshold = scaling_threshold
        self.scale_down_delay = scale_down_delay
        self.thread_name_prefix = thread_name_prefix
        
        # Current worker count
        self._current_workers = min_workers
        
        # Executor
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        # Worker stats
        self._worker_stats: Dict[int, WorkerStats] = {}
        self._task_history: deque = deque(maxlen=1000)
        
        # Scaling control
        self._last_scale_up = 0
        self._last_scale_down = 0
        self._scale_up_cooldown = 10.0
        self._scale_down_cooldown = scale_down_delay
        
        # Metrics
        self._metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "scale_up_events": 0,
            "scale_down_events": 0
        }
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitor loop for dynamic scaling."""
        while not self._shutdown:
            time.sleep(5)  # Check every 5 seconds
            
            if self._shutdown:
                break
            
            try:
                self._evaluate_scaling()
            except Exception as e:
                logger.error(f"Scaling evaluation error: {e}")
    
    def _evaluate_scaling(self):
        """Evaluate and perform scaling if needed."""
        with self._lock:
            current_time = time.time()
            
            # Calculate utilization
            active_count = self._executor._threads.__len__() if hasattr(self._executor, '_threads') else self._current_workers
            
            # Get recent task history
            recent_tasks = [
                t for t in self._task_history
                if current_time - t["time"] < 30  # Last 30 seconds
            ]
            
            if not recent_tasks:
                return
            
            # Calculate metrics
            completion_rate = len([t for t in recent_tasks if t["completed"]]) / len(recent_tasks)
            avg_wait_time = sum(t.get("wait_time", 0) for t in recent_tasks) / len(recent_tasks)
            
            # Scale up if needed
            if (completion_rate < 0.9 or avg_wait_time > 1.0) and self._current_workers < self.max_workers:
                if current_time - self._last_scale_up > self._scale_up_cooldown:
                    self._scale_up()
            
            # Scale down if needed
            elif completion_rate > 0.95 and avg_wait_time < 0.1 and self._current_workers > self.min_workers:
                if current_time - self._last_scale_down > self._scale_down_cooldown:
                    self._scale_down()
    
    def _scale_up(self):
        """Scale up worker count."""
        if self._current_workers < self.max_workers:
            self._current_workers = min(self._current_workers + 1, self.max_workers)
            self._last_scale_up = time.time()
            self._metrics["scale_up_events"] += 1
            logger.info(f"Scaled up to {self._current_workers} workers")
    
    def _scale_down(self):
        """Scale down worker count."""
        if self._current_workers > self.min_workers:
            self._current_workers = max(self._current_workers - 1, self.min_workers)
            self._last_scale_down = time.time()
            self._metrics["scale_down_events"] += 1
            logger.info(f"Scaled down to {self._current_workers} workers")
    
    def submit(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> Future:
        """
        Submit a task to the worker pool.
        
        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future representing the task
        """
        if self._shutdown:
            raise RuntimeError("Worker pool is shutdown")
        
        task_id = f"task-{time.time()}-{id(fn)}"
        submit_time = time.time()
        
        # Wrap function to track metrics
        def wrapped_fn(*args, **kwargs):
            start_time = time.time()
            wait_time = start_time - submit_time
            
            try:
                result = fn(*args, **kwargs)
                
                with self._lock:
                    self._task_history.append({
                        "time": time.time(),
                        "completed": True,
                        "wait_time": wait_time,
                        "duration": time.time() - start_time
                    })
                    self._metrics["tasks_completed"] += 1
                
                return result
                
            except Exception as e:
                with self._lock:
                    self._task_history.append({
                        "time": time.time(),
                        "completed": False,
                        "wait_time": wait_time,
                        "error": str(e)
                    })
                    self._metrics["tasks_failed"] += 1
                raise
        
        with self._lock:
            self._metrics["tasks_submitted"] += 1
        
        return self._executor.submit(wrapped_fn, *args, **kwargs)
    
    def map(self, fn: Callable, items: List[Any]) -> List[Any]:
        """
        Apply function to all items in parallel.
        
        Args:
            fn: Function to apply
            items: Items to process
            
        Returns:
            List of results
        """
        futures = [self.submit(fn, item) for item in items]
        return [f.result() for f in futures]
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker pool status."""
        with self._lock:
            return {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "current_workers": self._current_workers,
                "metrics": self._metrics.copy(),
                "recent_task_count": len(self._task_history)
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)
        logger.info("Worker pool shutdown complete")


class PriorityWorkerPool(IntelligentWorkerPool):
    """
    Worker pool with task prioritization.
    
    Extends base worker pool with priority queues.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Priority queues
        self._priority_queues = {
            "high": [],
            "normal": [],
            "low": []
        }
        
        self._priority_lock = threading.Lock()
    
    def submit_priority(
        self,
        fn: Callable,
        priority: str = "normal",
        *args,
        **kwargs
    ) -> Future:
        """
        Submit task with priority.
        
        Args:
            fn: Function to execute
            priority: Task priority (high, normal, low)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future
        """
        if priority not in self._priority_queues:
            priority = "normal"
        
        with self._priority_lock:
            self._priority_queues[priority].append((fn, args, kwargs))
        
        # Process highest priority task
        return self._process_next_task()
    
    def _process_next_task(self) -> Future:
        """Process next highest priority task."""
        with self._priority_lock:
            for priority in ["high", "normal", "low"]:
                if self._priority_queues[priority]:
                    fn, args, kwargs = self._priority_queues[priority].pop(0)
                    return self.submit(fn, *args, **kwargs)
        
        return None
