"""
Async Processing Engine
=======================

High-performance async processing with concurrency control,
task prioritization, and backpressure handling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List, Coroutine
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Represents a processing task."""
    id: str
    coro: Coroutine
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def duration_ms(self) -> float:
        """Get task duration in milliseconds."""
        if self.started_at:
            end = self.completed_at or time.time()
            return (end - self.started_at) * 1000
        return 0.0


class AsyncProcessingEngine:
    """
    High-performance async processing engine.
    
    Features:
    - Concurrent task execution with semaphore control
    - Priority-based task scheduling
    - Timeout handling
    - Backpressure management
    - Task cancellation
    - Performance metrics
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        queue_size: int = 1000,
        default_timeout: float = 30.0
    ):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.default_timeout = default_timeout
        
        # These async primitives are created lazily when an event loop is available
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._queues: Optional[Dict[TaskPriority, asyncio.Queue]] = None
        
        # Active tasks
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, ProcessingTask] = {}
        
        # Worker tasks
        self._workers: List[asyncio.Task] = []
        self._workers_started = False
        self._shutdown = False
        
        # Metrics
        self._metrics = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total_duration_ms": 0.0
        }
        
        self._lock = threading.Lock()
    
    def _ensure_async_primitives(self):
        """Create async primitives if not yet initialized (must be called inside event loop)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        if self._queues is None:
            self._queues = {
                priority: asyncio.Queue(maxsize=self.queue_size)
                for priority in TaskPriority
            }
    
    def _ensure_workers_started(self):
        """Start worker tasks if not yet started (must be called inside event loop)."""
        if not self._workers_started:
            self._ensure_async_primitives()
            self._start_workers()
            self._workers_started = True

    def _start_workers(self):
        """Start worker tasks for each priority level."""
        for priority in TaskPriority:
            worker = asyncio.ensure_future(
                self._worker_loop(priority)
            )
            self._workers.append(worker)
    
    async def _worker_loop(self, priority: TaskPriority):
        """Worker loop for processing tasks of a specific priority."""
        queue = self._queues[priority]
        
        while not self._shutdown:
            try:
                # Wait for task with timeout
                processing_task = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0
                )
                
                # Execute task with concurrency control
                async with self._semaphore:
                    await self._execute_task(processing_task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _execute_task(self, processing_task: ProcessingTask):
        """Execute a single task."""
        processing_task.state = TaskState.RUNNING
        processing_task.started_at = time.time()
        
        try:
            # Apply timeout if specified
            timeout = processing_task.timeout or self.default_timeout
            
            processing_task.result = await asyncio.wait_for(
                processing_task.coro,
                timeout=timeout
            )
            
            processing_task.state = TaskState.COMPLETED
            
            with self._lock:
                self._metrics["completed"] += 1
                self._metrics["total_duration_ms"] += processing_task.duration_ms
            
        except asyncio.TimeoutError:
            processing_task.error = Exception(f"Task timed out after {timeout}s")
            processing_task.state = TaskState.FAILED
            
            with self._lock:
                self._metrics["failed"] += 1
            
            logger.warning(f"Task {processing_task.id} timed out")
            
        except Exception as e:
            processing_task.error = e
            processing_task.state = TaskState.FAILED
            
            with self._lock:
                self._metrics["failed"] += 1
            
            logger.error(f"Task {processing_task.id} failed: {e}")
        
        finally:
            processing_task.completed_at = time.time()
            
            # Store result
            with self._lock:
                self._task_results[processing_task.id] = processing_task
                self._active_tasks.pop(processing_task.id, None)
    
    async def submit(
        self,
        coro: Coroutine,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> str:
        """
        Submit a task for async execution.
        
        Args:
            coro: Coroutine to execute
            task_id: Optional task ID (auto-generated if not provided)
            priority: Task priority
            timeout: Optional timeout in seconds
            
        Returns:
            Task ID
        """
        if self._shutdown:
            raise RuntimeError("Engine is shutdown")
        
        self._ensure_workers_started()
        
        task_id = task_id or f"task-{time.time()}-{id(coro)}"
        
        processing_task = ProcessingTask(
            id=task_id,
            coro=coro,
            priority=priority,
            timeout=timeout
        )
        
        # Add to appropriate queue
        queue = self._queues[priority]
        
        try:
            queue.put_nowait(processing_task)
            
            with self._lock:
                self._metrics["submitted"] += 1
            
            logger.debug(f"Task {task_id} submitted with priority {priority.name}")
            return task_id
            
        except asyncio.QueueFull:
            raise RuntimeError(f"Task queue full (priority={priority.name})")
    
    async def run(
        self,
        coro_func,
        *args,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Run a coroutine function and wait for result.
        
        Accepts either a coroutine or a coroutine function with args.
        
        Args:
            coro_func: Coroutine function (or coroutine) to execute
            *args: Arguments to pass to the coroutine function
            timeout: Optional timeout
            
        Returns:
            Coroutine result
        """
        self._ensure_async_primitives()
        
        effective_timeout = timeout or self.default_timeout
        
        # Build the coroutine
        if asyncio.iscoroutine(coro_func):
            coro = coro_func
        elif asyncio.iscoroutinefunction(coro_func):
            coro = coro_func(*args)
        elif callable(coro_func):
            # Sync function — wrap in executor
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, coro_func, *args)
        else:
            raise TypeError(f"Expected coroutine or callable, got {type(coro_func)}")
        
        async with self._semaphore:
            with self._lock:
                self._metrics["submitted"] += 1
            
            try:
                result = await asyncio.wait_for(coro, timeout=effective_timeout)
                
                with self._lock:
                    self._metrics["completed"] += 1
                
                return result
                
            except asyncio.TimeoutError:
                with self._lock:
                    self._metrics["failed"] += 1
                raise
            except Exception:
                with self._lock:
                    self._metrics["failed"] += 1
                raise
    
    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        # Check active tasks
        with self._lock:
            if task_id in self._active_tasks:
                self._active_tasks[task_id].cancel()
                self._metrics["cancelled"] += 1
                return True
        
        # Check queues
        for priority, queue in self._queues.items():
            # Note: This is a simplified approach - in production,
            # you'd need a more sophisticated queue removal mechanism
            pass
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        with self._lock:
            if task_id in self._task_results:
                task = self._task_results[task_id]
                return {
                    "id": task.id,
                    "state": task.state.value,
                    "priority": task.priority.name,
                    "duration_ms": task.duration_ms,
                    "has_result": task.result is not None,
                    "has_error": task.error is not None
                }
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            
            # Calculate average duration
            if metrics["completed"] > 0:
                metrics["avg_duration_ms"] = metrics["total_duration_ms"] / metrics["completed"]
            else:
                metrics["avg_duration_ms"] = 0.0
            
            # Queue sizes
            metrics["queue_sizes"] = {
                priority.name: queue.qsize()
                for priority, queue in self._queues.items()
            }
            
            # Active tasks
            metrics["active_tasks"] = len(self._active_tasks)
            
            return metrics
    
    def shutdown(self):
        """Shutdown the engine."""
        self._shutdown = True
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        logger.info("AsyncProcessingEngine shutdown complete")


class ParallelProcessor:
    """
    Utility for parallel processing of multiple items.
    
    Provides map/reduce style operations with async execution.
    """
    
    def __init__(self, engine: AsyncProcessingEngine):
        self.engine = engine
    
    async def map(
        self,
        func: Callable[[Any], Coroutine],
        items: List[Any],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """
        Apply async function to all items in parallel.
        
        Args:
            func: Async function to apply
            items: Items to process
            max_concurrent: Optional concurrency limit
            
        Returns:
            List of results
        """
        # Create coroutines for all items
        coros = [func(item) for item in items]
        
        # Submit all tasks
        task_ids = []
        for coro in coros:
            task_id = await self.engine.submit(coro)
            task_ids.append(task_id)
        
        # Wait for all results
        results = []
        for task_id in task_ids:
            while True:
                status = self.engine.get_task_status(task_id)
                if status and status["state"] in ("completed", "failed"):
                    with self.engine._lock:
                        if task_id in self.engine._task_results:
                            task = self.engine._task_results.pop(task_id)
                            if task.error:
                                results.append(None)  # Or raise
                            else:
                                results.append(task.result)
                    break
                await asyncio.sleep(0.01)
        
        return results
    
    async def filter(
        self,
        predicate: Callable[[Any], Coroutine],
        items: List[Any]
    ) -> List[Any]:
        """
        Filter items based on async predicate.
        
        Args:
            predicate: Async predicate function
            items: Items to filter
            
        Returns:
            Filtered items
        """
        results = await self.map(predicate, items)
        return [item for item, result in zip(items, results) if result]
