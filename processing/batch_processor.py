"""
Batch Processor
===============

Efficient batch processing with timeout handling and result aggregation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Any, Optional, Dict, Union
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class BatchState(Enum):
    """Batch processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    state: BatchState
    results: List[Any] = field(default_factory=list)
    errors: List[Optional[Exception]] = field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0
    duration_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.processed_count + self.failed_count
        return self.processed_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "state": self.state.value,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": round(self.success_rate, 2),
            "duration_ms": round(self.duration_ms, 2)
        }


class BatchProcessor:
    """
    Efficient batch processor with timeout and error handling.
    
    Features:
    - Dynamic batch sizing
    - Timeout handling
    - Error aggregation
    - Progress tracking
    - Parallel processing within batches
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        timeout: float = 30.0,
        max_retries: int = 2,
        parallel_within_batch: bool = True
    ):
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.parallel_within_batch = parallel_within_batch
        
        # Metrics
        self._metrics = {
            "batches_processed": 0,
            "items_processed": 0,
            "items_failed": 0,
            "total_duration_ms": 0.0
        }
        
        self._lock = threading.Lock()
    
    async def process_batch(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Any],
        batch_size: Optional[int] = None
    ) -> BatchResult:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            process_fn: Function to process each item
            batch_size: Optional override for batch size
            
        Returns:
            BatchResult with all results
        """
        batch_size = batch_size or self.batch_size
        batch_id = f"batch-{time.time()}"
        start_time = time.time()
        
        all_results = []
        all_errors = []
        processed = 0
        failed = 0
        
        # Split into batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} items)")
            
            if self.parallel_within_batch:
                batch_results, batch_errors = await self._process_batch_parallel(
                    batch, process_fn
                )
            else:
                batch_results, batch_errors = await self._process_batch_sequential(
                    batch, process_fn
                )
            
            all_results.extend(batch_results)
            all_errors.extend(batch_errors)
            
            processed += len([r for r in batch_results if r is not None])
            failed += len([e for e in batch_errors if e is not None])
        
        duration = (time.time() - start_time) * 1000
        
        # Determine state
        if failed == 0:
            state = BatchState.COMPLETED
        elif processed > 0:
            state = BatchState.PARTIAL
        else:
            state = BatchState.FAILED
        
        result = BatchResult(
            batch_id=batch_id,
            state=state,
            results=all_results,
            errors=all_errors,
            processed_count=processed,
            failed_count=failed,
            duration_ms=duration
        )
        
        # Update metrics
        with self._lock:
            self._metrics["batches_processed"] += 1
            self._metrics["items_processed"] += processed
            self._metrics["items_failed"] += failed
            self._metrics["total_duration_ms"] += duration
        
        logger.info(f"Batch {batch_id} completed: {processed} processed, {failed} failed in {duration:.1f}ms")
        
        return result
    
    async def _process_batch_parallel(
        self,
        batch: List[Any],
        process_fn: Callable[[Any], Any]
    ) -> tuple:
        """Process batch items in parallel."""
        tasks = []
        
        for item in batch:
            task = self._process_with_timeout(item, process_fn)
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate results and errors
        processed_results = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(None)
                errors.append(result)
            else:
                processed_results.append(result)
                errors.append(None)
        
        return processed_results, errors
    
    async def _process_batch_sequential(
        self,
        batch: List[Any],
        process_fn: Callable[[Any], Any]
    ) -> tuple:
        """Process batch items sequentially."""
        results = []
        errors = []
        
        for item in batch:
            try:
                result = await self._process_with_timeout(item, process_fn)
                results.append(result)
                errors.append(None)
            except Exception as e:
                results.append(None)
                errors.append(e)
        
        return results, errors
    
    async def _process_with_timeout(
        self,
        item: Any,
        process_fn: Callable[[Any], Any]
    ) -> Any:
        """Process single item with timeout and retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Handle both sync and async functions
                if asyncio.iscoroutinefunction(process_fn):
                    result = await asyncio.wait_for(
                        process_fn(item),
                        timeout=self.timeout
                    )
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, process_fn, item),
                        timeout=self.timeout
                    )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Processing timed out after {self.timeout}s")
                logger.warning(f"Attempt {attempt + 1} timed out for item")
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        # All retries exhausted
        raise last_error or Exception("All retry attempts failed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch processor metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            
            if metrics["batches_processed"] > 0:
                metrics["avg_batch_duration_ms"] = (
                    metrics["total_duration_ms"] / metrics["batches_processed"]
                )
            else:
                metrics["avg_batch_duration_ms"] = 0.0
            
            total_items = metrics["items_processed"] + metrics["items_failed"]
            metrics["overall_success_rate"] = (
                metrics["items_processed"] / total_items if total_items > 0 else 0.0
            )
            
            return metrics


class StreamingBatchProcessor(BatchProcessor):
    """
    Batch processor with streaming results.
    
    Yields results as they become available.
    """
    
    async def process_streaming(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Any]
    ):
        """
        Process items and yield results as they complete.
        
        Args:
            items: Items to process
            process_fn: Function to process each item
            
        Yields:
            Tuple of (item, result, error)
        """
        # Create tasks
        tasks = {
            asyncio.create_task(self._process_with_timeout(item, process_fn)): item
            for item in items
        }
        
        # Yield results as they complete
        for task in asyncio.as_completed(tasks.keys()):
            item = tasks[task]
            try:
                result = await task
                yield (item, result, None)
            except Exception as e:
                yield (item, None, e)


class AdaptiveBatchProcessor(BatchProcessor):
    """
    Batch processor with adaptive batch sizing.
    
    Adjusts batch size based on performance metrics.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._min_batch_size = 1
        self._max_batch_size = 100
        self._performance_history: deque = deque(maxlen=10)
        self._adaptive_interval = 5  # Adjust every 5 batches
        self._batches_since_adjustment = 0
    
    async def process_batch(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Any],
        batch_size: Optional[int] = None
    ) -> BatchResult:
        """Process with adaptive batch sizing."""
        # Use adaptive batch size if not specified
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size()
        
        result = await super().process_batch(items, process_fn, batch_size)
        
        # Record performance
        self._performance_history.append({
            "batch_size": batch_size,
            "duration_ms": result.duration_ms,
            "success_rate": result.success_rate
        })
        
        # Adjust batch size periodically
        self._batches_since_adjustment += 1
        if self._batches_since_adjustment >= self._adaptive_interval:
            self._adjust_batch_size()
            self._batches_since_adjustment = 0
        
        return result
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on history."""
        if not self._performance_history:
            return self.batch_size
        
        # Find batch size with best throughput
        best_size = self.batch_size
        best_throughput = 0.0
        
        for record in self._performance_history:
            if record["duration_ms"] > 0:
                throughput = record["batch_size"] / record["duration_ms"]
                if throughput > best_throughput and record["success_rate"] > 0.9:
                    best_throughput = throughput
                    best_size = record["batch_size"]
        
        return best_size
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance."""
        if not self._performance_history:
            return
        
        recent = list(self._performance_history)[-5:]
        avg_success_rate = sum(r["success_rate"] for r in recent) / len(recent)
        
        if avg_success_rate > 0.95:
            # Increase batch size
            self.batch_size = min(self.batch_size + 2, self._max_batch_size)
            logger.info(f"Increased batch size to {self.batch_size}")
        elif avg_success_rate < 0.8:
            # Decrease batch size
            self.batch_size = max(self.batch_size - 2, self._min_batch_size)
            logger.info(f"Decreased batch size to {self.batch_size}")
