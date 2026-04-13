"""
Test Suite for Hermes Agent Core
=================================

Comprehensive tests for the upgraded Hermes Agent.
"""

import unittest
import asyncio
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

# Import modules to test
import sys
sys.path.insert(0, '/mnt/okcomputer/output/hermes_agent_upgraded')

from core.hermes_core import (
    HermesAgentCore,
    AgentConfig,
    ProcessingMode,
    AgentState
)
from core.resilience_enhanced import (
    EnhancedResilienceEngine,
    AdaptiveTokenBucketV2,
    CircuitBreakerV2
)
from core.context_enhanced import (
    EnhancedTaskClassifier,
    EnhancedTokenAllocator,
    TaskType,
    TaskComplexity
)
from core.persistence_enhanced import EnhancedDeltaPersistence
from processing.async_engine import AsyncProcessingEngine
from processing.worker_pool import IntelligentWorkerPool
from processing.batch_processor import BatchProcessor
from efficiency.cache_manager import MultiLevelCache
from efficiency.load_balancer import SmartLoadBalancer, LoadStrategy
from monitoring.metrics_collector import MetricsCollector
from recovery.self_healing import SelfHealingRecovery, RecoveryStrategy


class TestHermesAgentCore(unittest.TestCase):
    """Test HermesAgentCore main functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AgentConfig(
            tokens_per_minute=1000,
            max_context=8000,
            enable_metrics=True
        )
    
    def test_initialization(self):
        """Test core initialization."""
        agent = HermesAgentCore(self.config)
        
        self.assertEqual(agent.state, AgentState.READY)
        self.assertIsNotNone(agent.resilience)
        self.assertIsNotNone(agent.classifier)
        self.assertIsNotNone(agent.allocator)
        self.assertIsNotNone(agent.persistence)
        
        agent.shutdown()
    
    def test_sync_processing(self):
        """Test synchronous message processing."""
        agent = HermesAgentCore(self.config)
        
        result = agent.process("Hello, how are you?")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.response)
        self.assertGreater(result.latency_ms, 0)
        
        agent.shutdown()
    
    def test_async_processing(self):
        """Test asynchronous message processing."""
        agent = HermesAgentCore(self.config)
        
        async def test_async():
            result = await agent.process_async("Test message")
            return result
        
        result = asyncio.run(test_async())
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.response)
        
        agent.shutdown()
    
    def test_batch_processing(self):
        """Test batch processing."""
        agent = HermesAgentCore(self.config)
        
        messages = [f"Message {i}" for i in range(5)]
        
        async def test_batch():
            results = await agent.process_batch(messages)
            return results
        
        results = asyncio.run(test_batch())
        
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.success for r in results))
        
        agent.shutdown()
    
    def test_health_metrics(self):
        """Test health and metrics reporting."""
        agent = HermesAgentCore(self.config)
        
        # Process some messages
        for i in range(5):
            agent.process(f"Test {i}")
        
        health = agent.get_health()
        
        self.assertIn("state", health)
        self.assertIn("processed_count", health)
        self.assertEqual(health["processed_count"], 5)
        
        metrics = agent.get_metrics()
        
        self.assertIn("processing", metrics)
        self.assertIn("resilience", metrics)
        
        agent.shutdown()
    
    def test_session_management(self):
        """Test session persistence."""
        agent = HermesAgentCore(self.config)
        
        # Initialize session
        agent.init_session("test-session-123")
        
        # Create checkpoint
        checkpoint = agent.save_checkpoint()
        
        self.assertIsNotNone(checkpoint)
        
        # Recover session
        success = agent.recover_session("test-session-123")
        
        self.assertTrue(success)
        
        agent.shutdown()


class TestResilienceEngine(unittest.TestCase):
    """Test enhanced resilience engine."""
    
    def test_token_bucket(self):
        """Test adaptive token bucket."""
        bucket = AdaptiveTokenBucketV2(
            rate=10.0,
            capacity=20.0
        )
        
        # Should be able to consume initially
        self.assertTrue(bucket.consume(5))
        
        # Record success
        bucket.record_success(100)
        
        status = bucket.get_status()
        
        self.assertIn("tokens", status)
        self.assertIn("adaptive_multiplier", status)
    
    def test_circuit_breaker(self):
        """Test enhanced circuit breaker."""
        cb = CircuitBreakerV2(
            failure_threshold=3,
            recovery_timeout=1.0
        )
        
        # Initially closed
        self.assertTrue(cb.can_execute())
        
        # Record failures
        for _ in range(3):
            cb.record_failure()
        
        # Should be open now
        self.assertFalse(cb.can_execute())
        
        # Wait for recovery
        time.sleep(1.5)
        
        # Should be half-open
        self.assertTrue(cb.can_execute())
    
    def test_resilience_engine(self):
        """Test full resilience engine."""
        engine = EnhancedResilienceEngine(
            tokens_per_minute=100,
            circuit_failure_threshold=3
        )
        
        # Check should pass initially
        result = engine.check("test", tokens=1)
        
        self.assertTrue(result.allowed)
        
        # Record success
        engine.record_success(100)
        
        # Get metrics
        metrics = engine.get_metrics()
        
        self.assertIn("token_bucket", metrics)
        self.assertIn("circuit_breaker", metrics)


class TestContextIntelligence(unittest.TestCase):
    """Test context intelligence modules."""
    
    def test_task_classifier(self):
        """Test task classification."""
        classifier = EnhancedTaskClassifier(min_confidence=0.5)
        
        # Test code classification
        profile = classifier.classify("Write a Python function to sort a list")
        
        self.assertEqual(profile.task_type, TaskType.CODE)
        self.assertGreater(profile.confidence, 0.5)
        
        # Test conversation classification
        profile = classifier.classify("Hello, how are you?")
        
        self.assertEqual(profile.task_type, TaskType.CONVERSATION)
    
    def test_token_allocator(self):
        """Test token allocation."""
        allocator = EnhancedTokenAllocator(max_context=16000)
        
        from core.context_enhanced import TaskProfile, TaskType, TaskComplexity
        
        profile = TaskProfile(
            task_type=TaskType.CODE,
            complexity=TaskComplexity.MODERATE
        )
        
        result = allocator.allocate(
            message="Write a function",
            profile=profile
        )
        
        self.assertGreater(result.allocated, 0)
        self.assertIn("system", result.allocation_map)
        self.assertIn("history", result.allocation_map)
        self.assertIn("working", result.allocation_map)


class TestPersistence(unittest.TestCase):
    """Test delta persistence."""
    
    def setUp(self):
        """Set up temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
    
    def tearDown(self):
        """Clean up temporary database."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_delta_persistence(self):
        """Test delta persistence operations."""
        persistence = EnhancedDeltaPersistence(
            db_path=self.db_path,
            auto_checkpoint=False
        )
        
        # Initialize session
        persistence.init_session("test-session")
        
        # Log some deltas
        from core.persistence_enhanced import DeltaType
        
        for i in range(5):
            persistence.log_delta(
                DeltaType.MESSAGE_ADDED,
                {"message": f"Test {i}"}
            )
        
        # Create checkpoint
        checkpoint = persistence.create_checkpoint(
            conversation_state={"messages": []},
            agent_state={}
        )
        
        self.assertIsNotNone(checkpoint)
        
        # Recover
        success, state, replayed = persistence.recover("test-session")
        
        self.assertTrue(success)
        self.assertEqual(replayed, 5)
        
        persistence.shutdown()


class TestProcessingModules(unittest.TestCase):
    """Test processing power modules."""
    
    def test_async_engine(self):
        """Test async processing engine."""
        engine = AsyncProcessingEngine(max_concurrent=5)
        
        async def test_task():
            await asyncio.sleep(0.01)
            return "success"
        
        async def run_test():
            result = await engine.run(test_task())
            return result
        
        result = asyncio.run(run_test())
        
        self.assertEqual(result, "success")
        
        engine.shutdown()
    
    def test_worker_pool(self):
        """Test intelligent worker pool."""
        pool = IntelligentWorkerPool(
            min_workers=2,
            max_workers=4
        )
        
        def task(x):
            time.sleep(0.01)
            return x * 2
        
        futures = [pool.submit(task, i) for i in range(5)]
        results = [f.result() for f in futures]
        
        self.assertEqual(results, [0, 2, 4, 6, 8])
        
        pool.shutdown()
    
    def test_batch_processor(self):
        """Test batch processor."""
        processor = BatchProcessor(batch_size=3)
        
        async def process_item(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        async def run_test():
            result = await processor.process_batch(
                [1, 2, 3, 4, 5],
                process_item
            )
            return result
        
        result = asyncio.run(run_test())
        
        self.assertEqual(result.processed_count, 5)
        self.assertEqual(result.failed_count, 0)


class TestEfficiencyModules(unittest.TestCase):
    """Test efficiency modules."""
    
    def test_cache_manager(self):
        """Test multi-level cache."""
        cache = MultiLevelCache(max_size=100, ttl=60)
        
        # Set and get
        cache.set("key1", "value1")
        value = cache.get("key1")
        
        self.assertEqual(value, "value1")
        
        # Cache miss
        value = cache.get("nonexistent")
        
        self.assertIsNone(value)
        
        # Check stats
        stats = cache.get_stats()
        
        self.assertIn("hit_rate", stats)
        self.assertGreater(stats["hits"], 0)
    
    def test_load_balancer(self):
        """Test smart load balancer."""
        lb = SmartLoadBalancer(strategy=LoadStrategy.ROUND_ROBIN)
        
        # Register resources
        lb.register("resource1", weight=1.0)
        lb.register("resource2", weight=1.0)
        
        # Select resources
        selections = [lb.select() for _ in range(4)]
        
        self.assertIn("resource1", selections)
        self.assertIn("resource2", selections)
        
        # Check stats
        stats = lb.get_stats()
        
        self.assertEqual(stats["total_resources"], 2)


class TestMonitoring(unittest.TestCase):
    """Test monitoring modules."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        # Record some metrics
        for i in range(10):
            collector.record_processing(
                success=True,
                latency_ms=100 + i * 10,
                tokens_used=50
            )
        
        # Get current metrics
        current = collector.get_current()
        
        self.assertEqual(current["processing_count"], 10)
        self.assertEqual(current["success_count"], 10)
        
        # Get summary
        summary = collector.get_summary()
        
        self.assertIn("latency_stats", summary)


class TestRecovery(unittest.TestCase):
    """Test recovery modules."""
    
    def test_self_healing(self):
        """Test self-healing recovery."""
        recovery = SelfHealingRecovery(
            max_retries=2,
            retry_delay=0.1,
            strategy=RecoveryStrategy.FIXED_DELAY
        )
        
        # Test successful execution
        def success_fn():
            return "success"
        
        result = recovery.execute_with_retry(success_fn)
        
        self.assertTrue(result.success)
        self.assertEqual(result.result, "success")
        self.assertEqual(result.attempts, 1)
    
    def test_retry_on_failure(self):
        """Test retry on failure."""
        recovery = SelfHealingRecovery(
            max_retries=2,
            retry_delay=0.1,
            strategy=RecoveryStrategy.FIXED_DELAY
        )
        
        call_count = [0]
        
        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = recovery.execute_with_retry(fail_then_succeed)
        
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHermesAgentCore))
    suite.addTests(loader.loadTestsFromTestCase(TestResilienceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestContextIntelligence))
    suite.addTests(loader.loadTestsFromTestCase(TestPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessingModules))
    suite.addTests(loader.loadTestsFromTestCase(TestEfficiencyModules))
    suite.addTests(loader.loadTestsFromTestCase(TestMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestRecovery))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
