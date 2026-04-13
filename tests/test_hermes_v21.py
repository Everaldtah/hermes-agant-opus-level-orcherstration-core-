"""
Hermes Agent v2.1 - Comprehensive Test Suite
=============================================

Run:  python tests/test_hermes_v21.py
"""

import sys
import os
import asyncio
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.expanduser("~/.hermes"), exist_ok=True)

PASSED = 0
FAILED = 0


def test(name):
    """Decorator that runs and reports a test."""
    def decorator(fn):
        global PASSED, FAILED
        try:
            fn()
            print(f"  ✅ {name}")
            PASSED += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            FAILED += 1
    return decorator


def run_async(coro):
    """Helper to run async code in tests."""
    return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 1. IMPORT TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("All core modules import")
def _():
    from core.resilience_enhanced import EnhancedResilienceEngine
    from core.context_enhanced import EnhancedTaskClassifier, HierarchicalContextV2
    from core.persistence_enhanced import EnhancedDeltaPersistence, SessionManager
    from core.llm_provider import MockProvider, ProviderManager, LLMRequest
    from core.event_bus import EventBus, get_event_bus

@test("All processing modules import")
def _():
    from processing.async_engine import AsyncProcessingEngine
    from processing.worker_pool import IntelligentWorkerPool, PriorityWorkerPool
    from processing.batch_processor import BatchProcessor, StreamingBatchProcessor

@test("All efficiency modules import")
def _():
    from efficiency.cache_manager import MultiLevelCache, PredictiveCacheLoader
    from efficiency.load_balancer import SmartLoadBalancer, AdaptiveLoadBalancer
    from efficiency.connection_pool import ConnectionPool, PooledConnection

@test("All monitoring/recovery modules import")
def _():
    from monitoring.metrics_collector import MetricsCollector, PrometheusExporter
    from recovery.self_healing import SelfHealingRecovery, CircuitBreakerRecovery

@test("Top-level HermesAgentCore imports")
def _():
    from core.hermes_core import HermesAgentCore, AgentConfig, ProcessingResult


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 2. EVENT BUS TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Publish and subscribe")
def _():
    from core.event_bus import EventBus
    bus = EventBus()
    received = []
    bus.subscribe("test.event", lambda e: received.append(e.data))
    bus.publish("test.event", {"msg": "hello"})
    assert len(received) == 1
    assert received[0]["msg"] == "hello"

@test("Wildcard subscriber receives all events")
def _():
    from core.event_bus import EventBus
    bus = EventBus()
    received = []
    bus.subscribe("*", lambda e: received.append(e.topic))
    bus.publish("a.event")
    bus.publish("b.event")
    assert received == ["a.event", "b.event"]

@test("Unsubscribe works")
def _():
    from core.event_bus import EventBus
    bus = EventBus()
    received = []
    unsub = bus.subscribe("x", lambda e: received.append(1))
    bus.publish("x")
    unsub()
    bus.publish("x")
    assert len(received) == 1

@test("Event history and stats")
def _():
    from core.event_bus import EventBus
    bus = EventBus()
    for i in range(5):
        bus.publish("counter", i)
    assert len(bus.get_recent("counter")) == 5
    stats = bus.get_stats()
    assert stats["total_events"] == 5


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 3. LLM PROVIDER TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("MockProvider returns sensible responses")
def _():
    from core.llm_provider import MockProvider, LLMRequest
    async def _run():
        provider = MockProvider(latency_ms=1)
        req = LLMRequest(messages=[{"role": "user", "content": "Hello there"}])
        resp = await provider.complete(req)
        assert resp.content and len(resp.content) > 5
        assert resp.input_tokens > 0
        assert resp.provider == "mock"
    run_async(_run())

@test("MockProvider classifies code requests")
def _():
    from core.llm_provider import MockProvider, LLMRequest
    async def _run():
        p = MockProvider(latency_ms=1)
        resp = await p.complete(LLMRequest(
            messages=[{"role": "user", "content": "Write a function to sort"}]
        ))
        assert "python" in resp.content.lower() or "def" in resp.content.lower()
    run_async(_run())

@test("ProviderManager failover to mock")
def _():
    from core.llm_provider import ProviderManager, MockProvider, LLMProvider, LLMRequest, LLMResponse
    class FailingProvider(LLMProvider):
        async def complete(self, req):
            raise ConnectionError("Simulated failure")
        async def stream(self, req):
            yield ""
    async def _run():
        mgr = ProviderManager()
        mgr.add_provider(FailingProvider(name="bad", model="bad"))
        mgr.add_provider(MockProvider(latency_ms=1))
        resp = await mgr.complete(LLMRequest(
            messages=[{"role": "user", "content": "test"}]
        ))
        assert resp.provider == "mock"  # fell back
    run_async(_run())

@test("Provider stats tracking")
def _():
    from core.llm_provider import MockProvider, LLMRequest
    async def _run():
        p = MockProvider(latency_ms=1)
        for _ in range(3):
            await p.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        stats = p.get_stats()
        assert stats["requests"] == 3
        assert stats["total_tokens"] > 0
    run_async(_run())


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 4. CONNECTION POOL TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Pool creates and acquires connections")
def _():
    from efficiency.connection_pool import ConnectionPool
    counter = {"n": 0}
    def factory():
        counter["n"] += 1
        return f"conn-{counter['n']}"
    pool = ConnectionPool(factory=factory, min_size=2, max_size=5)
    pc = pool.acquire(timeout=2)
    assert pc.connection.startswith("conn-")
    pool.release(pc)
    pool.close()

@test("Pool context manager (lease)")
def _():
    from efficiency.connection_pool import ConnectionPool
    pool = ConnectionPool(factory=lambda: "raw", min_size=1, max_size=3)
    with pool.lease() as pc:
        assert pc.connection == "raw"
        assert pc.use_count == 1
    stats = pool.get_stats()
    assert stats["acquired"] >= 1
    assert stats["released"] >= 1
    pool.close()

@test("Pool respects max_size")
def _():
    from efficiency.connection_pool import ConnectionPool
    pool = ConnectionPool(factory=lambda: "c", min_size=1, max_size=2)
    a = pool.acquire(timeout=1)
    b = pool.acquire(timeout=1)
    try:
        pool.acquire(timeout=0.2)
        assert False, "Should have timed out"
    except TimeoutError:
        pass
    pool.release(a)
    pool.release(b)
    pool.close()


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 5. TASK CLASSIFICATION TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Classifies code, debugging, analysis, conversation")
def _():
    from core.context_enhanced import EnhancedTaskClassifier, TaskType
    c = EnhancedTaskClassifier()
    cases = [
        ("Write a Python function to sort a list", TaskType.CODE),
        ("Analyze the quarterly sales data trends", TaskType.ANALYSIS),
        ("Hello, how are you?", TaskType.CONVERSATION),
        ("Debug this error: IndexError in line 42", TaskType.DEBUGGING),
        ("My code is not working, logs show a connection error", TaskType.DEBUGGING),
    ]
    for msg, expected in cases:
        p = c.classify(msg)
        assert p.task_type == expected, f"Expected {expected} for '{msg[:40]}', got {p.task_type}"

@test("Complexity detection works")
def _():
    from core.context_enhanced import EnhancedTaskClassifier, TaskComplexity
    c = EnhancedTaskClassifier()
    simple = c.classify("Just print hello")
    complex_ = c.classify("Write a comprehensive, detailed, in-depth analysis of multiple factors")
    assert simple.complexity == TaskComplexity.SIMPLE
    assert complex_.complexity == TaskComplexity.COMPLEX

@test("Token estimation varies by type/complexity")
def _():
    from core.context_enhanced import EnhancedTaskClassifier
    c = EnhancedTaskClassifier()
    simple_conv = c.classify("Hi")
    complex_code = c.classify("Write a comprehensive Python class with multiple methods and error handling")
    assert complex_code.estimated_tokens > simple_conv.estimated_tokens


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 6. RESILIENCE ENGINE TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Token bucket rate limiting")
def _():
    from core.resilience_enhanced import AdaptiveTokenBucketV2
    bucket = AdaptiveTokenBucketV2(rate=10, capacity=5, adaptive=False)
    # Consume all tokens
    for _ in range(5):
        assert bucket.consume(1)
    # Next should fail
    assert not bucket.consume(1)

@test("Circuit breaker opens after failures")
def _():
    from core.resilience_enhanced import CircuitBreakerV2, CircuitState
    cb = CircuitBreakerV2(failure_threshold=3, recovery_timeout=1)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert not cb.can_execute()

@test("Resilience engine check returns ProtectionResult")
def _():
    from core.resilience_enhanced import EnhancedResilienceEngine
    engine = EnhancedResilienceEngine(tokens_per_minute=6000, token_burst=100)
    result = engine.check("test", tokens=1)
    assert result.allowed
    assert result.reason == "OK"

@test("Failure prediction works")
def _():
    from core.resilience_enhanced import EnhancedResilienceEngine
    engine = EnhancedResilienceEngine(tokens_per_minute=6000, token_burst=100)
    pred = engine.predict_failure()
    assert 0 <= pred.predicted_failure_probability <= 1
    assert pred.recommended_action in ("normal_operation", "monitor_closely", "reduce_load")


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 7. CACHE TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Multi-level cache set/get")
def _():
    from efficiency.cache_manager import MultiLevelCache
    cache = MultiLevelCache(max_size=100, ttl=60)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

@test("Cache miss returns None")
def _():
    from efficiency.cache_manager import MultiLevelCache
    cache = MultiLevelCache(max_size=100, ttl=60)
    assert cache.get("nonexistent") is None

@test("Cache invalidation")
def _():
    from efficiency.cache_manager import MultiLevelCache
    cache = MultiLevelCache(max_size=100, ttl=60)
    cache.set("k", "v")
    assert cache.invalidate("k")
    assert cache.get("k") is None

@test("Cache hit rate tracking")
def _():
    from efficiency.cache_manager import MultiLevelCache
    cache = MultiLevelCache(max_size=100, ttl=60)
    cache.set("a", 1)
    cache.get("a")  # hit
    cache.get("b")  # miss
    rate = cache.get_hit_rate()
    assert 0.4 < rate < 0.6  # ~50%


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 8. LOAD BALANCER TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Register and select resources")
def _():
    from efficiency.load_balancer import SmartLoadBalancer
    lb = SmartLoadBalancer()
    lb.register("node-1", weight=1.0)
    lb.register("node-2", weight=2.0)
    selected = lb.select()
    assert selected in ("node-1", "node-2")

@test("No healthy resources returns None")
def _():
    from efficiency.load_balancer import SmartLoadBalancer
    lb = SmartLoadBalancer()
    assert lb.select() is None

@test("Request tracking updates metrics")
def _():
    from efficiency.load_balancer import SmartLoadBalancer
    lb = SmartLoadBalancer()
    lb.register("n1")
    lb.record_request_start("n1")
    lb.record_request_end("n1", success=True, latency_ms=50)
    status = lb.get_resource_status("n1")
    assert status["total_requests"] == 1
    assert status["avg_latency_ms"] > 0


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 9. PERSISTENCE TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Init session and create checkpoint")
def _():
    from core.persistence_enhanced import EnhancedDeltaPersistence
    db = os.path.expanduser("~/.hermes/test_persist.db")
    if os.path.exists(db):
        os.remove(db)
    p = EnhancedDeltaPersistence(db_path=db, auto_checkpoint=False)
    p.init_session("sess-1", {"test": True})
    cp = p.create_checkpoint({"messages": []}, {"state": "ready"})
    assert cp is not None
    assert cp.session_id == "sess-1"
    p.shutdown()

@test("Delta record compression and integrity")
def _():
    from core.persistence_enhanced import DeltaRecord, DeltaType
    record = DeltaRecord(
        sequence=1, timestamp=time.time(),
        delta_type=DeltaType.MESSAGE_ADDED,
        data={"content": "hello world " * 100}
    )
    record.checksum = record.calculate_checksum()
    assert record.verify()
    compressed = record.compress_data()
    assert len(compressed) < len(str(record.data).encode())
    decompressed = DeltaRecord.decompress_data(compressed)
    assert decompressed == record.data

@test("Session recovery from checkpoint + deltas")
def _():
    from core.persistence_enhanced import EnhancedDeltaPersistence, DeltaType
    db = os.path.expanduser("~/.hermes/test_recovery.db")
    if os.path.exists(db):
        os.remove(db)
    p = EnhancedDeltaPersistence(db_path=db, auto_checkpoint=False)
    p.init_session("recovery-test")
    p.create_checkpoint({"messages": ["msg1"]}, {"mode": "sync"})
    p.log_delta(DeltaType.MESSAGE_ADDED, {"content": "msg2"})
    p._flush_deltas()
    success, state, replayed = p.recover("recovery-test")
    assert success
    assert replayed >= 1
    p.shutdown()


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 10. SELF-HEALING RECOVERY TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Successful execution on first attempt")
def _():
    from recovery.self_healing import SelfHealingRecovery
    r = SelfHealingRecovery(max_retries=3, retry_delay=0.01)
    result = r.execute_with_retry(lambda: "ok")
    assert result.success
    assert result.result == "ok"
    assert result.attempts == 1

@test("Retry on transient failure then succeed")
def _():
    from recovery.self_healing import SelfHealingRecovery
    counter = {"n": 0}
    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"
    r = SelfHealingRecovery(max_retries=3, retry_delay=0.01)
    result = r.execute_with_retry(flaky)
    assert result.success
    assert result.attempts == 3

@test("Permanent failure exhausts retries")
def _():
    from recovery.self_healing import SelfHealingRecovery
    r = SelfHealingRecovery(max_retries=2, retry_delay=0.01)
    result = r.execute_with_retry(lambda: (_ for _ in ()).throw(ValueError("permanent")))
    assert not result.success
    assert result.attempts == 3


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 11. METRICS COLLECTOR TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Record and retrieve metrics")
def _():
    from monitoring.metrics_collector import MetricsCollector
    m = MetricsCollector(enable_alerts=False)
    m.record_processing(success=True, latency_ms=50, tokens_used=100)
    m.record_processing(success=True, latency_ms=80, tokens_used=200)
    m.record_processing(success=False, latency_ms=500, tokens_used=50)
    current = m.get_current()
    assert current["processing_count"] == 3
    assert current["error_count"] == 1
    assert current["tokens_used"] == 350

@test("Prometheus export format")
def _():
    from monitoring.metrics_collector import MetricsCollector, PrometheusExporter
    m = MetricsCollector(enable_alerts=False)
    m.record_processing(success=True, latency_ms=42, tokens_used=10)
    exporter = PrometheusExporter(m)
    output = exporter.export()
    assert "hermes_processing_total" in output
    assert "hermes_success_rate" in output

@test("Latency percentiles in summary")
def _():
    from monitoring.metrics_collector import MetricsCollector
    m = MetricsCollector(enable_alerts=False)
    for lat in [10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000]:
        m.record_processing(success=True, latency_ms=lat)
    summary = m.get_summary()
    assert "p50_ms" in summary["latency_stats"]
    assert "p95_ms" in summary["latency_stats"]


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 12. WORKER POOL TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Submit and get result from worker pool")
def _():
    from processing.worker_pool import IntelligentWorkerPool
    pool = IntelligentWorkerPool(min_workers=1, max_workers=4)
    future = pool.submit(lambda x: x * 2, 21)
    assert future.result(timeout=5) == 42
    pool.shutdown()

@test("Worker pool map over items")
def _():
    from processing.worker_pool import IntelligentWorkerPool
    pool = IntelligentWorkerPool(min_workers=2, max_workers=4)
    results = pool.map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    assert results == [1, 4, 9, 16, 25]
    pool.shutdown()

@test("Worker pool status reporting")
def _():
    from processing.worker_pool import IntelligentWorkerPool
    pool = IntelligentWorkerPool(min_workers=2, max_workers=8)
    status = pool.get_status()
    assert status["min_workers"] == 2
    assert status["max_workers"] == 8
    pool.shutdown()


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 13. BATCH PROCESSOR TESTS ═══")
# ══════════════════════════════════════════════════════════════════════

@test("Batch process items")
def _():
    from processing.batch_processor import BatchProcessor
    async def _run():
        bp = BatchProcessor(batch_size=3, timeout=5)
        async def double(x):
            return x * 2
        result = await bp.process_batch([1, 2, 3, 4, 5], double)
        assert result.processed_count == 5
        assert result.results == [2, 4, 6, 8, 10]
    run_async(_run())

@test("Batch handles failures gracefully")
def _():
    from processing.batch_processor import BatchProcessor, BatchState
    async def _run():
        bp = BatchProcessor(batch_size=5, timeout=5, max_retries=0)
        async def fail_on_3(x):
            if x == 3:
                raise ValueError("bad input")
            return x
        result = await bp.process_batch([1, 2, 3, 4], fail_on_3)
        assert result.state == BatchState.PARTIAL
        assert result.failed_count == 1
    run_async(_run())


# ══════════════════════════════════════════════════════════════════════
print("\n═══ 14. FULL INTEGRATION TEST ═══")
# ══════════════════════════════════════════════════════════════════════

@test("HermesAgentCore sync processing end-to-end")
def _():
    from core.hermes_core import HermesAgentCore, AgentConfig
    config = AgentConfig(tokens_per_minute=3000, max_context=16000)
    agent = HermesAgentCore(config)
    result = agent.process("Write a Python function to reverse a string")
    assert result.success, f"Failed: {result.error}"
    assert len(result.response) > 10
    assert result.metadata.get("provider") == "mock"
    agent.shutdown()

@test("HermesAgentCore async processing end-to-end")
def _():
    from core.hermes_core import HermesAgentCore, AgentConfig
    async def _run():
        config = AgentConfig(tokens_per_minute=3000, max_context=16000)
        agent = HermesAgentCore(config)
        result = await agent.process_async("Analyze some data trends")
        assert result.success, f"Failed: {result.error}"
        assert result.metadata.get("mode") == "async"
        agent.shutdown()
    run_async(_run())

@test("HermesAgentCore cache deduplication")
def _():
    from core.hermes_core import HermesAgentCore, AgentConfig
    agent = HermesAgentCore(AgentConfig())
    r1 = agent.process("Hello there")
    r2 = agent.process("Hello there")
    assert r2.metadata.get("cache_hit") is True
    agent.shutdown()

@test("HermesAgentCore health and metrics")
def _():
    from core.hermes_core import HermesAgentCore, AgentConfig
    agent = HermesAgentCore(AgentConfig())
    agent.process("test message")
    health = agent.get_health()
    assert health["state"] == "ready"
    assert health["processed_count"] >= 1
    metrics = agent.get_metrics()
    assert "resilience" in metrics
    assert "context" in metrics
    assert "persistence" in metrics
    agent.shutdown()

@test("HermesAgentCore event bus fires on processing")
def _():
    from core.hermes_core import HermesAgentCore, AgentConfig
    agent = HermesAgentCore(AgentConfig())
    events_received = []
    agent.events.subscribe("task.completed", lambda e: events_received.append(e))
    agent.process("Hello")
    assert len(events_received) >= 1
    assert events_received[0].data["task_type"] is not None
    agent.shutdown()


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 50}")
print(f"  Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
if FAILED == 0:
    print("  🎉 ALL TESTS PASSED")
else:
    print(f"  ⚠️  {FAILED} test(s) need attention")
print(f"{'═' * 50}\n")

sys.exit(1 if FAILED > 0 else 0)
