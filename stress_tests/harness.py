"""
Stress Testing Harness
========================

Comprehensive stress testing for the Hermes Agent under load:
  1. Concurrent load testing (N simultaneous requests)
  2. Burst testing (sudden spike)
  3. Sustained throughput testing (requests/sec over duration)
  4. Memory pressure testing (large context, many sessions)
  5. Chaos injection (random failures, latency spikes)
  6. Swarm stress testing (many sub-agents)

Usage:
    python -m stress_tests.harness --test all --concurrency 20
"""

import asyncio
import logging
import time
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Results from a stress test run."""
    test_name: str
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_duration_sec: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / max(1, self.total_requests)

    @property
    def throughput_rps(self) -> float:
        return self.total_requests / max(0.001, self.total_duration_sec)

    @property
    def p50_ms(self) -> float:
        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        return self._percentile(0.95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(0.99)

    def _percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        k = int(len(s) * p)
        return s[min(k, len(s) - 1)]

    def summary(self) -> Dict[str, Any]:
        return {
            "test": self.test_name,
            "total": self.total_requests,
            "success_rate": f"{self.success_rate:.1%}",
            "throughput_rps": round(self.throughput_rps, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
            "duration_sec": round(self.total_duration_sec, 1),
            "errors": len(self.errors),
        }


class StressHarness:
    """
    Stress testing harness for HermesAgentCore.
    """

    SAMPLE_MESSAGES = [
        "Write a Python function to reverse a linked list",
        "Analyze the security implications of JWT tokens",
        "Debug this error: TypeError in line 42",
        "Hello, how are you today?",
        "Create a comprehensive report on microservices architecture",
        "Explain buffer overflow vulnerabilities in C programs",
        "Write unit tests for a REST API endpoint",
        "Compare Redis vs Memcached for session storage",
        "What are the OWASP Top 10 vulnerabilities?",
        "Design a load balancer with health checking",
    ]

    def __init__(self, agent=None):
        self.agent = agent

    def _get_agent(self):
        if self.agent:
            return self.agent
        from core.hermes_core import HermesAgentCore, AgentConfig
        from core.llm_provider import ProviderType
        return HermesAgentCore(AgentConfig(
            llm_provider=ProviderType.MOCK,
            max_context=16000,
        ))

    # ── Test 1: Concurrent Load ───────────────────────────────────

    async def test_concurrent_load(
        self, concurrency: int = 20, total_requests: int = 100
    ) -> StressResult:
        """Fire N requests concurrently and measure throughput."""
        agent = self._get_agent()
        result = StressResult(test_name=f"concurrent_load_{concurrency}")
        semaphore = asyncio.Semaphore(concurrency)

        async def single_request(idx: int):
            msg = self.SAMPLE_MESSAGES[idx % len(self.SAMPLE_MESSAGES)]
            async with semaphore:
                start = time.time()
                try:
                    r = await agent.process_async(msg)
                    lat = (time.time() - start) * 1000
                    result.latencies_ms.append(lat)
                    if r.success:
                        result.successful += 1
                    else:
                        result.failed += 1
                        result.errors.append(r.error or "unknown")
                except Exception as e:
                    result.failed += 1
                    result.errors.append(str(e))

        start = time.time()
        tasks = [single_request(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        result.total_duration_sec = time.time() - start
        result.total_requests = total_requests

        agent.shutdown()
        return result

    # ── Test 2: Burst ─────────────────────────────────────────────

    async def test_burst(
        self, burst_size: int = 50
    ) -> StressResult:
        """Send burst_size requests all at once (no semaphore)."""
        agent = self._get_agent()
        result = StressResult(test_name=f"burst_{burst_size}")

        async def single():
            msg = random.choice(self.SAMPLE_MESSAGES)
            start = time.time()
            try:
                r = await agent.process_async(msg)
                result.latencies_ms.append((time.time() - start) * 1000)
                if r.success:
                    result.successful += 1
                else:
                    result.failed += 1
            except Exception as e:
                result.failed += 1
                result.errors.append(str(e))

        start = time.time()
        await asyncio.gather(*[single() for _ in range(burst_size)])
        result.total_duration_sec = time.time() - start
        result.total_requests = burst_size

        agent.shutdown()
        return result

    # ── Test 3: Sustained Throughput ──────────────────────────────

    async def test_sustained_throughput(
        self, target_rps: float = 10, duration_sec: float = 10
    ) -> StressResult:
        """Sustain target_rps requests per second for duration."""
        agent = self._get_agent()
        result = StressResult(test_name=f"sustained_{target_rps}rps_{duration_sec}s")
        interval = 1.0 / target_rps

        start = time.time()
        while time.time() - start < duration_sec:
            req_start = time.time()
            msg = random.choice(self.SAMPLE_MESSAGES)
            try:
                r = await agent.process_async(msg)
                result.latencies_ms.append((time.time() - req_start) * 1000)
                result.total_requests += 1
                if r.success:
                    result.successful += 1
                else:
                    result.failed += 1
            except Exception as e:
                result.total_requests += 1
                result.failed += 1
                result.errors.append(str(e))

            elapsed = time.time() - req_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

        result.total_duration_sec = time.time() - start
        agent.shutdown()
        return result

    # ── Test 4: Memory Pressure ───────────────────────────────────

    async def test_memory_pressure(
        self, sessions: int = 50, messages_per_session: int = 10
    ) -> StressResult:
        """Create many sessions with many messages to stress persistence."""
        agent = self._get_agent()
        result = StressResult(test_name=f"memory_pressure_{sessions}x{messages_per_session}")

        start = time.time()
        for s in range(sessions):
            session_id = f"stress-session-{s}"
            agent.init_session(session_id, {"test": True})

            for m in range(messages_per_session):
                msg = f"Session {s}, message {m}: {random.choice(self.SAMPLE_MESSAGES)}"
                req_start = time.time()
                try:
                    r = agent.process(msg)
                    result.latencies_ms.append((time.time() - req_start) * 1000)
                    result.total_requests += 1
                    if r.success:
                        result.successful += 1
                    else:
                        result.failed += 1
                except Exception as e:
                    result.total_requests += 1
                    result.failed += 1

        result.total_duration_sec = time.time() - start
        agent.shutdown()
        return result

    # ── Test 5: Chaos Injection ───────────────────────────────────

    async def test_chaos(
        self, requests: int = 50, failure_rate: float = 0.3
    ) -> StressResult:
        """
        Inject random failures to test resilience and recovery.
        The agent should gracefully handle failures and maintain
        acceptable success rate on non-injected requests.
        """
        agent = self._get_agent()
        result = StressResult(test_name=f"chaos_{failure_rate:.0%}_failure")

        # Monkey-patch the mock provider to randomly fail
        original_complete = agent.provider_manager._providers[-1].complete.__func__

        async def chaotic_complete(self_provider, request):
            if random.random() < failure_rate:
                raise ConnectionError("Chaos monkey: simulated failure")
            return await original_complete(self_provider, request)

        # Apply chaos to the last provider (mock)
        mock_provider = agent.provider_manager._providers[-1]
        import types
        mock_provider.complete = types.MethodType(chaotic_complete, mock_provider)

        start = time.time()
        for i in range(requests):
            msg = random.choice(self.SAMPLE_MESSAGES)
            req_start = time.time()
            try:
                r = agent.process(msg)
                result.latencies_ms.append((time.time() - req_start) * 1000)
                result.total_requests += 1
                if r.success:
                    result.successful += 1
                else:
                    result.failed += 1
                    result.errors.append(r.error or "chaos")
            except Exception as e:
                result.total_requests += 1
                result.failed += 1
                result.errors.append(str(e))

        result.total_duration_sec = time.time() - start
        agent.shutdown()
        return result

    # ── Run All ───────────────────────────────────────────────────

    async def run_all(self) -> List[StressResult]:
        """Run the complete stress test suite."""
        print("\n" + "=" * 60)
        print("  HERMES AGENT STRESS TEST SUITE")
        print("=" * 60)

        results = []

        tests = [
            ("Concurrent Load (20 workers, 100 requests)",
             self.test_concurrent_load(20, 100)),
            ("Burst (50 simultaneous)",
             self.test_burst(50)),
            ("Sustained Throughput (10 RPS, 5s)",
             self.test_sustained_throughput(10, 5)),
            ("Memory Pressure (20 sessions × 5 messages)",
             self.test_memory_pressure(20, 5)),
            ("Chaos Injection (30% failure rate)",
             self.test_chaos(50, 0.3)),
        ]

        for name, coro in tests:
            print(f"\n  Running: {name}...")
            try:
                result = await coro
                results.append(result)
                s = result.summary()
                print(f"    ✅ {s['success_rate']} success | "
                      f"{s['throughput_rps']} req/s | "
                      f"p50={s['p50_ms']}ms p95={s['p95_ms']}ms | "
                      f"{s['errors']} errors")
            except Exception as e:
                print(f"    ❌ FAILED: {e}")

        print("\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        for r in results:
            s = r.summary()
            print(f"  {s['test']:45s} | {s['success_rate']:>6s} | {s['throughput_rps']:>6.1f} rps | p95={s['p95_ms']:>7.1f}ms")
        print("=" * 60 + "\n")

        return results


# ── CLI ───────────────────────────────────────────────────────────

async def main():
    os.makedirs(os.path.expanduser("~/.hermes"), exist_ok=True)
    harness = StressHarness()
    await harness.run_all()


if __name__ == "__main__":
    asyncio.run(main())
