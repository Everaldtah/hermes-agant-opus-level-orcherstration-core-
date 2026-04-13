# Hermes Agent — Opus-Level Orchestration Core

> Multi-agent orchestration engine with sub-agent swarm delegation, three-tier cognitive memory, and multi-provider LLM failover.

**9,500+ lines of Python** | **49 unit tests** | **5 stress tests** | **Zero external dependencies for core operation**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HermesAgentCore                           │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────┐  │
│  │Resilience│ │  Context   │ │Persistence│ │  Event Bus   │  │
│  │ Engine   │ │Intelligence│ │  (Delta)  │ │  (pub/sub)   │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LLM Provider Manager                     │   │
│  │  GLM-5.1 → Kimi K2.5 → Claude → LM Studio → Mock    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌────────────────┐ ┌───────────────────────────────────┐   │
│  │  Task           │ │  Swarm Orchestrator               │   │
│  │  Orchestrator   │ │  • Orchestrator-Worker            │   │
│  │  (multi-step)   │ │  • Fan-Out / Fan-In              │   │
│  │                 │ │  • Pipeline                       │   │
│  │                 │ │  • Consensus Voting               │   │
│  └────────────────┘ └───────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Three-Tier Memory System                   │   │
│  │  Episodic (events) → Semantic (facts) → Procedural   │   │
│  │  (SOPs)  with automatic consolidation pipeline        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐   │
│  │  Cache    │ │  Load     │ │Connection│ │  Self-     │   │
│  │  (L1/L2/ │ │  Balancer │ │  Pool    │ │  Healing   │   │
│  │   L3)    │ │  (6 algo) │ │          │ │  Recovery  │   │
│  └──────────┘ └───────────┘ └──────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from core.hermes_core import HermesAgentCore, AgentConfig
from core.llm_provider import ProviderType

# AUTO mode reads API keys from env vars and builds the full failover chain
agent = HermesAgentCore(AgentConfig(
    llm_provider=ProviderType.AUTO,
    max_context=120000,
))

# Simple request
result = agent.process("Explain buffer overflows")
print(result.response)

# Complex multi-step task (auto-decomposes)
import asyncio
result = asyncio.run(agent.process_complex(
    "Do a comprehensive security audit of this web application"
))

# Sub-agent swarm: fan-out to specialists
from swarm.orchestrator import SubAgentConfig, AgentRole

async def run_swarm():
    result = await agent.swarm.fan_out_fan_in(
        task="Analyze this codebase",
        agents=[
            SubAgentConfig(name="security", role=AgentRole.SECURITY),
            SubAgentConfig(name="quality", role=AgentRole.REVIEWER),
            SubAgentConfig(name="perf", role=AgentRole.ANALYST),
        ],
    )
    print(f"Completed: {result.agents_succeeded} agents, {result.total_tokens} tokens")

asyncio.run(run_swarm())

# Consensus voting (for trading bots, risk decisions)
async def run_consensus():
    result = await agent.swarm.consensus(
        task="Should we go long on ETH based on current signals?",
        agents=[
            SubAgentConfig(name=f"analyst-{i}", role=AgentRole.ANALYST)
            for i in range(5)
        ],
        threshold=0.66,  # 2/3 must agree
    )
    print(result.final_output)

asyncio.run(run_consensus())

# Long-term memory
from memory.long_term import MemoryRecord, MemoryTier

agent.memory_store.store(MemoryRecord(
    tier=MemoryTier.EPISODIC,
    content="User prefers detailed security reports with CVSS scores",
    importance=0.8,
    tags=["preference", "security"],
))

# Retrieve relevant memories
memories = agent.memory_store.retrieve("security report format", limit=5)

# Consolidate episodic patterns into semantic knowledge
agent.memory_store.consolidate_episodes_to_semantic()

# Store a learned procedure (SOP)
agent.memory_store.store_procedure(
    name="VPS Security Hardening",
    steps=["Update packages", "Configure firewall", "Set up fail2ban", "Enable SSH keys only"],
    success_rate=0.95,
)
```

## Environment Variables

```bash
export GLM_API_KEY="your-zhipu-key"           # Primary: GLM-5.1
export OPENROUTER_API_KEY="your-or-key"       # Kimi K2.5 via OpenRouter
export KIMI_API_KEY="your-moonshot-key"       # Kimi direct API (backup)
export ANTHROPIC_API_KEY="sk-ant-..."         # Claude fallback
# LM Studio: just needs to be running on localhost:1234
```

## Provider Routing (AUTO mode)

| Priority | Provider | Model | Trigger |
|----------|----------|-------|---------|
| 1 | GLM (Zhipu AI) | glm-4-plus | Primary preference |
| 2 | Kimi K2.5 (OpenRouter) | moonshotai/kimi-k2 | Active session model |
| 3 | Kimi (direct API) | moonshot-v1-128k | OpenRouter backup |
| 4 | Claude (Anthropic) | claude-sonnet-4 | API limit fallback |
| 5 | LM Studio | local model | Offline / air-gapped |
| 6 | Mock | mock-v1 | Last resort (agent never crashes) |

Each provider auto-fails-over to the next. CJK-aware token counting for GLM/Kimi.

## Swarm Orchestration Patterns

| Pattern | How It Works | Best For |
|---------|-------------|----------|
| **Orchestrator-Worker** | Planner decomposes task → workers execute in parallel → aggregate | Large analysis tasks |
| **Pipeline** | Agent A output feeds Agent B input, sequential chain | Report generation, ETL |
| **Fan-Out/Fan-In** | All agents work same task in parallel → synthesizer combines | Multi-perspective analysis |
| **Consensus** | Agents vote on a decision, majority threshold wins | Trading bots, risk assessment |
| **Hierarchical** | Feature leads spawn their own specialists (tree delegation) | Complex codebases |

## Three-Tier Memory System

Inspired by MemGPT, CoALA framework, and the ICLR 2026 MemAgents workshop:

| Tier | What It Stores | Example |
|------|---------------|---------|
| **Episodic** | Specific events with timestamps | "User asked about SQL injection on Apr 13" |
| **Semantic** | Distilled facts and knowledge | "SQL injection is recurring concern in this project" |
| **Procedural** | Learned workflows / SOPs | "Security audit: 1) Map surface 2) OWASP check 3) Report" |

**Automatic consolidation**: Episodic memories with recurring patterns are automatically distilled into semantic facts. Strategic forgetting prunes low-importance memories over time.

## Stress Testing

```bash
# Run the full stress test suite
python -m stress_tests.harness

# Individual tests
python -c "
import asyncio
from stress_tests.harness import StressHarness
h = StressHarness()
# 20 concurrent workers, 100 total requests
asyncio.run(h.test_concurrent_load(20, 100))
# 30% random failure injection
asyncio.run(h.test_chaos(50, 0.3))
"
```

Stress test suite includes:
- **Concurrent load**: N workers × M requests with semaphore control
- **Burst**: All requests fired simultaneously
- **Sustained throughput**: Target RPS over duration
- **Memory pressure**: Many sessions × many messages
- **Chaos injection**: Random failures at configurable rate

## Running Tests

```bash
cd hermes_agent_upgraded

# Unit tests (49 tests, all modules)
python tests/test_hermes_v21.py

# Quick smoke test
python -c "
from core.hermes_core import HermesAgentCore, AgentConfig
agent = HermesAgentCore(AgentConfig())
r = agent.process('Hello')
print(f'Success: {r.success}, Provider: {r.metadata.get(\"provider\")}')
agent.shutdown()
"
```

## Project Structure

```
hermes_agent_upgraded/
├── core/                          # Core orchestration engine
│   ├── hermes_core.py             # Main agent class (928 lines)
│   ├── llm_provider.py            # 7 LLM providers with failover (780 lines)
│   ├── task_orchestrator.py       # Multi-step task decomposition
│   ├── event_bus.py               # Pub/sub inter-component messaging
│   ├── resilience_enhanced.py     # Circuit breaker, adaptive token bucket
│   ├── context_enhanced.py        # Task classification, token allocation
│   └── persistence_enhanced.py    # Delta persistence, session checkpoints
├── swarm/                         # Sub-agent orchestration
│   └── orchestrator.py            # 5 swarm patterns (450 lines)
├── memory/                        # Three-tier cognitive memory
│   └── long_term.py               # Episodic/semantic/procedural (360 lines)
├── processing/                    # Async engine, worker pool, batch
│   ├── async_engine.py
│   ├── worker_pool.py
│   └── batch_processor.py
├── efficiency/                    # Performance optimization
│   ├── cache_manager.py           # L1/L2/L3 multi-level cache
│   ├── load_balancer.py           # 6 load balancing strategies
│   └── connection_pool.py         # Reusable connection pool
├── monitoring/                    # Observability
│   └── metrics_collector.py       # Real-time metrics, Prometheus export
├── recovery/                      # Fault tolerance
│   └── self_healing.py            # 5 retry strategies, circuit breaker
├── stress_tests/                  # Load testing
│   └── harness.py                 # 5 stress test scenarios
├── tests/                         # Test suite
│   ├── test_hermes_v21.py         # 49 unit tests
│   └── test_hermes_core.py        # Legacy tests
├── setup.py
├── requirements.txt
└── README.md
```

## License

MIT
