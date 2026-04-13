"""
Long-Term Memory System
========================

Three-tier cognitive memory inspired by human memory architecture
(CoALA framework, MemGPT, and the ICLR 2026 MemAgents workshop).

Tiers:
  1. Episodic  — specific events/interactions with timestamps
  2. Semantic  — distilled facts, rules, knowledge (consolidated from episodes)
  3. Procedural — learned skills, SOPs, reusable workflows

Storage: SQLite + JSON (zero external dependencies).
Retrieval: TF-IDF keyword relevance scoring (no vector DB needed,
           but the interface is ready for a vector backend swap).

Key design decisions:
- Consolidation pipeline: episodic → semantic (automatic pattern extraction)
- Importance scoring: each memory gets a decaying importance score
- Strategic forgetting: low-importance memories are pruned on schedule
- Session-scoped working memory feeds into long-term on session close
"""

import json
import math
import os
import re
import sqlite3
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryRecord:
    """A single memory entry."""
    id: str = ""
    tier: MemoryTier = MemoryTier.EPISODIC
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5        # 0.0–1.0
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0
    session_id: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""               # which agent/step created this

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    @property
    def decayed_importance(self) -> float:
        """Importance decays logarithmically with age (hours)."""
        decay = 1.0 / (1.0 + 0.05 * math.log1p(self.age_hours))
        recency_boost = min(0.2, self.access_count * 0.02)
        return min(1.0, self.importance * decay + recency_boost)


class MemoryStore:
    """
    SQLite-backed three-tier memory store.

    Designed for single-node deployment (your VPS). For distributed
    swarm agents, swap this for a Redis or Postgres backend using
    the same interface.
    """

    def __init__(self, db_path: str = "~/.hermes/memory.db"):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._seq = 0
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    session_id TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_tier
                ON memories(tier)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_importance
                ON memories(importance DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_session
                ON memories(session_id)
            """)

    def _next_id(self, tier: MemoryTier) -> str:
        self._seq += 1
        return f"{tier.value[:3]}-{int(time.time())}-{self._seq}"

    # ── CRUD ──────────────────────────────────────────────────────

    def store(self, record: MemoryRecord) -> str:
        """Store a memory record. Returns the assigned ID."""
        with self._lock:
            if not record.id:
                record.id = self._next_id(record.tier)
            if not record.created_at:
                record.created_at = time.time()
            if not record.accessed_at:
                record.accessed_at = record.created_at

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories
                    (id, tier, content, metadata, importance, created_at,
                     accessed_at, access_count, session_id, tags, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id, record.tier.value, record.content,
                    json.dumps(record.metadata), record.importance,
                    record.created_at, record.accessed_at,
                    record.access_count, record.session_id,
                    json.dumps(record.tags), record.source,
                ))
            return record.id

    def retrieve(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        session_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve memories relevant to a query using TF-IDF keyword scoring.
        Falls back to recency if query is empty.
        """
        with self._lock:
            conditions = []
            params: list = []

            if tier:
                conditions.append("tier = ?")
                params.append(tier.value)
            if min_importance > 0:
                conditions.append("importance >= ?")
                params.append(min_importance)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            where = "WHERE " + " AND ".join(conditions) if conditions else ""

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM memories {where} ORDER BY importance DESC, accessed_at DESC LIMIT 500",
                    params,
                ).fetchall()

            records = [self._row_to_record(r) for r in rows]

            if query.strip():
                scored = self._rank_by_relevance(query, records)
                scored.sort(key=lambda x: x[1], reverse=True)
                records = [rec for rec, _score in scored[:limit]]
            else:
                records = records[:limit]

            # Update access stats
            now = time.time()
            ids = [r.id for r in records]
            if ids:
                with sqlite3.connect(self.db_path) as conn:
                    placeholders = ",".join("?" * len(ids))
                    conn.execute(
                        f"UPDATE memories SET accessed_at=?, access_count=access_count+1 WHERE id IN ({placeholders})",
                        [now] + ids,
                    )

            return records

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                return cur.rowcount > 0

    def prune(self, max_age_hours: float = 720, min_importance: float = 0.1) -> int:
        """
        Strategic forgetting: remove old, low-importance memories.
        Default: prune memories older than 30 days with importance < 0.1
        """
        cutoff = time.time() - (max_age_hours * 3600)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM memories WHERE created_at < ? AND importance < ?",
                    (cutoff, min_importance),
                )
                pruned = cur.rowcount
                conn.commit()

        if pruned:
            logger.info(f"Pruned {pruned} stale memories")
        return pruned

    # ── Consolidation ─────────────────────────────────────────────

    def consolidate_episodes_to_semantic(
        self,
        session_id: str = "",
        min_episodes: int = 3,
    ) -> List[str]:
        """
        MemGPT-style consolidation: find repeated patterns in episodic
        memories and distill them into semantic facts.

        Returns list of new semantic memory IDs.
        """
        episodes = self.retrieve(
            "", tier=MemoryTier.EPISODIC,
            session_id=session_id if session_id else None,
            limit=200,
        )

        if len(episodes) < min_episodes:
            return []

        # Extract repeated keywords/phrases
        word_counts: Counter = Counter()
        for ep in episodes:
            words = set(re.findall(r'\b\w{4,}\b', ep.content.lower()))
            word_counts.update(words)

        # Find themes (words appearing in 30%+ of episodes)
        threshold = max(2, len(episodes) * 0.3)
        common_themes = [w for w, c in word_counts.items() if c >= threshold]

        if not common_themes:
            return []

        new_ids = []
        # Create semantic memories from episode clusters
        theme_groups: Dict[str, List[MemoryRecord]] = {}
        for theme in common_themes[:10]:  # top 10 themes
            related = [
                ep for ep in episodes
                if theme in ep.content.lower()
            ]
            if len(related) >= min_episodes:
                theme_groups[theme] = related

        for theme, related_eps in theme_groups.items():
            # Compress into a semantic fact
            sources = [ep.content[:100] for ep in related_eps[:5]]
            semantic_content = (
                f"[Consolidated from {len(related_eps)} episodes] "
                f"Recurring pattern around '{theme}': "
                + " | ".join(sources)
            )

            avg_importance = sum(ep.importance for ep in related_eps) / len(related_eps)

            mid = self.store(MemoryRecord(
                tier=MemoryTier.SEMANTIC,
                content=semantic_content,
                importance=min(1.0, avg_importance + 0.1),  # boost consolidated
                tags=[theme, "consolidated"],
                source="consolidation_engine",
                metadata={
                    "episode_count": len(related_eps),
                    "episode_ids": [ep.id for ep in related_eps[:10]],
                },
            ))
            new_ids.append(mid)

        logger.info(f"Consolidated {len(new_ids)} semantic memories from episodes")
        return new_ids

    def store_procedure(
        self,
        name: str,
        steps: List[str],
        success_rate: float = 1.0,
        context: str = "",
    ) -> str:
        """
        Store a learned procedure (SOP) in procedural memory.
        These are reusable workflows the agent has learned.
        """
        return self.store(MemoryRecord(
            tier=MemoryTier.PROCEDURAL,
            content=f"Procedure: {name}\nSteps:\n" + "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(steps)
            ),
            importance=0.7 + (success_rate * 0.3),
            tags=["procedure", name.lower().replace(" ", "_")],
            source="procedural_learning",
            metadata={
                "name": name,
                "steps": steps,
                "success_rate": success_rate,
                "context": context,
            },
        ))

    # ── Helpers ────────────────────────────────────────────────────

    def _rank_by_relevance(
        self, query: str, records: List[MemoryRecord]
    ) -> List[Tuple[MemoryRecord, float]]:
        """Simple TF-IDF style relevance scoring."""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return [(r, r.decayed_importance) for r in records]

        scored = []
        for rec in records:
            rec_words = set(re.findall(r'\b\w{3,}\b', rec.content.lower()))
            overlap = query_words & rec_words
            if overlap:
                # Term overlap * importance * recency
                keyword_score = len(overlap) / len(query_words)
                relevance = keyword_score * 0.6 + rec.decayed_importance * 0.4
                scored.append((rec, relevance))
            else:
                scored.append((rec, rec.decayed_importance * 0.1))

        return scored

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            tier=MemoryTier(row["tier"]),
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            importance=row["importance"],
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
            session_id=row["session_id"],
            tags=json.loads(row["tags"]),
            source=row["source"],
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                by_tier = {}
                for tier in MemoryTier:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE tier=?", (tier.value,)
                    ).fetchone()[0]
                    by_tier[tier.value] = count

                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            return {
                "total_memories": total,
                "by_tier": by_tier,
                "db_size_bytes": db_size,
            }
