"""
Connection Pool Manager
========================

Reusable connection pool for HTTP/API connections.
Prevents socket exhaustion and reduces latency from
repeated connection setup when calling LLM APIs.
"""

import asyncio
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class PooledConnection:
    """Wrapper around a raw connection with lifecycle tracking."""
    id: str
    connection: Any  # The underlying connection object
    state: ConnectionState = ConnectionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    errors: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used


class ConnectionPool:
    """
    Generic connection pool with health checking and auto-scaling.

    Parameters:
        factory:      callable that creates a new raw connection
        destroyer:    callable that closes a raw connection
        min_size:     minimum idle connections to maintain
        max_size:     hard ceiling on total connections
        max_idle_sec: recycle connections idle longer than this
        max_age_sec:  recycle connections older than this
        health_check: optional callable(conn) -> bool
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        destroyer: Optional[Callable[[Any], None]] = None,
        min_size: int = 2,
        max_size: int = 20,
        max_idle_sec: float = 300.0,
        max_age_sec: float = 1800.0,
        health_check: Optional[Callable[[Any], bool]] = None,
    ):
        self._factory = factory
        self._destroyer = destroyer or (lambda c: None)
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_sec = max_idle_sec
        self.max_age_sec = max_age_sec
        self._health_check = health_check

        self._idle: deque[PooledConnection] = deque()
        self._active: Dict[str, PooledConnection] = {}
        self._seq = 0
        self._lock = threading.RLock()
        self._shutdown = False

        # Metrics
        self._stats = {
            "acquired": 0,
            "released": 0,
            "created": 0,
            "destroyed": 0,
            "errors": 0,
            "waits": 0,
        }

        # Pre-warm the pool
        self._warm_pool()

        # Background maintenance
        self._maint_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self._maint_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, timeout: float = 10.0) -> PooledConnection:
        """
        Acquire a connection from the pool.

        Blocks up to *timeout* seconds if the pool is exhausted.
        The caller MUST call release() when done.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                # Try to get an idle connection
                while self._idle:
                    pc = self._idle.popleft()
                    if self._is_healthy(pc):
                        pc.state = ConnectionState.ACTIVE
                        pc.last_used = time.time()
                        pc.use_count += 1
                        self._active[pc.id] = pc
                        self._stats["acquired"] += 1
                        return pc
                    else:
                        self._destroy(pc)

                # No idle connections — create if under limit
                total = len(self._active) + len(self._idle)
                if total < self.max_size:
                    pc = self._create()
                    pc.state = ConnectionState.ACTIVE
                    pc.use_count += 1
                    self._active[pc.id] = pc
                    self._stats["acquired"] += 1
                    return pc

            # Pool exhausted — wait briefly and retry
            self._stats["waits"] += 1
            time.sleep(0.05)

        raise TimeoutError(f"Could not acquire connection within {timeout}s")

    def release(self, pc: PooledConnection):
        """Return a connection to the pool."""
        with self._lock:
            self._active.pop(pc.id, None)
            pc.state = ConnectionState.IDLE
            pc.last_used = time.time()
            self._idle.append(pc)
            self._stats["released"] += 1

    def close(self):
        """Shut down the pool and destroy all connections."""
        self._shutdown = True
        with self._lock:
            for pc in list(self._idle):
                self._destroy(pc)
            for pc in list(self._active.values()):
                self._destroy(pc)
            self._idle.clear()
            self._active.clear()
        logger.info("Connection pool closed")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    class _Lease:
        """Context manager that auto-releases on exit."""
        def __init__(self, pool: "ConnectionPool"):
            self._pool = pool
            self.pc: Optional[PooledConnection] = None

        def __enter__(self) -> PooledConnection:
            self.pc = self._pool.acquire()
            return self.pc

        def __exit__(self, *exc):
            if self.pc is not None:
                if exc[0] is not None:
                    self.pc.errors += 1
                self._pool.release(self.pc)

    def lease(self) -> "_Lease":
        """Usage:  with pool.lease() as conn: ..."""
        return self._Lease(self)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create(self) -> PooledConnection:
        self._seq += 1
        cid = f"conn-{self._seq}"
        try:
            raw = self._factory()
        except Exception as e:
            self._stats["errors"] += 1
            raise RuntimeError(f"Failed to create connection: {e}") from e

        pc = PooledConnection(id=cid, connection=raw)
        self._stats["created"] += 1
        logger.debug(f"Created connection {cid}")
        return pc

    def _destroy(self, pc: PooledConnection):
        try:
            self._destroyer(pc.connection)
        except Exception as e:
            logger.warning(f"Error destroying {pc.id}: {e}")
        pc.state = ConnectionState.CLOSED
        self._stats["destroyed"] += 1

    def _is_healthy(self, pc: PooledConnection) -> bool:
        if pc.idle_seconds > self.max_idle_sec:
            return False
        if pc.age_seconds > self.max_age_sec:
            return False
        if self._health_check:
            try:
                return self._health_check(pc.connection)
            except Exception:
                return False
        return True

    def _warm_pool(self):
        with self._lock:
            for _ in range(self.min_size):
                try:
                    pc = self._create()
                    self._idle.append(pc)
                except Exception as e:
                    logger.warning(f"Pool warm-up failed: {e}")

    def _maintenance_loop(self):
        while not self._shutdown:
            time.sleep(30)
            if self._shutdown:
                break
            self._evict_stale()
            self._ensure_minimum()

    def _evict_stale(self):
        with self._lock:
            to_remove = []
            for pc in self._idle:
                if not self._is_healthy(pc):
                    to_remove.append(pc)
            for pc in to_remove:
                self._idle.remove(pc)
                self._destroy(pc)

    def _ensure_minimum(self):
        with self._lock:
            deficit = self.min_size - len(self._idle)
            for _ in range(max(0, deficit)):
                try:
                    pc = self._create()
                    self._idle.append(pc)
                except Exception:
                    break

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "idle": len(self._idle),
                "active": len(self._active),
                "total": len(self._idle) + len(self._active),
                "min_size": self.min_size,
                "max_size": self.max_size,
                **self._stats,
            }
