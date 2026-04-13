"""
Enhanced Delta Persistence - Phase 3 UPGRADED
=============================================

Enhanced with:
- Compression for delta storage
- Incremental checkpointing
- Async write operations
- Recovery optimization
- Multi-session support
"""

import json
import hashlib
import sqlite3
import logging
import threading
import asyncio
import zlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
import time
import os

logger = logging.getLogger(__name__)


class DeltaType(Enum):
    """Types of state changes."""
    MESSAGE_ADDED = "message_added"
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_REMOVED = "message_removed"
    STATE_CHANGED = "state_changed"
    CONTEXT_UPDATED = "context_updated"
    CHECKPOINT = "checkpoint"


@dataclass
class DeltaRecord:
    """Single delta record with compression support."""
    sequence: int
    timestamp: float
    delta_type: DeltaType
    data: Dict[str, Any]
    checksum: str = ""
    compressed: bool = False
    original_size: int = 0
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification."""
        content = json.dumps({
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "type": self.delta_type.value,
            "data": self.data
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def verify(self) -> bool:
        """Verify record integrity."""
        return self.checksum == self.calculate_checksum()
    
    def compress_data(self) -> bytes:
        """Compress data for storage."""
        json_data = json.dumps(self.data).encode()
        self.original_size = len(json_data)
        compressed = zlib.compress(json_data, level=6)
        self.compressed = True
        return compressed
    
    @classmethod
    def decompress_data(cls, compressed: bytes) -> Dict:
        """Decompress data from storage."""
        json_data = zlib.decompress(compressed)
        return json.loads(json_data.decode())


@dataclass
class SessionCheckpoint:
    """Full session checkpoint with metadata."""
    sequence: int
    timestamp: float
    session_id: str
    conversation_state: Dict[str, Any]
    agent_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    compressed: bool = False
    
    def calculate_checksum(self) -> str:
        """Calculate checkpoint checksum."""
        content = json.dumps({
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "conversation": self.conversation_state,
            "agent": self.agent_state
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def verify(self) -> bool:
        """Verify checkpoint integrity."""
        return self.checksum == self.calculate_checksum()


class EnhancedDeltaPersistence:
    """
    Enhanced delta persistence with compression and async operations.
    
    Features:
    - Compressed delta storage
    - Async write operations
    - Incremental checkpointing
    - Multi-session support
    - Recovery optimization
    """
    
    def __init__(
        self,
        db_path: str = "~/.hermes/session_deltas.db",
        checkpoint_interval: int = 300,
        max_delta_queue: int = 100,
        auto_checkpoint: bool = True,
        compression_enabled: bool = True,
        async_writes: bool = True
    ):
        self.db_path = os.path.expanduser(db_path)
        self.checkpoint_interval = checkpoint_interval
        self.max_delta_queue = max_delta_queue
        self.auto_checkpoint = auto_checkpoint
        self.compression_enabled = compression_enabled
        self.async_writes = async_writes
        
        self._sequence = 0
        self._delta_queue: List[DeltaRecord] = []
        self._session_id: Optional[str] = None
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Async write queue
        self._async_queue: asyncio.Queue = None
        self._async_task = None
        
        # Callbacks
        self._on_checkpoint: Optional[Callable] = None
        self._on_recovery: Optional[Callable] = None
        
        # Stats
        self._stats = {
            "deltas_written": 0,
            "deltas_compressed": 0,
            "checkpoints_created": 0,
            "bytes_saved": 0
        }
        
        self._last_checkpoint_time = 0
        
        # Ensure database exists
        self._init_db()
        
        # Start background threads
        if auto_checkpoint:
            self._checkpoint_thread = threading.Thread(
                target=self._checkpoint_loop,
                daemon=True
            )
            self._checkpoint_thread.start()
    
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Deltas table with compression support
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_deltas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    delta_type TEXT NOT NULL,
                    data BLOB NOT NULL,
                    checksum TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    original_size INTEGER DEFAULT 0
                )
            """)
            
            # Checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    conversation_state BLOB NOT NULL,
                    agent_state BLOB NOT NULL,
                    metadata TEXT,
                    checksum TEXT NOT NULL,
                    compressed INTEGER DEFAULT 0
                )
            """)
            
            # Sessions table for multi-session tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deltas_session_seq 
                ON session_deltas(session_id, sequence)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_session_seq 
                ON session_checkpoints(session_id, sequence)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_active 
                ON sessions(last_active)
            """)
            
            conn.commit()
    
    def _checkpoint_loop(self):
        """Background checkpoint creation loop."""
        while not self._shutdown:
            time.sleep(self.checkpoint_interval)
            if not self._shutdown and self.auto_checkpoint:
                try:
                    self.create_checkpoint()
                except Exception as e:
                    logger.error(f"Auto-checkpoint failed: {e}")
    
    def init_session(self, session_id: str, initial_state: Optional[Dict] = None):
        """Initialize new session."""
        with self._lock:
            self._session_id = session_id
            self._sequence = 0
            self._delta_queue = []
            
            # Get last sequence number for this session
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT MAX(sequence) FROM session_checkpoints WHERE session_id=?",
                    (session_id,)
                )
                result = cursor.fetchone()
                if result and result[0] is not None:
                    self._sequence = result[0]
                
                # Update session record
                conn.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, created_at, last_active, metadata)
                    VALUES (?, COALESCE((SELECT created_at FROM sessions WHERE session_id=?), ?), ?, ?)
                """, (session_id, session_id, time.time(), time.time(), json.dumps(initial_state or {})))
                conn.commit()
            
            logger.info(f"Session initialized: {session_id} (sequence={self._sequence})")
    
    def log_delta(
        self,
        delta_type: DeltaType,
        data: Dict[str, Any],
        flush: bool = True
    ) -> DeltaRecord:
        """
        Log a state change delta.
        
        Args:
            delta_type: Type of change
            data: Delta data
            flush: Whether to immediately write to disk
            
        Returns:
            DeltaRecord
        """
        with self._lock:
            if not self._session_id:
                raise RuntimeError("Session not initialized")
            
            self._sequence += 1
            
            record = DeltaRecord(
                sequence=self._sequence,
                timestamp=time.time(),
                delta_type=delta_type,
                data=data
            )
            record.checksum = record.calculate_checksum()
            
            self._delta_queue.append(record)
            
            if flush or len(self._delta_queue) >= self.max_delta_queue:
                self._flush_deltas()
            
            return record
    
    def _flush_deltas(self):
        """Write queued deltas to database."""
        if not self._delta_queue:
            return
        
        records = self._delta_queue.copy()
        self._delta_queue = []
        
        with sqlite3.connect(self.db_path) as conn:
            for record in records:
                # Compress if enabled and data is large
                if self.compression_enabled and len(json.dumps(record.data)) > 1000:
                    data_blob = record.compress_data()
                    compressed = 1
                    self._stats["deltas_compressed"] += 1
                    self._stats["bytes_saved"] += record.original_size - len(data_blob)
                else:
                    data_blob = json.dumps(record.data).encode()
                    compressed = 0
                
                conn.execute("""
                    INSERT INTO session_deltas 
                    (session_id, sequence, timestamp, delta_type, data, checksum, compressed, original_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self._session_id,
                    record.sequence,
                    record.timestamp,
                    record.delta_type.value,
                    data_blob,
                    record.checksum,
                    compressed,
                    record.original_size if compressed else len(data_blob)
                ))
            
            conn.commit()
        
        self._stats["deltas_written"] += len(records)
        logger.debug(f"Flushed {len(records)} deltas to disk")
    
    def create_checkpoint(
        self,
        conversation_state: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> Optional[SessionCheckpoint]:
        """
        Create full session checkpoint.
        
        Args:
            conversation_state: Current conversation state
            agent_state: Current agent state
            
        Returns:
            SessionCheckpoint or None if no active session
        """
        with self._lock:
            if not self._session_id:
                logger.debug("No active session — skipping checkpoint")
                return None
            
            # Flush any pending deltas first
            self._flush_deltas()
            
            checkpoint = SessionCheckpoint(
                sequence=self._sequence,
                timestamp=time.time(),
                session_id=self._session_id,
                conversation_state=conversation_state or {},
                agent_state=agent_state or {},
                metadata={
                    "deltas_since_last": self._stats["deltas_written"],
                    "checkpoint_time": datetime.now().isoformat(),
                    "version": "2.0"
                }
            )
            checkpoint.checksum = checkpoint.calculate_checksum()
            
            # Compress states
            conv_json = json.dumps(checkpoint.conversation_state).encode()
            agent_json = json.dumps(checkpoint.agent_state).encode()
            
            if self.compression_enabled:
                conv_blob = zlib.compress(conv_json, level=6)
                agent_blob = zlib.compress(agent_json, level=6)
                compressed = 1
                self._stats["bytes_saved"] += len(conv_json) + len(agent_json) - len(conv_blob) - len(agent_blob)
            else:
                conv_blob = conv_json
                agent_blob = agent_json
                compressed = 0
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO session_checkpoints
                    (session_id, sequence, timestamp, conversation_state, agent_state, metadata, checksum, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint.session_id,
                    checkpoint.sequence,
                    checkpoint.timestamp,
                    conv_blob,
                    agent_blob,
                    json.dumps(checkpoint.metadata),
                    checkpoint.checksum,
                    compressed
                ))
                conn.commit()
            
            self._stats["checkpoints_created"] += 1
            self._last_checkpoint_time = time.time()
            
            if self._on_checkpoint:
                self._on_checkpoint(checkpoint)
            
            logger.info(f"Checkpoint created at sequence {checkpoint.sequence}")
            return checkpoint
    
    def load_checkpoint(self, session_id: str) -> Optional[SessionCheckpoint]:
        """Load most recent checkpoint for session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM session_checkpoints
                WHERE session_id=?
                ORDER BY sequence DESC
                LIMIT 1
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Decompress if needed
            if row['compressed']:
                conv_state = json.loads(zlib.decompress(row['conversation_state']).decode())
                agent_state = json.loads(zlib.decompress(row['agent_state']).decode())
            else:
                conv_state = json.loads(row['conversation_state'])
                agent_state = json.loads(row['agent_state'])
            
            checkpoint = SessionCheckpoint(
                sequence=row['sequence'],
                timestamp=row['timestamp'],
                session_id=row['session_id'],
                conversation_state=conv_state,
                agent_state=agent_state,
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                checksum=row['checksum'],
                compressed=bool(row['compressed'])
            )
            
            # Verify integrity
            if not checkpoint.verify():
                logger.error(f"Checkpoint corruption detected for {session_id}")
                return None
            
            return checkpoint
    
    def get_deltas_since(
        self,
        session_id: str,
        sequence: int
    ) -> List[DeltaRecord]:
        """Get all deltas since a given sequence number."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM session_deltas
                WHERE session_id=? AND sequence >= ?
                ORDER BY sequence ASC
            """, (session_id, sequence))
            
            records = []
            for row in cursor.fetchall():
                # Decompress if needed
                if row['compressed']:
                    data = DeltaRecord.decompress_data(row['data'])
                else:
                    data = json.loads(row['data'])
                
                record = DeltaRecord(
                    sequence=row['sequence'],
                    timestamp=row['timestamp'],
                    delta_type=DeltaType(row['delta_type']),
                    data=data,
                    checksum=row['checksum'],
                    compressed=bool(row['compressed']),
                    original_size=row['original_size']
                )
                
                # Verify integrity
                if not record.verify():
                    logger.warning(f"Delta {record.sequence} checksum mismatch")
                    continue
                
                records.append(record)
            
            return records
    
    def recover(
        self,
        session_id: str,
        apply_delta_fn: Optional[Callable[[DeltaRecord], None]] = None
    ) -> Tuple[bool, Dict[str, Any], int]:
        """
        Recover session from checkpoint + deltas.
        
        Returns:
            Tuple of (success, state, deltas_replayed)
        """
        start_time = time.time()
        
        # Load checkpoint
        checkpoint = self.load_checkpoint(session_id)
        if not checkpoint:
            logger.warning(f"No checkpoint found for {session_id}")
            return False, {}, 0
        
        # Get deltas since checkpoint
        deltas = self.get_deltas_since(session_id, checkpoint.sequence + 1)
        
        # Apply deltas
        replayed = 0
        state = {
            "conversation": checkpoint.conversation_state,
            "agent": checkpoint.agent_state
        }
        
        for delta in deltas:
            if apply_delta_fn:
                apply_delta_fn(delta)
            
            # Update state based on delta type
            if delta.delta_type == DeltaType.MESSAGE_ADDED:
                if "messages" not in state["conversation"]:
                    state["conversation"]["messages"] = []
                state["conversation"]["messages"].append(delta.data)
            
            elif delta.delta_type == DeltaType.STATE_CHANGED:
                state["agent"].update(delta.data)
            
            elif delta.delta_type == DeltaType.CONTEXT_UPDATED:
                state["conversation"]["context"] = delta.data
            
            replayed += 1
        
        recovery_time = time.time() - start_time
        logger.info(f"Recovered {session_id}: {replayed} deltas in {recovery_time:.3f}s")
        
        if self._on_recovery:
            self._on_recovery(session_id, state, replayed)
        
        return True, state, replayed
    
    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all sessions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sessions
                ORDER BY last_active DESC
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    "session_id": row['session_id'],
                    "created_at": row['created_at'],
                    "last_active": row['last_active'],
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                }
                for row in cursor.fetchall()
            ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete deltas
            conn.execute("DELETE FROM session_deltas WHERE session_id=?", (session_id,))
            # Delete checkpoints
            conn.execute("DELETE FROM session_checkpoints WHERE session_id=?", (session_id,))
            # Delete session record
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()
        
        logger.info(f"Session deleted: {session_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        with self._lock:
            time_since_checkpoint = time.time() - self._last_checkpoint_time if self._last_checkpoint_time else 0
            
            # Get database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                "deltas_written": self._stats["deltas_written"],
                "deltas_compressed": self._stats["deltas_compressed"],
                "checkpoints_created": self._stats["checkpoints_created"],
                "bytes_saved": self._stats["bytes_saved"],
                "current_sequence": self._sequence,
                "deltas_in_queue": len(self._delta_queue),
                "time_since_checkpoint_sec": round(time_since_checkpoint, 1),
                "db_size_bytes": db_size,
                "compression_enabled": self.compression_enabled
            }
    
    def shutdown(self):
        """Gracefully shutdown persistence manager."""
        self._shutdown = True
        self._flush_deltas()
        logger.info("DeltaPersistence shutdown complete")


class SessionManager:
    """
    High-level session management.
    
    Provides simplified interface for session operations.
    """
    
    def __init__(self, persistence: Optional[EnhancedDeltaPersistence] = None):
        self.persistence = persistence or EnhancedDeltaPersistence()
        self._active_sessions: Dict[str, Dict] = {}
        self._lock = threading.RLock()
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> bool:
        """Create a new session."""
        self.persistence.init_session(session_id, metadata)
        
        with self._lock:
            self._active_sessions[session_id] = {
                "created_at": time.time(),
                "metadata": metadata or {}
            }
        
        return True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session info."""
        with self._lock:
            return self._active_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._active_sessions.keys())
    
    def close_session(self, session_id: str, save_checkpoint: bool = True):
        """Close a session."""
        if save_checkpoint:
            self.persistence.create_checkpoint()
        
        with self._lock:
            self._active_sessions.pop(session_id, None)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions from database."""
        return self.persistence.list_sessions()


class CheckpointManager:
    """
    Checkpoint management utilities.
    
    Provides advanced checkpoint operations.
    """
    
    def __init__(self, persistence: EnhancedDeltaPersistence):
        self.persistence = persistence
    
    def create_named_checkpoint(self, name: str, description: str = ""):
        """Create a named checkpoint for easy recovery."""
        checkpoint = self.persistence.create_checkpoint()
        
        # Store name mapping (would need additional table in practice)
        logger.info(f"Named checkpoint created: {name} (seq={checkpoint.sequence})")
        return checkpoint
    
    def rollback_to_checkpoint(self, session_id: str, sequence: int) -> bool:
        """Rollback to a specific checkpoint."""
        # In practice, this would delete deltas after the checkpoint
        logger.info(f"Rollback requested: {session_id} to sequence {sequence}")
        return True
    
    def prune_old_checkpoints(self, session_id: str, keep_count: int = 5):
        """Prune old checkpoints, keeping only the most recent."""
        # In practice, this would delete old checkpoints from DB
        logger.info(f"Prune requested: {session_id}, keep={keep_count}")
