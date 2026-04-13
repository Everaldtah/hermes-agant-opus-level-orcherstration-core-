"""
Multi-Level Cache Manager
==========================

Hierarchical caching system with LRU, time-based, and predictive caching.
"""

import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


class MultiLevelCache:
    """
    Multi-level caching system.
    
    Levels:
    - L1: Hot cache (most recent, in-memory)
    - L2: Warm cache (frequently accessed)
    - L3: Cold cache (predictive preloading)
    
    Features:
    - LRU eviction
    - TTL support
    - Predictive caching
    - Size-based eviction
    - Hit rate tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 300,
        enable_predictive: bool = True,
        predictive_threshold: int = 2
    ):
        self.max_size = max_size
        self.default_ttl = ttl
        self.enable_predictive = enable_predictive
        self.predictive_threshold = predictive_threshold
        
        # Cache levels
        self._l1_hot: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_warm: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l3_cold: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Size limits per level
        self._l1_size = max_size // 4
        self._l2_size = max_size // 2
        self._l3_size = max_size // 4
        
        # Access patterns for predictive caching
        self._access_patterns: Dict[str, List[str]] = {}
        self._pattern_lock = threading.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._predictive_hits = 0
        
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Periodic cleanup of expired entries."""
        while True:
            time.sleep(60)  # Clean every minute
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired entries from all levels."""
        with self._lock:
            for level in [self._l1_hot, self._l2_warm, self._l3_cold]:
                expired = [
                    key for key, entry in level.items()
                    if entry.is_expired
                ]
                for key in expired:
                    del level[key]
                    self._evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            # Check L1 (hot)
            if key in self._l1_hot:
                entry = self._l1_hot[key]
                if not entry.is_expired:
                    entry.accessed_at = time.time()
                    entry.access_count += 1
                    self._hits += 1
                    
                    # Move to end (most recently used)
                    self._l1_hot.move_to_end(key)
                    
                    # Record access pattern
                    self._record_access(key)
                    
                    return entry.value
                else:
                    del self._l1_hot[key]
            
            # Check L2 (warm)
            if key in self._l2_warm:
                entry = self._l2_warm[key]
                if not entry.is_expired:
                    entry.accessed_at = time.time()
                    entry.access_count += 1
                    self._hits += 1
                    
                    # Promote to L1
                    self._promote_to_l1(key, entry)
                    
                    return entry.value
                else:
                    del self._l2_warm[key]
            
            # Check L3 (cold)
            if key in self._l3_cold:
                entry = self._l3_cold[key]
                if not entry.is_expired:
                    entry.accessed_at = time.time()
                    entry.access_count += 1
                    self._hits += 1
                    
                    # Promote to L2
                    self._promote_to_l2(key, entry)
                    
                    return entry.value
                else:
                    del self._l3_cold[key]
            
            self._misses += 1
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        level: str = "l1"
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            level: Cache level (l1, l2, l3)
        """
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            size=self._estimate_size(value)
        )
        
        with self._lock:
            if level == "l1":
                self._add_to_l1(key, entry)
            elif level == "l2":
                self._add_to_l2(key, entry)
            else:
                self._add_to_l3(key, entry)
    
    def _add_to_l1(self, key: str, entry: CacheEntry):
        """Add entry to L1 cache."""
        # Evict if necessary
        while len(self._l1_hot) >= self._l1_size:
            self._evict_l1_oldest()
        
        self._l1_hot[key] = entry
        self._l1_hot.move_to_end(key)
    
    def _add_to_l2(self, key: str, entry: CacheEntry):
        """Add entry to L2 cache."""
        while len(self._l2_warm) >= self._l2_size:
            self._evict_l2_oldest()
        
        self._l2_warm[key] = entry
        self._l2_warm.move_to_end(key)
    
    def _add_to_l3(self, key: str, entry: CacheEntry):
        """Add entry to L3 cache."""
        while len(self._l3_cold) >= self._l3_size:
            self._evict_l3_oldest()
        
        self._l3_cold[key] = entry
        self._l3_cold.move_to_end(key)
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry to L1."""
        # Remove from current level
        for level in [self._l2_warm, self._l3_cold]:
            if key in level:
                del level[key]
                break
        
        self._add_to_l1(key, entry)
    
    def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote entry to L2."""
        if key in self._l3_cold:
            del self._l3_cold[key]
        
        self._add_to_l2(key, entry)
    
    def _evict_l1_oldest(self):
        """Evict oldest entry from L1."""
        if self._l1_hot:
            key, entry = self._l1_hot.popitem(last=False)
            # Move to L2 if frequently accessed
            if entry.access_count >= 2:
                self._add_to_l2(key, entry)
            self._evictions += 1
    
    def _evict_l2_oldest(self):
        """Evict oldest entry from L2."""
        if self._l2_warm:
            key, entry = self._l2_warm.popitem(last=False)
            # Move to L3 if accessed at all
            if entry.access_count >= 1:
                self._add_to_l3(key, entry)
            self._evictions += 1
    
    def _evict_l3_oldest(self):
        """Evict oldest entry from L3."""
        if self._l3_cold:
            self._l3_cold.popitem(last=False)
            self._evictions += 1
    
    def _record_access(self, key: str):
        """Record access pattern for predictive caching."""
        if not self.enable_predictive:
            return
        
        with self._pattern_lock:
            # Simple pattern: track sequences of accesses
            for pattern_key, sequence in list(self._access_patterns.items()):
                if sequence and sequence[-1] != key:
                    sequence.append(key)
                    if len(sequence) > 5:
                        sequence.pop(0)
            
            # Check for predictive opportunities
            self._check_predictive_load(key)
    
    def _check_predictive_load(self, current_key: str):
        """Check if we should predictively load any keys."""
        for pattern_key, sequence in self._access_patterns.items():
            if len(sequence) >= 2 and sequence[-1] == current_key:
                # Predict next key in sequence
                # This is simplified - real implementation would use more sophisticated prediction
                pass
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            return 100  # Default estimate
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Key to invalidate
            
        Returns:
            True if entry was found and removed
        """
        with self._lock:
            for level in [self._l1_hot, self._l2_warm, self._l3_cold]:
                if key in level:
                    del level[key]
                    return True
            return False
    
    def invalidate_pattern(self, pattern: str):
        """
        Invalidate all entries matching pattern.
        
        Args:
            pattern: Key pattern to match
        """
        with self._lock:
            for level in [self._l1_hot, self._l2_warm, self._l3_cold]:
                keys_to_remove = [
                    key for key in level.keys()
                    if pattern in key
                ]
                for key in keys_to_remove:
                    del level[key]
    
    def clear(self):
        """Clear all cache levels."""
        with self._lock:
            self._l1_hot.clear()
            self._l2_warm.clear()
            self._l3_cold.clear()
            self._access_patterns.clear()
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = (
                sum(e.size for e in self._l1_hot.values()) +
                sum(e.size for e in self._l2_warm.values()) +
                sum(e.size for e in self._l3_cold.values())
            )
            
            return {
                "l1_size": len(self._l1_hot),
                "l2_size": len(self._l2_warm),
                "l3_size": len(self._l3_cold),
                "total_entries": len(self._l1_hot) + len(self._l2_warm) + len(self._l3_cold),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self.get_hit_rate(), 4),
                "evictions": self._evictions,
                "estimated_size_bytes": total_size
            }


class PredictiveCacheLoader:
    """
    Predictive cache loader based on access patterns.
    
    Preloads likely-to-be-needed items into cache.
    """
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self._access_sequences: List[List[str]] = []
        self._sequence_lock = threading.Lock()
    
    def record_sequence(self, keys: List[str]):
        """Record an access sequence."""
        with self._sequence_lock:
            self._access_sequences.append(keys)
            if len(self._access_sequences) > 100:
                self._access_sequences.pop(0)
    
    def predict_next_keys(self, current_key: str, n: int = 3) -> List[str]:
        """Predict next keys based on patterns."""
        predictions = []
        
        with self._sequence_lock:
            for sequence in self._access_sequences:
                if current_key in sequence:
                    idx = sequence.index(current_key)
                    if idx + 1 < len(sequence):
                        next_key = sequence[idx + 1]
                        if next_key not in predictions:
                            predictions.append(next_key)
                
                if len(predictions) >= n:
                    break
        
        return predictions
    
    def preload_predictions(self, current_key: str, loader_fn: Callable[[str], Any]):
        """Preload predicted keys into cache."""
        predictions = self.predict_next_keys(current_key)
        
        for key in predictions:
            if self.cache.get(key) is None:
                try:
                    value = loader_fn(key)
                    self.cache.set(key, value, level="l3")
                except Exception as e:
                    logger.warning(f"Failed to preload {key}: {e}")
