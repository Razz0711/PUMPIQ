"""
NEXYPHER In-Memory Cache Layer
============================
Thread-safe TTL-based caching to reduce external API calls
(CoinGecko, DexScreener, Gemini) and improve response times.

Usage:
    from cache import cache

    # Store with 5-minute TTL
    cache.set("top_coins", data, ttl=300)

    # Retrieve (returns None if expired)
    data = cache.get("top_coins")

    # Decorator for async functions
    @cache.cached(ttl=300, key_prefix="top_coins")
    async def get_top_coins(limit=50):
        ...
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Used to distinguish 'not cached' from 'cached None'


@dataclass
class CacheEntry:
    """A single cached value with expiration."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.monotonic)
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class InMemoryCache:
    """
    Thread-safe in-memory cache with TTL support.

    Features:
    - Per-key TTL (time to live)
    - Automatic cleanup of expired entries
    - Thread-safe via RLock
    - Hit/miss statistics
    - Max size with LRU-like eviction
    - Async decorator support
    """

    def __init__(self, max_size: int = 1000, cleanup_interval: int = 300):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.monotonic()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None if missing or expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                return None
            entry.hits += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Store a value with TTL (seconds). Default: 5 minutes."""
        with self._lock:
            self._maybe_cleanup()
            if len(self._store) >= self._max_size:
                self._evict_expired()
                # If still full, remove oldest entries
                if len(self._store) >= self._max_size:
                    oldest_keys = sorted(
                        self._store.keys(),
                        key=lambda k: self._store[k].created_at,
                    )[:self._max_size // 4]
                    for k in oldest_keys:
                        del self._store[k]

            self._store[key] = CacheEntry(
                value=value,
                expires_at=time.monotonic() + ttl,
            )

    def delete(self, key: str) -> bool:
        """Remove a key from cache."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys starting with a prefix."""
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1) * 100, 1),
                "total_requests": total,
            }

    def cached(self, ttl: int = 300, key_prefix: str = ""):
        """
        Decorator for caching async function results.

        Usage:
            @cache.cached(ttl=300, key_prefix="top_coins")
            async def get_top_coins(limit=50):
                ...
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Build cache key from function name + args
                key_parts = [key_prefix or func.__name__]
                if args:
                    # Skip 'self' if it's a method
                    start_idx = 1 if args and hasattr(args[0], '__class__') and not isinstance(args[0], (str, int, float, bool)) else 0
                    for a in args[start_idx:]:
                        key_parts.append(str(a))
                if kwargs:
                    for k, v in sorted(kwargs.items()):
                        key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

                # Check cache
                result = self.get(cache_key)
                if result is not None:
                    logger.debug("Cache HIT: %s", cache_key)
                    return result

                # Call function
                logger.debug("Cache MISS: %s", cache_key)
                result = await func(*args, **kwargs)

                # Store result (cache everything except _SENTINEL; None is valid)
                if result is not _SENTINEL:
                    self.set(cache_key, result, ttl=ttl)

                return result
            return wrapper
        return decorator

    def _maybe_cleanup(self):
        """Periodically remove expired entries."""
        now = time.monotonic()
        if now - self._last_cleanup > self._cleanup_interval:
            self._evict_expired()
            self._last_cleanup = now

    def _evict_expired(self):
        """Remove all expired entries."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        if expired:
            logger.debug("Cache cleanup: removed %d expired entries", len(expired))


# ── Singleton ────────────────────────────────────────────────────
cache = InMemoryCache(max_size=2000, cleanup_interval=120)
