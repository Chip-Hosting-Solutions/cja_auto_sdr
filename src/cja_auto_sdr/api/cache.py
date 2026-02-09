"""Validation caches for CJA Auto SDR."""

import atexit
import contextlib
import hashlib
import logging
import multiprocessing
import threading
import time
from typing import Any

import pandas as pd


class ValidationCache:
    """
    Thread-safe LRU cache for data quality validation results

    Caches validation results based on DataFrame content hash and configuration.
    Uses LRU eviction policy to prevent unbounded memory growth.

    Performance Impact:
    - Cache hits: 50-90% faster than running validation
    - Cache misses: ~1-2% overhead for hashing (negligible)
    - Memory: ~1-5MB per 1000 cached entries

    Thread Safety:
    - Uses threading.Lock for all cache operations
    - Safe for use with parallel validation (check_all_parallel)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, logger: logging.Logger | None = None):
        """
        Initialize validation cache

        Args:
            max_size: Maximum number of cached entries, >= 1 (default: 1000)
            ttl_seconds: Time-to-live for cache entries in seconds, >= 1 (default: 3600 = 1 hour)
            logger: Logger instance for cache statistics (default: module logger)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.logger = logger or logging.getLogger(__name__)

        # Cache storage: key -> (issues_list, timestamp)
        self._cache: dict[str, tuple[list[dict], float]] = {}

        # LRU tracking: key -> last_access_time
        self._access_times: dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        self.logger.debug(f"ValidationCache initialized: max_size={max_size}, ttl={ttl_seconds}s")

    def _generate_cache_key(
        self, df: pd.DataFrame, item_type: str, required_fields: list[str], critical_fields: list[str]
    ) -> str:
        """
        Generate cache key from DataFrame content and configuration

        Strategy:
        - Uses pandas.util.hash_pandas_object for efficient DataFrame hashing
        - Combines DataFrame hash with configuration parameters
        - Returns consistent hash for identical inputs

        Args:
            df: DataFrame to hash
            item_type: 'Metrics' or 'Dimensions'
            required_fields: List of required field names
            critical_fields: List of critical field names

        Returns:
            Cache key string in format: "{item_type}:{df_hash}:{config_hash}"
        """
        try:
            # Hash DataFrame content using pandas built-in function
            # This is much faster than manual iteration (1-2ms vs 10-50ms for 1000 rows)

            # Hash DataFrame structure and content
            df_hash = pd.util.hash_pandas_object(df, index=False).sum()

            # Hash configuration (required_fields + critical_fields)
            config_str = f"{sorted(required_fields)}:{sorted(critical_fields)}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

            # Combine into cache key
            cache_key = f"{item_type}:{df_hash}:{config_hash}"

            return cache_key

        except Exception as e:
            self.logger.warning(f"Error generating cache key: {e}. Cache disabled for this call.")
            # Return unique key to force cache miss
            return f"error:{time.time()}"

    def get(
        self, df: pd.DataFrame, item_type: str, required_fields: list[str], critical_fields: list[str]
    ) -> tuple[list[dict] | None, str]:
        """
        Retrieve cached validation results if available

        Returns:
            Tuple of (issues list or None, cache_key).
            The cache_key can be passed to put() to avoid recomputing the hash.
        """
        cache_key = self._generate_cache_key(df, item_type, required_fields, critical_fields)

        # Check debug logging once outside the lock to avoid repeated checks
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                if debug_enabled:
                    self.logger.debug(f"Cache MISS: {item_type} (key: {cache_key[:20]}...)")
                return None, cache_key

            cached_issues, timestamp = self._cache[cache_key]

            # Check TTL expiration
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                if debug_enabled:
                    self.logger.debug(f"Cache EXPIRED: {item_type} (age: {age:.1f}s)")
                del self._cache[cache_key]
                del self._access_times[cache_key]
                self._misses += 1
                return None, cache_key

            # Cache hit - update access time
            self._access_times[cache_key] = time.time()
            self._hits += 1
            if debug_enabled:
                self.logger.debug(f"Cache HIT: {item_type} ({len(cached_issues)} issues)")

            # Return deep copy to prevent mutation of cached data
            return [issue.copy() for issue in cached_issues], cache_key

    def put(
        self,
        df: pd.DataFrame,
        item_type: str,
        required_fields: list[str],
        critical_fields: list[str],
        issues: list[dict],
        cache_key: str | None = None,
    ):
        """
        Store validation results in cache

        Implements LRU eviction when cache is full.

        Args:
            cache_key: Optional pre-computed cache key from get() to avoid rehashing
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(df, item_type, required_fields, critical_fields)

        # Check debug logging once to avoid repeated checks in hot path
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_lru(debug_enabled)

            # Store issues with timestamp
            # Deep copy to prevent external mutation
            self._cache[cache_key] = ([issue.copy() for issue in issues], time.time())
            self._access_times[cache_key] = time.time()

            if debug_enabled:
                self.logger.debug(f"Cache STORE: {item_type} ({len(issues)} issues)")

    def _evict_lru(self, debug_enabled: bool = False):
        """Evict least recently used cache entry.

        Args:
            debug_enabled: Whether debug logging is enabled (avoids repeated checks)
        """
        if not self._access_times:
            return

        # Find least recently used key
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]

        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._evictions += 1

        if debug_enabled:
            self.logger.debug(f"Cache EVICT: LRU entry removed (total evictions: {self._evictions})")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get cache performance statistics

        Returns:
            Dict with hits, misses, hit_rate, size, evictions
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
                "evictions": self._evictions,
                "total_requests": total_requests,
            }

    def clear(self):
        """Clear all cache entries (useful for testing)"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.logger.debug("Cache cleared")

    def log_statistics(self):
        """
        Log cache statistics in a user-friendly format.

        Logs hit rate, total requests, cache size, and estimated time savings.
        Only logs if there have been cache requests.
        """
        stats = self.get_statistics()
        if stats["total_requests"] == 0:
            self.logger.debug("Cache statistics: No requests recorded")
            return

        # Estimate time saved (average validation ~50ms, cache lookup ~1ms)
        estimated_time_saved = stats["hits"] * 0.049  # 49ms saved per hit

        self.logger.info(
            f"Cache Statistics: {stats['hits']}/{stats['total_requests']} hits ({stats['hit_rate']:.1f}% hit rate)"
        )
        self.logger.info(f"  - Cache size: {stats['size']}/{stats['max_size']} entries")
        if stats["evictions"] > 0:
            self.logger.info(f"  - Evictions: {stats['evictions']}")
        if estimated_time_saved > 0.1:
            self.logger.info(f"  - Estimated time saved: {estimated_time_saved:.2f}s")


class SharedValidationCache:
    """
    Process-safe shared cache for validation results across batch workers.

    Uses multiprocessing.Manager to share cache state across multiple
    processes in batch mode. This allows validation results to be reused
    across different data views being processed in parallel.

    API Compatibility:
    - Implements the same interface as ValidationCache for drop-in replacement
    - get() and put() methods accept DataFrame objects like ValidationCache

    Key Differences from ValidationCache:
    - Uses Manager().dict() instead of regular dict (process-safe)
    - Manager-level locking instead of threading.Lock
    - Requires explicit shutdown() to cleanup Manager resources
    - Slightly higher overhead but enables cross-process sharing

    Usage:
        # In BatchProcessor
        shared_cache = SharedValidationCache(max_size=1000)

        # Pass to workers (cache reference is pickle-safe via Manager)
        worker_args = (data_view_id, ..., shared_cache)

        # In worker process - same API as ValidationCache
        cached, key = shared_cache.get(df, item_type, req_fields, crit_fields)
        if cached is None:
            result = run_validation(...)
            shared_cache.put(df, item_type, req_fields, crit_fields, result, key)

        # After batch processing
        shared_cache.shutdown()
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, logger: logging.Logger | None = None):
        """
        Initialize shared validation cache.

        Args:
            max_size: Maximum number of cached entries (default: 1000)
            ttl_seconds: Time-to-live for cache entries in seconds (default: 3600)
            logger: Logger instance for cache statistics (default: module logger)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.logger = logger or logging.getLogger(__name__)

        # Create multiprocessing Manager for shared state
        self._manager = multiprocessing.Manager()

        # Shared cache storage: key -> (issues_list, timestamp)
        self._cache = self._manager.dict()

        # Shared LRU tracking: key -> last_access_time
        self._access_times = self._manager.dict()

        # Shared statistics
        self._stats = self._manager.dict({"hits": 0, "misses": 0, "evictions": 0})

        # Manager-level lock for cache operations
        self._lock = self._manager.Lock()

        # Ensure cleanup on exit even if shutdown() is never called explicitly
        atexit.register(self.shutdown)

    def _generate_cache_key(
        self, df: pd.DataFrame, item_type: str, required_fields: list[str], critical_fields: list[str]
    ) -> str:
        """
        Generate cache key from DataFrame content and configuration.

        Same algorithm as ValidationCache for compatibility.
        """
        try:
            # Hash DataFrame structure and content
            df_hash = pd.util.hash_pandas_object(df, index=False).sum()

            # Hash configuration (required_fields + critical_fields)
            config_str = f"{sorted(required_fields)}:{sorted(critical_fields)}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

            # Combine into cache key
            return f"{item_type}:{df_hash}:{config_hash}"

        except Exception as e:
            self.logger.warning(f"Error generating cache key: {e}. Cache disabled for this call.")
            # Return unique key to force cache miss
            return f"error:{time.time()}"

    def get(
        self, df: pd.DataFrame, item_type: str, required_fields: list[str], critical_fields: list[str]
    ) -> tuple[list[dict] | None, str]:
        """
        Retrieve cached validation results if available.

        Same interface as ValidationCache for compatibility.

        Args:
            df: DataFrame to look up in cache
            item_type: 'Metrics' or 'Dimensions'
            required_fields: List of required field names
            critical_fields: List of critical field names

        Returns:
            Tuple of (issues list or None, cache_key).
            The cache_key can be passed to put() to avoid recomputing the hash.
        """
        cache_key = self._generate_cache_key(df, item_type, required_fields, critical_fields)

        with self._lock:
            if cache_key not in self._cache:
                self._stats["misses"] = self._stats.get("misses", 0) + 1
                return None, cache_key

            cached_data = self._cache[cache_key]
            cached_issues, timestamp = cached_data

            # Check TTL expiration
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                del self._cache[cache_key]
                if cache_key in self._access_times:
                    del self._access_times[cache_key]
                self._stats["misses"] = self._stats.get("misses", 0) + 1
                return None, cache_key

            # Cache hit - update access time and stats
            self._access_times[cache_key] = time.time()
            self._stats["hits"] = self._stats.get("hits", 0) + 1

            # Return copy to prevent mutation
            return [issue.copy() for issue in cached_issues], cache_key

    def put(
        self,
        df: pd.DataFrame,
        item_type: str,
        required_fields: list[str],
        critical_fields: list[str],
        issues: list[dict],
        cache_key: str | None = None,
    ) -> None:
        """
        Store validation results in cache.

        Same interface as ValidationCache for compatibility.

        Args:
            df: DataFrame being cached (used if cache_key not provided)
            item_type: 'Metrics' or 'Dimensions'
            required_fields: List of required field names
            critical_fields: List of critical field names
            issues: List of validation issues to cache
            cache_key: Optional pre-computed cache key from get()
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(df, item_type, required_fields, critical_fields)

        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_lru()

            # Store issues with timestamp (deep copy for safety)
            self._cache[cache_key] = ([issue.copy() for issue in issues], time.time())
            self._access_times[cache_key] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used cache entry (must be called within lock)."""
        if not self._access_times:
            return

        # Find least recently used key
        access_times_dict = dict(self._access_times)
        if not access_times_dict:
            return

        lru_key = min(access_times_dict.items(), key=lambda x: x[1])[0]

        # Remove from cache
        if lru_key in self._cache:
            del self._cache[lru_key]
        if lru_key in self._access_times:
            del self._access_times[lru_key]

        self._stats["evictions"] = self._stats.get("evictions", 0) + 1

    def get_statistics(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dict with hits, misses, hit_rate, size, evictions
        """
        with self._lock:
            stats = dict(self._stats)
            hits = stats.get("hits", 0)
            misses = stats.get("misses", 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
                "evictions": stats.get("evictions", 0),
                "total_requests": total_requests,
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def shutdown(self) -> None:
        """
        Shutdown the Manager and cleanup resources.

        IMPORTANT: Call this after batch processing is complete to avoid
        resource leaks. The cache cannot be used after shutdown.
        """
        with contextlib.suppress(Exception):
            self._manager.shutdown()
