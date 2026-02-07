"""
Org-wide report caching functionality.

This module provides caching for data view component data to speed up
repeat analysis runs, and a lock mechanism to prevent concurrent runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cja_auto_sdr.org.models import DataViewSummary


class OrgReportLock:
    """File-based lock to prevent concurrent org-report runs for the same org.

    Uses a lock file with PID and timestamp to detect stale locks from
    crashed processes. Lock is automatically released when context exits.

    Usage:
        with OrgReportLock(org_id) as lock:
            if not lock.acquired:
                print("Another org-report is running")
                return
            # ... run analysis ...

    Args:
        org_id: Organization ID to lock
        lock_dir: Directory for lock files. Defaults to ~/.cja_auto_sdr/locks/
        stale_threshold_seconds: Consider lock stale after this many seconds (default: 1 hour)
    """

    def __init__(
        self,
        org_id: str,
        lock_dir: Path | None = None,
        stale_threshold_seconds: int = 3600,
    ):
        if lock_dir is None:
            lock_dir = Path.home() / ".cja_auto_sdr" / "locks"
        self.lock_dir = lock_dir
        # Sanitize org_id for filename (keep only safe chars)
        import re

        safe_org_id = re.sub(r"[^a-zA-Z0-9_-]", "_", org_id)
        self.lock_file = lock_dir / f"org_report_{safe_org_id}.lock"
        self.stale_threshold = stale_threshold_seconds
        self.acquired = False
        self._pid = os.getpid()

    def __enter__(self) -> OrgReportLock:
        self.acquired = self._try_acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.acquired:
            self._release()

    def _try_acquire(self) -> bool:
        """Attempt to acquire the lock.

        Uses atomic file creation (O_CREAT|O_EXCL) to prevent TOCTOU races.

        Returns:
            True if lock acquired, False if another process holds it
        """
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        lock_payload = json.dumps(
            {
                "pid": self._pid,
                "timestamp": time.time(),
                "started_at": datetime.now().isoformat(),
            }
        )

        # First, try atomic exclusive creation (no race condition)
        try:
            fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, lock_payload.encode())
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            pass  # Lock file exists, check if stale below
        except OSError:
            return False

        # Lock file exists — check if stale or held by a dead process
        try:
            with open(self.lock_file) as f:
                lock_data = json.load(f)

            lock_pid = lock_data.get("pid")
            lock_time = lock_data.get("timestamp", 0)

            age_seconds = time.time() - lock_time
            if age_seconds <= self.stale_threshold and lock_pid and self._is_process_running(lock_pid):
                # Process is still running and lock is fresh — cannot acquire
                return False

            # Lock is stale or holder is dead — remove and retry atomically
            with contextlib.suppress(OSError):
                os.unlink(str(self.lock_file))

            try:
                fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, lock_payload.encode())
                finally:
                    os.close(fd)
                return True
            except FileExistsError:
                # Another process beat us to it after we removed the stale lock
                return False
            except OSError:
                return False

        except OSError, json.JSONDecodeError, KeyError:
            # Corrupted lock file — remove and retry
            with contextlib.suppress(OSError):
                os.unlink(str(self.lock_file))
            try:
                fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, lock_payload.encode())
                finally:
                    os.close(fd)
                return True
            except OSError, FileExistsError:
                return False

    def _release(self) -> None:
        """Release the lock."""
        try:
            if self.lock_file.exists():
                # Only remove if we own it
                with open(self.lock_file) as f:
                    lock_data = json.load(f)
                if lock_data.get("pid") == self._pid:
                    self.lock_file.unlink()
        except OSError, json.JSONDecodeError, KeyError:
            # Best effort removal
            with contextlib.suppress(Exception):
                self.lock_file.unlink(missing_ok=True)

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process with the given PID is running."""
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
            return True
        except OSError, ProcessLookupError:
            return False

    def get_lock_info(self) -> dict[str, Any] | None:
        """Get information about the current lock holder.

        Returns:
            Dict with pid, timestamp, started_at or None if no lock
        """
        if not self.lock_file.exists():
            return None
        try:
            with open(self.lock_file) as f:
                return json.load(f)
        except OSError, json.JSONDecodeError:
            return None


class OrgReportCache:
    """Cache for org-wide report data view components.

    Stores fetched DataViewSummary objects to disk for faster repeat runs.
    Cache is JSON-based and stores component IDs, counts, and metadata.
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cja_auto_sdr/cache/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cja_auto_sdr" / "cache"
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "org_report_cache.json"
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except OSError, json.JSONDecodeError:
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except OSError:
            pass

    def get(
        self,
        dv_id: str,
        max_age_hours: int = 24,
        required_flags: dict[str, bool] | None = None,
        current_modified: str | None = None,
    ) -> DataViewSummary | None:
        """Get cached data view summary if fresh enough.

        Args:
            dv_id: Data view ID
            max_age_hours: Maximum cache age in hours
            required_flags: Optional flags indicating required cached fields
            current_modified: Current data view modification timestamp for validation.
                If provided and differs from cached timestamp, returns None.

        Returns:
            Cached DataViewSummary or None if not cached or stale
        """
        if dv_id not in self._cache:
            return None

        entry = self._cache[dv_id]
        fetched_at = entry.get("fetched_at")
        if not fetched_at:
            return None

        try:
            fetched_time = datetime.fromisoformat(fetched_at)
            if datetime.now() - fetched_time > timedelta(hours=max_age_hours):
                return None  # Cache is stale
        except ValueError, TypeError:
            return None

        # Validate modification timestamp if provided
        # NOTE: If current_modified is None (API didn't return timestamp),
        # we skip validation and treat cached entry as valid (optimistic approach).
        # This prevents unnecessary refetches when metadata is unavailable.
        if current_modified is not None:
            cached_modified = entry.get("modified")
            if cached_modified != current_modified:
                return None  # Data view has been modified since cached

        if required_flags:
            include_names = required_flags.get("include_names", False)
            include_metadata = required_flags.get("include_metadata", False)
            include_component_types = required_flags.get("include_component_types", False)

            if include_names and not entry.get("include_names", False):
                return None
            if include_metadata and not entry.get("include_metadata", False):
                return None
            if include_component_types and not entry.get("include_component_types", False):
                return None

        # Reconstruct DataViewSummary from cached data
        try:
            return DataViewSummary(
                data_view_id=entry.get("data_view_id", dv_id),
                data_view_name=entry.get("data_view_name", "Unknown"),
                metric_ids=set(entry.get("metric_ids", [])),
                dimension_ids=set(entry.get("dimension_ids", [])),
                metric_count=entry.get("metric_count", 0),
                dimension_count=entry.get("dimension_count", 0),
                status=entry.get("status", "active"),
                fetch_duration=entry.get("fetch_duration", 0.0),
                error=entry.get("error"),
                metric_names=entry.get("metric_names"),
                dimension_names=entry.get("dimension_names"),
                standard_metric_count=entry.get("standard_metric_count", 0),
                derived_metric_count=entry.get("derived_metric_count", 0),
                standard_dimension_count=entry.get("standard_dimension_count", 0),
                derived_dimension_count=entry.get("derived_dimension_count", 0),
                owner=entry.get("owner"),
                owner_id=entry.get("owner_id"),
                created=entry.get("created"),
                modified=entry.get("modified"),
                description=entry.get("description"),
                has_description=entry.get("has_description", False),
            )
        except Exception:
            return None

    def put(
        self,
        summary: DataViewSummary,
        include_names: bool = False,
        include_metadata: bool = False,
        include_component_types: bool = False,
    ) -> None:
        """Store a DataViewSummary in the cache.

        Args:
            summary: DataViewSummary to cache
            include_names: Whether names were included in the summary
            include_metadata: Whether metadata was included in the summary
            include_component_types: Whether component type counts were included
        """
        self._cache[summary.data_view_id] = self._build_entry(
            summary,
            include_names=include_names,
            include_metadata=include_metadata,
            include_component_types=include_component_types,
        )
        self._save_cache()

    def put_many(
        self,
        summaries: list[DataViewSummary],
        include_names: bool = False,
        include_metadata: bool = False,
        include_component_types: bool = False,
    ) -> None:
        """Store multiple DataViewSummary objects in the cache with a single disk write."""
        if not summaries:
            return
        for summary in summaries:
            self._cache[summary.data_view_id] = self._build_entry(
                summary,
                include_names=include_names,
                include_metadata=include_metadata,
                include_component_types=include_component_types,
            )
        self._save_cache()

    def _build_entry(
        self,
        summary: DataViewSummary,
        *,
        include_names: bool,
        include_metadata: bool,
        include_component_types: bool,
    ) -> dict[str, Any]:
        return {
            "data_view_id": summary.data_view_id,
            "data_view_name": summary.data_view_name,
            "metric_ids": list(summary.metric_ids),
            "dimension_ids": list(summary.dimension_ids),
            "metric_count": summary.metric_count,
            "dimension_count": summary.dimension_count,
            "status": summary.status,
            "fetch_duration": summary.fetch_duration,
            "error": summary.error,
            "metric_names": summary.metric_names,
            "dimension_names": summary.dimension_names,
            "standard_metric_count": summary.standard_metric_count,
            "derived_metric_count": summary.derived_metric_count,
            "standard_dimension_count": summary.standard_dimension_count,
            "derived_dimension_count": summary.derived_dimension_count,
            "owner": summary.owner,
            "owner_id": summary.owner_id,
            "created": summary.created,
            "modified": summary.modified,
            "description": summary.description,
            "has_description": summary.has_description,
            "include_names": include_names,
            "include_metadata": include_metadata,
            "include_component_types": include_component_types,
            "fetched_at": datetime.now().isoformat(),
        }

    def invalidate(self, dv_id: str | None = None) -> None:
        """Clear cache entries.

        Args:
            dv_id: Specific data view ID to clear, or None to clear all
        """
        if dv_id is None:
            self._cache = {}
        elif dv_id in self._cache:
            del self._cache[dv_id]
        self._save_cache()

    def has_valid_entry(self, dv_id: str, max_age_hours: int = 24) -> bool:
        """Check if a cache entry exists and is within age limit.

        Args:
            dv_id: Data view ID
            max_age_hours: Maximum cache age in hours

        Returns:
            True if entry exists and is fresh enough to potentially use
        """
        if dv_id not in self._cache:
            return False

        entry = self._cache[dv_id]
        fetched_at = entry.get("fetched_at")
        if not fetched_at:
            return False

        try:
            fetched_time = datetime.fromisoformat(fetched_at)
            if datetime.now() - fetched_time > timedelta(hours=max_age_hours):
                return False  # Cache is stale anyway
        except ValueError, TypeError:
            return False

        return True

    def get_cached_modified(self, dv_id: str) -> str | None:
        """Get the cached modification timestamp for a data view.

        Args:
            dv_id: Data view ID

        Returns:
            Cached modification timestamp or None if not cached
        """
        if dv_id not in self._cache:
            return None
        return self._cache[dv_id].get("modified")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        return {
            "entries": len(self._cache),
            "cache_file": str(self.cache_file),
            "cache_size_bytes": self.cache_file.stat().st_size if self.cache_file.exists() else 0,
        }
