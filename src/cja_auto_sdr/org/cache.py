"""
Org-wide report caching functionality.

This module provides caching for data view component data to speed up
repeat analysis runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from cja_auto_sdr.org.models import DataViewSummary


class OrgReportCache:
    """Cache for org-wide report data view components.

    Stores fetched DataViewSummary objects to disk for faster repeat runs.
    Cache is JSON-based and stores component IDs, counts, and metadata.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cja_auto_sdr/cache/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cja_auto_sdr" / "cache"
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "org_report_cache.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, default=str)
        except IOError:
            pass

    def get(
        self,
        dv_id: str,
        max_age_hours: int = 24,
        required_flags: Optional[Dict[str, bool]] = None
    ) -> Optional[DataViewSummary]:
        """Get cached data view summary if fresh enough.

        Args:
            dv_id: Data view ID
            max_age_hours: Maximum cache age in hours
            required_flags: Optional flags indicating required cached fields

        Returns:
            Cached DataViewSummary or None if not cached or stale
        """
        if dv_id not in self._cache:
            return None

        entry = self._cache[dv_id]
        fetched_at = entry.get('fetched_at')
        if not fetched_at:
            return None

        try:
            fetched_time = datetime.fromisoformat(fetched_at)
            if datetime.now() - fetched_time > timedelta(hours=max_age_hours):
                return None  # Cache is stale
        except (ValueError, TypeError):
            return None

        if required_flags:
            include_names = required_flags.get('include_names', False)
            include_metadata = required_flags.get('include_metadata', False)
            include_component_types = required_flags.get('include_component_types', False)

            if include_names and not entry.get('include_names', False):
                return None
            if include_metadata and not entry.get('include_metadata', False):
                return None
            if include_component_types and not entry.get('include_component_types', False):
                return None

        # Reconstruct DataViewSummary from cached data
        try:
            return DataViewSummary(
                data_view_id=entry.get('data_view_id', dv_id),
                data_view_name=entry.get('data_view_name', 'Unknown'),
                metric_ids=set(entry.get('metric_ids', [])),
                dimension_ids=set(entry.get('dimension_ids', [])),
                metric_count=entry.get('metric_count', 0),
                dimension_count=entry.get('dimension_count', 0),
                status=entry.get('status', 'active'),
                fetch_duration=entry.get('fetch_duration', 0.0),
                error=entry.get('error'),
                metric_names=entry.get('metric_names'),
                dimension_names=entry.get('dimension_names'),
                standard_metric_count=entry.get('standard_metric_count', 0),
                calculated_metric_count=entry.get('calculated_metric_count', 0),
                derived_metric_count=entry.get('derived_metric_count', 0),
                standard_dimension_count=entry.get('standard_dimension_count', 0),
                derived_dimension_count=entry.get('derived_dimension_count', 0),
                owner=entry.get('owner'),
                owner_id=entry.get('owner_id'),
                created=entry.get('created'),
                modified=entry.get('modified'),
                description=entry.get('description'),
                has_description=entry.get('has_description', False),
            )
        except Exception:
            return None

    def put(
        self,
        summary: DataViewSummary,
        include_names: bool = False,
        include_metadata: bool = False,
        include_component_types: bool = False
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
    ) -> Dict[str, Any]:
        return {
            'data_view_id': summary.data_view_id,
            'data_view_name': summary.data_view_name,
            'metric_ids': list(summary.metric_ids),
            'dimension_ids': list(summary.dimension_ids),
            'metric_count': summary.metric_count,
            'dimension_count': summary.dimension_count,
            'status': summary.status,
            'fetch_duration': summary.fetch_duration,
            'error': summary.error,
            'metric_names': summary.metric_names,
            'dimension_names': summary.dimension_names,
            'standard_metric_count': summary.standard_metric_count,
            'calculated_metric_count': summary.calculated_metric_count,
            'derived_metric_count': summary.derived_metric_count,
            'standard_dimension_count': summary.standard_dimension_count,
            'derived_dimension_count': summary.derived_dimension_count,
            'owner': summary.owner,
            'owner_id': summary.owner_id,
            'created': summary.created,
            'modified': summary.modified,
            'description': summary.description,
            'has_description': summary.has_description,
            'include_names': include_names,
            'include_metadata': include_metadata,
            'include_component_types': include_component_types,
            'fetched_at': datetime.now().isoformat(),
        }

    def invalidate(self, dv_id: Optional[str] = None) -> None:
        """Clear cache entries.

        Args:
            dv_id: Specific data view ID to clear, or None to clear all
        """
        if dv_id is None:
            self._cache = {}
        elif dv_id in self._cache:
            del self._cache[dv_id]
        self._save_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        return {
            'entries': len(self._cache),
            'cache_file': str(self.cache_file),
            'cache_size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
        }
