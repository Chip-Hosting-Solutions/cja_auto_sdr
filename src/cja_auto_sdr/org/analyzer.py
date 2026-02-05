"""
Org-wide component analyzer.

This module provides the main analysis engine for org-wide component analysis.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from cja_auto_sdr.core.constants import (
    DEFAULT_ORG_REPORT_WORKERS,
    GOVERNANCE_MAX_OVERLAP_THRESHOLD,
)
from cja_auto_sdr.inventory.utils import extract_owner
from cja_auto_sdr.org.models import (
    ComponentDistribution,
    ComponentInfo,
    DataViewCluster,
    DataViewSummary,
    OrgReportConfig,
    OrgReportResult,
    SimilarityPair,
)

if TYPE_CHECKING:
    import cjapy

from cja_auto_sdr.org.cache import OrgReportCache, OrgReportLock


class OrgComponentAnalyzer:
    """Analyze component distribution across all data views in an organization.

    This class provides comprehensive org-wide analysis including:
    - Component distribution (core/common/limited/isolated)
    - Pairwise similarity matrix using Jaccard similarity
    - Governance recommendations based on analysis results

    Args:
        cja: Initialized CJA API client
        config: OrgReportConfig with analysis parameters
        logger: Logger instance for output
        org_id: Organization ID
        cache: Optional OrgReportCache for incremental analysis
    """

    def __init__(
        self,
        cja: "cjapy.CJA",
        config: OrgReportConfig,
        logger: logging.Logger,
        org_id: str = "unknown",
        cache: Optional[OrgReportCache] = None
    ):
        self.cja = cja
        self.config = config
        self.logger = logger
        self.org_id = org_id
        self.cache = cache
        self._thread_local = threading.local()

        # Handle cache clear option
        if config.clear_cache and self.cache:
            self.cache.invalidate()
            self.logger.info("Cache cleared")

    def run_analysis(self) -> OrgReportResult:
        """Execute the full org-wide component analysis.

        Returns:
            OrgReportResult containing all analysis data

        Raises:
            ConcurrentOrgReportError: If another org-report is already running for this org
        """
        # Check for concurrent runs (unless skip_lock is set)
        if not self.config.skip_lock:
            from cja_auto_sdr.core.exceptions import ConcurrentOrgReportError

            lock = OrgReportLock(self.org_id)
            lock_info = lock.get_lock_info()

            # Try to acquire the lock before starting
            with lock:
                if not lock.acquired:
                    raise ConcurrentOrgReportError(
                        org_id=self.org_id,
                        lock_holder_pid=lock_info.get("pid") if lock_info else None,
                        started_at=lock_info.get("started_at") if lock_info else None,
                    )
                # Run the analysis while holding the lock
                return self._run_analysis_impl()
        else:
            # Skip lock (for testing)
            return self._run_analysis_impl()

    def _run_analysis_impl(self) -> OrgReportResult:
        """Internal implementation of org-wide analysis (called within lock)."""
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # 1. List and filter data views
        self.logger.info("Fetching data view list...")
        data_views, is_sampled, total_available = self._list_and_filter_data_views()

        if not data_views:
            self.logger.warning("No data views found matching criteria")
            return OrgReportResult(
                timestamp=timestamp,
                org_id=self.org_id,
                parameters=self.config,
                data_view_summaries=[],
                component_index={},
                distribution=ComponentDistribution(),
                similarity_pairs=None,
                recommendations=[],
                duration=time.time() - start_time,
                is_sampled=is_sampled,
                total_available_data_views=total_available,
            )

        if is_sampled:
            self.logger.info(f"Sampled {len(data_views)} from {total_available} available data views")
        else:
            self.logger.info(f"Found {len(data_views)} data views to analyze")

        # 2. Fetch components for each data view in parallel
        self.logger.info("Fetching components from all data views...")
        summaries = self._fetch_all_data_views(data_views)

        # 3. Build component index
        self.logger.info("Building component index...")
        component_index = self._build_component_index(summaries)
        self.logger.info(f"Indexed {len(component_index)} unique components")

        # 3.5. Check memory warning
        self._check_memory_warning(component_index)

        # 4. Compute distribution
        self.logger.info("Computing distribution buckets...")
        successful_summaries = [summary for summary in summaries if not summary.error]
        distribution = self._compute_distribution(component_index, len(successful_summaries))

        # 5. Compute similarity matrix (if not skipped and not org-stats mode)
        similarity_pairs = None
        if not self.config.skip_similarity and not self.config.org_stats_only:
            max_dvs = self.config.similarity_max_dvs
            if (
                max_dvs is not None
                and len(successful_summaries) > max_dvs
                and not self.config.force_similarity
            ):
                self.logger.warning(
                    "Skipping similarity matrix: %s data views exceed guardrail of %s. "
                    "Use --force-similarity or increase --similarity-max-dvs to override.",
                    len(successful_summaries),
                    max_dvs,
                )
            else:
                self.logger.info("Computing similarity matrix...")
                similarity_pairs = self._compute_similarity_matrix(summaries)
                effective_threshold = min(self.config.overlap_threshold, GOVERNANCE_MAX_OVERLAP_THRESHOLD)
                self.logger.info(
                    f"Found {len(similarity_pairs)} pairs above threshold (>= {effective_threshold})"
                )
        elif self.config.org_stats_only:
            self.logger.info("Skipping similarity matrix (--org-stats mode)")
        else:
            self.logger.info("Skipping similarity matrix (--skip-similarity)")

        # 6. Compute clusters (if enabled and not org-stats mode)
        clusters = None
        if self.config.enable_clustering and not self.config.org_stats_only:
            self.logger.info("Computing data view clusters...")
            clusters = self._compute_clusters(summaries)
            if clusters:
                self.logger.info(f"Found {len(clusters)} clusters")

        # 7. Generate recommendations
        self.logger.info("Generating recommendations...")
        recommendations = self._generate_recommendations(
            summaries, component_index, distribution, similarity_pairs
        )

        # 8. Check governance thresholds (Feature 1)
        governance_violations = None
        thresholds_exceeded = False
        if self.config.duplicate_threshold is not None or self.config.isolated_threshold is not None:
            self.logger.info("Checking governance thresholds...")
            governance_violations, thresholds_exceeded = self._check_governance_thresholds(
                similarity_pairs, distribution, len(component_index)
            )
            if thresholds_exceeded:
                self.logger.warning(f"Governance thresholds exceeded: {len(governance_violations)} violation(s)")

        # 9. Naming audit (Feature 3)
        naming_audit = None
        if self.config.audit_naming or self.config.flag_stale:
            self.logger.info("Auditing naming conventions...")
            naming_audit = self._audit_naming_conventions(component_index)

        # 10. Owner summary (Feature 5)
        owner_summary = None
        if self.config.include_owner_summary:
            if self.config.include_metadata:
                self.logger.info("Computing owner summary...")
                owner_summary = self._compute_owner_summary(summaries)
            else:
                self.logger.warning("--owner-summary requires --include-metadata")

        # 11. Stale components (Feature 6)
        stale_components = None
        if self.config.flag_stale:
            self.logger.info("Detecting stale components...")
            stale_components = self._detect_stale_components(component_index)
            if stale_components:
                self.logger.info(f"Found {len(stale_components)} components with stale naming patterns")

        duration = time.time() - start_time
        self.logger.info(f"Analysis complete in {duration:.2f}s")

        return OrgReportResult(
            timestamp=timestamp,
            org_id=self.org_id,
            parameters=self.config,
            data_view_summaries=summaries,
            component_index=component_index,
            distribution=distribution,
            similarity_pairs=similarity_pairs,
            recommendations=recommendations,
            duration=duration,
            clusters=clusters,
            is_sampled=is_sampled,
            total_available_data_views=total_available,
            governance_violations=governance_violations,
            thresholds_exceeded=thresholds_exceeded,
            naming_audit=naming_audit,
            owner_summary=owner_summary,
            stale_components=stale_components,
        )

    def _list_and_filter_data_views(self) -> Tuple[List[Dict[str, Any]], bool, int]:
        """List all data views and apply filter/exclude/sample patterns.

        Returns:
            Tuple of (data_view_list, is_sampled, total_available_count)
        """
        try:
            all_data_views = self.cja.getDataViews()
        except Exception as e:
            self.logger.error(f"Failed to list data views: {e}")
            return [], False, 0

        if all_data_views is None or len(all_data_views) == 0:
            return [], False, 0

        # Convert to list of dicts if needed
        if isinstance(all_data_views, pd.DataFrame):
            all_data_views = all_data_views.to_dict('records')

        filtered = all_data_views

        # Apply include filter
        if self.config.filter_pattern:
            try:
                pattern = re.compile(self.config.filter_pattern, re.IGNORECASE)
                filtered = [dv for dv in filtered if pattern.search(dv.get('name', ''))]
                self.logger.info(f"Filter '{self.config.filter_pattern}' matched {len(filtered)} data views")
            except re.error as e:
                self.logger.warning(f"Invalid filter pattern: {e}")

        # Apply exclude filter
        if self.config.exclude_pattern:
            try:
                pattern = re.compile(self.config.exclude_pattern, re.IGNORECASE)
                before = len(filtered)
                filtered = [dv for dv in filtered if not pattern.search(dv.get('name', ''))]
                self.logger.info(f"Exclude '{self.config.exclude_pattern}' removed {before - len(filtered)} data views")
            except re.error as e:
                self.logger.warning(f"Invalid exclude pattern: {e}")

        # Track total available before sampling/limiting
        total_available = len(filtered)
        is_sampled = False

        # Apply sampling (before limit)
        if self.config.sample_size and len(filtered) > self.config.sample_size:
            import random
            if self.config.sample_stratified:
                filtered = self._stratified_sample(filtered, self.config.sample_size)
            else:
                random.seed(self.config.sample_seed)
                filtered = random.sample(filtered, self.config.sample_size)
            is_sampled = True
            self.logger.info(f"Sampled {len(filtered)} data views (seed={self.config.sample_seed})")

        # Apply limit (after sampling)
        if self.config.limit and len(filtered) > self.config.limit:
            self.logger.info(f"Limiting to first {self.config.limit} data views")
            filtered = filtered[:self.config.limit]

        return filtered, is_sampled, total_available

    def _stratified_sample(self, data_views: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Stratified sampling by data view name prefix.

        Groups data views by common prefix and samples proportionally from each group.

        Args:
            data_views: List of data view dicts
            sample_size: Target sample size

        Returns:
            Stratified sample of data views
        """
        import random

        random.seed(self.config.sample_seed)

        # Group by prefix (first word or chars before common separators)
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for dv in data_views:
            name = dv.get('name', '')
            # Extract prefix: first word or chars before - _ or space
            prefix = name.split()[0] if name.split() else name[:10]
            for sep in ['-', '_', ' ']:
                if sep in prefix:
                    prefix = prefix.split(sep)[0]
                    break
            groups[prefix.lower()].append(dv)

        # Sample proportionally from each group
        sampled = []
        total = len(data_views)
        for prefix, group in groups.items():
            # Proportional allocation
            group_sample_size = max(1, int(sample_size * len(group) / total))
            if len(group) <= group_sample_size:
                sampled.extend(group)
            else:
                sampled.extend(random.sample(group, group_sample_size))

        # Adjust if we have too many or too few
        if len(sampled) > sample_size:
            sampled = random.sample(sampled, sample_size)
        elif len(sampled) < sample_size and len(sampled) < len(data_views):
            # Add more randomly from remaining
            remaining = [dv for dv in data_views if dv not in sampled]
            needed = min(sample_size - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, needed))

        return sampled

    def _fetch_all_data_views(self, data_views: List[Dict[str, Any]]) -> List[DataViewSummary]:
        """Fetch components for all data views in parallel.

        Uses cache when enabled to avoid redundant API calls.

        Args:
            data_views: List of data view dicts to fetch

        Returns:
            List of DataViewSummary objects
        """
        summaries = []
        to_fetch = []
        cache_hits = 0
        cache_stale = 0

        # Check cache first if enabled
        if self.config.use_cache and self.cache:
            # Use smart validation if enabled
            if self.config.validate_cache:
                self.logger.info("Validating cached entries against modification timestamps...")
                to_fetch, valid_summaries, valid_count, stale_count = self._validate_cache_entries(data_views)
                summaries.extend(valid_summaries)
                cache_hits = valid_count
                cache_stale = stale_count
                if cache_hits > 0 or cache_stale > 0:
                    self.logger.info(f"Cache validation: {cache_hits} valid, {cache_stale} stale, {len(to_fetch)} to fetch")
            else:
                # Standard age-based cache lookup
                required_flags = {
                    'include_names': self.config.include_names,
                    'include_metadata': self.config.include_metadata,
                    'include_component_types': self.config.include_component_types,
                }
                for dv in data_views:
                    dv_id = dv.get('id', '')
                    cached = self.cache.get(
                        dv_id,
                        self.config.cache_max_age_hours,
                        required_flags=required_flags
                    )
                    if cached:
                        summaries.append(cached)
                        cache_hits += 1
                    else:
                        to_fetch.append(dv)

                if cache_hits > 0:
                    self.logger.info(f"Cache: {cache_hits} hits, {len(to_fetch)} to fetch")
        else:
            to_fetch = data_views

        if not to_fetch:
            return summaries

        pending_cache: List[DataViewSummary] = []

        # Use ThreadPoolExecutor for parallel fetches
        with ThreadPoolExecutor(max_workers=min(DEFAULT_ORG_REPORT_WORKERS, len(to_fetch))) as executor:
            futures = {
                executor.submit(self._fetch_data_view_components, dv): dv
                for dv in to_fetch
            }

            desc = "Fetching data views" if not cache_hits else f"Fetching {len(to_fetch)} uncached"
            with tqdm(
                total=len(to_fetch),
                desc=desc,
                unit="dv",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                leave=True,
                disable=self.config.quiet  # Suppress progress bar in quiet mode
            ) as pbar:
                for future in as_completed(futures):
                    dv = futures[future]
                    try:
                        summary = future.result()
                        summaries.append(summary)

                        # Store in cache if enabled (batch to reduce disk writes)
                        if self.config.use_cache and self.cache and not summary.error:
                            pending_cache.append(summary)

                        if summary.error:
                            pbar.set_postfix_str(f"✗ {dv.get('name', dv.get('id', '?'))[:20]}")
                        else:
                            pbar.set_postfix_str(f"✓ {summary.metric_count}m/{summary.dimension_count}d")
                    except Exception as e:
                        error_msg = str(e) or f"{type(e).__name__}"
                        summaries.append(DataViewSummary(
                            data_view_id=dv.get('id', 'unknown'),
                            data_view_name=dv.get('name', 'Unknown'),
                            error=error_msg
                        ))
                    pbar.update(1)

        if self.config.use_cache and self.cache and pending_cache:
            self.cache.put_many(
                pending_cache,
                include_names=self.config.include_names,
                include_metadata=self.config.include_metadata,
                include_component_types=self.config.include_component_types,
            )

        return summaries

    def _get_thread_client(self):
        if not self.config.cja_per_thread:
            return self.cja

        client = getattr(self._thread_local, "cja", None)
        if client is None:
            import cjapy
            client = cjapy.CJA()
            self._thread_local.cja = client
        return client

    def _fetch_data_view_components(self, dv: Dict[str, Any]) -> DataViewSummary:
        """Fetch metrics and dimensions for a single data view.

        Args:
            dv: Data view dict with 'id' and 'name' keys

        Returns:
            DataViewSummary with component IDs (and names if include_names=True)
        """
        dv_id = dv.get('id', '')
        dv_name = dv.get('name', 'Unknown')
        start_time = time.time()

        try:
            cja = self._get_thread_client()
            # Fetch metrics
            metrics_df = cja.getMetrics(dv_id, inclType=True, full=True)
            metric_ids = set()
            metric_names = None
            standard_metric_count = 0
            derived_metric_count = 0
            if metrics_df is not None and not metrics_df.empty:
                if 'id' in metrics_df.columns:
                    metric_ids = set(metrics_df['id'].dropna().astype(str).tolist())
                # Capture names if requested
                if self.config.include_names and 'name' in metrics_df.columns:
                    metric_names = {}
                    for _, row in metrics_df.iterrows():
                        if pd.notna(row.get('id')) and pd.notna(row.get('name')):
                            metric_names[str(row['id'])] = str(row['name'])
                # Count standard vs derived metrics
                if self.config.include_component_types:
                    for _, row in metrics_df.iterrows():
                        # Check type or sourceFieldType for derived indicator
                        comp_type = str(row.get('type', '')).lower()
                        source_type = str(row.get('sourceFieldType', '')).lower()
                        if 'derived' in comp_type or 'derived' in source_type:
                            derived_metric_count += 1
                        else:
                            standard_metric_count += 1

            # Fetch dimensions
            dimensions_df = cja.getDimensions(dv_id, inclType=True, full=True)
            dimension_ids = set()
            dimension_names = None
            standard_dimension_count = 0
            derived_dimension_count = 0
            if dimensions_df is not None and not dimensions_df.empty:
                if 'id' in dimensions_df.columns:
                    dimension_ids = set(dimensions_df['id'].dropna().astype(str).tolist())
                # Capture names if requested
                if self.config.include_names and 'name' in dimensions_df.columns:
                    dimension_names = {}
                    for _, row in dimensions_df.iterrows():
                        if pd.notna(row.get('id')) and pd.notna(row.get('name')):
                            dimension_names[str(row['id'])] = str(row['name'])
                # Count standard vs derived dimensions
                if self.config.include_component_types:
                    for _, row in dimensions_df.iterrows():
                        # Check type or sourceFieldType for derived indicator
                        comp_type = str(row.get('type', '')).lower()
                        source_type = str(row.get('sourceFieldType', '')).lower()
                        if 'derived' in comp_type or 'derived' in source_type:
                            derived_dimension_count += 1
                        else:
                            standard_dimension_count += 1

            # Fetch calculated metrics count if component types enabled
            calculated_metric_count = 0
            if self.config.include_component_types:
                try:
                    calc_metrics = cja.getCalculatedMetrics(
                        dataViewId=dv_id,
                        full=False
                    )
                    if calc_metrics is not None:
                        if isinstance(calc_metrics, pd.DataFrame):
                            calculated_metric_count = len(calc_metrics)
                        elif isinstance(calc_metrics, list):
                            calculated_metric_count = len(calc_metrics)
                except Exception:
                    # Calculated metrics API may fail - continue without it
                    pass

            # Fetch metadata if enabled
            owner = None
            owner_id = None
            created = None
            modified = None
            description = None
            has_description = False

            if self.config.include_metadata:
                try:
                    # Try to get detailed data view info
                    dv_details = cja.getDataView(dv_id)
                    if dv_details is not None:
                        if isinstance(dv_details, dict):
                            # Extract owner using utility function
                            owner, owner_id = extract_owner(dv_details.get('owner'))

                            # Extract dates
                            created = dv_details.get('created') or dv_details.get('createdDate')
                            modified = dv_details.get('modified') or dv_details.get('modifiedDate')

                            # Extract description
                            description = dv_details.get('description', '')
                            has_description = bool(description and description.strip())
                except Exception:
                    # Metadata fetch may fail - continue without it
                    pass

            return DataViewSummary(
                data_view_id=dv_id,
                data_view_name=dv_name,
                metric_ids=metric_ids,
                dimension_ids=dimension_ids,
                metric_count=len(metric_ids),
                dimension_count=len(dimension_ids),
                status=dv.get('status', 'active'),
                fetch_duration=time.time() - start_time,
                metric_names=metric_names,
                dimension_names=dimension_names,
                standard_metric_count=standard_metric_count,
                calculated_metric_count=calculated_metric_count,
                derived_metric_count=derived_metric_count,
                standard_dimension_count=standard_dimension_count,
                derived_dimension_count=derived_dimension_count,
                owner=owner,
                owner_id=owner_id,
                created=created,
                modified=modified,
                description=description,
                has_description=has_description,
            )

        except Exception as e:
            error_msg = str(e) or f"{type(e).__name__}"
            return DataViewSummary(
                data_view_id=dv_id,
                data_view_name=dv_name,
                error=error_msg,
                fetch_duration=time.time() - start_time
            )

    def _build_component_index(self, summaries: List[DataViewSummary]) -> Dict[str, ComponentInfo]:
        """Build index mapping component_id -> ComponentInfo.

        Args:
            summaries: List of DataViewSummary objects

        Returns:
            Dict mapping component ID to ComponentInfo (with names if include_names=True)
        """
        index: Dict[str, ComponentInfo] = {}

        for summary in summaries:
            if summary.error:
                continue

            # Index metrics
            for metric_id in summary.metric_ids:
                if metric_id not in index:
                    # Get name from summary if available (first encountered wins)
                    metric_name = None
                    if summary.metric_names:
                        metric_name = summary.metric_names.get(metric_id)
                    index[metric_id] = ComponentInfo(
                        component_id=metric_id,
                        component_type='metric',
                        name=metric_name,
                        data_views=set()
                    )
                index[metric_id].data_views.add(summary.data_view_id)

            # Index dimensions
            for dim_id in summary.dimension_ids:
                if dim_id not in index:
                    # Get name from summary if available (first encountered wins)
                    dim_name = None
                    if summary.dimension_names:
                        dim_name = summary.dimension_names.get(dim_id)
                    index[dim_id] = ComponentInfo(
                        component_id=dim_id,
                        component_type='dimension',
                        name=dim_name,
                        data_views=set()
                    )
                index[dim_id].data_views.add(summary.data_view_id)

        return index

    def _estimate_component_index_memory(self, index: Dict[str, ComponentInfo]) -> float:
        """Estimate memory usage of the component index in MB.

        Uses a heuristic formula:
        per_component = 200 (base object overhead)
                      + len(comp_id)
                      + len(name) if name else 0
                      + 50 * len(data_views) (set entries)
                      + 50 (misc overhead)

        Args:
            index: Component index to estimate

        Returns:
            Estimated memory usage in megabytes
        """
        total_bytes = 0
        for comp_id, info in index.items():
            base_overhead = 200
            id_size = len(comp_id)
            name_size = len(info.name) if info.name else 0
            data_views_size = 50 * len(info.data_views)
            misc_overhead = 50
            total_bytes += base_overhead + id_size + name_size + data_views_size + misc_overhead

        return total_bytes / (1024 * 1024)

    def _check_memory_warning(self, index: Dict[str, ComponentInfo]) -> None:
        """Check if component index memory exceeds threshold and log warning.

        Args:
            index: Component index to check
        """
        threshold = self.config.memory_warning_threshold_mb
        if threshold is None or threshold <= 0:
            return  # Warning disabled

        estimated_mb = self._estimate_component_index_memory(index)
        if estimated_mb > threshold:
            self.logger.warning(
                "High memory usage detected: component index estimated at %.1fMB (threshold: %dMB). "
                "Consider using --limit, --sample, or --skip-similarity to reduce memory footprint.",
                estimated_mb,
                threshold
            )

    def _compute_distribution(
        self,
        index: Dict[str, ComponentInfo],
        total_dvs: int
    ) -> ComponentDistribution:
        """Classify components into distribution buckets.

        Buckets:
        - Core: In >= core_threshold% of data views (or core_min_count)
        - Common: In 25-49% of data views
        - Limited: In 2+ data views but < 25%
        - Isolated: In exactly 1 data view

        Args:
            index: Component index
            total_dvs: Total number of data views analyzed

        Returns:
            ComponentDistribution with bucket assignments
        """
        if total_dvs == 0:
            return ComponentDistribution()

        # Determine core threshold using math.ceil for proper "X% or more" semantics
        # e.g., with 3 DVs and 50% threshold: ceil(3 * 0.5) = ceil(1.5) = 2 (correct)
        if self.config.core_min_count is not None:
            core_threshold_count = max(1, self.config.core_min_count)
        else:
            core_threshold_count = max(1, math.ceil(total_dvs * self.config.core_threshold))

        # Common threshold: 25% of DVs, but at least 2 to avoid overlap with isolated
        common_threshold_count = max(2, math.ceil(total_dvs * 0.25))

        distribution = ComponentDistribution()

        for comp_id, info in index.items():
            presence = info.presence_count

            # Check core threshold first to handle --core-min-count 1 correctly
            # (otherwise presence==1 would always go to isolated before being considered for core)
            if presence >= core_threshold_count:
                # Core
                if info.component_type == 'metric':
                    distribution.core_metrics.append(comp_id)
                else:
                    distribution.core_dimensions.append(comp_id)
            elif presence >= common_threshold_count:
                # Common
                if info.component_type == 'metric':
                    distribution.common_metrics.append(comp_id)
                else:
                    distribution.common_dimensions.append(comp_id)
            elif presence == 1:
                # Isolated (exactly 1 DV)
                if info.component_type == 'metric':
                    distribution.isolated_metrics.append(comp_id)
                else:
                    distribution.isolated_dimensions.append(comp_id)
            else:
                # Limited (in 2+ data views but below common threshold)
                if info.component_type == 'metric':
                    distribution.limited_metrics.append(comp_id)
                else:
                    distribution.limited_dimensions.append(comp_id)

        return distribution

    def _compute_similarity_matrix(self, summaries: List[DataViewSummary]) -> List[SimilarityPair]:
        """Compute pairwise Jaccard similarity between data views.

        Jaccard similarity = |A ∩ B| / |A ∪ B|

        When include_drift is enabled, also captures which components are unique to each DV.
        Always includes pairs with >=90% similarity to support governance checks,
        even if the overlap threshold is higher.

        Args:
            summaries: List of DataViewSummary objects

        Returns:
            List of SimilarityPair objects above threshold, sorted by similarity
        """
        pairs = []
        valid_summaries = [s for s in summaries if s.error is None]
        min_similarity_threshold = min(self.config.overlap_threshold, GOVERNANCE_MAX_OVERLAP_THRESHOLD)

        for i, dv1 in enumerate(valid_summaries):
            set1 = dv1.all_component_ids
            if not set1:
                continue

            for dv2 in valid_summaries[i + 1:]:
                set2 = dv2.all_component_ids
                if not set2:
                    continue

                intersection = len(set1 & set2)
                union = len(set1 | set2)

                if union > 0:
                    similarity = intersection / union

                    if similarity >= min_similarity_threshold:
                        # Compute drift details if enabled
                        only_in_dv1 = []
                        only_in_dv2 = []
                        only_in_dv1_names = None
                        only_in_dv2_names = None

                        if self.config.include_drift:
                            only_in_dv1 = sorted(list(set1 - set2))
                            only_in_dv2 = sorted(list(set2 - set1))

                            # Get names if available
                            if self.config.include_names:
                                only_in_dv1_names = {}
                                for comp_id in only_in_dv1:
                                    name = dv1.get_component_name(comp_id)
                                    if name:
                                        only_in_dv1_names[comp_id] = name

                                only_in_dv2_names = {}
                                for comp_id in only_in_dv2:
                                    name = dv2.get_component_name(comp_id)
                                    if name:
                                        only_in_dv2_names[comp_id] = name

                        pairs.append(SimilarityPair(
                            dv1_id=dv1.data_view_id,
                            dv1_name=dv1.data_view_name,
                            dv2_id=dv2.data_view_id,
                            dv2_name=dv2.data_view_name,
                            jaccard_similarity=round(similarity, 4),
                            shared_count=intersection,
                            union_count=union,
                            only_in_dv1=only_in_dv1,
                            only_in_dv2=only_in_dv2,
                            only_in_dv1_names=only_in_dv1_names,
                            only_in_dv2_names=only_in_dv2_names,
                        ))

        return sorted(pairs, key=lambda p: p.jaccard_similarity, reverse=True)

    def _compute_clusters(self, summaries: List[DataViewSummary]) -> Optional[List[DataViewCluster]]:
        """Compute hierarchical clusters of related data views.

        Uses scipy for hierarchical clustering based on Jaccard distances.

        Args:
            summaries: List of DataViewSummary objects

        Returns:
            List of DataViewCluster objects, or None if clustering fails
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            import numpy as np
        except ImportError:
            self.logger.warning(
                "scipy not available - skipping clustering. "
                "Install with: uv pip install 'cja-auto-sdr[clustering]'"
            )
            return None

        valid_summaries = [s for s in summaries if s.error is None and s.all_component_ids]
        if len(valid_summaries) < 2:
            self.logger.info("Not enough data views for clustering")
            return None

        # Warn if ward method is used - it assumes Euclidean distances
        if self.config.cluster_method == "ward":
            self.logger.warning(
                "Cluster method 'ward' assumes Euclidean distances but Jaccard distances are used. "
                "Results may be suboptimal. Consider using 'average' or 'complete' instead."
            )

        n = len(valid_summaries)

        # Build distance matrix (1 - Jaccard similarity)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            set_i = valid_summaries[i].all_component_ids
            for j in range(i + 1, n):
                set_j = valid_summaries[j].all_component_ids
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / union if union > 0 else 0
                distance = 1 - jaccard
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance

        # Convert to condensed form for scipy
        condensed_dist = squareform(dist_matrix)

        # Perform hierarchical clustering
        try:
            Z = linkage(condensed_dist, method=self.config.cluster_method)
        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            return None

        # Determine optimal number of clusters using silhouette or fixed threshold
        # Using a distance threshold of 0.5 (50% dissimilarity)
        threshold = 0.5
        labels = fcluster(Z, t=threshold, criterion='distance')

        # Build cluster objects
        cluster_members: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_members[label].append(idx)

        clusters = []
        for cluster_id, member_indices in cluster_members.items():
            member_summaries = [valid_summaries[i] for i in member_indices]
            dv_ids = [s.data_view_id for s in member_summaries]
            dv_names = [s.data_view_name for s in member_summaries]

            # Compute cohesion (average within-cluster similarity)
            cohesion = 0.0
            if len(member_indices) > 1:
                similarities = []
                for i in range(len(member_indices)):
                    for j in range(i + 1, len(member_indices)):
                        idx_i, idx_j = member_indices[i], member_indices[j]
                        similarities.append(1 - dist_matrix[idx_i, idx_j])
                cohesion = sum(similarities) / len(similarities) if similarities else 0.0

            # Infer cluster name from common prefix
            cluster_name = self._infer_cluster_name(dv_names)

            clusters.append(DataViewCluster(
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                data_view_ids=dv_ids,
                data_view_names=dv_names,
                cohesion_score=round(cohesion, 4),
            ))

        # Sort by cluster size descending
        return sorted(clusters, key=lambda c: c.size, reverse=True)

    def _infer_cluster_name(self, names: List[str]) -> Optional[str]:
        """Infer a cluster name from common prefixes of member names.

        Args:
            names: List of data view names in the cluster

        Returns:
            Inferred cluster name or None
        """
        if not names:
            return None
        if len(names) == 1:
            return names[0]

        # Find longest common prefix
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix) and prefix:
                prefix = prefix[:-1]
            if not prefix:
                break

        # Clean up prefix (remove trailing separators)
        prefix = prefix.rstrip(' -_')

        if len(prefix) >= 3:
            return prefix

        # Try first word if no common prefix
        first_words = [n.split()[0] if n.split() else n for n in names]
        if len(set(first_words)) == 1:
            return first_words[0]

        return None

    def _generate_recommendations(
        self,
        summaries: List[DataViewSummary],
        index: Dict[str, ComponentInfo],
        distribution: ComponentDistribution,
        similarity_pairs: Optional[List[SimilarityPair]]
    ) -> List[Dict[str, Any]]:
        """Generate governance recommendations based on analysis.

        Args:
            summaries: Data view summaries
            index: Component index
            distribution: Component distribution
            similarity_pairs: Similarity matrix results

        Returns:
            List of recommendation dicts
        """
        recommendations = []

        # Build isolated components by data view
        isolated_by_dv: Dict[str, List[str]] = {}
        for comp_id, info in index.items():
            if info.presence_count == 1:
                dv_id = next(iter(info.data_views))
                if dv_id not in isolated_by_dv:
                    isolated_by_dv[dv_id] = []
                isolated_by_dv[dv_id].append(comp_id)

        # Recommendation: Data views with many isolated components
        for dv_id, isolated in isolated_by_dv.items():
            if len(isolated) > self.config.isolated_review_threshold:
                dv_name = next((s.data_view_name for s in summaries if s.data_view_id == dv_id), 'Unknown')
                recommendations.append({
                    'type': 'review_isolated',
                    'severity': 'medium',
                    'data_view': dv_id,
                    'data_view_name': dv_name,
                    'isolated_count': len(isolated),
                    'reason': f"Data view has {len(isolated)} components not used elsewhere. "
                              "Consider if these are needed or if this DV serves a specialized purpose."
                })

        # Recommendation: High overlap pairs
        if similarity_pairs:
            for pair in similarity_pairs:
                if pair.jaccard_similarity >= GOVERNANCE_MAX_OVERLAP_THRESHOLD:
                    recommendations.append({
                        'type': 'review_overlap',
                        'severity': 'high',
                        'data_view_1': pair.dv1_id,
                        'data_view_1_name': pair.dv1_name,
                        'data_view_2': pair.dv2_id,
                        'data_view_2_name': pair.dv2_name,
                        'similarity': pair.jaccard_similarity,
                        'reason': f"{pair.jaccard_similarity*100:.0f}% similarity - "
                                  "likely prod/staging pair or potential duplicate. Verify if intentional."
                    })

        # Recommendation: Core component standardization
        total_successful = len([s for s in summaries if s.error is None])
        if total_successful > 3:
            # Check for near-core components (in 70-99% of DVs but not all)
            near_core_count = 0
            for comp_id, info in index.items():
                pct = info.presence_count / total_successful
                if 0.7 <= pct < 1.0:
                    near_core_count += 1

            if near_core_count > 5:
                recommendations.append({
                    'type': 'standardization_opportunity',
                    'severity': 'low',
                    'count': near_core_count,
                    'reason': f"{near_core_count} components are in 70-99% of data views. "
                              "Consider standardizing these across all data views."
                })

        # Recommendation: Fetch errors
        error_count = len([s for s in summaries if s.error is not None])
        if error_count > 0:
            recommendations.append({
                'type': 'fetch_errors',
                'severity': 'high',
                'count': error_count,
                'reason': f"{error_count} data view(s) could not be analyzed due to errors. "
                          "Check permissions and data view status."
            })

        # Recommendation: High derived ratio (if component types enabled)
        if self.config.include_component_types:
            for summary in summaries:
                if summary.error:
                    continue
                total_components = summary.metric_count + summary.dimension_count
                derived_count = summary.derived_metric_count + summary.derived_dimension_count
                if total_components > 0 and derived_count / total_components > 0.5:
                    recommendations.append({
                        'type': 'high_derived_ratio',
                        'severity': 'low',
                        'data_view': summary.data_view_id,
                        'data_view_name': summary.data_view_name,
                        'derived_count': derived_count,
                        'total_count': total_components,
                        'ratio': round(derived_count / total_components, 2),
                        'reason': f"Data view has {derived_count}/{total_components} ({derived_count*100//total_components}%) derived components. "
                                  "High derived ratios may indicate complex transformations or maintenance burden."
                    })

        # Recommendation: Stale data views (if metadata enabled)
        if self.config.include_metadata:
            six_months_ago = datetime.now() - timedelta(days=180)
            for summary in summaries:
                if summary.error or not summary.modified:
                    continue
                try:
                    modified_date = datetime.fromisoformat(summary.modified.replace('Z', '+00:00'))
                    if modified_date.replace(tzinfo=None) < six_months_ago:
                        recommendations.append({
                            'type': 'stale_data_view',
                            'severity': 'low',
                            'data_view': summary.data_view_id,
                            'data_view_name': summary.data_view_name,
                            'modified': summary.modified,
                            'reason': f"Data view not modified since {summary.modified[:10]}. "
                                      "Consider reviewing if still needed or if updates are required."
                        })
                except (ValueError, TypeError):
                    pass

            # Missing descriptions
            no_desc_count = len([s for s in summaries if not s.error and not s.has_description])
            if no_desc_count > 0 and no_desc_count >= len(summaries) * 0.3:
                recommendations.append({
                    'type': 'missing_descriptions',
                    'severity': 'low',
                    'count': no_desc_count,
                    'reason': f"{no_desc_count} data view(s) have no description. "
                              "Adding descriptions improves discoverability and governance."
                })

        # Recommendation: Drift in high-overlap pairs (if drift enabled)
        if self.config.include_drift and similarity_pairs:
            for pair in similarity_pairs:
                drift_count = len(pair.only_in_dv1) + len(pair.only_in_dv2)
                if pair.jaccard_similarity >= GOVERNANCE_MAX_OVERLAP_THRESHOLD and drift_count > 0:
                    # Update existing overlap recommendation with drift info
                    for rec in recommendations:
                        if (rec.get('type') == 'review_overlap' and
                            rec.get('data_view_1') == pair.dv1_id and
                            rec.get('data_view_2') == pair.dv2_id):
                            rec['drift_count'] = drift_count
                            rec['reason'] += f" Differs by {drift_count} components."
                            break

        return recommendations

    def _check_governance_thresholds(
        self,
        similarity_pairs: Optional[List[SimilarityPair]],
        distribution: ComponentDistribution,
        total_components: int
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Check if governance thresholds are exceeded (Feature 1).

        Args:
            similarity_pairs: List of similarity pairs from analysis
            distribution: Component distribution across buckets
            total_components: Total unique components

        Returns:
            Tuple of (violations_list, thresholds_exceeded)
        """
        violations = []
        exceeded = False

        # Check duplicate threshold (high-similarity pairs >= 90%)
        if self.config.duplicate_threshold is not None and similarity_pairs:
            high_sim_pairs = [p for p in similarity_pairs if p.jaccard_similarity >= GOVERNANCE_MAX_OVERLAP_THRESHOLD]
            if len(high_sim_pairs) > self.config.duplicate_threshold:
                violations.append({
                    'type': 'duplicate_threshold_exceeded',
                    'threshold': self.config.duplicate_threshold,
                    'actual': len(high_sim_pairs),
                    'message': f"Found {len(high_sim_pairs)} high-similarity pairs (>=90%), threshold is {self.config.duplicate_threshold}"
                })
                exceeded = True

        # Check isolated threshold (percentage of isolated components)
        if self.config.isolated_threshold is not None and total_components > 0:
            isolated_count = distribution.total_isolated
            isolated_pct = isolated_count / total_components
            if isolated_pct > self.config.isolated_threshold:
                violations.append({
                    'type': 'isolated_threshold_exceeded',
                    'threshold': self.config.isolated_threshold,
                    'actual': round(isolated_pct, 4),
                    'isolated_count': isolated_count,
                    'total_components': total_components,
                    'message': f"Isolated components at {isolated_pct*100:.1f}% ({isolated_count}/{total_components}), threshold is {self.config.isolated_threshold*100:.1f}%"
                })
                exceeded = True

        return violations, exceeded

    def _audit_naming_conventions(
        self,
        component_index: Dict[str, ComponentInfo]
    ) -> Dict[str, Any]:
        """Audit naming convention consistency (Feature 3).

        Detects:
        - snake_case vs camelCase vs PascalCase mixing
        - Prefix groupings (e.g., "evar_", "prop_")
        - Stale patterns (test, old, temp, dates)

        Args:
            component_index: Component index from analysis

        Returns:
            Dict with naming audit results
        """
        audit = {
            'total_components': len(component_index),
            'case_styles': {'snake_case': 0, 'camelCase': 0, 'PascalCase': 0, 'other': 0},
            'prefix_groups': {},
            'stale_patterns': [],
            'recommendations': []
        }

        # Stale pattern regexes
        stale_keywords = re.compile(r'(^|[_\-\s])(test|old|temp|tmp|backup|copy|deprecated|legacy|archive)([_\-\s]|$)', re.IGNORECASE)
        version_suffix = re.compile(r'[_\-]v\d+$', re.IGNORECASE)
        date_pattern = re.compile(r'[_\-]?(20\d{2}[01]\d[0-3]\d|20\d{2}[_\-][01]\d[_\-][0-3]\d)([_\-]|$)')

        for comp_id, info in component_index.items():
            # Use name if available, otherwise use ID
            name = info.name or comp_id

            # Detect case style
            if '_' in name and name == name.lower():
                audit['case_styles']['snake_case'] += 1
            elif name and name[0].islower() and any(c.isupper() for c in name):
                audit['case_styles']['camelCase'] += 1
            elif name and name[0].isupper() and any(c.isupper() for c in name[1:]):
                audit['case_styles']['PascalCase'] += 1
            else:
                audit['case_styles']['other'] += 1

            # Detect prefix groupings
            if '/' in name:
                prefix = name.split('/')[0]
            elif '_' in name:
                prefix = name.split('_')[0]
            elif name and len(name) > 3:
                prefix = name[:3]
            else:
                prefix = 'other'

            prefix_lower = prefix.lower()
            if prefix_lower not in audit['prefix_groups']:
                audit['prefix_groups'][prefix_lower] = 0
            audit['prefix_groups'][prefix_lower] += 1

            # Detect stale patterns
            if stale_keywords.search(name):
                audit['stale_patterns'].append({
                    'component_id': comp_id,
                    'name': name,
                    'pattern': 'stale_keyword',
                    'data_views': list(info.data_views)[:3]  # First 3 DVs
                })
            elif version_suffix.search(name):
                audit['stale_patterns'].append({
                    'component_id': comp_id,
                    'name': name,
                    'pattern': 'version_suffix',
                    'data_views': list(info.data_views)[:3]
                })
            elif date_pattern.search(name):
                audit['stale_patterns'].append({
                    'component_id': comp_id,
                    'name': name,
                    'pattern': 'date_pattern',
                    'data_views': list(info.data_views)[:3]
                })

        # Generate recommendations
        styles = audit['case_styles']
        dominant_style = max(styles.items(), key=lambda x: x[1])[0]
        non_dominant = sum(v for k, v in styles.items() if k != dominant_style and k != 'other')
        if non_dominant > 5:
            audit['recommendations'].append({
                'type': 'naming_inconsistency',
                'severity': 'low',
                'message': f"Mixed naming conventions detected. Dominant style is {dominant_style}, "
                          f"but {non_dominant} components use different styles."
            })

        if len(audit['stale_patterns']) > 0:
            audit['recommendations'].append({
                'type': 'stale_naming_patterns',
                'severity': 'medium',
                'count': len(audit['stale_patterns']),
                'message': f"Found {len(audit['stale_patterns'])} components with stale naming patterns "
                          "(test, old, temp, version suffixes, or date stamps)."
            })

        return audit

    def _compute_owner_summary(
        self,
        summaries: List[DataViewSummary]
    ) -> Dict[str, Any]:
        """Compute statistics grouped by data view owner (Feature 5).

        Args:
            summaries: List of DataViewSummary objects

        Returns:
            Dict with owner-grouped statistics
        """
        owner_stats: Dict[str, Dict[str, Any]] = {}

        for summary in summaries:
            if summary.error:
                continue

            owner = summary.owner or 'Unknown'
            owner_id = summary.owner_id or 'unknown'

            if owner not in owner_stats:
                owner_stats[owner] = {
                    'owner_id': owner_id,
                    'data_view_count': 0,
                    'data_view_names': [],
                    'data_view_ids': [],
                    'total_metrics': 0,
                    'total_dimensions': 0,
                    'total_derived': 0,
                    'total_calculated': 0,
                }

            stats = owner_stats[owner]
            stats['data_view_count'] += 1
            stats['data_view_names'].append(summary.data_view_name)
            stats['data_view_ids'].append(summary.data_view_id)
            stats['total_metrics'] += summary.metric_count
            stats['total_dimensions'] += summary.dimension_count
            stats['total_derived'] += summary.derived_metric_count + summary.derived_dimension_count
            stats['total_calculated'] += summary.calculated_metric_count

        # Compute averages
        for owner, stats in owner_stats.items():
            dv_count = stats['data_view_count']
            stats['avg_metrics_per_dv'] = round(stats['total_metrics'] / dv_count, 1) if dv_count > 0 else 0
            stats['avg_dimensions_per_dv'] = round(stats['total_dimensions'] / dv_count, 1) if dv_count > 0 else 0
            stats['avg_components_per_dv'] = round(
                (stats['total_metrics'] + stats['total_dimensions']) / dv_count, 1
            ) if dv_count > 0 else 0

        return {
            'by_owner': owner_stats,
            'total_owners': len(owner_stats),
            'owners_sorted_by_dv_count': sorted(
                owner_stats.keys(),
                key=lambda o: owner_stats[o]['data_view_count'],
                reverse=True
            )
        }

    def _fetch_modification_date(self, dv_id: str) -> Optional[str]:
        """Fetch the current modification date for a data view.

        Makes a lightweight API call to get just the data view metadata.

        Args:
            dv_id: Data view ID

        Returns:
            Modification timestamp string or None if unavailable
        """
        try:
            cja = self._get_thread_client()
            dv_details = cja.getDataView(dv_id)
            if dv_details is not None and isinstance(dv_details, dict):
                return dv_details.get('modified') or dv_details.get('modifiedDate')
        except Exception:
            pass
        return None

    def _validate_cache_entries(
        self,
        data_views: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[DataViewSummary], int, int]:
        """Validate cached entries against current modification timestamps.

        For each data view, checks if the cached entry's modification timestamp
        matches the current modification timestamp from the API.

        Args:
            data_views: List of data view dicts to validate

        Returns:
            Tuple of (dvs_to_fetch, valid_cached_summaries, valid_count, stale_count)
        """
        if not self.cache:
            return data_views, [], 0, 0

        to_fetch = []
        valid_summaries = []
        valid_count = 0
        stale_count = 0

        required_flags = {
            'include_names': self.config.include_names,
            'include_metadata': self.config.include_metadata,
            'include_component_types': self.config.include_component_types,
        }

        for dv in data_views:
            dv_id = dv.get('id', '')

            # Check if entry exists and is within age limit
            if not self.cache.needs_validation(dv_id, self.config.cache_max_age_hours):
                to_fetch.append(dv)
                continue

            # Fetch current modification date
            current_modified = self._fetch_modification_date(dv_id)

            # Try to get from cache with validation
            cached = self.cache.get(
                dv_id,
                self.config.cache_max_age_hours,
                required_flags=required_flags,
                current_modified=current_modified
            )

            if cached:
                valid_summaries.append(cached)
                valid_count += 1
            else:
                to_fetch.append(dv)
                stale_count += 1

        return to_fetch, valid_summaries, valid_count, stale_count

    def _detect_stale_components(
        self,
        component_index: Dict[str, ComponentInfo]
    ) -> List[Dict[str, Any]]:
        """Detect components with stale naming patterns (Feature 6).

        Detects:
        - Keywords: test, old, temp, tmp, backup, copy, deprecated
        - Version suffixes: _v1, _v2, _V1
        - Date patterns: _20240101, _2024-01-01

        Args:
            component_index: Component index from analysis

        Returns:
            List of stale component entries
        """
        stale_components = []

        # Pattern definitions
        stale_keywords = re.compile(
            r'(^|[_\-\s])(test|old|temp|tmp|backup|copy|deprecated|legacy|archive|obsolete|unused)([_\-\s]|$)',
            re.IGNORECASE
        )
        version_suffix = re.compile(r'[_\-]v\d+$', re.IGNORECASE)
        date_pattern = re.compile(r'[_\-]?(20\d{2}[01]\d[0-3]\d|20\d{2}[_\-][01]\d[_\-][0-3]\d)([_\-]|$)')

        for comp_id, info in component_index.items():
            name = info.name or comp_id
            pattern_matched = None

            if stale_keywords.search(name):
                pattern_matched = 'stale_keyword'
            elif version_suffix.search(name):
                pattern_matched = 'version_suffix'
            elif date_pattern.search(name):
                pattern_matched = 'date_pattern'

            if pattern_matched:
                stale_components.append({
                    'component_id': comp_id,
                    'name': name,
                    'type': info.component_type,
                    'pattern': pattern_matched,
                    'presence_count': info.presence_count,
                    'data_views': list(info.data_views)[:5]  # First 5 DVs
                })

        return stale_components
