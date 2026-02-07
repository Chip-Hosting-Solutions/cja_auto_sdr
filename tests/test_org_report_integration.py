"""
Integration tests for org-wide report functionality.

These tests exercise the full org-report flow end-to-end with mocked CJA API,
verifying that all components work together correctly.
"""

import json
import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest

sys.path.insert(0, ".")

from cja_auto_sdr.core.exceptions import MemoryLimitExceeded
from cja_auto_sdr.generator import (
    OrgComponentAnalyzer,
    OrgReportConfig,
    write_org_report_excel,
    write_org_report_json,
)


def create_mock_data_views(count: int = 5, prefix: str = "DV") -> list:
    """Create a list of mock data view dicts.

    Args:
        count: Number of data views to create
        prefix: Name prefix for data views

    Returns:
        List of data view dicts
    """
    return [
        {
            "id": f"dv_{i:05d}",
            "name": f"{prefix} {i}",
            "status": "active",
            "modified": f"2024-01-{i + 1:02d}T10:00:00Z",
            "created": "2023-01-01T10:00:00Z",
        }
        for i in range(1, count + 1)
    ]


def create_mock_metrics(dv_id: str, count: int = 10, overlap_ids: list | None = None) -> pd.DataFrame:
    """Create mock metrics DataFrame for a data view.

    Args:
        dv_id: Data view ID
        count: Number of metrics to create
        overlap_ids: List of metric IDs that should overlap with other DVs

    Returns:
        DataFrame with metric data
    """
    overlap_ids = overlap_ids or []
    metrics = []

    # Add overlapping metrics first
    for metric_id in overlap_ids:
        metrics.append(
            {
                "id": metric_id,
                "name": f"Metric {metric_id}",
                "type": "standard",
                "sourceFieldType": "field",
            }
        )

    # Add unique metrics for this DV
    for i in range(count - len(overlap_ids)):
        metrics.append(
            {
                "id": f"{dv_id}_metric_{i}",
                "name": f"Metric {i} for {dv_id}",
                "type": "standard",
                "sourceFieldType": "field",
            }
        )

    return pd.DataFrame(metrics)


def create_mock_dimensions(dv_id: str, count: int = 10, overlap_ids: list | None = None) -> pd.DataFrame:
    """Create mock dimensions DataFrame for a data view.

    Args:
        dv_id: Data view ID
        count: Number of dimensions to create
        overlap_ids: List of dimension IDs that should overlap with other DVs

    Returns:
        DataFrame with dimension data
    """
    overlap_ids = overlap_ids or []
    dimensions = []

    # Add overlapping dimensions first
    for dim_id in overlap_ids:
        dimensions.append(
            {
                "id": dim_id,
                "name": f"Dimension {dim_id}",
                "type": "standard",
                "sourceFieldType": "field",
            }
        )

    # Add unique dimensions for this DV
    for i in range(count - len(overlap_ids)):
        dimensions.append(
            {
                "id": f"{dv_id}_dim_{i}",
                "name": f"Dimension {i} for {dv_id}",
                "type": "standard",
                "sourceFieldType": "field",
            }
        )

    return pd.DataFrame(dimensions)


class TestOrgReportIntegration:
    """End-to-end integration tests for --org-report."""

    @pytest.fixture
    def mock_cja_with_realistic_data(self):
        """Create mock CJA with realistic multi-DV data.

        Creates 5 data views with overlapping components to test
        similarity detection and distribution classification.
        """
        mock = Mock()

        # Setup 5 data views
        data_views = create_mock_data_views(5)
        mock.getDataViews.return_value = data_views

        # Define overlapping components
        # Core components (in all 5 DVs)
        core_metrics = ["metrics/pageviews", "metrics/visits", "metrics/revenue"]
        core_dims = ["variables/page", "variables/date"]

        # Common components (in 3 DVs)
        common_metrics = ["metrics/orders", "metrics/units"]
        common_dims = ["variables/product"]

        # Setup metrics/dimensions per DV
        def get_metrics(dv_id, **kwargs):
            if dv_id == "dv_00001":
                return create_mock_metrics(dv_id, 15, core_metrics + common_metrics)
            elif dv_id == "dv_00002":
                return create_mock_metrics(dv_id, 12, core_metrics + common_metrics)
            elif dv_id == "dv_00003":
                return create_mock_metrics(dv_id, 10, core_metrics + common_metrics)
            elif dv_id == "dv_00004":
                return create_mock_metrics(dv_id, 8, core_metrics)
            else:
                return create_mock_metrics(dv_id, 6, core_metrics)

        def get_dimensions(dv_id, **kwargs):
            if dv_id == "dv_00001":
                return create_mock_dimensions(dv_id, 10, [*core_dims, common_dims[0]])
            elif dv_id == "dv_00002":
                return create_mock_dimensions(dv_id, 8, [*core_dims, common_dims[0]])
            elif dv_id == "dv_00003":
                return create_mock_dimensions(dv_id, 7, [*core_dims, common_dims[0]])
            elif dv_id == "dv_00004":
                return create_mock_dimensions(dv_id, 6, core_dims)
            else:
                return create_mock_dimensions(dv_id, 5, core_dims)

        mock.getMetrics.side_effect = get_metrics
        mock.getDimensions.side_effect = get_dimensions

        return mock

    @pytest.fixture
    def mock_cja_with_duplicates(self):
        """Create mock CJA with near-duplicate data views (high similarity)."""
        mock = Mock()

        # Setup 3 data views where 2 are nearly identical
        data_views = create_mock_data_views(3)
        data_views[0]["name"] = "Production Analytics"
        data_views[1]["name"] = "Staging Analytics"  # Near-duplicate of production
        data_views[2]["name"] = "Test Environment"
        mock.getDataViews.return_value = data_views

        # Production and Staging share 95% of components
        shared_metrics = [f"metrics/shared_{i}" for i in range(19)]
        shared_dims = [f"dims/shared_{i}" for i in range(10)]

        def get_metrics(dv_id, **kwargs):
            if dv_id == "dv_00001":
                # Production: 19 shared + 1 unique
                df = create_mock_metrics(dv_id, 20, shared_metrics)
                return df
            elif dv_id == "dv_00002":
                # Staging: 19 shared + 1 unique (different unique)
                df = create_mock_metrics(dv_id, 20, shared_metrics)
                return df
            else:
                # Test: only 5 shared
                return create_mock_metrics(dv_id, 10, shared_metrics[:5])

        def get_dimensions(dv_id, **kwargs):
            if dv_id in ["dv_00001", "dv_00002"]:
                return create_mock_dimensions(dv_id, 10, shared_dims)
            else:
                return create_mock_dimensions(dv_id, 6, shared_dims[:3])

        mock.getMetrics.side_effect = get_metrics
        mock.getDimensions.side_effect = get_dimensions

        return mock

    def test_full_analysis_flow_console_output(self, mock_cja_with_realistic_data, capsys):
        """Test complete analysis with console output."""
        import logging

        logger = logging.getLogger("test_console")
        # Use cja_per_thread=False to avoid creating per-thread clients that bypass mocking
        config = OrgReportConfig(skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Verify basic result structure
        assert result is not None
        assert result.org_id == "test@AdobeOrg"
        assert len(result.data_view_summaries) == 5
        assert result.successful_data_views == 5
        assert result.total_unique_components > 0

        # Verify distribution was computed
        assert result.distribution is not None
        assert result.distribution.total_core > 0  # Core metrics + dims

        # Verify similarity was computed
        assert result.similarity_pairs is not None

    def test_full_analysis_flow_json_output(self, mock_cja_with_realistic_data, tmp_path):
        """Test complete analysis with JSON file output."""
        import logging

        logger = logging.getLogger("test_json")
        config = OrgReportConfig(skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Write JSON output
        json_path = tmp_path / "org_report.json"
        write_org_report_json(result, json_path, str(tmp_path), logger)

        # Verify JSON was written and is valid
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)

        assert data["report_type"] == "org_analysis"
        assert data["org_id"] == "test@AdobeOrg"
        assert "summary" in data
        assert "distribution" in data
        assert data["summary"]["data_views_analyzed"] == 5

    def test_full_analysis_flow_excel_output(self, mock_cja_with_realistic_data, tmp_path):
        """Test complete analysis with Excel output."""
        import logging

        logger = logging.getLogger("test_excel")
        config = OrgReportConfig(skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Write Excel output
        excel_path = tmp_path / "org_report.xlsx"
        write_org_report_excel(result, excel_path, str(tmp_path), logger)

        # Verify Excel was written
        assert excel_path.exists()
        assert excel_path.stat().st_size > 0

    def test_analysis_with_filtering(self, mock_cja_with_realistic_data):
        """Test --filter and --exclude patterns work correctly."""
        import logging

        logger = logging.getLogger("test_filter")

        # Filter to include only DVs with "1" or "2" in name
        config = OrgReportConfig(filter_pattern=r"DV [12]", skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Should only have DV 1 and DV 2
        assert len(result.data_view_summaries) == 2
        names = {s.data_view_name for s in result.data_view_summaries}
        assert "DV 1" in names
        assert "DV 2" in names

    def test_analysis_with_exclude(self, mock_cja_with_realistic_data):
        """Test --exclude pattern works correctly."""
        import logging

        logger = logging.getLogger("test_exclude")

        # Exclude DVs with "5" in name
        config = OrgReportConfig(exclude_pattern=r"DV 5", skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Should have 4 DVs (DV 5 excluded)
        assert len(result.data_view_summaries) == 4
        names = {s.data_view_name for s in result.data_view_summaries}
        assert "DV 5" not in names

    def test_analysis_with_limit(self, mock_cja_with_realistic_data):
        """Test --limit works correctly."""
        import logging

        logger = logging.getLogger("test_limit")

        config = OrgReportConfig(limit=2, skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        assert len(result.data_view_summaries) == 2

    def test_analysis_with_sampling(self, mock_cja_with_realistic_data):
        """Test --sample with reproducible seed."""
        import logging

        logger = logging.getLogger("test_sample")

        config = OrgReportConfig(sample_size=3, sample_seed=42, skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result1 = analyzer.run_analysis()

        # Run again with same seed
        analyzer2 = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result2 = analyzer2.run_analysis()

        # Should get same DVs
        assert len(result1.data_view_summaries) == 3
        assert len(result2.data_view_summaries) == 3

        ids1 = {s.data_view_id for s in result1.data_view_summaries}
        ids2 = {s.data_view_id for s in result2.data_view_summaries}
        assert ids1 == ids2

        assert result1.is_sampled is True
        assert result1.total_available_data_views == 5

    def test_analysis_with_skip_similarity(self, mock_cja_with_realistic_data):
        """Test --skip-similarity works correctly."""
        import logging

        logger = logging.getLogger("test_skip_sim")

        config = OrgReportConfig(skip_similarity=True, skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Similarity should be None
        assert result.similarity_pairs is None
        # But distribution should still work
        assert result.distribution is not None

    def test_analysis_org_stats_mode(self, mock_cja_with_realistic_data):
        """Test --org-stats quick mode."""
        import logging

        logger = logging.getLogger("test_stats")

        config = OrgReportConfig(org_stats_only=True, skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Should skip similarity and clustering
        assert result.similarity_pairs is None
        assert result.clusters is None
        # But should still have basic stats
        assert len(result.data_view_summaries) == 5
        assert result.distribution is not None

    def test_high_similarity_detection(self, mock_cja_with_duplicates):
        """Test that high-similarity pairs are detected correctly."""
        import logging

        logger = logging.getLogger("test_duplicates")

        config = OrgReportConfig(overlap_threshold=0.8, skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_duplicates, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Should detect high similarity between Production and Staging
        assert result.similarity_pairs is not None
        assert len(result.similarity_pairs) > 0

        # Find the prod/staging pair
        high_sim_pairs = [p for p in result.similarity_pairs if p.jaccard_similarity >= 0.8]
        assert len(high_sim_pairs) >= 1

    def test_governance_threshold_violations(self, mock_cja_with_duplicates):
        """Test governance threshold detection."""
        import logging

        logger = logging.getLogger("test_governance")

        config = OrgReportConfig(
            duplicate_threshold=0,  # Any duplicate is a violation
            fail_on_threshold=True,
            skip_lock=True,
            cja_per_thread=False,
        )

        analyzer = OrgComponentAnalyzer(mock_cja_with_duplicates, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Should have violations
        assert result.governance_violations is not None
        assert len(result.governance_violations) > 0
        assert result.thresholds_exceeded is True

    def test_memory_limit_exceeded(self, mock_cja_with_realistic_data):
        """Test --memory-limit aborts when exceeded."""
        import logging

        logger = logging.getLogger("test_memory_limit")

        # Set a very low memory limit that will be exceeded
        config = OrgReportConfig(
            memory_limit_mb=1,  # 1MB - will be exceeded by any real data
            skip_lock=True,
            cja_per_thread=False,
        )

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")

        # This might or might not raise depending on the actual data size
        # For this test, we patch the estimate to ensure it exceeds
        with patch.object(analyzer, "_estimate_component_index_memory", return_value=50.0):
            with pytest.raises(MemoryLimitExceeded) as exc_info:
                analyzer.run_analysis()

            assert exc_info.value.limit_mb == 1
            assert exc_info.value.estimated_mb == 50.0

    def test_clustering_enabled(self, mock_cja_with_realistic_data):
        """Test --cluster produces cluster results."""
        import logging

        pytest.importorskip("scipy", reason="scipy required for clustering")

        logger = logging.getLogger("test_cluster")

        config = OrgReportConfig(enable_clustering=True, cluster_method="average", skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Should have clusters
        assert result.clusters is not None
        assert len(result.clusters) >= 1

        # Each cluster should have data
        for cluster in result.clusters:
            assert len(cluster.data_view_ids) >= 1
            assert cluster.cohesion_score >= 0

    def test_core_component_detection(self, mock_cja_with_realistic_data):
        """Test that core components are detected correctly."""
        import logging

        logger = logging.getLogger("test_core")

        config = OrgReportConfig(
            core_threshold=0.8,  # 80% = 4 out of 5 DVs
            skip_lock=True,
            cja_per_thread=False,
        )

        analyzer = OrgComponentAnalyzer(mock_cja_with_realistic_data, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        # Core metrics should include the shared ones
        assert result.distribution.total_core > 0

        # Verify core components are in the index
        for comp_id in result.distribution.core_metrics + result.distribution.core_dimensions:
            assert comp_id in result.component_index
            # Core components should be in >= 80% of DVs
            info = result.component_index[comp_id]
            assert info.presence_count >= 4  # 80% of 5

    def test_empty_org_handling(self):
        """Test handling of organization with no data views."""
        import logging

        mock_cja = Mock()
        mock_cja.getDataViews.return_value = []

        logger = logging.getLogger("test_empty")

        config = OrgReportConfig(skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja, config, logger, org_id="empty@AdobeOrg")
        result = analyzer.run_analysis()

        assert len(result.data_view_summaries) == 0
        assert result.total_unique_components == 0

    def test_data_view_with_error(self):
        """Test handling of data views that return errors."""
        import logging

        mock_cja = Mock()
        mock_cja.getDataViews.return_value = create_mock_data_views(2)

        # First DV succeeds, second raises error
        def get_metrics(dv_id, **kwargs):
            if dv_id == "dv_00001":
                return create_mock_metrics(dv_id, 5)
            else:
                raise Exception("API Error")

        def get_dimensions(dv_id, **kwargs):
            if dv_id == "dv_00001":
                return create_mock_dimensions(dv_id, 5)
            else:
                raise Exception("API Error")

        mock_cja.getMetrics.side_effect = get_metrics
        mock_cja.getDimensions.side_effect = get_dimensions

        logger = logging.getLogger("test_error")

        config = OrgReportConfig(skip_lock=True, cja_per_thread=False)

        analyzer = OrgComponentAnalyzer(mock_cja, config, logger, org_id="test@AdobeOrg")
        result = analyzer.run_analysis()

        assert len(result.data_view_summaries) == 2
        assert result.successful_data_views == 1

        # One should have error
        errors = [s for s in result.data_view_summaries if s.error is not None]
        assert len(errors) == 1

        # Should have recommendation about fetch errors
        error_recs = [r for r in result.recommendations if r["type"] == "fetch_errors"]
        assert len(error_recs) == 1


class TestCacheValidationBatch:
    """Test batch cache validation optimization."""

    def test_cache_validation_uses_batch_metadata(self):
        """Test that cache validation uses metadata from getDataViews() instead of individual calls."""
        import logging

        from cja_auto_sdr.org.cache import OrgReportCache

        mock_cja = Mock()

        # Data views with modification dates
        data_views = [
            {"id": "dv_1", "name": "DV 1", "modified": "2024-01-15T10:00:00Z"},
            {"id": "dv_2", "name": "DV 2", "modified": "2024-01-16T10:00:00Z"},
        ]
        mock_cja.getDataViews.return_value = data_views

        # Mock the getDataView method (should NOT be called in batch mode)
        mock_cja.getDataView = Mock()

        # Setup mock cache
        mock_cache = Mock(spec=OrgReportCache)
        mock_cache.has_valid_entry.return_value = True
        mock_cache.get.return_value = None  # All stale

        logger = logging.getLogger("test_batch_cache")

        config = OrgReportConfig(use_cache=True, validate_cache=True, skip_lock=True)

        analyzer = OrgComponentAnalyzer(mock_cja, config, logger, org_id="test@AdobeOrg", cache=mock_cache)

        # Call the validation method directly
        _to_fetch, _valid, _valid_count, _stale_count = analyzer._validate_cache_entries(data_views)

        # getDataView should NOT be called (batch optimization)
        mock_cja.getDataView.assert_not_called()

        # Cache.get should have been called with the modification dates from the list
        assert mock_cache.get.call_count == 2

        # Verify modification dates were passed
        calls = mock_cache.get.call_args_list
        for i, call in enumerate(calls):
            assert call.kwargs.get("current_modified") == data_views[i]["modified"]
