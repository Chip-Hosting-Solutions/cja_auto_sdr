"""
Edge-case tests for cja_auto_sdr.org.analyzer to boost coverage from ~75% to ~85%.

Targets untested branches in:
  _validate_regex_pattern, _list_and_filter_data_views, _stratified_sample,
  _check_memory_warning, _compute_distribution, _check_governance_thresholds,
  _audit_naming_conventions, _detect_stale_components.
"""

from __future__ import annotations

import logging
import re
from unittest.mock import Mock

import pandas as pd
import pytest

from cja_auto_sdr.core.exceptions import MemoryLimitExceeded
from cja_auto_sdr.org.analyzer import OrgComponentAnalyzer
from cja_auto_sdr.org.models import (
    ComponentDistribution,
    ComponentInfo,
    OrgReportConfig,
    SimilarityPair,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logger():
    return logging.getLogger("test_analyzer_coverage")


@pytest.fixture
def mock_cja():
    return Mock()


def _make_analyzer(
    mock_cja,
    logger,
    *,
    config: OrgReportConfig | None = None,
) -> OrgComponentAnalyzer:
    """Shorthand to create an analyzer with defaults suitable for unit tests."""
    cfg = config or OrgReportConfig(skip_lock=True, cja_per_thread=False)
    return OrgComponentAnalyzer(mock_cja, cfg, logger, org_id="unit@AdobeOrg")


def _make_component(
    comp_id: str,
    comp_type: str = "metric",
    name: str | None = None,
    data_views: set[str] | None = None,
) -> ComponentInfo:
    return ComponentInfo(
        component_id=comp_id,
        component_type=comp_type,
        name=name,
        data_views=data_views or {"dv1"},
    )


# ===================================================================
# 1. _validate_regex_pattern
# ===================================================================


class TestValidateRegexPattern:
    """Tests for _validate_regex_pattern ReDoS guard and compilation."""

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(?:a+b+c+)+",  # nested +
            r"(?:x*y*z*)*",  # nested *
            r".*.*.*",  # multiple .* in sequence
            r".+.+.+",  # multiple .+ in sequence
        ],
        ids=["nested_plus", "nested_star", "triple_dotstar", "triple_dotplus"],
    )
    def test_dangerous_redos_patterns_rejected(self, pattern, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        result = analyzer._validate_regex_pattern(pattern, "filter")
        assert result is None

    @pytest.mark.parametrize(
        "pattern",
        [
            r"[invalid",
            r"(?P<oops",
            r"*leading_quantifier",
        ],
        ids=["unclosed_bracket", "unclosed_group", "leading_quantifier"],
    )
    def test_invalid_regex_returns_none(self, pattern, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        result = analyzer._validate_regex_pattern(pattern, "exclude")
        assert result is None

    @pytest.mark.parametrize(
        "pattern",
        [
            r"^prod_.*",
            r"DV\s+\d+",
            r"(?i)staging",
            r"analytics|reporting",
        ],
        ids=["anchored_prefix", "dv_number", "case_insensitive_group", "alternation"],
    )
    def test_valid_patterns_compile_successfully(self, pattern, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        result = analyzer._validate_regex_pattern(pattern, "filter")
        assert isinstance(result, re.Pattern)

    def test_valid_pattern_is_case_insensitive(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        compiled = analyzer._validate_regex_pattern("prod", "filter")
        assert compiled is not None
        assert compiled.search("Production") is not None


# ===================================================================
# 2. _list_and_filter_data_views
# ===================================================================


class TestListAndFilterDataViews:
    """Tests for _list_and_filter_data_views edge cases."""

    def test_sample_size_below_one_raises(self, mock_cja, logger):
        config = OrgReportConfig(sample_size=0, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        with pytest.raises(ValueError, match="at least 1"):
            analyzer._list_and_filter_data_views()

    def test_sample_size_negative_raises(self, mock_cja, logger):
        config = OrgReportConfig(sample_size=-5, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        with pytest.raises(ValueError, match="at least 1"):
            analyzer._list_and_filter_data_views()

    def test_api_failure_returns_empty(self, mock_cja, logger):
        mock_cja.getDataViews.side_effect = RuntimeError("network error")
        analyzer = _make_analyzer(mock_cja, logger)
        data_views, is_sampled, total = analyzer._list_and_filter_data_views()
        assert data_views == []
        assert is_sampled is False
        assert total == 0

    def test_dataframe_result_converted_to_list(self, mock_cja, logger):
        df = pd.DataFrame(
            [
                {"id": "dv1", "name": "Alpha"},
                {"id": "dv2", "name": "Beta"},
            ]
        )
        mock_cja.getDataViews.return_value = df
        analyzer = _make_analyzer(mock_cja, logger)
        data_views, _, total = analyzer._list_and_filter_data_views()
        assert isinstance(data_views, list)
        assert len(data_views) == 2
        assert total == 2

    def test_combined_filter_and_exclude(self, mock_cja, logger):
        dvs = [
            {"id": "dv1", "name": "Prod Analytics"},
            {"id": "dv2", "name": "Prod Test"},
            {"id": "dv3", "name": "Dev Analytics"},
        ]
        mock_cja.getDataViews.return_value = dvs
        config = OrgReportConfig(
            filter_pattern="Prod",
            exclude_pattern="Test",
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        result, _, _total = analyzer._list_and_filter_data_views()
        assert len(result) == 1
        assert result[0]["name"] == "Prod Analytics"

    def test_none_return_treated_as_empty(self, mock_cja, logger):
        mock_cja.getDataViews.return_value = None
        analyzer = _make_analyzer(mock_cja, logger)
        data_views, _is_sampled, total = analyzer._list_and_filter_data_views()
        assert data_views == []
        assert total == 0

    def test_empty_list_return(self, mock_cja, logger):
        mock_cja.getDataViews.return_value = []
        analyzer = _make_analyzer(mock_cja, logger)
        data_views, _, total = analyzer._list_and_filter_data_views()
        assert data_views == []
        assert total == 0


# ===================================================================
# 3. _stratified_sample
# ===================================================================


class TestStratifiedSample:
    """Tests for _stratified_sample proportional allocation."""

    def _make_dvs(self, names: list[str]) -> list[dict]:
        return [{"id": f"dv_{i}", "name": n} for i, n in enumerate(names)]

    def test_proportional_allocation(self, mock_cja, logger):
        """Each prefix group should get proportional representation."""
        names = [f"GroupA Item {i}" for i in range(6)] + [f"GroupB Item {i}" for i in range(4)]
        dvs = self._make_dvs(names)
        config = OrgReportConfig(sample_size=5, sample_seed=42, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        result = analyzer._stratified_sample(dvs, 5)
        assert len(result) == 5

    def test_over_sampling_trimmed(self, mock_cja, logger):
        """When many single-item groups exist, over-sampling is trimmed down."""
        # 10 unique prefixes, each with 1 item -- proportional = max(1, ...) = 1 each = 10
        names = [f"Prefix{i} Item" for i in range(10)]
        dvs = self._make_dvs(names)
        config = OrgReportConfig(sample_size=5, sample_seed=0, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        result = analyzer._stratified_sample(dvs, 5)
        assert len(result) == 5

    def test_under_sampling_backfill(self, mock_cja, logger):
        """When proportional allocation yields fewer items, backfill kicks in."""
        # 2 groups with 50 items each -> proportional gives 2 * max(1, int(4*50/100))=2 each => 4
        # Needs to backfill 2 more to reach 6
        names = [f"Alpha Item {i}" for i in range(50)] + [f"Beta Item {i}" for i in range(50)]
        dvs = self._make_dvs(names)
        config = OrgReportConfig(sample_size=6, sample_seed=7, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        result = analyzer._stratified_sample(dvs, 6)
        assert len(result) == 6

    def test_single_item_groups_all_included(self, mock_cja, logger):
        """Groups smaller than their proportional allocation are fully included."""
        names = ["Solo Item"] + [f"Big Group {i}" for i in range(9)]
        dvs = self._make_dvs(names)
        config = OrgReportConfig(sample_size=8, sample_seed=1, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        result = analyzer._stratified_sample(dvs, 8)
        assert len(result) == 8
        # The solo item should always be included since its group has only 1 element
        solo_included = any(dv["name"] == "Solo Item" for dv in result)
        assert solo_included


# ===================================================================
# 4. _check_memory_warning
# ===================================================================


class TestCheckMemoryWarning:
    """Tests for _check_memory_warning thresholds and hard limit."""

    def _make_index(self, count: int = 5) -> dict[str, ComponentInfo]:
        return {f"c{i}": _make_component(f"c{i}", name=f"Component {i}") for i in range(count)}

    def test_below_threshold_no_warning(self, mock_cja, logger, caplog):
        config = OrgReportConfig(
            memory_warning_threshold_mb=100,
            memory_limit_mb=None,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(2)
        with caplog.at_level(logging.WARNING, logger="test_analyzer_coverage"):
            analyzer._check_memory_warning(index)
        assert "High memory usage" not in caplog.text

    def test_above_threshold_warning_logged(self, mock_cja, logger, caplog):
        config = OrgReportConfig(
            memory_warning_threshold_mb=1,  # very low threshold
            memory_limit_mb=None,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(5)
        # Fake the estimate to exceed threshold
        analyzer._estimate_component_index_memory = Mock(return_value=50.0)
        with caplog.at_level(logging.WARNING):
            analyzer._check_memory_warning(index)
        assert "High memory usage" in caplog.text

    def test_threshold_disabled_with_none(self, mock_cja, logger, caplog):
        config = OrgReportConfig(
            memory_warning_threshold_mb=None,
            memory_limit_mb=None,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(5)
        analyzer._estimate_component_index_memory = Mock(return_value=200.0)
        with caplog.at_level(logging.WARNING):
            analyzer._check_memory_warning(index)
        assert "High memory usage" not in caplog.text

    def test_threshold_disabled_with_zero(self, mock_cja, logger, caplog):
        config = OrgReportConfig(
            memory_warning_threshold_mb=0,
            memory_limit_mb=None,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(5)
        analyzer._estimate_component_index_memory = Mock(return_value=200.0)
        with caplog.at_level(logging.WARNING):
            analyzer._check_memory_warning(index)
        assert "High memory usage" not in caplog.text

    def test_hard_limit_exceeded_raises(self, mock_cja, logger):
        config = OrgReportConfig(
            memory_warning_threshold_mb=100,
            memory_limit_mb=10,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(5)
        analyzer._estimate_component_index_memory = Mock(return_value=50.0)
        with pytest.raises(MemoryLimitExceeded):
            analyzer._check_memory_warning(index)

    def test_hard_limit_none_does_not_raise(self, mock_cja, logger):
        config = OrgReportConfig(
            memory_warning_threshold_mb=100,
            memory_limit_mb=None,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(2)
        analyzer._estimate_component_index_memory = Mock(return_value=5.0)
        # Should not raise
        analyzer._check_memory_warning(index)

    def test_hard_limit_zero_does_not_raise(self, mock_cja, logger):
        """memory_limit_mb=0 should be treated as disabled."""
        config = OrgReportConfig(
            memory_warning_threshold_mb=100,
            memory_limit_mb=0,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = self._make_index(2)
        analyzer._estimate_component_index_memory = Mock(return_value=200.0)
        # Should not raise because limit is 0 (disabled)
        analyzer._check_memory_warning(index)


# ===================================================================
# 5. _compute_distribution
# ===================================================================


class TestComputeDistribution:
    """Tests for _compute_distribution bucket classification."""

    def test_zero_dvs_returns_empty(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "m1": _make_component("m1", data_views={"dv1"}),
        }
        result = analyzer._compute_distribution(index, 0)
        assert isinstance(result, ComponentDistribution)
        assert result.total_core == 0
        assert result.total_isolated == 0

    def test_core_min_count_override(self, mock_cja, logger):
        config = OrgReportConfig(core_min_count=2, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = {
            "m1": _make_component("m1", data_views={"dv1", "dv2", "dv3"}),
            "m2": _make_component("m2", data_views={"dv1"}),
        }
        result = analyzer._compute_distribution(index, 3)
        assert "m1" in result.core_metrics
        assert "m2" in result.isolated_metrics

    def test_core_min_count_one_does_not_misclassify_isolated(self, mock_cja, logger):
        """When core_min_count=1, everything should be core regardless of presence."""
        config = OrgReportConfig(core_min_count=1, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = {
            "m1": _make_component("m1", data_views={"dv1"}),
            "d1": _make_component("d1", comp_type="dimension", data_views={"dv1"}),
        }
        result = analyzer._compute_distribution(index, 5)
        assert "m1" in result.core_metrics
        assert "d1" in result.core_dimensions
        assert result.total_isolated == 0

    def test_boundary_ceil_vs_floor(self, mock_cja, logger):
        """With 3 DVs and 50% threshold: ceil(1.5)=2, so 2+ is core, 1 is isolated."""
        config = OrgReportConfig(core_threshold=0.5, skip_lock=True, cja_per_thread=False)
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        index = {
            "core_m": _make_component("core_m", data_views={"dv1", "dv2"}),
            "iso_m": _make_component("iso_m", data_views={"dv1"}),
        }
        result = analyzer._compute_distribution(index, 3)
        assert "core_m" in result.core_metrics
        assert "iso_m" in result.isolated_metrics

    def test_isolated_equals_one_exactly(self, mock_cja, logger):
        """Component in exactly 1 DV should land in isolated bucket."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "iso_d": _make_component("iso_d", comp_type="dimension", data_views={"dv1"}),
        }
        result = analyzer._compute_distribution(index, 10)
        assert "iso_d" in result.isolated_dimensions

    def test_limited_bucket_classification(self, mock_cja, logger):
        """Component in 2 DVs with 10 total DVs (< 25% = 3 for common) => limited."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "lim_m": _make_component("lim_m", data_views={"dv1", "dv2"}),
        }
        result = analyzer._compute_distribution(index, 10)
        assert "lim_m" in result.limited_metrics

    def test_common_bucket_classification(self, mock_cja, logger):
        """Component in 3 DVs with 10 total DVs (25%=3 for common, core at 50%=5) => common."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "com_d": _make_component("com_d", comp_type="dimension", data_views={"dv1", "dv2", "dv3"}),
        }
        result = analyzer._compute_distribution(index, 10)
        assert "com_d" in result.common_dimensions


# ===================================================================
# 6. _check_governance_thresholds
# ===================================================================


class TestCheckGovernanceThresholds:
    """Tests for _check_governance_thresholds violation detection."""

    def _sim_pair(self, similarity: float) -> SimilarityPair:
        return SimilarityPair(
            dv1_id="dv1",
            dv1_name="DV 1",
            dv2_id="dv2",
            dv2_name="DV 2",
            jaccard_similarity=similarity,
            shared_count=10,
            union_count=11,
        )

    def test_no_violations(self, mock_cja, logger):
        config = OrgReportConfig(
            duplicate_threshold=5,
            isolated_threshold=0.5,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        dist = ComponentDistribution()
        violations, exceeded = analyzer._check_governance_thresholds([], dist, 100)
        assert violations == []
        assert exceeded is False

    def test_duplicate_threshold_exceeded(self, mock_cja, logger):
        config = OrgReportConfig(
            duplicate_threshold=0,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        pairs = [self._sim_pair(0.95)]
        dist = ComponentDistribution()
        violations, exceeded = analyzer._check_governance_thresholds(pairs, dist, 10)
        assert exceeded is True
        assert any(v["type"] == "duplicate_threshold_exceeded" for v in violations)

    def test_isolated_threshold_exceeded(self, mock_cja, logger):
        config = OrgReportConfig(
            isolated_threshold=0.1,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        dist = ComponentDistribution(
            isolated_metrics=["m1", "m2", "m3"],
            isolated_dimensions=["d1", "d2"],
        )
        # 5 isolated / 10 total = 50% > 10%
        violations, exceeded = analyzer._check_governance_thresholds(None, dist, 10)
        assert exceeded is True
        assert any(v["type"] == "isolated_threshold_exceeded" for v in violations)

    def test_both_thresholds_exceeded(self, mock_cja, logger):
        config = OrgReportConfig(
            duplicate_threshold=0,
            isolated_threshold=0.1,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        pairs = [self._sim_pair(0.95)]
        dist = ComponentDistribution(isolated_metrics=["m1", "m2", "m3"])
        violations, exceeded = analyzer._check_governance_thresholds(pairs, dist, 5)
        assert exceeded is True
        types = [v["type"] for v in violations]
        assert "duplicate_threshold_exceeded" in types
        assert "isolated_threshold_exceeded" in types

    def test_no_similarity_pairs_passed(self, mock_cja, logger):
        """When similarity_pairs is None, duplicate check should be skipped gracefully."""
        config = OrgReportConfig(
            duplicate_threshold=0,
            isolated_threshold=None,
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        dist = ComponentDistribution()
        violations, exceeded = analyzer._check_governance_thresholds(None, dist, 10)
        assert violations == []
        assert exceeded is False

    def test_isolated_threshold_not_exceeded(self, mock_cja, logger):
        config = OrgReportConfig(
            isolated_threshold=0.9,  # very permissive
            skip_lock=True,
            cja_per_thread=False,
        )
        analyzer = _make_analyzer(mock_cja, logger, config=config)
        dist = ComponentDistribution(isolated_metrics=["m1"])
        _violations, exceeded = analyzer._check_governance_thresholds(None, dist, 100)
        assert exceeded is False


# ===================================================================
# 7. _audit_naming_conventions
# ===================================================================


class TestAuditNamingConventions:
    """Tests for _audit_naming_conventions detection logic."""

    def test_snake_case_detected(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "my_metric": _make_component("my_metric", name="page_view_count"),
        }
        audit = analyzer._audit_naming_conventions(index)
        assert audit["case_styles"]["snake_case"] >= 1

    def test_camel_case_detected(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "myMetric": _make_component("myMetric", name="pageViewCount"),
        }
        audit = analyzer._audit_naming_conventions(index)
        assert audit["case_styles"]["camelCase"] >= 1

    def test_pascal_case_detected(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "MyMetric": _make_component("MyMetric", name="PageViewCount"),
        }
        audit = analyzer._audit_naming_conventions(index)
        assert audit["case_styles"]["PascalCase"] >= 1

    def test_stale_keyword_word_boundary(self, mock_cja, logger):
        """Stale keywords must match at word boundaries, not inside words."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            # Should match - stale keyword at word boundary
            "test_metric": _make_component("test_metric", name="test_metric"),
            # Should NOT match - 'test' is inside the word 'attestation'
            "attestation": _make_component("attestation", name="attestation"),
        }
        audit = analyzer._audit_naming_conventions(index)
        stale_ids = [s["component_id"] for s in audit["stale_patterns"]]
        assert "test_metric" in stale_ids
        assert "attestation" not in stale_ids

    def test_version_suffix_at_end_only(self, mock_cja, logger):
        """Version suffix _v1 should only match at end of string."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "metric_v2": _make_component("metric_v2", name="metric_v2"),
            "v2_metric": _make_component("v2_metric", name="v2_metric"),
        }
        audit = analyzer._audit_naming_conventions(index)
        stale_ids = [s["component_id"] for s in audit["stale_patterns"]]
        assert "metric_v2" in stale_ids
        # v2_metric should NOT match the version suffix pattern (not at end)
        version_suffix_matches = [
            s for s in audit["stale_patterns"] if s["component_id"] == "v2_metric" and s["pattern"] == "version_suffix"
        ]
        assert len(version_suffix_matches) == 0

    def test_prefix_grouping(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "evar/one": _make_component("evar/one", name="evar/one"),
            "evar/two": _make_component("evar/two", name="evar/two"),
            "prop_click": _make_component("prop_click", name="prop_click"),
        }
        audit = analyzer._audit_naming_conventions(index)
        assert "evar" in audit["prefix_groups"]
        assert audit["prefix_groups"]["evar"] == 2
        assert "prop" in audit["prefix_groups"]

    def test_naming_inconsistency_recommendation(self, mock_cja, logger):
        """When >5 non-dominant style components exist, a recommendation is generated."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {}
        # 10 snake_case
        for i in range(10):
            index[f"snake_{i}"] = _make_component(f"snake_{i}", name=f"page_view_{i}")
        # 6 camelCase (enough to trigger recommendation)
        for i in range(6):
            index[f"camel{i}"] = _make_component(f"camel{i}", name=f"pageView{chr(65 + i)}")
        audit = analyzer._audit_naming_conventions(index)
        rec_types = [r["type"] for r in audit["recommendations"]]
        assert "naming_inconsistency" in rec_types

    def test_stale_patterns_recommendation_generated(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "old_metric": _make_component("old_metric", name="old_metric"),
        }
        audit = analyzer._audit_naming_conventions(index)
        rec_types = [r["type"] for r in audit["recommendations"]]
        assert "stale_naming_patterns" in rec_types

    def test_component_without_name_uses_id(self, mock_cja, logger):
        """When name is None, the component ID is used for analysis."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "temp_data": _make_component("temp_data", name=None),
        }
        audit = analyzer._audit_naming_conventions(index)
        stale_ids = [s["component_id"] for s in audit["stale_patterns"]]
        assert "temp_data" in stale_ids


# ===================================================================
# 8. _detect_stale_components
# ===================================================================


class TestDetectStaleComponents:
    """Tests for _detect_stale_components heuristic matching."""

    @pytest.mark.parametrize(
        ("name", "expected_pattern"),
        [
            ("test_metric", "stale_keyword"),
            ("old_revenue", "stale_keyword"),
            ("temp_session", "stale_keyword"),
            ("tmp-count", "stale_keyword"),
            ("backup_data", "stale_keyword"),
            ("copy_of_metric", "stale_keyword"),
            ("deprecated_field", "stale_keyword"),
            ("legacy_evar", "stale_keyword"),
            ("archive-dim", "stale_keyword"),
            ("obsolete_prop", "stale_keyword"),
            ("unused_segment", "stale_keyword"),
        ],
        ids=[
            "test",
            "old",
            "temp",
            "tmp",
            "backup",
            "copy",
            "deprecated",
            "legacy",
            "archive",
            "obsolete",
            "unused",
        ],
    )
    def test_stale_keywords_detected(self, name, expected_pattern, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {name: _make_component(name, name=name)}
        result = analyzer._detect_stale_components(index)
        assert len(result) == 1
        assert result[0]["pattern"] == expected_pattern

    def test_version_suffix_detected(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "metric_v3": _make_component("metric_v3", name="metric_v3"),
            "metric-V12": _make_component("metric-V12", name="metric-V12"),
        }
        result = analyzer._detect_stale_components(index)
        patterns = {r["component_id"]: r["pattern"] for r in result}
        assert patterns.get("metric_v3") == "version_suffix"
        assert patterns.get("metric-V12") == "version_suffix"

    def test_date_pattern_yyyymmdd(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "metric_20240115": _make_component("metric_20240115", name="metric_20240115"),
        }
        result = analyzer._detect_stale_components(index)
        assert len(result) == 1
        assert result[0]["pattern"] == "date_pattern"

    def test_date_pattern_yyyy_mm_dd(self, mock_cja, logger):
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "metric_2024-01-15": _make_component("metric_2024-01-15", name="metric_2024-01-15"),
        }
        result = analyzer._detect_stale_components(index)
        assert len(result) == 1
        assert result[0]["pattern"] == "date_pattern"

    def test_active_components_excluded(self, mock_cja, logger):
        """Normal component names should not be flagged."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "page_views": _make_component("page_views", name="page_views"),
            "revenue_total": _make_component("revenue_total", name="revenue_total"),
            "conversion_rate": _make_component("conversion_rate", name="conversion_rate"),
        }
        result = analyzer._detect_stale_components(index)
        assert len(result) == 0

    def test_stale_component_includes_metadata(self, mock_cja, logger):
        """Stale entries should include component_id, name, type, pattern, presence_count, data_views."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "old_field": _make_component(
                "old_field",
                comp_type="dimension",
                name="old_field",
                data_views={"dv1", "dv2"},
            ),
        }
        result = analyzer._detect_stale_components(index)
        assert len(result) == 1
        entry = result[0]
        assert entry["component_id"] == "old_field"
        assert entry["name"] == "old_field"
        assert entry["type"] == "dimension"
        assert entry["pattern"] == "stale_keyword"
        assert entry["presence_count"] == 2
        assert len(entry["data_views"]) <= 5

    def test_component_without_name_uses_id(self, mock_cja, logger):
        """When info.name is None, the component_id is used for detection."""
        analyzer = _make_analyzer(mock_cja, logger)
        index = {
            "temp_data": _make_component("temp_data", name=None),
        }
        result = analyzer._detect_stale_components(index)
        assert len(result) == 1
        assert result[0]["name"] == "temp_data"

    def test_data_views_capped_at_five(self, mock_cja, logger):
        """data_views list should be capped at 5 entries max."""
        analyzer = _make_analyzer(mock_cja, logger)
        many_dvs = {f"dv{i}" for i in range(10)}
        index = {
            "test_metric": _make_component("test_metric", name="test_metric", data_views=many_dvs),
        }
        result = analyzer._detect_stale_components(index)
        assert len(result) == 1
        assert len(result[0]["data_views"]) <= 5
