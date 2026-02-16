"""Coverage-focused tests for interactive and org-report console helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cja_auto_sdr import generator
from cja_auto_sdr.org.models import (
    ComponentDistribution,
    ComponentInfo,
    DataViewCluster,
    DataViewSummary,
    OrgReportComparison,
    OrgReportConfig,
    OrgReportResult,
    SimilarityPair,
)


def _mock_data_views() -> list[dict[str, object]]:
    return [
        {"id": "dv_001", "name": "Marketing DV", "owner": {"name": "Alice"}},
        {"id": "dv_002", "name": "Product DV", "owner": {"name": "Bob"}},
        {"id": "dv_003", "name": "Finance DV", "owner": {"name": "Carol"}},
    ]


def _build_rich_result() -> OrgReportResult:
    config = OrgReportConfig(
        core_threshold=0.5,
        include_metadata=True,
        overlap_threshold=0.95,
        summary_only=False,
        include_component_types=True,
        include_drift=True,
    )

    summaries = [
        DataViewSummary(
            data_view_id="dv_001",
            data_view_name="Primary Business Data View With A Long Name",
            metric_ids={"metric/core/1", "metric/common/1", "metric/limited/1"},
            dimension_ids={"dimension/core/1", "dimension/common/1"},
            metric_count=3,
            dimension_count=2,
            standard_metric_count=2,
            derived_metric_count=1,
            standard_dimension_count=1,
            derived_dimension_count=1,
            owner="Alice",
            owner_id="owner_1",
            created="2026-01-01T00:00:00+00:00",
            modified="2026-02-15T10:00:00+00:00",
            has_description=True,
        ),
        DataViewSummary(
            data_view_id="dv_002",
            data_view_name="Secondary Data View",
            metric_count=0,
            dimension_count=0,
            error="permission denied",
            status="error",
        ),
        DataViewSummary(
            data_view_id="dv_003",
            data_view_name="Tertiary Data View",
            metric_ids={"metric/core/2", "metric/isolated/1"},
            dimension_ids={"dimension/isolated/1"},
            metric_count=2,
            dimension_count=1,
            standard_metric_count=1,
            derived_metric_count=1,
            standard_dimension_count=1,
            derived_dimension_count=0,
            owner="Carol",
            owner_id="owner_3",
            created="2026-01-03T00:00:00+00:00",
            modified="2026-02-15T11:00:00+00:00",
        ),
    ]

    distribution = ComponentDistribution(
        core_metrics=["metric/core/1", "metric/core/2"],
        core_dimensions=["dimension/core/1"],
        common_metrics=["metric/common/1"],
        common_dimensions=["dimension/common/1"],
        limited_metrics=["metric/limited/1"],
        limited_dimensions=["dimension/limited/1"],
        isolated_metrics=["metric/isolated/1"],
        isolated_dimensions=["dimension/isolated/1"],
    )

    component_index = {
        "metric/core/1": ComponentInfo(
            component_id="metric/core/1",
            component_type="metric",
            name="Core Metric One",
            data_views={"dv_001", "dv_003"},
        ),
        "metric/core/2": ComponentInfo(
            component_id="metric/core/2",
            component_type="metric",
            name="Core Metric Two",
            data_views={"dv_001", "dv_003"},
        ),
        "metric/common/1": ComponentInfo(
            component_id="metric/common/1",
            component_type="metric",
            name="Common Metric",
            data_views={"dv_001", "dv_002"},
        ),
        "metric/limited/1": ComponentInfo(
            component_id="metric/limited/1",
            component_type="metric",
            name="Limited Metric",
            data_views={"dv_001", "dv_002"},
        ),
        "metric/isolated/1": ComponentInfo(
            component_id="metric/isolated/1",
            component_type="metric",
            name="Isolated Metric",
            data_views={"dv_003"},
        ),
        "dimension/core/1": ComponentInfo(
            component_id="dimension/core/1",
            component_type="dimension",
            name="Core Dimension",
            data_views={"dv_001", "dv_003"},
        ),
        "dimension/common/1": ComponentInfo(
            component_id="dimension/common/1",
            component_type="dimension",
            name="Common Dimension",
            data_views={"dv_001", "dv_002"},
        ),
        "dimension/limited/1": ComponentInfo(
            component_id="dimension/limited/1",
            component_type="dimension",
            name="Limited Dimension",
            data_views={"dv_001", "dv_002"},
        ),
        "dimension/isolated/1": ComponentInfo(
            component_id="dimension/isolated/1",
            component_type="dimension",
            name="Isolated Dimension",
            data_views={"dv_003"},
        ),
    }

    similarity_pairs = [
        SimilarityPair(
            dv1_id="dv_001",
            dv1_name="Primary Business Data View With A Very Long Name",
            dv2_id="dv_003",
            dv2_name="Tertiary Data View Also Quite Long",
            jaccard_similarity=0.93,
            shared_count=15,
            union_count=18,
            only_in_dv1=["metric/limited/1", "dimension/common/1", "metric/common/1", "metric/x"],
            only_in_dv2=["metric/isolated/1", "dimension/isolated/1", "dimension/y", "metric/z"],
            only_in_dv1_names={
                "metric/limited/1": "Limited Metric",
                "dimension/common/1": "Common Dimension",
            },
            only_in_dv2_names={
                "metric/isolated/1": "Isolated Metric",
                "dimension/isolated/1": "Isolated Dimension",
            },
        )
    ]
    similarity_pairs.extend(
        [
            SimilarityPair(
                dv1_id=f"dv_{i:03d}",
                dv1_name=f"Data View {i}",
                dv2_id=f"dv_{i + 1:03d}",
                dv2_name=f"Data View {i + 1}",
                jaccard_similarity=0.90,
                shared_count=20,
                union_count=22,
            )
            for i in range(10, 31)
        ]
    )

    clusters = [
        DataViewCluster(
            cluster_id=i,
            cluster_name=f"Cluster {i}",
            data_view_ids=[f"dv_{i:03d}", f"dv_{i + 100:03d}", f"dv_{i + 200:03d}", f"dv_{i + 300:03d}"],
            data_view_names=[
                f"Data View {i}",
                f"Data View {i + 100}",
                f"Data View {i + 200}",
                f"Data View {i + 300}",
            ],
            cohesion_score=0.78,
        )
        for i in range(1, 12)
    ]

    owner_data = {
        f"Owner {i}": {
            "data_view_count": i,
            "total_metrics": i * 10,
            "total_dimensions": i * 5,
            "avg_components_per_dv": float(i * 3),
        }
        for i in range(1, 18)
    }

    stale_components = [
        {"pattern": "deprecated_prefix", "name": f"old_metric_{i}", "component_id": f"metric/old/{i}"}
        for i in range(6)
    ]
    stale_components.extend(
        [{"pattern": "legacy_suffix", "name": f"segment_{i}_old", "component_id": f"segment/old/{i}"} for i in range(3)]
    )

    recommendations = [
        {
            "severity": "high",
            "reason": "A data view has many isolated components",
            "data_view": "dv_001",
            "data_view_name": "Primary Business Data View",
        },
        {
            "severity": "medium",
            "reason": "Two data views are highly similar",
            "data_view_1": "dv_001",
            "data_view_1_name": "Primary Business Data View",
            "data_view_2": "dv_003",
            "data_view_2_name": "Tertiary Data View",
        },
    ]

    return OrgReportResult(
        timestamp="2026-02-16T12:00:00+00:00",
        org_id="test_org@AdobeOrg",
        parameters=config,
        data_view_summaries=summaries,
        component_index=component_index,
        distribution=distribution,
        similarity_pairs=similarity_pairs,
        recommendations=recommendations,
        duration=12.34,
        clusters=clusters,
        is_sampled=True,
        total_available_data_views=25,
        governance_violations=[
            {"message": "Duplicate threshold exceeded", "threshold": 5, "actual": 11},
        ],
        naming_audit={
            "case_styles": {"snake_case": 9, "camelCase": 4, "UPPER_CASE": 2},
            "total_components": 15,
            "recommendations": [{"severity": "medium", "message": "Prefer snake_case for new components"}],
        },
        owner_summary={
            "by_owner": owner_data,
            "owners_sorted_by_dv_count": list(owner_data.keys()),
        },
        stale_components=stale_components,
    )


@patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", None))
@patch("cja_auto_sdr.generator.cjapy")
def test_interactive_select_dataviews_retries_and_parses_ranges(mock_cjapy: Mock, _mock_config: Mock):
    mock_cja = Mock()
    mock_cja.getDataViews.return_value = _mock_data_views()
    mock_cjapy.CJA.return_value = mock_cja

    with patch("builtins.input", side_effect=["", "1-a", "10", "1,2-3"]):
        selected = generator.interactive_select_dataviews(profile="dev")

    assert selected == ["dv_001", "dv_002", "dv_003"]


@patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", None))
@patch("cja_auto_sdr.generator.cjapy")
def test_interactive_select_dataviews_cancelled(mock_cjapy: Mock, _mock_config: Mock):
    mock_cja = Mock()
    mock_cja.getDataViews.return_value = _mock_data_views()
    mock_cjapy.CJA.return_value = mock_cja

    with patch("builtins.input", side_effect=["q"]):
        assert generator.interactive_select_dataviews(profile="dev") == []


@patch("cja_auto_sdr.generator.configure_cjapy", return_value=(False, "bad credentials", None))
def test_interactive_select_dataviews_config_failure(_mock_config: Mock):
    assert generator.interactive_select_dataviews(config_file="bad.json") == []


@patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", None))
@patch("cja_auto_sdr.generator.cjapy")
def test_interactive_wizard_full_flow_with_retries(mock_cjapy: Mock, _mock_config: Mock):
    mock_cja = Mock()
    mock_cja.getDataViews.return_value = _mock_data_views()
    mock_cjapy.CJA.return_value = mock_cja

    user_inputs = [
        "",
        "2-1,5",
        "1,2",
        "99",
        "",
        "maybe",
        "y",
        "n",
        "",
        "",
    ]
    with patch("builtins.input", side_effect=user_inputs):
        result = generator.interactive_wizard(profile="dev")

    assert result is not None
    assert result.data_view_ids == ["dv_001", "dv_002"]
    assert result.output_format == "excel"
    assert result.include_segments is True
    assert result.include_calculated is False
    assert result.include_derived is False


@patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", None))
@patch("cja_auto_sdr.generator.cjapy")
def test_interactive_wizard_cancelled_at_confirmation(mock_cjapy: Mock, _mock_config: Mock):
    mock_cja = Mock()
    mock_cja.getDataViews.return_value = _mock_data_views()
    mock_cjapy.CJA.return_value = mock_cja

    user_inputs = ["1", "1", "n", "n", "n", "n"]
    with patch("builtins.input", side_effect=user_inputs):
        assert generator.interactive_wizard(profile="dev") is None


@patch("cja_auto_sdr.generator.configure_cjapy", return_value=(False, "bad credentials", None))
def test_interactive_wizard_config_failure(_mock_config: Mock):
    assert generator.interactive_wizard(config_file="bad.json") is None


def test_write_org_report_console_renders_all_major_sections(capsys):
    result = _build_rich_result()
    generator.write_org_report_console(result, result.parameters, quiet=False)
    output = capsys.readouterr().out

    assert "ORG-WIDE COMPONENT ANALYSIS REPORT" in output
    assert "[SAMPLED]" in output
    assert "DATA VIEWS" in output
    assert "COMPONENT SUMMARY" in output
    assert "HIGH OVERLAP PAIRS" in output
    assert "DATA VIEW CLUSTERS" in output
    assert "GOVERNANCE VIOLATIONS" in output
    assert "OWNER SUMMARY" in output
    assert "NAMING AUDIT" in output
    assert "STALE COMPONENTS" in output
    assert "RECOMMENDATIONS" in output


def test_write_org_report_stats_and_comparison_consoles(capsys):
    result = _build_rich_result()
    generator.write_org_report_stats_only(result, quiet=False)

    comparison = OrgReportComparison(
        current_timestamp="2026-02-16T12:00:00+00:00",
        previous_timestamp="2026-02-01T12:00:00+00:00",
        data_views_added=["dv_new_1", "dv_new_2", "dv_new_3", "dv_new_4", "dv_new_5", "dv_new_6"],
        data_views_removed=["dv_old_1", "dv_old_2", "dv_old_3", "dv_old_4", "dv_old_5", "dv_old_6"],
        data_views_added_names=["New 1", "New 2", "New 3", "New 4", "New 5", "New 6"],
        data_views_removed_names=["Old 1", "Old 2", "Old 3", "Old 4", "Old 5", "Old 6"],
        new_high_similarity_pairs=[{"dv1_id": "dv_a", "dv2_id": "dv_b"}],
        resolved_pairs=[{"dv1_id": "dv_x", "dv2_id": "dv_y"}],
        summary={"data_views_delta": 5, "components_delta": -3, "core_delta": 2, "isolated_delta": 0},
    )
    generator.write_org_report_comparison_console(comparison, quiet=False)
    output = capsys.readouterr().out

    assert "ORG STATS" in output
    assert "ORG REPORT COMPARISON (TRENDING)" in output
    assert "Data Views Added" in output
    assert "Resolved High-Similarity Pairs" in output


def test_write_org_report_file_outputs_all_formats(tmp_path: Path):
    result = _build_rich_result()
    logger = logging.getLogger("test_org_report_outputs")

    json_path = generator.write_org_report_json(result, tmp_path / "org_report", str(tmp_path), logger)
    excel_path = generator.write_org_report_excel(result, tmp_path / "org_report", str(tmp_path), logger)
    markdown_path = generator.write_org_report_markdown(result, tmp_path / "org_report", str(tmp_path), logger)
    html_path = generator.write_org_report_html(result, tmp_path / "org_report", str(tmp_path), logger)
    csv_dir = generator.write_org_report_csv(result, tmp_path / "org_report.csv", str(tmp_path), logger)

    assert Path(json_path).exists()
    assert Path(excel_path).exists()
    assert Path(markdown_path).exists()
    assert Path(html_path).exists()
    assert Path(csv_dir).is_dir()
    assert (Path(csv_dir) / "org_report_summary.csv").exists()
    assert (Path(csv_dir) / "org_report_data_views.csv").exists()
    assert (Path(csv_dir) / "org_report_components.csv").exists()
    assert (Path(csv_dir) / "org_report_distribution.csv").exists()
    assert (Path(csv_dir) / "org_report_similarity.csv").exists()
    assert (Path(csv_dir) / "org_report_recommendations.csv").exists()


def test_org_report_branches_for_core_min_count_and_quiet_modes(capsys):
    result = _build_rich_result()
    result.parameters.core_min_count = 2
    result.is_sampled = False

    comparison = OrgReportComparison(
        current_timestamp="2026-02-16T12:00:00+00:00",
        previous_timestamp="2026-02-15T12:00:00+00:00",
        summary={"data_views_delta": 0, "components_delta": 0, "core_delta": 0, "isolated_delta": 0},
    )

    generator.write_org_report_console(result, result.parameters, quiet=False)
    out = capsys.readouterr().out
    assert "Core (in >=2 DVs)" in out
    assert "Data Views Analyzed: 2 / 3" in out

    generator.write_org_report_stats_only(result, quiet=True)
    generator.write_org_report_comparison_console(comparison, quiet=True)


def test_run_org_report_all_formats_branch(tmp_path: Path):
    result = _build_rich_result()
    result.thresholds_exceeded = True
    result.governance_violations = [{"message": "threshold exceeded"}]
    org_config = OrgReportConfig(fail_on_threshold=True)

    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
        patch("cja_auto_sdr.generator.write_org_report_console") as mock_console,
        patch("cja_auto_sdr.generator.write_org_report_json", return_value=str(tmp_path / "report.json")) as mock_json,
        patch(
            "cja_auto_sdr.generator.write_org_report_excel", return_value=str(tmp_path / "report.xlsx")
        ) as mock_excel,
        patch(
            "cja_auto_sdr.generator.write_org_report_markdown", return_value=str(tmp_path / "report.md")
        ) as mock_markdown,
        patch("cja_auto_sdr.generator.write_org_report_html", return_value=str(tmp_path / "report.html")) as mock_html,
        patch("cja_auto_sdr.generator.write_org_report_csv", return_value=str(tmp_path / "report_csv")) as mock_csv,
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary") as mock_append_summary,
    ):
        mock_cjapy.CJA.return_value = Mock()
        mock_analyzer = Mock()
        mock_analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = mock_analyzer

        success, thresholds_exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="all",
            output_path=str(tmp_path / "org_output"),
            output_dir=str(tmp_path),
            org_config=org_config,
            profile="dev",
            quiet=False,
        )

    assert success is True
    assert thresholds_exceeded is True
    mock_console.assert_called_once()
    mock_json.assert_called_once()
    mock_excel.assert_called_once()
    mock_markdown.assert_called_once()
    mock_html.assert_called_once()
    mock_csv.assert_called_once()
    mock_append_summary.assert_called_once()


def test_run_org_report_stats_only_and_csv_stdout_guard(tmp_path: Path):
    result = _build_rich_result()

    stats_config = OrgReportConfig(org_stats_only=True)
    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
        patch("cja_auto_sdr.generator.write_org_report_stats_only") as mock_stats_only,
        patch("cja_auto_sdr.generator.write_org_report_json", return_value=str(tmp_path / "stats.json")) as mock_json,
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary"),
    ):
        mock_cjapy.CJA.return_value = Mock()
        mock_analyzer = Mock()
        mock_analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = mock_analyzer

        success, thresholds_exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="json",
            output_path=str(tmp_path / "org_stats"),
            output_dir=str(tmp_path),
            org_config=stats_config,
            profile=None,
            quiet=False,
        )

    assert success is True
    assert thresholds_exceeded is False
    mock_stats_only.assert_called_once()
    mock_json.assert_called_once()

    csv_config = OrgReportConfig()
    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary"),
    ):
        mock_cjapy.CJA.return_value = Mock()
        mock_analyzer = Mock()
        mock_analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = mock_analyzer

        success, thresholds_exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="csv",
            output_path="stdout",
            output_dir=str(tmp_path),
            org_config=csv_config,
            quiet=False,
        )

    assert success is False
    assert thresholds_exceeded is False


def test_run_org_report_reports_alias_and_individual_format_branches(tmp_path: Path):
    result = _build_rich_result()
    base_output = tmp_path / "org_report_out"
    common_patches = (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy"),
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer"),
        patch("cja_auto_sdr.generator.write_org_report_console", return_value=None),
        patch("cja_auto_sdr.generator.write_org_report_json", return_value=str(tmp_path / "report.json")),
        patch("cja_auto_sdr.generator.write_org_report_excel", return_value=str(tmp_path / "report.xlsx")),
        patch("cja_auto_sdr.generator.write_org_report_markdown", return_value=str(tmp_path / "report.md")),
        patch("cja_auto_sdr.generator.write_org_report_html", return_value=str(tmp_path / "report.html")),
        patch("cja_auto_sdr.generator.write_org_report_csv", return_value=str(tmp_path / "report_csv")),
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary"),
    )

    with (
        common_patches[0] as _cfg,
        common_patches[1] as mock_cjapy,
        common_patches[2] as mock_analyzer_cls,
        common_patches[3],
        common_patches[4],
        common_patches[5],
        common_patches[6],
        common_patches[7],
        common_patches[8],
        common_patches[9],
        common_patches[10],
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = analyzer

        # Alias branch
        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="reports",
            output_path=str(base_output),
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
        assert ok is True
        assert exceeded is False

        # Single-format branches
        for fmt in ("console", "json", "excel", "markdown", "html", "csv"):
            ok, exceeded = generator.run_org_report(
                config_file="config.json",
                output_format=fmt,
                output_path=str(base_output),
                output_dir=str(tmp_path),
                org_config=OrgReportConfig(),
                quiet=False,
            )
            assert ok is True
            assert exceeded is False


def test_run_org_report_cache_and_comparison_error_paths(tmp_path: Path):
    result = _build_rich_result()
    config = OrgReportConfig(use_cache=True, compare_org_report="missing_previous.json")

    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgReportCache") as mock_cache_cls,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
        patch("cja_auto_sdr.generator.compare_org_reports", side_effect=FileNotFoundError),
        patch("cja_auto_sdr.generator.write_org_report_console"),
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary"),
    ):
        mock_cjapy.CJA.return_value = Mock()
        cache = Mock()
        cache.get_stats.return_value = {"entries": 3}
        mock_cache_cls.return_value = cache

        analyzer = Mock()
        analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = analyzer

        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="console",
            output_path=None,
            output_dir=str(tmp_path),
            org_config=config,
            quiet=False,
        )

    assert ok is True
    assert exceeded is False


def test_run_org_report_failure_paths(tmp_path: Path):
    with patch("cja_auto_sdr.generator.configure_cjapy", return_value=(False, "bad credentials", None)):
        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="console",
            output_path=None,
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
    assert ok is False
    assert exceeded is False

    empty_result = _build_rich_result()
    empty_result.data_view_summaries = []
    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.return_value = empty_result
        mock_analyzer_cls.return_value = analyzer

        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="console",
            output_path=None,
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
    assert ok is False
    assert exceeded is False

    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.side_effect = RuntimeError("analysis failed")
        mock_analyzer_cls.return_value = analyzer

        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="console",
            output_path=None,
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
    assert ok is False
    assert exceeded is False


def test_run_org_report_org_stats_json_stdout_branch(tmp_path: Path, capsys):
    result = _build_rich_result()
    config = OrgReportConfig(org_stats_only=True)

    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
        patch("cja_auto_sdr.generator.write_org_report_stats_only"),
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary"),
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = analyzer

        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="json",
            output_path="stdout",
            output_dir=str(tmp_path),
            org_config=config,
            quiet=False,
        )

    stdout = capsys.readouterr().out
    assert ok is True
    assert exceeded is False
    assert '"report_type": "org_analysis"' in stdout


def test_org_report_renderers_core_min_count_and_unnamed_component_branches(tmp_path: Path, capsys):
    result = _build_rich_result()
    result.parameters.core_min_count = 3

    # Force markdown/html/console branches that depend on unnamed components and large core lists.
    # This exercises the "more rows" truncation and no-name formatting paths.
    result.distribution.core_metrics = [f"metric/core_extra/{i}" for i in range(25)]
    result.distribution.core_dimensions = [f"dimension/core_extra/{i}" for i in range(25)]
    result.similarity_pairs = []

    for comp_id in result.distribution.core_metrics:
        result.component_index[comp_id] = ComponentInfo(
            component_id=comp_id,
            component_type="metric",
            name=None,
            data_views={"dv_001", "dv_003"},
        )
    for comp_id in result.distribution.core_dimensions:
        result.component_index[comp_id] = ComponentInfo(
            component_id=comp_id,
            component_type="dimension",
            name=None,
            data_views={"dv_001", "dv_003"},
        )

    logger = logging.getLogger("test_org_report_renderer_branches")
    md_path = generator.write_org_report_markdown(result, tmp_path / "core_min", str(tmp_path), logger)
    html_path = generator.write_org_report_html(result, tmp_path / "core_min_html", str(tmp_path), logger)
    generator.write_org_report_console(result, result.parameters, quiet=False)
    output = capsys.readouterr().out

    assert Path(md_path).exists()
    assert Path(html_path).exists()
    assert "Core (in >=3 DVs)" in output


def test_run_org_report_defensive_unsupported_format_fallback(tmp_path: Path):
    result = _build_rich_result()
    with (
        patch("cja_auto_sdr.generator._validate_org_report_output_request", return_value=True),
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = analyzer

        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="unsupported_internal_mode",
            output_path=None,
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )

    assert ok is False
    assert exceeded is False


def test_run_org_report_data_alias_and_internal_csv_stdout_branch(tmp_path: Path):
    result = _build_rich_result()
    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
        patch("cja_auto_sdr.generator.write_org_report_json", return_value=str(tmp_path / "report.json")) as mock_json,
        patch("cja_auto_sdr.generator.write_org_report_csv", return_value=str(tmp_path / "report_csv")) as mock_csv,
        patch("cja_auto_sdr.generator.build_org_step_summary", return_value="summary"),
        patch("cja_auto_sdr.generator.append_github_step_summary"),
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = analyzer

        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="data",
            output_path=str(tmp_path / "org_data"),
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
    assert ok is True
    assert exceeded is False
    mock_json.assert_called()
    mock_csv.assert_called()

    # Internal defensive CSV-stdout branch (normally blocked by upfront validation).
    with (
        patch("cja_auto_sdr.generator._validate_org_report_output_request", return_value=True),
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.return_value = result
        mock_analyzer_cls.return_value = analyzer
        ok, exceeded = generator.run_org_report(
            config_file="config.json",
            output_format="csv",
            output_path="stdout",
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
    assert ok is False
    assert exceeded is False


def test_run_org_report_file_not_found_and_keyboard_interrupt_paths(tmp_path: Path):
    with patch("cja_auto_sdr.generator.configure_cjapy", side_effect=FileNotFoundError):
        ok, exceeded = generator.run_org_report(
            config_file="missing.json",
            output_format="console",
            output_path=None,
            output_dir=str(tmp_path),
            org_config=OrgReportConfig(),
            quiet=False,
        )
    assert ok is False
    assert exceeded is False

    with (
        patch("cja_auto_sdr.generator.configure_cjapy", return_value=(True, "mock", {"org_id": "test_org@AdobeOrg"})),
        patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
        patch("cja_auto_sdr.generator.OrgComponentAnalyzer") as mock_analyzer_cls,
    ):
        mock_cjapy.CJA.return_value = Mock()
        analyzer = Mock()
        analyzer.run_analysis.side_effect = KeyboardInterrupt
        mock_analyzer_cls.return_value = analyzer
        with pytest.raises(KeyboardInterrupt):
            generator.run_org_report(
                config_file="config.json",
                output_format="console",
                output_path=None,
                output_dir=str(tmp_path),
                org_config=OrgReportConfig(),
                quiet=False,
            )


def test_recommendation_context_and_normalization_helpers():
    rec = {
        "data_view_name": "Primary",
        "data_view": "dv_1",
        "data_view_1_name": "Left",
        "data_view_1": "dv_left",
        "similarity": 0.88,
        "isolated_count": 4,
        "derived_count": 2,
        "total_count": 10,
        "count": 5,
        "drift_count": 1,
        "ratio": 0.4,
        "modified": "2026-02-10T00:00:00+00:00",
    }
    entries = generator._format_recommendation_context_entries(rec)
    assert ("Data View", "Primary (dv_1)") in entries
    assert ("Pair", "Left (dv_left)") in entries
    assert ("Similarity", "88.0%") in entries
    assert ("Isolated Count", "4") in entries
    assert ("Derived Count", "2") in entries
    assert ("Total Count", "10") in entries
    assert ("Count", "5") in entries
    assert ("Drift Count", "1") in entries
    assert ("Ratio", "40.0%") in entries
    assert ("Last Modified", "2026-02-10T00:00:00+00:00") in entries

    assert generator._normalize_recommendation_severity("HIGH") == "high"
    assert generator._normalize_recommendation_severity("unexpected") == "low"

    normalized = generator._normalize_recommendation_for_json("raw text recommendation")
    assert normalized["type"] == "unknown"
    assert normalized["severity"] == "low"
    assert "raw text recommendation" in normalized["reason"]

    flattened = generator._flatten_recommendation_for_tabular(
        {"type": "naming", "severity": "MEDIUM", "reason": "Use one style", "custom_field": {"x": 1}}
    )
    assert flattened["Severity"] == "medium"
    assert "custom_field" in flattened["Extra Details"]
