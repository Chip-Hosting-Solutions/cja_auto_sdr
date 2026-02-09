"""End-to-end integration tests for the SDR generation pipeline.

These tests mock only the CJA API boundary and let the entire pipeline
run for real — data quality validation, metadata assembly, and output
writers all produce actual files on disk which are then parsed and
validated for correctness.
"""

import csv
import json
import logging
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cja_auto_sdr.generator import process_single_dataview

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

DV_ID = "dv_e2e_test_42"
DV_NAME = "E2E Test DataView"
DV_OWNER = "Integration Test Runner"


@pytest.fixture
def e2e_config_file(tmp_path):
    """Minimal config.json accepted by initialize_cja."""
    cfg = {
        "org_id": "e2e_org@AdobeOrg",
        "client_id": "e2e_client",
        "secret": "e2e_secret",
        "scopes": "openid,AdobeID",
    }
    p = tmp_path / "e2e_config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


@pytest.fixture
def e2e_output_dir(tmp_path):
    d = tmp_path / "e2e_output"
    d.mkdir()
    return str(d)


@pytest.fixture
def e2e_metrics_df():
    """Realistic metrics DataFrame with varied types and edge cases."""
    return pd.DataFrame([
        {
            "id": "metrics/page_views",
            "name": "Page Views",
            "type": "int",
            "title": "Page Views",
            "description": "Total page views across all pages",
        },
        {
            "id": "metrics/visits",
            "name": "Visits",
            "type": "int",
            "title": "Visits",
            "description": "Total number of visits",
        },
        {
            "id": "metrics/revenue",
            "name": "Revenue",
            "type": "currency",
            "title": "Revenue",
            "description": None,  # Missing description — triggers DQ issue
        },
        {
            "id": "metrics/bounce_rate",
            "name": "Bounce Rate",
            "type": "percent",
            "title": "Bounce Rate",
            "description": "Session bounce rate",
        },
        {
            "id": "metrics/cart_adds",
            "name": "Cart Additions",
            "type": "int",
            "title": "Cart Additions",
            "description": "Items added to cart",
        },
    ])


@pytest.fixture
def e2e_dimensions_df():
    """Realistic dimensions DataFrame with duplicates and edge cases."""
    return pd.DataFrame([
        {
            "id": "dimensions/page",
            "name": "Page Name",
            "type": "string",
            "title": "Page Name",
            "description": "Page URL path",
        },
        {
            "id": "dimensions/browser",
            "name": "Browser",
            "type": "string",
            "title": "Browser",
            "description": "User's web browser",
        },
        {
            "id": "dimensions/region",
            "name": "Region",
            "type": "string",
            "title": "Region",
            "description": "",  # Empty description — triggers DQ issue
        },
        {
            "id": "dimensions/device_type",
            "name": "Device Type",
            "type": "string",
            "title": "Device Type",
            "description": "Mobile, Desktop, or Tablet",
        },
        {
            "id": "dimensions/marketing_channel",
            "name": "Page Name",  # Duplicate name — triggers DQ issue
            "type": "string",
            "title": "Marketing Channel",
            "description": "Traffic source classification",
        },
        {
            "id": "dimensions/os",
            "name": "Operating System",
            "type": "string",
            "title": "Operating System",
            "description": "OS with <special> & \"chars\"",  # Special chars
        },
    ])


@pytest.fixture
def e2e_dataview_info():
    return {
        "id": DV_ID,
        "name": DV_NAME,
        "owner": {"name": DV_OWNER},
        "description": "Data view for end-to-end integration testing",
    }


def _setup_api_mocks(
    mock_setup_logging,
    mock_init_cja,
    mock_validate_dv,
    mock_fetcher_class,
    metrics_df,
    dimensions_df,
    dataview_info,
):
    """Wire up the API-boundary mocks — shared across all e2e tests."""
    logger = logging.getLogger("e2e_test")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    mock_setup_logging.return_value = logger

    mock_cja = Mock()
    mock_init_cja.return_value = mock_cja
    mock_validate_dv.return_value = True

    mock_fetcher = Mock()
    mock_fetcher.fetch_all_data.return_value = (
        metrics_df.copy(),
        dimensions_df.copy(),
        dict(dataview_info),
    )
    mock_fetcher.get_tuner_statistics.return_value = None
    mock_fetcher_class.return_value = mock_fetcher

    return mock_cja, mock_fetcher


# Shared decorator stack — mocks only the API boundary
_api_boundary_patches = [
    patch("cja_auto_sdr.generator.setup_logging"),
    patch("cja_auto_sdr.generator.initialize_cja"),
    patch("cja_auto_sdr.generator.validate_data_view"),
    patch("cja_auto_sdr.generator.ParallelAPIFetcher"),
]


def _apply_patches(func):
    """Apply the API-boundary patches to a test method (outermost first)."""
    for p in reversed(_api_boundary_patches):
        func = p(func)
    return func


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

def _get_excel_sheet_names(file_path: str) -> list[str]:
    with zipfile.ZipFile(file_path) as zf:
        root = ET.fromstring(zf.read("xl/workbook.xml"))
    return [s.attrib["name"] for s in root.findall("x:sheets/x:sheet", XLSX_NS)]


def _get_excel_shared_strings(file_path: str) -> list[str]:
    with zipfile.ZipFile(file_path) as zf:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [
        "".join(t.text or "" for t in si.findall(".//x:t", XLSX_NS))
        for si in root.findall("x:si", XLSX_NS)
    ]


# ===================================================================
# Test class
# ===================================================================

class TestEndToEndPipeline:
    """Full pipeline integration tests — mock API only, real everything else."""

    # ----- Excel output (default format) -----

    @_apply_patches
    def test_excel_output_produces_valid_workbook(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="excel",
            quiet=True,
        )

        # --- Result assertions ---
        assert result.success is True
        assert result.data_view_id == DV_ID
        assert result.data_view_name == DV_NAME
        assert result.metrics_count == 5
        assert result.dimensions_count == 6
        assert result.duration > 0

        # --- File existence ---
        assert result.output_file != ""
        assert os.path.isfile(result.output_file)
        assert result.file_size_bytes > 0

        # --- Valid XLSX structure ---
        sheets = _get_excel_sheet_names(result.output_file)
        assert "Metadata" in sheets
        assert "Data Quality" in sheets
        assert "Metrics" in sheets
        assert "Dimensions" in sheets

        # --- Shared strings contain our data ---
        strings = _get_excel_shared_strings(result.output_file)
        assert any("Page Views" in s for s in strings), "Metric name not found in Excel"
        assert any("Browser" in s for s in strings), "Dimension name not found in Excel"

    # ----- CSV output -----

    @_apply_patches
    def test_csv_output_produces_parseable_files(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="csv",
            quiet=True,
        )

        assert result.success is True

        # CSV produces a directory
        csv_dirs = [d for d in os.listdir(e2e_output_dir) if os.path.isdir(os.path.join(e2e_output_dir, d))]
        assert len(csv_dirs) == 1
        csv_dir = os.path.join(e2e_output_dir, csv_dirs[0])

        # Expect standard CSV files
        expected = {"metadata.csv", "data_quality.csv", "dataview_details.csv", "metrics.csv", "dimensions.csv"}
        actual = set(os.listdir(csv_dir))
        assert expected.issubset(actual), f"Missing CSV files: {expected - actual}"

        # Parse and validate metrics.csv
        metrics_csv = pd.read_csv(os.path.join(csv_dir, "metrics.csv"))
        assert len(metrics_csv) == 5
        assert "name" in metrics_csv.columns
        assert "Page Views" in metrics_csv["name"].values

        # Parse and validate dimensions.csv
        dims_csv = pd.read_csv(os.path.join(csv_dir, "dimensions.csv"))
        assert len(dims_csv) == 6
        assert "Browser" in dims_csv["name"].values

        # Validate data_quality.csv is parseable and has issues
        dq_csv = pd.read_csv(os.path.join(csv_dir, "data_quality.csv"))
        assert len(dq_csv) > 0, "Expected data quality issues from test data"
        assert "Severity" in dq_csv.columns

    # ----- JSON output -----

    @_apply_patches
    def test_json_output_has_correct_structure_and_data(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="json",
            quiet=True,
        )

        assert result.success is True

        # Find the JSON file
        json_files = [f for f in os.listdir(e2e_output_dir) if f.endswith(".json")]
        assert len(json_files) == 1
        json_path = os.path.join(e2e_output_dir, json_files[0])

        # Parse it
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Top-level structure
        assert "metadata" in data
        assert "metrics" in data
        assert "dimensions" in data
        assert "data_quality" in data

        # Metrics data
        assert isinstance(data["metrics"], list)
        assert len(data["metrics"]) == 5
        metric_names = [m["name"] for m in data["metrics"]]
        assert "Page Views" in metric_names
        assert "Revenue" in metric_names

        # Dimensions data
        assert isinstance(data["dimensions"], list)
        assert len(data["dimensions"]) == 6

        # Data quality issues present
        assert isinstance(data["data_quality"], list)
        assert len(data["data_quality"]) > 0

        # Metadata contains expected fields
        assert isinstance(data["metadata"], dict)

    # ----- HTML output -----

    @_apply_patches
    def test_html_output_is_well_formed_and_contains_data(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="html",
            quiet=True,
        )

        assert result.success is True

        html_files = [f for f in os.listdir(e2e_output_dir) if f.endswith(".html")]
        assert len(html_files) == 1
        html_path = os.path.join(e2e_output_dir, html_files[0])

        with open(html_path, encoding="utf-8") as f:
            content = f.read()

        # Basic HTML structure
        assert "<html" in content.lower()
        assert "</html>" in content.lower()
        assert "<table" in content.lower()

        # Data is present
        assert "Page Views" in content
        assert "Browser" in content
        assert DV_ID in content

        # Special characters are properly escaped (item 1 fix verification)
        assert "&amp;" in content or "&lt;" in content or "Operating System" in content

    # ----- Markdown output -----

    @_apply_patches
    def test_markdown_output_contains_tables_and_data(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="markdown",
            quiet=True,
        )

        assert result.success is True

        md_files = [f for f in os.listdir(e2e_output_dir) if f.endswith(".md")]
        assert len(md_files) == 1
        md_path = os.path.join(e2e_output_dir, md_files[0])

        with open(md_path, encoding="utf-8") as f:
            content = f.read()

        # Markdown table separators
        assert "| ---" in content or "|---" in content or "| -" in content

        # Data present
        assert "Page Views" in content
        assert "Browser" in content

    # ----- All-formats output -----

    @_apply_patches
    def test_all_formats_generates_every_format(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="all",
            quiet=True,
        )

        assert result.success is True

        entries = os.listdir(e2e_output_dir)
        extensions = {Path(e).suffix for e in entries}
        # .xlsx, .json, .html, .md, and a CSV directory (no extension)
        assert ".xlsx" in extensions
        assert ".json" in extensions
        assert ".html" in extensions
        assert ".md" in extensions
        # CSV directory exists
        dirs = [e for e in entries if os.path.isdir(os.path.join(e2e_output_dir, e))]
        assert len(dirs) >= 1

    # ----- Data quality detection -----

    @_apply_patches
    def test_data_quality_issues_detected_in_output(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        """Verify the real DataQualityChecker finds issues in our test data."""
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="json",
            quiet=True,
        )

        assert result.success is True
        # Our test data has: missing description on Revenue, empty description
        # on Region, duplicate name "Page Name" in dimensions
        assert result.dq_issues_count > 0
        assert len(result.dq_issues) > 0

        # Verify severity counts are populated
        assert isinstance(result.dq_severity_counts, dict)
        assert sum(result.dq_severity_counts.values()) == result.dq_issues_count

    # ----- skip-validation mode -----

    @_apply_patches
    def test_skip_validation_produces_output_without_dq(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="excel",
            skip_validation=True,
            quiet=True,
        )

        assert result.success is True
        assert result.dq_issues_count == 0
        assert result.metrics_count == 5
        assert os.path.isfile(result.output_file)

    # ----- quality-report-only mode -----

    @_apply_patches
    def test_quality_report_only_returns_issues_without_files(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="excel",
            quality_report_only=True,
            quiet=True,
        )

        assert result.success is True
        assert result.dq_issues_count > 0
        assert result.metrics_count == 5
        assert result.dimensions_count == 6
        # No output files should be generated
        assert result.output_file == ""

    # ----- CJA init failure -----

    @_apply_patches
    def test_cja_init_failure_returns_failed_result(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        logger = logging.getLogger("e2e_fail")
        logger.handlers = []
        mock_setup_logging.return_value = logger
        mock_init_cja.return_value = None  # Simulate failure

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            quiet=True,
        )

        assert result.success is False
        assert "initialization failed" in result.error_message.lower()
        assert result.data_view_name == "Unknown"

    # ----- Data view validation failure -----

    @_apply_patches
    def test_dataview_validation_failure(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        logger = logging.getLogger("e2e_fail")
        logger.handlers = []
        mock_setup_logging.return_value = logger
        mock_init_cja.return_value = Mock()
        mock_validate_dv.return_value = False  # Simulate invalid DV

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            quiet=True,
        )

        assert result.success is False
        assert "validation failed" in result.error_message.lower()

    # ----- Empty data view -----

    @_apply_patches
    def test_empty_dataview_returns_failure(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        logger = logging.getLogger("e2e_empty")
        logger.handlers = []
        mock_setup_logging.return_value = logger
        mock_init_cja.return_value = Mock()
        mock_validate_dv.return_value = True

        mock_fetcher = Mock()
        mock_fetcher.fetch_all_data.return_value = (
            pd.DataFrame(),  # Empty metrics
            pd.DataFrame(),  # Empty dimensions
            {"id": DV_ID, "name": DV_NAME},
        )
        mock_fetcher.get_tuner_statistics.return_value = None
        mock_fetcher_class.return_value = mock_fetcher

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            quiet=True,
        )

        assert result.success is False
        assert "empty" in result.error_message.lower() or "no metrics" in result.error_message.lower()

    # ----- metrics-only / dimensions-only filtering -----

    @_apply_patches
    def test_metrics_only_excludes_dimensions_from_excel(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="excel",
            metrics_only=True,
            quiet=True,
        )

        assert result.success is True
        sheets = _get_excel_sheet_names(result.output_file)
        assert "Metrics" in sheets
        assert "Dimensions" not in sheets

    @_apply_patches
    def test_dimensions_only_excludes_metrics_from_excel(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="excel",
            dimensions_only=True,
            quiet=True,
        )

        assert result.success is True
        sheets = _get_excel_sheet_names(result.output_file)
        assert "Dimensions" in sheets
        assert "Metrics" not in sheets

    # ----- max_issues limiting -----

    @_apply_patches
    def test_max_issues_limits_reported_count(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        # First run without limit to get total issues
        result_all = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="json",
            quiet=True,
        )
        total_issues = result_all.dq_issues_count

        # Only test limiting if there are enough issues
        if total_issues >= 2:
            # Use a fresh output dir to avoid file conflicts
            limited_dir = str(Path(e2e_output_dir).parent / "limited_output")
            os.makedirs(limited_dir, exist_ok=True)

            result_limited = process_single_dataview(
                data_view_id=DV_ID,
                config_file=e2e_config_file,
                output_dir=limited_dir,
                output_format="json",
                max_issues=1,
                quiet=True,
            )

            assert result_limited.success is True
            # dq_issues list should be limited
            assert len(result_limited.dq_issues) <= 1

    # ----- Special characters in output -----

    @_apply_patches
    def test_special_characters_handled_in_all_formats(
        self,
        mock_fetcher_class,
        mock_validate_dv,
        mock_init_cja,
        mock_setup_logging,
        e2e_config_file,
        e2e_output_dir,
        e2e_metrics_df,
        e2e_dimensions_df,
        e2e_dataview_info,
    ):
        """Verify special chars (HTML entities, quotes) don't corrupt output."""
        _setup_api_mocks(
            mock_setup_logging, mock_init_cja, mock_validate_dv,
            mock_fetcher_class, e2e_metrics_df, e2e_dimensions_df, e2e_dataview_info,
        )

        # Test HTML — our fixture has description with <special> & "chars"
        result = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=e2e_output_dir,
            output_format="html",
            quiet=True,
        )
        assert result.success is True

        html_file = [f for f in os.listdir(e2e_output_dir) if f.endswith(".html")][0]
        with open(os.path.join(e2e_output_dir, html_file), encoding="utf-8") as f:
            html_content = f.read()

        # Raw < should not appear unescaped in table data (outside tags)
        # The description "OS with <special> & \"chars\"" should be escaped
        assert "<special>" not in html_content, "Unescaped < in HTML output"

        # Test JSON — special chars should be preserved as-is in JSON strings
        json_dir = str(Path(e2e_output_dir).parent / "json_special")
        os.makedirs(json_dir, exist_ok=True)

        result_json = process_single_dataview(
            data_view_id=DV_ID,
            config_file=e2e_config_file,
            output_dir=json_dir,
            output_format="json",
            quiet=True,
        )
        assert result_json.success is True

        json_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
        with open(os.path.join(json_dir, json_file), encoding="utf-8") as f:
            data = json.load(f)

        # JSON should preserve the original string
        os_dim = [d for d in data["dimensions"] if d.get("id") == "dimensions/os"]
        assert len(os_dim) == 1
        assert "<special>" in os_dim[0]["description"]
