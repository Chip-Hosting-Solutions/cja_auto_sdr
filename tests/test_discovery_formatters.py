"""Tests for shared discovery command formatting helpers."""

import json

import pytest

from cja_auto_sdr.core.constants import BANNER_WIDTH
from cja_auto_sdr.generator import (
    WorkerArgs,
    _apply_discovery_filters_and_sort,
    _exit_error,
    _format_as_csv,
    _format_as_json,
    _format_as_table,
)

# ==================== _format_as_json ====================


class TestFormatAsJson:
    """Tests for _format_as_json helper."""

    def test_basic_payload(self):
        """Produces valid JSON with indent=2."""
        payload = {"items": [{"id": "a"}], "count": 1}
        result = _format_as_json(payload)
        assert json.loads(result) == payload
        # Check indentation
        assert '  "items"' in result

    def test_empty_list(self):
        """Empty list payload serializes correctly."""
        payload = {"items": [], "count": 0}
        result = _format_as_json(payload)
        assert json.loads(result) == payload

    def test_unicode_values(self):
        """Unicode characters are preserved."""
        payload = {"name": "Données françaises"}
        result = _format_as_json(payload)
        parsed = json.loads(result)
        assert parsed["name"] == "Données françaises"

    def test_nested_dict(self):
        """Nested dicts are serialized."""
        payload = {"a": {"b": {"c": 1}}}
        result = _format_as_json(payload)
        assert json.loads(result)["a"]["b"]["c"] == 1


# ==================== _format_as_csv ====================


class TestFormatAsCsv:
    """Tests for _format_as_csv helper."""

    def test_basic_rows(self):
        """Produces CSV with header and data rows."""
        cols = ["id", "name"]
        rows = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]
        result = _format_as_csv(cols, rows)
        lines = result.strip().split("\n")
        assert lines[0] == "id,name"
        assert lines[1] == "1,Alice"
        assert lines[2] == "2,Bob"

    def test_empty_rows(self):
        """Empty row list produces header only."""
        result = _format_as_csv(["a", "b"], [])
        assert result.strip() == "a,b"

    def test_missing_keys(self):
        """Missing keys default to empty string."""
        result = _format_as_csv(["id", "name"], [{"id": "1"}])
        lines = result.strip().split("\n")
        assert lines[1] == "1,"

    def test_values_with_commas(self):
        """Values containing commas are properly quoted."""
        result = _format_as_csv(["name"], [{"name": "Last, First"}])
        lines = result.strip().split("\n")
        assert '"Last, First"' in lines[1]

    def test_unicode_values(self):
        """Unicode values in CSV."""
        result = _format_as_csv(["name"], [{"name": "données"}])
        assert "données" in result

    def test_newline_terminator(self):
        """Each row ends with \\n, not \\r\\n."""
        result = _format_as_csv(["a"], [{"a": "1"}])
        assert "\r\n" not in result


# ==================== _format_as_table ====================


class TestFormatAsTable:
    """Tests for _format_as_table helper."""

    def test_basic_table(self):
        """Produces a table with header, separator, and data."""
        items = [{"id": "abc", "name": "Test"}]
        result = _format_as_table("Found 1 item(s):", items, ["id", "name"])
        assert "Found 1 item(s):" in result
        assert "Id" in result  # default title-cased label
        assert "Name" in result
        assert "abc" in result
        assert "Test" in result
        # Should contain a dash separator line
        assert "---" in result

    def test_custom_labels(self):
        """Custom col_labels override defaults."""
        items = [{"x": "1"}]
        result = _format_as_table("Header:", items, ["x"], col_labels=["Custom"])
        assert "Custom" in result
        assert "X" not in result  # default label not used

    def test_column_width_adapts(self):
        """Columns are wide enough for the longest value."""
        items = [{"id": "short"}, {"id": "a_very_long_value_here"}]
        result = _format_as_table("Header:", items, ["id"])
        lines = result.split("\n")
        # Find the data lines — they should all be consistently wide
        data_lines = [line for line in lines if "a_very_long_value_here" in line]
        assert len(data_lines) == 1

    def test_empty_items(self):
        """Empty items list produces header but no data rows."""
        result = _format_as_table("No items:", [], ["id", "name"])
        assert "No items:" in result
        # Still has column labels and separator
        assert "Id" in result
        assert "---" in result

    def test_missing_keys_in_items(self):
        """Items with missing keys show empty string."""
        items = [{"id": "1"}]
        result = _format_as_table("Header:", items, ["id", "name"])
        assert "1" in result

    def test_leading_trailing_blank_lines(self):
        """Result starts and ends with blank lines for spacing."""
        items = [{"a": "1"}]
        result = _format_as_table("H:", items, ["a"])
        assert result.startswith("\n")


# ==================== WorkerArgs ====================


class TestWorkerArgs:
    """Tests for WorkerArgs dataclass."""

    def test_required_field(self):
        """data_view_id is required."""
        wa = WorkerArgs(data_view_id="dv_123")
        assert wa.data_view_id == "dv_123"

    def test_defaults(self):
        """All optional fields have sensible defaults."""
        wa = WorkerArgs(data_view_id="dv_123")
        assert wa.config_file == "config.json"
        assert wa.output_dir == "."
        assert wa.log_level == "INFO"
        assert wa.output_format == "excel"
        assert wa.enable_cache is False
        assert wa.cache_size == 1000
        assert wa.cache_ttl == 3600
        assert wa.quiet is False
        assert wa.skip_validation is False
        assert wa.max_issues == 0
        assert wa.profile is None
        assert wa.shared_cache is None
        assert wa.inventory_only is False
        assert wa.inventory_order is None

    def test_custom_values(self):
        """Custom values are stored correctly."""
        wa = WorkerArgs(
            data_view_id="dv_456",
            config_file="/path/to/config.json",
            output_format="json",
            enable_cache=True,
            cache_size=500,
            profile="staging",
            inventory_only=True,
            inventory_order="alpha",
        )
        assert wa.data_view_id == "dv_456"
        assert wa.config_file == "/path/to/config.json"
        assert wa.output_format == "json"
        assert wa.enable_cache is True
        assert wa.cache_size == 500
        assert wa.profile == "staging"
        assert wa.inventory_only is True
        assert wa.inventory_order == "alpha"

    def test_missing_required_field_raises(self):
        """Omitting data_view_id raises TypeError."""
        with pytest.raises(TypeError):
            WorkerArgs()


# ==================== _exit_error ====================


class TestExitError:
    """Tests for _exit_error helper."""

    def test_prints_error_and_exits(self, capsys):
        """Should print to stderr and exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            _exit_error("something went wrong")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "something went wrong" in captured.err
        assert "ERROR:" in captured.err


# ==================== BANNER_WIDTH ====================


class TestBannerWidth:
    """Tests for BANNER_WIDTH constant."""

    def test_value(self):
        """BANNER_WIDTH is 60."""
        assert BANNER_WIDTH == 60

    def test_type(self):
        """BANNER_WIDTH is an int."""
        assert isinstance(BANNER_WIDTH, int)


# ==================== _apply_discovery_filters_and_sort ====================


class TestApplyDiscoveryFiltersAndSort:
    """Tests for discovery filtering/sorting helper behavior."""

    def test_numeric_sort_uses_numeric_order(self):
        """Numeric values should sort as numbers (2 before 10), not text."""
        rows = [
            {"id": "a", "dataview_count": 10},
            {"id": "b", "dataview_count": 2},
        ]
        sorted_rows = _apply_discovery_filters_and_sort(
            rows,
            sort_expression="dataview_count",
            searchable_fields=["id", "dataview_count"],
            default_sort_field="id",
        )
        assert [row["id"] for row in sorted_rows] == ["b", "a"]

    def test_numeric_sort_descending_uses_numeric_order(self):
        """Descending numeric sort should place larger numbers first."""
        rows = [
            {"id": "a", "dataview_count": 10},
            {"id": "b", "dataview_count": 2},
        ]
        sorted_rows = _apply_discovery_filters_and_sort(
            rows,
            sort_expression="-dataview_count",
            searchable_fields=["id", "dataview_count"],
            default_sort_field="id",
        )
        assert [row["id"] for row in sorted_rows] == ["a", "b"]

    def test_numeric_like_strings_sort_numerically_when_column_is_numeric(self):
        """Numeric-like strings should sort numerically when all values are numeric-like."""
        rows = [
            {"id": "a", "count": "10"},
            {"id": "b", "count": "2"},
            {"id": "c", "count": "3"},
        ]
        sorted_rows = _apply_discovery_filters_and_sort(
            rows,
            sort_expression="count",
            searchable_fields=["id", "count"],
            default_sort_field="id",
        )
        assert [row["id"] for row in sorted_rows] == ["b", "c", "a"]

    def test_mixed_numeric_and_text_values_fall_back_to_text_sort(self):
        """Mixed value types should use deterministic text sorting instead of numeric coercion."""
        rows = [
            {"id": "a", "count": "10"},
            {"id": "b", "count": "unknown"},
            {"id": "c", "count": "2"},
        ]
        sorted_rows = _apply_discovery_filters_and_sort(
            rows,
            sort_expression="count",
            searchable_fields=["id", "count"],
            default_sort_field="id",
        )
        assert [row["id"] for row in sorted_rows] == ["a", "c", "b"]
