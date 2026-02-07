"""
Tests for shared CJA inventory utilities.
"""

import logging

import pandas as pd

from cja_auto_sdr.inventory.utils import (
    BatchProcessingStats,
    compute_complexity_score,
    extract_owner,
    extract_short_name,
    extract_tags,
    format_iso_date,
    normalize_api_response,
    validate_required_id,
)


class TestFormatIsoDate:
    """Tests for format_iso_date function."""

    def test_empty_string_returns_dash(self):
        assert format_iso_date("") == "-"

    def test_none_returns_dash(self):
        assert format_iso_date(None) == "-"

    def test_iso_datetime_formatted(self):
        result = format_iso_date("2024-01-15T10:30:00Z")
        assert result == "2024-01-15 10:30"

    def test_iso_datetime_with_timezone(self):
        result = format_iso_date("2024-01-15T10:30:00+00:00")
        assert result == "2024-01-15 10:30"

    def test_date_only_preserved(self):
        result = format_iso_date("2024-01-15")
        assert result == "2024-01-15"

    def test_invalid_date_truncated(self):
        # Invalid dates without 'T' return first 10 chars (date-only format)
        result = format_iso_date("invalid-date-string-with-lots-of-text")
        assert result == "invalid-da"  # First 10 characters


class TestExtractOwner:
    """Tests for extract_owner function."""

    def test_dict_owner_with_name_and_id(self):
        owner_data = {"name": "John Doe", "id": "user123"}
        name, owner_id = extract_owner(owner_data)
        assert name == "John Doe"
        assert owner_id == "user123"

    def test_dict_owner_with_login_fallback(self):
        owner_data = {"name": "Jane Doe", "login": "jane@example.com"}
        name, owner_id = extract_owner(owner_data)
        assert name == "Jane Doe"
        assert owner_id == "jane@example.com"

    def test_string_owner(self):
        name, owner_id = extract_owner("Simple Owner")
        assert name == "Simple Owner"
        assert owner_id == ""

    def test_none_owner(self):
        name, owner_id = extract_owner(None)
        assert name == ""
        assert owner_id == ""

    def test_empty_dict_owner(self):
        name, owner_id = extract_owner({})
        assert name == ""
        assert owner_id == ""


class TestExtractTags:
    """Tests for extract_tags function."""

    def test_list_of_dicts(self):
        tags_data = [{"name": "Tag1"}, {"name": "Tag2"}]
        result = extract_tags(tags_data)
        assert result == ["Tag1", "Tag2"]

    def test_list_of_strings(self):
        tags_data = ["Tag1", "Tag2", "Tag3"]
        result = extract_tags(tags_data)
        assert result == ["Tag1", "Tag2", "Tag3"]

    def test_mixed_list(self):
        tags_data = [{"name": "DictTag"}, "StringTag"]
        result = extract_tags(tags_data)
        assert result == ["DictTag", "StringTag"]

    def test_empty_list(self):
        result = extract_tags([])
        assert result == []

    def test_none_returns_empty_list(self):
        result = extract_tags(None)
        assert result == []


class TestNormalizeApiResponse:
    """Tests for normalize_api_response function."""

    def test_none_returns_none(self, caplog):
        with caplog.at_level(logging.INFO):
            result = normalize_api_response(None, "test items")
        assert result is None
        assert "No test items found" in caplog.text

    def test_empty_dataframe_returns_none(self, caplog):
        with caplog.at_level(logging.INFO):
            result = normalize_api_response(pd.DataFrame(), "test items")
        assert result is None
        assert "No test items found" in caplog.text

    def test_dataframe_converted_to_list(self):
        df = pd.DataFrame([{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}])
        result = normalize_api_response(df, "items")
        assert result == [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

    def test_list_returned_as_is(self):
        items = [{"id": 1}, {"id": 2}]
        result = normalize_api_response(items, "items")
        assert result == items

    def test_empty_list_returns_none(self, caplog):
        with caplog.at_level(logging.INFO):
            result = normalize_api_response([], "items")
        assert result is None
        assert "No items found" in caplog.text

    def test_unexpected_type_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = normalize_api_response("unexpected string", "items")
        assert result is None
        assert "Unexpected response type" in caplog.text


class TestExtractShortName:
    """Tests for extract_short_name function."""

    def test_slash_path(self):
        assert extract_short_name("metrics/revenue") == "revenue"

    def test_multiple_slash_path(self):
        assert extract_short_name("path/to/field/name") == "name"

    def test_dot_notation(self):
        assert extract_short_name("_experience.analytics.eVars.eVar1") == "eVar1"

    def test_simple_id(self):
        assert extract_short_name("simple_id") == "simple_id"

    def test_empty_string(self):
        assert extract_short_name("") == ""

    def test_none_returns_empty(self):
        assert extract_short_name(None) == ""


class TestComputeComplexityScore:
    """Tests for compute_complexity_score function."""

    def test_zero_factors_returns_zero(self):
        factors = {"operators": 0, "nesting": 0}
        weights = {"operators": 0.5, "nesting": 0.5}
        max_values = {"operators": 10, "nesting": 5}
        assert compute_complexity_score(factors, weights, max_values) == 0.0

    def test_max_factors_returns_100(self):
        factors = {"operators": 100, "nesting": 100}  # Over max
        weights = {"operators": 0.5, "nesting": 0.5}
        max_values = {"operators": 10, "nesting": 5}
        assert compute_complexity_score(factors, weights, max_values) == 100.0

    def test_partial_complexity(self):
        factors = {"operators": 5, "nesting": 2}  # Half of max
        weights = {"operators": 0.5, "nesting": 0.5}
        max_values = {"operators": 10, "nesting": 4}
        # (0.5 * 0.5) + (0.5 * 0.5) = 0.5 * 100 = 50
        assert compute_complexity_score(factors, weights, max_values) == 50.0


class TestBatchProcessingStats:
    """Tests for BatchProcessingStats class."""

    def test_initial_counts_zero(self):
        stats = BatchProcessingStats()
        assert stats.processed == 0
        assert stats.skipped == 0

    def test_record_success(self):
        stats = BatchProcessingStats()
        stats.record_success()
        stats.record_success()
        assert stats.processed == 2

    def test_record_skip(self, caplog):
        with caplog.at_level(logging.WARNING):
            stats = BatchProcessingStats()
            stats.record_skip("missing field", "item123")
        assert stats.skipped == 1
        assert "item123" in caplog.text

    def test_record_error(self, caplog):
        with caplog.at_level(logging.WARNING):
            stats = BatchProcessingStats()
            stats.record_error("parse error", "item456")
        assert stats.skipped == 1
        assert len(stats.errors) == 1
        assert "item456" in caplog.text

    def test_has_issues(self):
        stats = BatchProcessingStats()
        assert not stats.has_issues

        stats.record_skip("reason")
        assert stats.has_issues

    def test_log_summary_with_skips(self, caplog):
        with caplog.at_level(logging.WARNING):
            stats = BatchProcessingStats()
            stats.record_success()
            stats.record_success()
            stats.record_skip("reason")
            stats.log_summary("items")
        assert "2 items processed" in caplog.text
        assert "1 skipped" in caplog.text


class TestValidateRequiredId:
    """Tests for validate_required_id function."""

    def test_valid_id_returned(self):
        data = {"id": "abc123", "name": "Test Item"}
        result = validate_required_id(data)
        assert result == "abc123"

    def test_empty_id_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            data = {"id": "", "name": "Test Item"}
            result = validate_required_id(data)
        assert result is None
        assert "Test Item" in caplog.text
        assert "no id" in caplog.text.lower()

    def test_missing_id_returns_none(self, caplog):
        with caplog.at_level(logging.WARNING):
            data = {"name": "Test Item"}
            result = validate_required_id(data)
        assert result is None

    def test_custom_id_field(self):
        data = {"metric_id": "met123", "name": "Test"}
        result = validate_required_id(data, id_field="metric_id")
        assert result == "met123"
