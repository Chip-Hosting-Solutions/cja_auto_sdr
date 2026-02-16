"""Tests targeting uncovered utility functions in generator.py to maximize line coverage.

These tests exercise pure utility functions that do not require API mocking.
"""

from __future__ import annotations

import math

import pytest

from cja_auto_sdr.diff.models import ChangeType, ComponentDiff
from cja_auto_sdr.generator import (
    RunMode,
    _canonical_quality_policy_key,
    _coerce_run_mode,
    _format_diff_value,
    _get_change_detail,
    _get_change_symbol,
    _get_colored_symbol,
    _infer_run_status,
    _normalize_exit_code,
    _parse_non_negative_policy_int,
    count_quality_issues_by_severity,
    find_similar_names,
    is_data_view_id,
    levenshtein_distance,
    mask_sensitive_value,
    normalize_quality_severity,
)

# ==================== _coerce_run_mode ====================


class TestCoerceRunMode:
    """Tests for _coerce_run_mode() — converts values to RunMode enum."""

    def test_returns_run_mode_unchanged(self):
        assert _coerce_run_mode(RunMode.SDR) is RunMode.SDR

    def test_valid_string_returns_enum(self):
        assert _coerce_run_mode("sdr") is RunMode.SDR

    def test_all_enum_members_round_trip(self):
        for member in RunMode:
            assert _coerce_run_mode(member.value) is member

    def test_invalid_string_returns_none(self):
        assert _coerce_run_mode("not_a_mode") is None

    def test_none_returns_none(self):
        assert _coerce_run_mode(None) is None

    def test_int_returns_none(self):
        assert _coerce_run_mode(42) is None

    def test_bool_returns_none(self):
        assert _coerce_run_mode(True) is None

    def test_empty_string_returns_none(self):
        assert _coerce_run_mode("") is None


# ==================== _canonical_quality_policy_key ====================


class TestCanonicalQualityPolicyKey:
    """Tests for _canonical_quality_policy_key() — normalizes policy key strings."""

    def test_hyphen_to_underscore(self):
        assert _canonical_quality_policy_key("fail-on-quality") == "fail_on_quality"

    def test_uppercase_lowered(self):
        assert _canonical_quality_policy_key("FAIL_ON_QUALITY") == "fail_on_quality"

    def test_whitespace_stripped(self):
        assert _canonical_quality_policy_key("  max_issues  ") == "max_issues"

    def test_mixed_hyphens_uppercase_whitespace(self):
        assert _canonical_quality_policy_key("  Fail-On-Quality  ") == "fail_on_quality"

    def test_non_string_coerced(self):
        assert _canonical_quality_policy_key(123) == "123"

    def test_none_coerced(self):
        assert _canonical_quality_policy_key(None) == "none"


# ==================== _parse_non_negative_policy_int ====================


class TestParseNonNegativePolicyInt:
    """Tests for _parse_non_negative_policy_int() — validates policy integer fields."""

    def test_valid_zero(self):
        assert _parse_non_negative_policy_int(0, key="max_issues") == 0

    def test_valid_positive(self):
        assert _parse_non_negative_policy_int(42, key="max_issues") == 42

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            _parse_non_negative_policy_int(-1, key="max_issues")

    def test_bool_true_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_non_negative_policy_int(True, key="max_issues")

    def test_bool_false_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_non_negative_policy_int(False, key="max_issues")

    def test_float_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_non_negative_policy_int(3.14, key="max_issues")

    def test_string_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_non_negative_policy_int("10", key="max_issues")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_non_negative_policy_int(None, key="max_issues")


# ==================== normalize_quality_severity ====================


class TestNormalizeQualitySeverity:
    """Tests for normalize_quality_severity() — normalizes severity strings."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("critical", "CRITICAL"),
            ("CRITICAL", "CRITICAL"),
            ("high", "HIGH"),
            ("High", "HIGH"),
            ("medium", "MEDIUM"),
            ("low", "LOW"),
            ("info", "INFO"),
        ],
    )
    def test_valid_severities(self, raw, expected):
        assert normalize_quality_severity(raw) == expected

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="Invalid quality severity"):
            normalize_quality_severity("UNKNOWN")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid quality severity"):
            normalize_quality_severity("")


# ==================== count_quality_issues_by_severity ====================


class TestCountQualityIssuesBySeverity:
    """Tests for count_quality_issues_by_severity() — counts issues by severity."""

    def test_empty_list(self):
        result = count_quality_issues_by_severity([])
        assert result == {}

    def test_single_severity(self):
        issues = [{"Severity": "HIGH"}, {"Severity": "HIGH"}]
        result = count_quality_issues_by_severity(issues)
        assert result == {"HIGH": 2}

    def test_mixed_severities(self):
        issues = [
            {"Severity": "HIGH"},
            {"Severity": "LOW"},
            {"Severity": "HIGH"},
            {"Severity": "CRITICAL"},
        ]
        result = count_quality_issues_by_severity(issues)
        assert result == {"CRITICAL": 1, "HIGH": 2, "LOW": 1}

    def test_unknown_severity_ignored(self):
        issues = [{"Severity": "BOGUS"}, {"Severity": "HIGH"}]
        result = count_quality_issues_by_severity(issues)
        assert result == {"HIGH": 1}

    def test_missing_severity_key_ignored(self):
        issues = [{"other": "field"}, {"Severity": "LOW"}]
        result = count_quality_issues_by_severity(issues)
        assert result == {"LOW": 1}

    def test_lowercase_severity_normalised(self):
        """Lowercase severity in data is uppercased and counted."""
        issues = [{"Severity": "high"}]
        result = count_quality_issues_by_severity(issues)
        assert result == {"HIGH": 1}


# ==================== _normalize_exit_code ====================


class TestNormalizeExitCode:
    """Tests for _normalize_exit_code() — normalizes SystemExit.code to int."""

    def test_none_returns_zero(self):
        assert _normalize_exit_code(None) == 0

    def test_int_returns_same(self):
        assert _normalize_exit_code(0) == 0
        assert _normalize_exit_code(1) == 1
        assert _normalize_exit_code(2) == 2
        assert _normalize_exit_code(127) == 127

    def test_bool_true_returns_one(self):
        # bool is subclass of int, so isinstance(True, int) is True
        # True is handled by the int branch and returns int(True) == 1
        assert _normalize_exit_code(True) == 1

    def test_bool_false_returns_zero(self):
        assert _normalize_exit_code(False) == 0

    def test_string_returns_one(self):
        assert _normalize_exit_code("error message") == 1

    def test_empty_string_returns_one(self):
        assert _normalize_exit_code("") == 1

    def test_list_returns_one(self):
        assert _normalize_exit_code([1, 2, 3]) == 1


# ==================== _infer_run_status ====================


class TestInferRunStatus:
    """Tests for _infer_run_status() — classifies run status from exit code + state."""

    def test_exit_zero_is_success(self):
        assert _infer_run_status(0, {}) == "success"

    def test_exit_zero_ignores_state(self):
        assert _infer_run_status(0, {"quality_gate_failed": True}) == "success"

    def test_quality_gate_exit_two_is_policy_exit(self):
        run_state = {"quality_gate_failed": True}
        assert _infer_run_status(2, run_state) == "policy_exit"

    def test_quality_gate_exit_one_is_error(self):
        run_state = {"quality_gate_failed": True}
        assert _infer_run_status(1, run_state) == "error"

    def test_org_threshold_exit_two_is_policy_exit(self):
        run_state = {
            "mode": RunMode.ORG_REPORT,
            "details": {
                "thresholds_exceeded": True,
                "fail_on_threshold": True,
            },
        }
        assert _infer_run_status(2, run_state) == "policy_exit"

    def test_org_threshold_missing_fail_on_is_error(self):
        run_state = {
            "mode": RunMode.ORG_REPORT,
            "details": {
                "thresholds_exceeded": True,
                "fail_on_threshold": False,
            },
        }
        assert _infer_run_status(2, run_state) == "error"

    @pytest.mark.parametrize("mode", [RunMode.DIFF, RunMode.DIFF_SNAPSHOT, RunMode.COMPARE_SNAPSHOTS])
    @pytest.mark.parametrize("exit_code", [2, 3])
    def test_diff_modes_policy_exit(self, mode, exit_code):
        run_state = {
            "mode": mode,
            "details": {"operation_success": True},
        }
        assert _infer_run_status(exit_code, run_state) == "policy_exit"

    def test_diff_mode_operation_failed_is_error(self):
        run_state = {
            "mode": RunMode.DIFF,
            "details": {"operation_success": False},
        }
        assert _infer_run_status(2, run_state) == "error"

    def test_unknown_exit_code_is_error(self):
        assert _infer_run_status(1, {}) == "error"

    def test_none_details_is_error(self):
        assert _infer_run_status(1, {"details": None}) == "error"


# ==================== mask_sensitive_value ====================


class TestMaskSensitiveValue:
    """Tests for mask_sensitive_value() — masks strings for display."""

    def test_empty_string_returns_empty_label(self):
        assert mask_sensitive_value("") == "(empty)"

    def test_short_string_all_asterisks(self):
        # Length 8 with show_chars=4 => 4*2=8 => all asterisks
        assert mask_sensitive_value("abcdefgh") == "********"

    def test_short_string_below_threshold(self):
        assert mask_sensitive_value("abc") == "***"

    def test_long_string_partial_mask(self):
        # "abcdefghijklm" (13 chars) with show_chars=4:
        # first 4 = "abcd", last 4 = "jklm", middle = 13 - 8 = 5 asterisks
        result = mask_sensitive_value("abcdefghijklm")
        assert result.startswith("abcd")
        assert result.endswith("jklm")
        assert result == "abcd*****jklm"

    def test_custom_show_chars(self):
        # "abcdefghij" (10 chars) with show_chars=2 => "ab******ij"
        result = mask_sensitive_value("abcdefghij", show_chars=2)
        assert result == "ab******ij"

    def test_show_chars_one(self):
        # "secret" (6 chars) with show_chars=1: "s****t"
        result = mask_sensitive_value("secret", show_chars=1)
        assert result == "s****t"

    def test_single_char(self):
        assert mask_sensitive_value("x") == "*"


# ==================== _get_change_symbol ====================


class TestGetChangeSymbol:
    """Tests for _get_change_symbol() — maps ChangeType to plain symbols."""

    @pytest.mark.parametrize(
        ("change_type", "expected"),
        [
            (ChangeType.ADDED, "+"),
            (ChangeType.REMOVED, "-"),
            (ChangeType.MODIFIED, "~"),
            (ChangeType.UNCHANGED, " "),
        ],
    )
    def test_known_types(self, change_type, expected):
        assert _get_change_symbol(change_type) == expected

    def test_unknown_type_returns_question_mark(self):
        # Pass something that is not in the dict
        assert _get_change_symbol("not_a_change_type") == "?"


# ==================== _get_colored_symbol ====================


class TestGetColoredSymbol:
    """Tests for _get_colored_symbol() — color-coded symbols."""

    def test_color_disabled_returns_plain_symbol(self):
        for ct in ChangeType:
            symbol = _get_colored_symbol(ct, use_color=False)
            assert symbol == _get_change_symbol(ct)

    def test_added_with_color_has_ansi(self):
        result = _get_colored_symbol(ChangeType.ADDED, use_color=True)
        assert "+" in result
        assert "\033[" in result  # ANSI escape present

    def test_removed_with_color_has_ansi(self):
        result = _get_colored_symbol(ChangeType.REMOVED, use_color=True)
        assert "-" in result
        assert "\033[" in result

    def test_modified_with_color_has_ansi(self):
        result = _get_colored_symbol(ChangeType.MODIFIED, use_color=True)
        assert "~" in result
        assert "\033[" in result

    def test_unchanged_with_color_returns_plain(self):
        # UNCHANGED does not get coloured
        result = _get_colored_symbol(ChangeType.UNCHANGED, use_color=True)
        assert result == " "


# ==================== _format_diff_value ====================


class TestFormatDiffValue:
    """Tests for _format_diff_value() — formats values for diff display."""

    def test_none_returns_empty_label(self):
        assert _format_diff_value(None) == "(empty)"

    def test_nan_returns_empty_label(self):
        assert _format_diff_value(float("nan")) == "(empty)"

    def test_numpy_nan_returns_empty_label(self):
        assert _format_diff_value(math.nan) == "(empty)"

    def test_string_returned_as_is(self):
        assert _format_diff_value("hello") == "hello"

    def test_integer_stringified(self):
        assert _format_diff_value(42) == "42"

    def test_truncation_at_default_max(self):
        long_val = "a" * 50
        result = _format_diff_value(long_val, truncate=True, max_len=30)
        assert len(result) == 30
        assert result == "a" * 30

    def test_no_truncation(self):
        long_val = "a" * 50
        result = _format_diff_value(long_val, truncate=False)
        assert result == long_val

    def test_custom_max_len(self):
        result = _format_diff_value("abcdefghij", truncate=True, max_len=5)
        assert result == "abcde"

    def test_exact_max_len_not_truncated(self):
        result = _format_diff_value("abcde", truncate=True, max_len=5)
        assert result == "abcde"


# ==================== _get_change_detail ====================


class TestGetChangeDetail:
    """Tests for _get_change_detail() — formats changed fields for display."""

    def test_modified_with_changes(self):
        diff = ComponentDiff(
            id="dim1",
            name="Dimension 1",
            change_type=ChangeType.MODIFIED,
            changed_fields={"description": ("old desc", "new desc")},
        )
        result = _get_change_detail(diff)
        assert "description: 'old desc' -> 'new desc'" in result

    def test_modified_multiple_fields(self):
        diff = ComponentDiff(
            id="dim1",
            name="Dimension 1",
            change_type=ChangeType.MODIFIED,
            changed_fields={
                "description": ("old", "new"),
                "type": ("string", "integer"),
            },
        )
        result = _get_change_detail(diff)
        assert "; " in result
        assert "description:" in result
        assert "type:" in result

    def test_added_returns_empty(self):
        diff = ComponentDiff(
            id="dim1",
            name="Dimension 1",
            change_type=ChangeType.ADDED,
        )
        assert _get_change_detail(diff) == ""

    def test_removed_returns_empty(self):
        diff = ComponentDiff(
            id="dim1",
            name="Dimension 1",
            change_type=ChangeType.REMOVED,
        )
        assert _get_change_detail(diff) == ""

    def test_modified_no_changes_returns_empty(self):
        diff = ComponentDiff(
            id="dim1",
            name="Dimension 1",
            change_type=ChangeType.MODIFIED,
            changed_fields={},
        )
        assert _get_change_detail(diff) == ""

    def test_none_values_shown_as_empty(self):
        diff = ComponentDiff(
            id="dim1",
            name="Dimension 1",
            change_type=ChangeType.MODIFIED,
            changed_fields={"description": (None, "new desc")},
        )
        result = _get_change_detail(diff)
        assert "(empty)" in result
        assert "'new desc'" in result


# ==================== is_data_view_id ====================


class TestIsDataViewId:
    """Tests for is_data_view_id() — checks 'dv_' prefix."""

    def test_valid_id(self):
        assert is_data_view_id("dv_abc123") is True

    def test_just_prefix(self):
        assert is_data_view_id("dv_") is True

    def test_wrong_prefix(self):
        assert is_data_view_id("my_dataview") is False

    def test_empty_string(self):
        assert is_data_view_id("") is False

    def test_similar_prefix(self):
        assert is_data_view_id("DV_abc123") is False

    def test_name_containing_dv(self):
        assert is_data_view_id("some_dv_name") is False


# ==================== levenshtein_distance ====================


class TestLevenshteinDistance:
    """Tests for levenshtein_distance() — edit distance calculation."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_first(self):
        assert levenshtein_distance("", "abc") == 3

    def test_empty_second(self):
        assert levenshtein_distance("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein_distance("", "") == 0

    def test_single_char_diff(self):
        assert levenshtein_distance("cat", "car") == 1

    def test_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1

    def test_deletion(self):
        assert levenshtein_distance("cats", "cat") == 1

    def test_reversed_args_same_distance(self):
        assert levenshtein_distance("kitten", "sitting") == levenshtein_distance("sitting", "kitten")

    def test_known_distance(self):
        # kitten -> sitting: 3 edits
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3


# ==================== find_similar_names ====================


class TestFindSimilarNames:
    """Tests for find_similar_names() — similar name finder using edit distance."""

    def test_exact_match_case_insensitive(self):
        result = find_similar_names("Hello", ["hello", "world"])
        assert len(result) >= 1
        assert result[0] == ("hello", 0)

    def test_close_match(self):
        result = find_similar_names("tset", ["test", "best", "zzzz"])
        # "tset" -> "test" is distance 2 (swap); should be found
        names = [name for name, _ in result]
        assert "test" in names

    def test_no_match_beyond_threshold(self):
        result = find_similar_names("abc", ["xyzxyzxyz", "qwertyuiop"])
        assert result == []

    def test_max_suggestions_limits_results(self):
        names = [f"test{i}" for i in range(10)]
        result = find_similar_names("test0", names, max_suggestions=2)
        assert len(result) <= 2

    def test_max_distance_parameter(self):
        result = find_similar_names("abc", ["abd", "xyz"], max_distance=1)
        names = [name for name, _ in result]
        assert "abd" in names
        assert "xyz" not in names

    def test_empty_available_names(self):
        result = find_similar_names("test", [])
        assert result == []

    def test_results_sorted_by_distance(self):
        result = find_similar_names("test", ["test", "tset", "xxxx"], max_distance=5)
        distances = [d for _, d in result]
        assert distances == sorted(distances)
