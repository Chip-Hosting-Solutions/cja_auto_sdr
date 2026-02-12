"""Tests for quality policy functions and run summary/status inference.

Covers: normalize_quality_severity, count_quality_issues_by_severity,
has_quality_issues_at_or_above, aggregate_quality_issues, load_quality_policy,
apply_quality_policy_defaults, _normalize_exit_code, _infer_run_status,
_coerce_run_mode.
"""

import argparse
import json

import pytest

from cja_auto_sdr.generator import (
    ProcessingResult,
    RunMode,
    _coerce_run_mode,
    _infer_run_status,
    _normalize_exit_code,
    aggregate_quality_issues,
    apply_quality_policy_defaults,
    count_quality_issues_by_severity,
    has_quality_issues_at_or_above,
    load_quality_policy,
    normalize_quality_severity,
)

# ==================== normalize_quality_severity ====================


class TestNormalizeQualitySeverity:
    def test_valid_lowercase(self):
        assert normalize_quality_severity("critical") == "CRITICAL"

    def test_valid_uppercase(self):
        assert normalize_quality_severity("HIGH") == "HIGH"

    def test_valid_mixed_case(self):
        assert normalize_quality_severity("Medium") == "MEDIUM"

    def test_all_valid_levels(self):
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            assert normalize_quality_severity(level) == level

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid quality severity"):
            normalize_quality_severity("UNKNOWN")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid quality severity"):
            normalize_quality_severity("")


# ==================== count_quality_issues_by_severity ====================


class TestCountQualityIssuesBySeverity:
    def test_empty_list(self):
        assert count_quality_issues_by_severity([]) == {}

    def test_mixed_severities(self):
        issues = [
            {"Severity": "HIGH"},
            {"Severity": "HIGH"},
            {"Severity": "LOW"},
            {"Severity": "CRITICAL"},
        ]
        result = count_quality_issues_by_severity(issues)
        assert result == {"CRITICAL": 1, "HIGH": 2, "LOW": 1}

    def test_unknown_severity_ignored(self):
        issues = [
            {"Severity": "HIGH"},
            {"Severity": "BOGUS"},
        ]
        result = count_quality_issues_by_severity(issues)
        assert result == {"HIGH": 1}

    def test_missing_severity_key(self):
        issues = [{"other_field": "value"}]
        result = count_quality_issues_by_severity(issues)
        assert result == {}

    def test_lowercase_severity_treated_as_unknown(self):
        """count_quality_issues_by_severity upper-cases internally."""
        issues = [{"Severity": "high"}]
        result = count_quality_issues_by_severity(issues)
        assert result == {"HIGH": 1}


# ==================== has_quality_issues_at_or_above ====================


class TestHasQualityIssuesAtOrAbove:
    def test_critical_above_info(self):
        issues = [{"Severity": "CRITICAL"}]
        assert has_quality_issues_at_or_above(issues, "INFO") is True

    def test_info_not_above_critical(self):
        issues = [{"Severity": "INFO"}]
        assert has_quality_issues_at_or_above(issues, "CRITICAL") is False

    def test_exact_threshold_match(self):
        issues = [{"Severity": "MEDIUM"}]
        assert has_quality_issues_at_or_above(issues, "MEDIUM") is True

    def test_empty_issues(self):
        assert has_quality_issues_at_or_above([], "LOW") is False

    def test_high_above_medium(self):
        issues = [{"Severity": "HIGH"}]
        assert has_quality_issues_at_or_above(issues, "MEDIUM") is True

    def test_low_not_above_high(self):
        issues = [{"Severity": "LOW"}]
        assert has_quality_issues_at_or_above(issues, "HIGH") is False

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="Invalid quality severity"):
            has_quality_issues_at_or_above([], "BOGUS")

    def test_case_insensitive_threshold(self):
        issues = [{"Severity": "HIGH"}]
        assert has_quality_issues_at_or_above(issues, "high") is True

    def test_all_severity_boundaries(self):
        """Verify rank ordering: CRITICAL < HIGH < MEDIUM < LOW < INFO."""
        ordered = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        for i, severity in enumerate(ordered):
            issues = [{"Severity": severity}]
            # Should match at own level and all levels below (higher index)
            for j, threshold in enumerate(ordered):
                expected = i <= j
                assert has_quality_issues_at_or_above(issues, threshold) is expected, (
                    f"severity={severity}, threshold={threshold}"
                )


# ==================== aggregate_quality_issues ====================


class TestAggregateQualityIssues:
    def _make_result(self, dv_id, dv_name, issues):
        return ProcessingResult(
            data_view_id=dv_id,
            data_view_name=dv_name,
            success=True,
            duration=1.0,
            dq_issues=issues,
        )

    def test_empty_results(self):
        assert aggregate_quality_issues([]) == []

    def test_single_result_no_issues(self):
        result = self._make_result("dv1", "Test View", [])
        assert aggregate_quality_issues([result]) == []

    def test_adds_data_view_context(self):
        issues = [{"Severity": "HIGH", "Message": "test"}]
        result = self._make_result("dv1", "Test View", issues)
        aggregated = aggregate_quality_issues([result])
        assert len(aggregated) == 1
        assert aggregated[0]["Data View ID"] == "dv1"
        assert aggregated[0]["Data View Name"] == "Test View"
        assert aggregated[0]["Severity"] == "HIGH"

    def test_multiple_results_flattened(self):
        r1 = self._make_result("dv1", "View 1", [{"Severity": "HIGH"}])
        r2 = self._make_result("dv2", "View 2", [{"Severity": "LOW"}, {"Severity": "MEDIUM"}])
        aggregated = aggregate_quality_issues([r1, r2])
        assert len(aggregated) == 3

    def test_preserves_existing_context(self):
        """If issue already has Data View ID, it should NOT be overwritten."""
        issues = [{"Severity": "HIGH", "Data View ID": "original"}]
        result = self._make_result("dv1", "View 1", issues)
        aggregated = aggregate_quality_issues([result])
        assert aggregated[0]["Data View ID"] == "original"


# ==================== load_quality_policy ====================


class TestLoadQualityPolicy:
    def test_valid_policy_file(self, tmp_path):
        policy = {"fail_on_quality": "HIGH", "quality_report": "json"}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(policy))
        result = load_quality_policy(policy_file)
        assert result["fail_on_quality"] == "HIGH"
        assert result["quality_report"] == "json"

    def test_nested_quality_policy_key(self, tmp_path):
        payload = {"quality_policy": {"fail_on_quality": "MEDIUM"}}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(payload))
        result = load_quality_policy(policy_file)
        assert result["fail_on_quality"] == "MEDIUM"

    def test_nested_quality_key(self, tmp_path):
        payload = {"quality": {"fail_on_quality": "LOW"}}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(payload))
        result = load_quality_policy(policy_file)
        assert result["fail_on_quality"] == "LOW"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Policy file not found"):
            load_quality_policy("/nonexistent/policy.json")

    def test_invalid_json(self, tmp_path):
        policy_file = tmp_path / "bad.json"
        policy_file.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_quality_policy(policy_file)

    def test_non_dict_payload(self, tmp_path):
        policy_file = tmp_path / "array.json"
        policy_file.write_text('["not", "a", "dict"]')
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_quality_policy(policy_file)

    def test_unknown_keys_rejected(self, tmp_path):
        policy = {"fail_on_quality": "HIGH", "unknown_key": True}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(policy))
        with pytest.raises(ValueError, match="Unsupported quality policy key"):
            load_quality_policy(policy_file)

    def test_max_issues_non_negative_int(self, tmp_path):
        policy = {"max_issues": 10}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(policy))
        result = load_quality_policy(policy_file)
        assert result["max_issues"] == 10

    def test_max_issues_negative_rejected(self, tmp_path):
        policy = {"max_issues": -1}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(policy))
        with pytest.raises(ValueError, match="must be >= 0"):
            load_quality_policy(policy_file)

    def test_empty_fail_on_quality_rejected(self, tmp_path):
        policy = {"fail_on_quality": ""}
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(policy))
        with pytest.raises(ValueError, match="cannot be empty"):
            load_quality_policy(policy_file)


# ==================== apply_quality_policy_defaults ====================


class TestApplyQualityPolicyDefaults:
    def test_applies_defaults_when_not_specified(self):
        args = argparse.Namespace(fail_on_quality=None, quality_report=None, max_issues=None)
        policy = {"fail_on_quality": "MEDIUM", "quality_report": "csv", "max_issues": 50}
        applied = apply_quality_policy_defaults(args, policy, argv=["cja_auto_sdr", "dv_123"])
        assert args.fail_on_quality == "MEDIUM"
        assert args.quality_report == "csv"
        assert args.max_issues == 50
        assert applied == {"fail_on_quality": "MEDIUM", "quality_report": "csv", "max_issues": 50}

    def test_cli_flag_overrides_policy(self):
        args = argparse.Namespace(fail_on_quality="HIGH", quality_report=None)
        policy = {"fail_on_quality": "LOW"}
        applied = apply_quality_policy_defaults(
            args, policy, argv=["cja_auto_sdr", "--fail-on-quality", "HIGH", "dv_123"]
        )
        # CLI flag was specified so policy should NOT override
        assert args.fail_on_quality == "HIGH"
        assert "fail_on_quality" not in applied

    def test_empty_policy(self):
        args = argparse.Namespace(fail_on_quality=None)
        applied = apply_quality_policy_defaults(args, {}, argv=[])
        assert applied == {}


# ==================== _normalize_exit_code ====================


class TestNormalizeExitCode:
    def test_none_returns_zero(self):
        assert _normalize_exit_code(None) == 0

    def test_int_passthrough(self):
        assert _normalize_exit_code(0) == 0
        assert _normalize_exit_code(1) == 1
        assert _normalize_exit_code(42) == 42

    def test_bool_true(self):
        # bool is subclass of int; True -> 1
        assert _normalize_exit_code(True) == 1

    def test_bool_false(self):
        assert _normalize_exit_code(False) == 0

    def test_string_returns_one(self):
        assert _normalize_exit_code("error message") == 1

    def test_negative_int(self):
        assert _normalize_exit_code(-1) == -1


# ==================== _infer_run_status ====================


class TestInferRunStatus:
    def test_exit_code_zero_is_success(self):
        assert _infer_run_status(0, {}) == "success"

    def test_quality_gate_exit_2(self):
        run_state = {"quality_gate_failed": True}
        assert _infer_run_status(2, run_state) == "policy_exit"

    def test_org_report_threshold_exit(self):
        run_state = {
            "mode": RunMode.ORG_REPORT,
            "details": {
                "thresholds_exceeded": True,
                "fail_on_threshold": True,
            },
        }
        assert _infer_run_status(2, run_state) == "policy_exit"

    def test_diff_changes_found_exit_2(self):
        run_state = {
            "mode": RunMode.DIFF,
            "details": {"operation_success": True},
        }
        assert _infer_run_status(2, run_state) == "policy_exit"

    def test_diff_warn_threshold_exit_3(self):
        run_state = {
            "mode": RunMode.DIFF_SNAPSHOT,
            "details": {"operation_success": True},
        }
        assert _infer_run_status(3, run_state) == "policy_exit"

    def test_generic_error(self):
        assert _infer_run_status(1, {}) == "error"

    def test_exit_2_without_quality_gate_is_error(self):
        assert _infer_run_status(2, {}) == "error"

    def test_diff_failure_not_policy_exit(self):
        """If operation_success is False, exit 2 should be error, not policy_exit."""
        run_state = {
            "mode": RunMode.DIFF,
            "details": {"operation_success": False},
        }
        assert _infer_run_status(2, run_state) == "error"


# ==================== _coerce_run_mode ====================


class TestCoerceRunMode:
    def test_passthrough_enum(self):
        assert _coerce_run_mode(RunMode.SDR) is RunMode.SDR

    def test_valid_string(self):
        assert _coerce_run_mode("sdr") is RunMode.SDR
        assert _coerce_run_mode("diff") is RunMode.DIFF

    def test_invalid_string(self):
        assert _coerce_run_mode("nonexistent") is None

    def test_none_input(self):
        assert _coerce_run_mode(None) is None

    def test_int_input(self):
        assert _coerce_run_mode(42) is None
