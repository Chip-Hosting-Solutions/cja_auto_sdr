"""Tests for config status, validation, stats, data view resolution, and misc helpers.

Covers uncovered lines in generator.py:
- show_config_status (lines 9894-10027)
- validate_config_only (lines 10035-10220)
- show_stats (lines 10226-10410)
- resolve_data_view_names (lines 8208-8361)
- prompt_for_selection (lines 8165-8205)
- DataViewCache (lines 8046-8088)
- _format_diff_value (lines 3262-3263)
- _safe_env_number (line 6707)
- levenshtein_distance (lines 7947-7977)
- is_data_view_id (line 7934)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

logger = logging.getLogger("test_config_and_resolution")


# ---------------------------------------------------------------------------
# 1. show_config_status — lines 9894-10027
# ---------------------------------------------------------------------------


class TestShowConfigStatusProfile:
    """Lines 9894-9907: profile credential loading paths."""

    @patch("cja_auto_sdr.generator.load_profile_credentials")
    def test_profile_found_complete(self, mock_load, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_config_status

        mock_load.return_value = {"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}
        result = show_config_status(profile="myprofile")
        assert result is True
        output = capsys.readouterr().out
        assert "Profile: myprofile" in output

    @patch("cja_auto_sdr.generator.load_profile_credentials")
    def test_profile_not_found_error(self, mock_load) -> None:
        from cja_auto_sdr.generator import ProfileNotFoundError, show_config_status

        mock_load.side_effect = ProfileNotFoundError("not found", profile_name="bad")
        result = show_config_status(profile="bad")
        assert result is False

    @patch("cja_auto_sdr.generator.load_profile_credentials")
    def test_profile_not_found_json(self, mock_load, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import ProfileNotFoundError, show_config_status

        mock_load.side_effect = ProfileNotFoundError("not found", profile_name="bad")
        result = show_config_status(profile="bad", output_json=True)
        assert result is False
        data = json.loads(capsys.readouterr().out)
        assert data["valid"] is False


class TestShowConfigStatusEnvVars:
    """Lines 9910-9915: environment variable credential detection."""

    @patch("cja_auto_sdr.generator.load_credentials_from_env")
    @patch("cja_auto_sdr.generator.validate_env_credentials", return_value=True)
    def test_env_vars_detected(self, _mock_validate, mock_load_env, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_config_status

        mock_load_env.return_value = {"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}
        result = show_config_status()
        assert result is True
        output = capsys.readouterr().out
        assert "Environment variables" in output


class TestShowConfigStatusFile:
    """Lines 9918-9954: config file loading and error paths."""

    def test_config_file_valid(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_config_status

        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}),
        )
        result = show_config_status(config_file=str(config))
        assert result is True

    def test_config_file_invalid_json(self, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import show_config_status

        config = tmp_path / "config.json"
        config.write_text("{bad json")
        result = show_config_status(config_file=str(config))
        assert result is False

    def test_config_file_invalid_json_output_json(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_config_status

        config = tmp_path / "config.json"
        config.write_text("{bad json")
        result = show_config_status(config_file=str(config), output_json=True)
        assert result is False
        data = json.loads(capsys.readouterr().out)
        assert data["valid"] is False

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import show_config_status

        result = show_config_status(config_file=str(tmp_path / "nonexistent.json"))
        assert result is False

    def test_config_file_not_found_json(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_config_status

        result = show_config_status(config_file=str(tmp_path / "nonexistent.json"), output_json=True)
        assert result is False
        data = json.loads(capsys.readouterr().out)
        assert data["valid"] is False


class TestShowConfigStatusJsonOutput:
    """Lines 9988-9997: JSON output format."""

    def test_json_output_all_fields(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_config_status

        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}),
        )
        result = show_config_status(config_file=str(config), output_json=True)
        assert result is True
        data = json.loads(capsys.readouterr().out)
        assert data["valid"] is True
        assert data["source_type"] == "file"

    def test_missing_required_field(self, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import show_config_status

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"org_id": "org@Adobe"}))
        result = show_config_status(config_file=str(config))
        assert result is False


# ---------------------------------------------------------------------------
# 2. validate_config_only — lines 10035-10220
# ---------------------------------------------------------------------------


class TestValidateConfigOnly:
    """Lines 10100-10220: profile/env/file validation + API test."""

    @patch("cja_auto_sdr.generator._config_from_env")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.load_profile_credentials")
    def test_profile_valid_api_success(
        self, mock_load, mock_cjapy, _mock_config_env, capsys: pytest.CaptureFixture
    ) -> None:
        from cja_auto_sdr.generator import validate_config_only

        mock_load.return_value = {"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv1"}]
        mock_cjapy.CJA.return_value = mock_cja
        result = validate_config_only(profile="myprofile")
        assert result is True
        assert "VALIDATION PASSED" in capsys.readouterr().out

    @patch("cja_auto_sdr.generator.load_profile_credentials")
    def test_profile_not_found(self, mock_load) -> None:
        from cja_auto_sdr.generator import ProfileNotFoundError, validate_config_only

        mock_load.side_effect = ProfileNotFoundError("not found", profile_name="bad")
        result = validate_config_only(profile="bad")
        assert result is False

    @patch("cja_auto_sdr.generator.load_profile_credentials")
    def test_profile_config_error(self, mock_load) -> None:
        from cja_auto_sdr.generator import ProfileConfigError, validate_config_only

        mock_load.side_effect = ProfileConfigError("invalid config", profile_name="bad")
        result = validate_config_only(profile="bad")
        assert result is False

    def test_no_config_file_no_env(self, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import validate_config_only

        with patch("cja_auto_sdr.generator.load_credentials_from_env", return_value=None):
            result = validate_config_only(config_file=str(tmp_path / "nonexistent.json"))
        assert result is False

    @patch("cja_auto_sdr.generator.cjapy")
    def test_api_connection_failure(self, mock_cjapy, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import validate_config_only

        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}),
        )
        mock_cja = MagicMock()
        mock_cja.getDataViews.side_effect = RuntimeError("connection refused")
        mock_cjapy.CJA.return_value = mock_cja
        with patch("cja_auto_sdr.generator.load_credentials_from_env", return_value=None):
            result = validate_config_only(config_file=str(config))
        assert result is False

    @patch("cja_auto_sdr.generator._config_from_env")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.load_credentials_from_env")
    @patch("cja_auto_sdr.generator.validate_env_credentials", return_value=True)
    def test_env_credentials_api_returns_none(
        self, _mock_validate, mock_load_env, mock_cjapy, _mock_config_env, capsys: pytest.CaptureFixture
    ) -> None:
        """Line 10201: API returns None response."""
        from cja_auto_sdr.generator import validate_config_only

        mock_load_env.return_value = {"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = None
        mock_cjapy.CJA.return_value = mock_cja
        validate_config_only()
        output = capsys.readouterr().out
        assert "empty response" in output or "unstable" in output

    def test_config_file_invalid_json(self, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import validate_config_only

        config = tmp_path / "config.json"
        config.write_text("{bad json")
        with patch("cja_auto_sdr.generator.load_credentials_from_env", return_value=None):
            result = validate_config_only(config_file=str(config))
        assert result is False

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.load_credentials_from_env")
    @patch("cja_auto_sdr.generator.validate_env_credentials", return_value=True)
    def test_env_credentials_incomplete_then_file(
        self, _mock_validate, mock_load_env, mock_cjapy, tmp_path: Path
    ) -> None:
        """Line 10131: env creds incomplete, fallback to file."""
        from cja_auto_sdr.generator import validate_config_only

        # Return creds that look present but validate_env_credentials returns True
        # but display_credentials says missing fields
        mock_load_env.return_value = {"org_id": "org@Adobe"}  # missing client_id, secret
        _mock_validate.return_value = False  # Actually fail env validation
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"org_id": "org@Adobe", "client_id": "abcd1234efgh", "secret": "secret12345678"}),
        )
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv1"}]
        mock_cjapy.CJA.return_value = mock_cja
        validate_config_only(config_file=str(config))
        # Should fall through to config file and pass


# ---------------------------------------------------------------------------
# 3. show_stats — lines 10226-10410
# ---------------------------------------------------------------------------


class TestShowStats:
    """Lines 10248-10410: stats command output formats and error handling."""

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_table_format(self, mock_config, mock_cjapy, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = {"name": "Test DV", "owner": {"name": "Alice"}, "description": "Desc"}
        mock_cja.getMetrics.return_value = pd.DataFrame({"id": ["m1", "m2"]})
        mock_cja.getDimensions.return_value = pd.DataFrame({"id": ["d1"]})
        mock_cjapy.CJA.return_value = mock_cja
        result = show_stats(["dv_test"])
        assert result is True
        assert "Test DV" in capsys.readouterr().out

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_json_format(self, mock_config, mock_cjapy, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = {"name": "Test DV", "owner": {"name": "Alice"}, "description": ""}
        mock_cja.getMetrics.return_value = pd.DataFrame({"id": ["m1"]})
        mock_cja.getDimensions.return_value = pd.DataFrame({"id": ["d1"]})
        mock_cjapy.CJA.return_value = mock_cja
        result = show_stats(["dv_test"], output_format="json")
        assert result is True
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_csv_format(self, mock_config, mock_cjapy, capsys: pytest.CaptureFixture) -> None:
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = {"name": "Test DV", "owner": {"name": "Alice"}, "description": ""}
        mock_cja.getMetrics.return_value = pd.DataFrame({"id": ["m1"]})
        mock_cja.getDimensions.return_value = pd.DataFrame({"id": ["d1"]})
        mock_cjapy.CJA.return_value = mock_cja
        result = show_stats(["dv_test"], output_format="csv")
        assert result is True
        assert "id,name" in capsys.readouterr().out

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_json_to_stdout(self, mock_config, mock_cjapy) -> None:
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = {"name": "DV", "owner": {}, "description": ""}
        mock_cja.getMetrics.return_value = pd.DataFrame()
        mock_cja.getDimensions.return_value = pd.DataFrame()
        mock_cjapy.CJA.return_value = mock_cja
        assert show_stats(["dv_test"], output_file="-") is True

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_json_to_file(self, mock_config, mock_cjapy, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = {"name": "DV", "owner": {}, "description": ""}
        mock_cja.getMetrics.return_value = pd.DataFrame()
        mock_cja.getDimensions.return_value = pd.DataFrame()
        mock_cjapy.CJA.return_value = mock_cja
        outfile = str(tmp_path / "stats.json")
        assert show_stats(["dv_test"], output_format="json", output_file=outfile) is True
        assert Path(outfile).exists()

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_csv_to_file(self, mock_config, mock_cjapy, tmp_path: Path) -> None:
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = {"name": "DV", "owner": {}, "description": ""}
        mock_cja.getMetrics.return_value = pd.DataFrame()
        mock_cja.getDimensions.return_value = pd.DataFrame()
        mock_cjapy.CJA.return_value = mock_cja
        outfile = str(tmp_path / "stats.csv")
        assert show_stats(["dv_test"], output_format="csv", output_file=outfile) is True
        assert Path(outfile).exists()

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_per_dv_exception(self, mock_config, mock_cjapy, capsys: pytest.CaptureFixture) -> None:
        """Lines 10301-10312: exception per data view caught gracefully."""
        from cja_auto_sdr.generator import show_stats

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = RuntimeError("API error")
        mock_cjapy.CJA.return_value = mock_cja
        result = show_stats(["dv_test"], output_format="json")
        assert result is True
        data = json.loads(capsys.readouterr().out)
        assert data["stats"][0]["name"] == "ERROR"

    @patch("cja_auto_sdr.generator.configure_cjapy", return_value=(False, "Config error", None))
    def test_config_failure(self, _mock_config) -> None:
        from cja_auto_sdr.generator import show_stats

        assert show_stats(["dv_test"]) is False

    @patch("cja_auto_sdr.generator.configure_cjapy", side_effect=FileNotFoundError("not found"))
    def test_file_not_found(self, _mock_config) -> None:
        """Lines 10390-10396: FileNotFoundError handler."""
        from cja_auto_sdr.generator import show_stats

        assert show_stats(["dv_test"]) is False

    @patch("cja_auto_sdr.generator.configure_cjapy", side_effect=FileNotFoundError("not found"))
    def test_file_not_found_machine_readable(self, _mock_config) -> None:
        """Lines 10391-10393: FileNotFoundError with JSON output."""
        from cja_auto_sdr.generator import show_stats

        assert show_stats(["dv_test"], output_format="json") is False

    @patch("cja_auto_sdr.generator.configure_cjapy", side_effect=RuntimeError("boom"))
    def test_generic_exception(self, _mock_config) -> None:
        """Lines 10404-10410: generic exception handler."""
        from cja_auto_sdr.generator import show_stats

        assert show_stats(["dv_test"]) is False

    @patch("cja_auto_sdr.generator.configure_cjapy", side_effect=RuntimeError("boom"))
    def test_generic_exception_machine_readable(self, _mock_config) -> None:
        """Lines 10405-10407: generic exception with JSON output."""
        from cja_auto_sdr.generator import show_stats

        assert show_stats(["dv_test"], output_format="json") is False


# ---------------------------------------------------------------------------
# 4. resolve_data_view_names — lines 8208-8361
# ---------------------------------------------------------------------------


class TestResolveDataViewNames:
    """Lines 8252-8360: resolution modes, name matching, suggestions."""

    @pytest.fixture(autouse=True)
    def _clear_dv_cache(self):
        """Clear global DataViewCache before each test."""
        from cja_auto_sdr.generator import _data_view_cache

        _data_view_cache.clear()

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_exact_match(self, mock_config, mock_cjapy) -> None:
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv_1", "name": "My DV"}]
        mock_cjapy.CJA.return_value = mock_cja
        ids, name_map = resolve_data_view_names(["My DV"], match_mode="exact")
        assert ids == ["dv_1"]
        assert "My DV" in name_map

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_insensitive_match(self, mock_config, mock_cjapy) -> None:
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv_1", "name": "My DV"}]
        mock_cjapy.CJA.return_value = mock_cja
        ids, _ = resolve_data_view_names(["my dv"], match_mode="insensitive")
        assert ids == ["dv_1"]

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_fuzzy_match_nearest(self, mock_config, mock_cjapy) -> None:
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv_1", "name": "My Data View"}]
        mock_cjapy.CJA.return_value = mock_cja
        ids, _ = resolve_data_view_names(["My Dta View"], match_mode="fuzzy")
        assert len(ids) >= 1

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_name_not_found_with_suggestions(self, mock_config, mock_cjapy) -> None:
        """Lines 8334-8342: suggestions when exact match fails."""
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv_1", "name": "Production DV"}]
        mock_cjapy.CJA.return_value = mock_cja
        ids, _ = resolve_data_view_names(["Productin DV"], match_mode="exact")
        assert len(ids) == 0

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_id_passthrough(self, mock_config, mock_cjapy) -> None:
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv_test", "name": "Test"}]
        mock_cjapy.CJA.return_value = mock_cja
        ids, _ = resolve_data_view_names(["dv_test"])
        assert ids == ["dv_test"]

    @patch("cja_auto_sdr.generator.configure_cjapy", return_value=(False, "Error", None))
    def test_config_failure(self, _mock_config) -> None:
        from cja_auto_sdr.generator import resolve_data_view_names

        ids, _ = resolve_data_view_names(["dv_test"])
        assert ids == []

    @patch("cja_auto_sdr.generator.configure_cjapy", side_effect=FileNotFoundError("not found"))
    def test_file_not_found(self, _mock_config) -> None:
        """Line 8355-8357: FileNotFoundError handler."""
        from cja_auto_sdr.generator import resolve_data_view_names

        ids, _ = resolve_data_view_names(["dv_test"])
        assert ids == []

    @patch("cja_auto_sdr.generator.configure_cjapy", side_effect=RuntimeError("unexpected"))
    def test_generic_exception(self, _mock_config) -> None:
        """Lines 8358-8360: generic exception handler."""
        from cja_auto_sdr.generator import resolve_data_view_names

        ids, _ = resolve_data_view_names(["dv_test"])
        assert ids == []

    def test_invalid_match_mode(self) -> None:
        from cja_auto_sdr.generator import resolve_data_view_names

        with pytest.raises(ValueError, match="Invalid match_mode"):
            resolve_data_view_names(["dv_test"], match_mode="invalid")

    @patch("cja_auto_sdr.generator.get_cached_data_views", return_value=[])
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_no_data_views_available(self, mock_config, mock_cjapy, _mock_cache) -> None:
        """Line 8267: no data views accessible."""
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cjapy.CJA.return_value = MagicMock()
        ids, _ = resolve_data_view_names(["My DV"])
        assert ids == []

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_insensitive_no_match(self, mock_config, mock_cjapy) -> None:
        """Lines 8346-8347: insensitive mode error message."""
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [{"id": "dv_1", "name": "Other DV"}]
        mock_cjapy.CJA.return_value = mock_cja
        ids, _ = resolve_data_view_names(["nonexistent"], match_mode="insensitive")
        assert len(ids) == 0

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_multiple_ids_for_name(self, mock_config, mock_cjapy) -> None:
        """Lines 8329-8330: multiple data views with same name."""
        from cja_auto_sdr.generator import resolve_data_view_names

        mock_config.return_value = (True, "file", None)
        mock_cja = MagicMock()
        mock_cja.getDataViews.return_value = [
            {"id": "dv_1", "name": "My DV"},
            {"id": "dv_2", "name": "My DV"},
        ]
        mock_cjapy.CJA.return_value = mock_cja
        ids, name_map = resolve_data_view_names(["My DV"], match_mode="exact")
        assert len(ids) == 2
        assert len(name_map["My DV"]) == 2


# ---------------------------------------------------------------------------
# 5. prompt_for_selection — lines 8165-8205
# ---------------------------------------------------------------------------


class TestPromptForSelection:
    """Lines 8177-8205: interactive selection, non-TTY, cancel, input."""

    def test_non_tty_returns_none(self) -> None:
        from cja_auto_sdr.generator import prompt_for_selection

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = prompt_for_selection([("dv_1", "DV 1")], "Select:")
        assert result is None

    @patch("sys.stdin")
    @patch("builtins.input", return_value="1")
    def test_valid_selection(self, _mock_input, mock_stdin) -> None:
        from cja_auto_sdr.generator import prompt_for_selection

        mock_stdin.isatty.return_value = True
        result = prompt_for_selection([("dv_1", "DV 1"), ("dv_2", "DV 2")], "Select:")
        assert result == "dv_1"

    @patch("sys.stdin")
    @patch("builtins.input", return_value="0")
    def test_cancel_selection(self, _mock_input, mock_stdin) -> None:
        from cja_auto_sdr.generator import prompt_for_selection

        mock_stdin.isatty.return_value = True
        assert prompt_for_selection([("dv_1", "DV 1")], "Select:") is None

    @patch("sys.stdin")
    @patch("builtins.input", return_value="quit")
    def test_quit_selection(self, _mock_input, mock_stdin) -> None:
        from cja_auto_sdr.generator import prompt_for_selection

        mock_stdin.isatty.return_value = True
        assert prompt_for_selection([("dv_1", "DV 1")], "Select:") is None

    @patch("sys.stdin")
    @patch("builtins.input", side_effect=["abc", "1"])
    def test_invalid_then_valid(self, _mock_input, mock_stdin) -> None:
        """Line 8201-8202: ValueError on non-numeric input."""
        from cja_auto_sdr.generator import prompt_for_selection

        mock_stdin.isatty.return_value = True
        assert prompt_for_selection([("dv_1", "DV 1")], "Select:") == "dv_1"

    @patch("sys.stdin")
    @patch("builtins.input", side_effect=["99", "1"])
    def test_out_of_range_then_valid(self, _mock_input, mock_stdin) -> None:
        """Line 8200: out-of-range number."""
        from cja_auto_sdr.generator import prompt_for_selection

        mock_stdin.isatty.return_value = True
        assert prompt_for_selection([("dv_1", "DV 1")], "Select:") == "dv_1"

    @patch("sys.stdin")
    @patch("builtins.input", side_effect=EOFError)
    def test_eof_returns_none(self, _mock_input, mock_stdin) -> None:
        """Line 8203-8205: EOFError handler."""
        from cja_auto_sdr.generator import prompt_for_selection

        mock_stdin.isatty.return_value = True
        assert prompt_for_selection([("dv_1", "DV 1")], "Select:") is None


# ---------------------------------------------------------------------------
# 6. DataViewCache — lines 8046-8088
# ---------------------------------------------------------------------------


class TestDataViewCache:
    """Lines 8046-8088: cache hit/miss, TTL expiration."""

    def test_cache_set_and_get(self) -> None:
        from cja_auto_sdr.generator import DataViewCache

        cache = DataViewCache.__new__(DataViewCache)
        cache._initialized = False
        cache.__init__()
        cache.set("key1", [{"id": "dv1"}])
        assert cache.get("key1") == [{"id": "dv1"}]

    def test_cache_miss(self) -> None:
        from cja_auto_sdr.generator import DataViewCache

        cache = DataViewCache.__new__(DataViewCache)
        cache._initialized = False
        cache.__init__()
        assert cache.get("nonexistent") is None

    def test_cache_clear(self) -> None:
        from cja_auto_sdr.generator import DataViewCache

        cache = DataViewCache.__new__(DataViewCache)
        cache._initialized = False
        cache.__init__()
        cache.set("key1", [{"id": "dv1"}])
        cache.clear()
        assert cache.get("key1") is None

    def test_cache_ttl_expiry(self) -> None:
        """Line 8088: set_ttl then expiry."""
        import time

        from cja_auto_sdr.generator import DataViewCache

        cache = DataViewCache.__new__(DataViewCache)
        cache._initialized = False
        cache.__init__()
        cache.set_ttl(1)
        cache.set("key1", [{"id": "dv1"}])
        # Manually set timestamp to past to simulate expiry
        cache._cache["key1"] = (cache._cache["key1"][0], time.time() - 2)
        assert cache.get("key1") is None


# ---------------------------------------------------------------------------
# 7. _format_diff_value — line 3262-3263
# ---------------------------------------------------------------------------


class TestFormatDiffValueTypeError:
    """Line 3262: pd.isna raises TypeError for unhashable types like list."""

    def test_list_value(self) -> None:
        from cja_auto_sdr.generator import _format_diff_value

        assert _format_diff_value([1, 2, 3]) == "[1, 2, 3]"

    def test_dict_value(self) -> None:
        from cja_auto_sdr.generator import _format_diff_value

        result = _format_diff_value({"key": "value"})
        assert "key" in result

    def test_none_value(self) -> None:
        from cja_auto_sdr.generator import _format_diff_value

        assert _format_diff_value(None) == "(empty)"

    def test_long_value_truncated(self) -> None:
        from cja_auto_sdr.generator import _format_diff_value

        long_str = "x" * 200
        result = _format_diff_value(long_str, truncate=True)
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# 8. _safe_env_number — line 6707
# ---------------------------------------------------------------------------


class TestSafeEnvNumber:
    """Line 6707: except TypeError, ValueError fallback."""

    def test_valid_int(self) -> None:
        from cja_auto_sdr.generator import _safe_env_number

        with patch.dict("os.environ", {"TEST_VAR": "42"}):
            assert _safe_env_number("TEST_VAR", 10, int) == 42

    def test_invalid_value_returns_default(self) -> None:
        from cja_auto_sdr.generator import _safe_env_number

        with patch.dict("os.environ", {"TEST_VAR": "not_a_number"}):
            assert _safe_env_number("TEST_VAR", 10, int) == 10

    def test_missing_env_var_returns_default(self) -> None:
        from cja_auto_sdr.generator import _safe_env_number

        with patch.dict("os.environ", {}, clear=True):
            assert _safe_env_number("MISSING_VAR", 99, int) == 99


# ---------------------------------------------------------------------------
# 9. levenshtein_distance + find_similar_names — lines 7947-7977
# ---------------------------------------------------------------------------


class TestLevenshteinDistance:
    def test_identical_strings(self) -> None:
        from cja_auto_sdr.generator import levenshtein_distance

        assert levenshtein_distance("abc", "abc") == 0

    def test_empty_string(self) -> None:
        from cja_auto_sdr.generator import levenshtein_distance

        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self) -> None:
        from cja_auto_sdr.generator import levenshtein_distance

        assert levenshtein_distance("abc", "axc") == 1

    def test_different_strings(self) -> None:
        from cja_auto_sdr.generator import levenshtein_distance

        assert levenshtein_distance("kitten", "sitting") == 3


class TestFindSimilarNames:
    def test_find_close_match(self) -> None:
        from cja_auto_sdr.generator import find_similar_names

        names = ["Production DV", "Staging DV", "Dev DV"]
        similar = find_similar_names("Productin DV", names)
        assert len(similar) > 0
        assert similar[0][0] == "Production DV"


# ---------------------------------------------------------------------------
# 10. is_data_view_id — line 7934
# ---------------------------------------------------------------------------


class TestIsDataViewId:
    def test_valid_id(self) -> None:
        from cja_auto_sdr.generator import is_data_view_id

        assert is_data_view_id("dv_abc123") is True

    def test_name_not_id(self) -> None:
        from cja_auto_sdr.generator import is_data_view_id

        assert is_data_view_id("My Data View") is False

    def test_empty_string(self) -> None:
        from cja_auto_sdr.generator import is_data_view_id

        assert is_data_view_id("") is False
