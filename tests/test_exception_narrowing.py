"""Tests verifying command-boundary exception handling in generator.py.

These tests enforce a strict CLI boundary contract:
  - recoverable API/bootstrap failures (including ConfigurationError) are
    surfaced as controlled user-facing failures.
  - non-recoverable failures outside the configured tuples still propagate.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from cja_auto_sdr import generator
from cja_auto_sdr.core.exceptions import ConfigurationError
from cja_auto_sdr.generator import (
    _require_accessible_dataview,
    validate_data_view,
)
from cja_auto_sdr.generator import test_profile as run_test_profile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger() -> logging.Logger:
    """Return a quiet logger for test use."""
    logger = logging.getLogger("test_exception_narrowing")
    logger.setLevel(logging.CRITICAL)
    return logger


# ===========================================================================
# Group B: test_profile  (catches RECOVERABLE_CONFIG_API_EXCEPTIONS)
# ===========================================================================


class TestTestProfileExceptionNarrowing:
    """Verify test_profile catches recoverable API/bootstrap failures and propagates others."""

    def _run_with_side_effect(self, exc, tmp_path, capsys):
        """Helper: invoke test_profile with cjapy.importConfigFile raising *exc*."""
        profile_dir = tmp_path / "orgs" / "narrowing"
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.json").write_text('{"client_id":"x","secret":"s","org_id":"o@AdobeOrg"}')

        with (
            patch("cja_auto_sdr.generator.get_profile_path", return_value=profile_dir),
            patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
            patch("cja_auto_sdr.generator.ConfigValidator") as mock_validator,
        ):
            mock_cjapy.importConfigFile.side_effect = exc
            mock_validator.validate_all.return_value = []
            return run_test_profile("narrowing")

    def test_os_error_caught(self, tmp_path, capsys):
        """OSError (in RECOVERABLE_API_EXCEPTIONS) is caught -> returns False."""
        result = self._run_with_side_effect(OSError("network down"), tmp_path, capsys)
        assert result is False
        assert "FAILED" in capsys.readouterr().err

    def test_value_error_caught(self, tmp_path, capsys):
        """ValueError (in RECOVERABLE_API_EXCEPTIONS) is caught -> returns False."""
        result = self._run_with_side_effect(ValueError("bad token"), tmp_path, capsys)
        assert result is False

    def test_configuration_error_caught(self, tmp_path, capsys):
        """ConfigurationError should return a controlled failure, not a traceback."""
        result = self._run_with_side_effect(ConfigurationError("invalid profile credentials"), tmp_path, capsys)
        assert result is False
        assert "FAILED" in capsys.readouterr().err

    def test_runtime_error_propagates(self, tmp_path, capsys):
        """RuntimeError is NOT in RECOVERABLE_API_EXCEPTIONS -> propagates."""
        with pytest.raises(RuntimeError, match="should escape"):
            self._run_with_side_effect(RuntimeError("should escape"), tmp_path, capsys)

    def test_system_error_propagates(self, tmp_path, capsys):
        """SystemError is NOT in RECOVERABLE_API_EXCEPTIONS -> propagates."""
        with pytest.raises(SystemError, match="should escape"):
            self._run_with_side_effect(SystemError("should escape"), tmp_path, capsys)


# ===========================================================================
# Group B: validate_data_view  (outer guard catches RECOVERABLE_CONFIG_API_EXCEPTIONS)
# ===========================================================================


class TestValidateDataViewExceptionNarrowing:
    """Verify validate_data_view catches recoverable failures and propagates others."""

    def test_type_error_caught(self):
        """TypeError (in RECOVERABLE_API_EXCEPTIONS) is caught -> returns False."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = TypeError("bad payload")
        assert validate_data_view(mock_cja, "dv_test", _make_logger()) is False

    def test_key_error_caught(self):
        """KeyError (in RECOVERABLE_API_EXCEPTIONS) is caught -> returns False."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = KeyError("missing key")
        assert validate_data_view(mock_cja, "dv_test", _make_logger()) is False

    def test_runtime_error_propagates(self):
        """RuntimeError is NOT in RECOVERABLE_API_EXCEPTIONS -> propagates."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = RuntimeError("should escape")
        with pytest.raises(RuntimeError, match="should escape"):
            validate_data_view(mock_cja, "dv_test", _make_logger())

    def test_system_error_propagates(self):
        """SystemError is NOT in RECOVERABLE_API_EXCEPTIONS -> propagates."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = SystemError("should escape")
        with pytest.raises(SystemError, match="should escape"):
            validate_data_view(mock_cja, "dv_test", _make_logger())

    def test_configuration_error_while_listing_available_views_caught(self):
        """ConfigurationError from fallback list call should not escape."""
        mock_cja = MagicMock()
        mock_cja.getDataView.return_value = None
        mock_cja.getDataViews.side_effect = ConfigurationError("token bootstrap failed")
        assert validate_data_view(mock_cja, "dv_test", _make_logger()) is False


# ===========================================================================
# Group B: _require_accessible_dataview  (now catches RECOVERABLE_API_EXCEPTIONS)
# ===========================================================================


class TestRequireAccessibleDataviewExceptionNarrowing:
    """Verify _require_accessible_dataview catches RECOVERABLE_API_EXCEPTIONS and re-raises."""

    def test_os_error_reraises(self):
        """OSError (in RECOVERABLE_API_EXCEPTIONS) is caught and re-raised."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = OSError("connection reset")
        # The handler re-raises if _is_inaccessible_dataview_lookup_error returns False
        with pytest.raises(OSError, match="connection reset"):
            _require_accessible_dataview(mock_cja, "dv_test")

    def test_value_error_reraises(self):
        """ValueError (in RECOVERABLE_API_EXCEPTIONS) is caught and re-raised."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = ValueError("bad id")
        with pytest.raises(ValueError, match="bad id"):
            _require_accessible_dataview(mock_cja, "dv_test")

    def test_runtime_error_propagates(self):
        """RuntimeError is NOT in RECOVERABLE_API_EXCEPTIONS -> propagates uncaught."""
        mock_cja = MagicMock()
        mock_cja.getDataView.side_effect = RuntimeError("should escape")
        with pytest.raises(RuntimeError, match="should escape"):
            _require_accessible_dataview(mock_cja, "dv_test")


# ===========================================================================
# Group A: process_inventory_summary  (fallback -> (RuntimeError, AttributeError))
# ===========================================================================


class TestProcessInventorySummaryExceptionNarrowing:
    """Verify process_inventory_summary fallback catches (RuntimeError, AttributeError)."""

    def _run_with_side_effect(self, exc):
        """Mock CJA init + dataviews.get_single raising *exc*."""
        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.side_effect = exc

        with (
            patch("cja_auto_sdr.generator.initialize_cja", return_value=mock_cja),
            patch("cja_auto_sdr.generator.setup_logging", return_value=_make_logger()),
            patch("cja_auto_sdr.generator.with_log_context", return_value=_make_logger()),
        ):
            return generator.process_inventory_summary(
                data_view_id="dv_test",
                config_file="config.json",
            )

    def test_runtime_error_caught(self):
        """RuntimeError (in fallback tuple) is caught -> returns error dict."""
        result = self._run_with_side_effect(RuntimeError("cjapy internal"))
        assert "error" in result

    def test_attribute_error_caught(self):
        """AttributeError (in fallback tuple) is caught -> returns error dict."""
        result = self._run_with_side_effect(AttributeError("no attribute"))
        # AttributeError is in BOTH tuples — first handler catches it
        assert "error" in result

    def test_configuration_error_caught(self):
        """ConfigurationError should be returned as structured summary error payload."""
        result = self._run_with_side_effect(ConfigurationError("invalid auth bootstrap"))
        assert "error" in result
        assert "invalid auth bootstrap" in result["error"]

    def test_system_error_propagates(self):
        """SystemError is NOT in either handler -> propagates."""
        with pytest.raises(SystemError, match="should escape"):
            self._run_with_side_effect(SystemError("should escape"))


# ===========================================================================
# Group A: validate_config_only  (fallback -> (RuntimeError, AttributeError))
# ===========================================================================


class TestValidateConfigOnlyExceptionNarrowing:
    """Verify validate_config_only fallback catches (RuntimeError, AttributeError)."""

    def _run_with_cja_side_effect(self, exc):
        """Mock credential resolution + cjapy.CJA() raising *exc*."""
        fake_creds = {"org_id": "X@AdobeOrg", "client_id": "cid", "secret": "sec"}
        with (
            patch("cja_auto_sdr.generator.load_credentials_from_env", return_value=fake_creds),
            patch("cja_auto_sdr.generator.validate_env_credentials", return_value=True),
            patch("cja_auto_sdr.generator._config_from_env"),
            patch("cja_auto_sdr.generator.cjapy") as mock_cjapy,
            patch("cja_auto_sdr.generator._check_output_dir_access", return_value=(True, ".", None, None)),
        ):
            mock_cjapy.CJA.side_effect = exc
            return generator.validate_config_only(config_file="config.json")

    def test_runtime_error_caught(self):
        """RuntimeError (in fallback tuple) is caught -> returns False."""
        result = self._run_with_cja_side_effect(RuntimeError("cjapy init failed"))
        assert result is False

    def test_attribute_error_caught(self):
        """AttributeError (in fallback tuple) is caught -> returns False."""
        result = self._run_with_cja_side_effect(AttributeError("missing method"))
        # AttributeError is in BOTH RECOVERABLE_API and fallback; first handler catches
        assert result is False

    def test_configuration_error_caught(self):
        """ConfigurationError should return False (controlled validation failure)."""
        result = self._run_with_cja_side_effect(ConfigurationError("invalid oauth configuration"))
        assert result is False

    def test_system_error_propagates(self):
        """SystemError is NOT in either handler -> propagates."""
        with pytest.raises(SystemError, match="should escape"):
            self._run_with_cja_side_effect(SystemError("should escape"))


class TestResolveDataViewNamesExceptionBoundary:
    """Verify resolve_data_view_names handles recoverable bootstrap failures."""

    def test_configuration_error_returns_connectivity_diagnostics(self):
        """ConfigurationError should produce controlled connectivity diagnostics."""
        with patch("cja_auto_sdr.generator.configure_cjapy", side_effect=ConfigurationError("bad org credentials")):
            ids, name_map, diagnostics = generator.resolve_data_view_names(
                ["My DV"],
                include_diagnostics=True,
            )

        assert ids == []
        assert name_map == {}
        assert diagnostics.error_type == "connectivity_error"
        assert "bad org credentials" in diagnostics.error_message


# ===========================================================================
# Group A: _count_component_items_for_fetch_spec_with_retry (best-effort)
# ===========================================================================


class TestComponentCountRetryExceptionBoundary:
    """Verify retry-based component counting degrades recoverable optional failures."""

    def _run_count_with_retry_side_effect(self, exc):
        """Run component count helper with a patched retry call that raises *exc*."""
        with patch("cja_auto_sdr.generator.make_api_call_with_retry", side_effect=exc):
            return generator._count_component_items_for_fetch_spec_with_retry(
                MagicMock(),
                "dv_test",
                generator._METRICS_COMPONENT_FETCH_SPEC,
                logger=_make_logger(),
            )

    def test_discovery_not_found_error_degrades_to_zero(self):
        """Missing component methods should degrade to zero instead of raising."""
        result = self._run_count_with_retry_side_effect(
            generator.DiscoveryNotFoundError("missing getMetrics"),
        )
        assert result == 0

    def test_runtime_error_degrades_to_zero(self):
        """Unexpected runtime errors in optional count collection should degrade to zero."""
        result = self._run_count_with_retry_side_effect(RuntimeError("cjapy internal bug"))
        assert result == 0

    def test_keyboard_interrupt_propagates(self):
        """BaseException subclasses must still propagate through the helper."""
        with pytest.raises(KeyboardInterrupt):
            self._run_count_with_retry_side_effect(KeyboardInterrupt())
