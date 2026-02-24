"""CLI smoke tests for core command groups.

These tests intentionally cover top-level dispatch for core modes while mocking
heavy integrations, so regressions in argument routing are caught early.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import pytest

from cja_auto_sdr.generator import _main_impl


def _run_main_impl(argv_tail: list[str]) -> int:
    """Execute _main_impl with argv and return the SystemExit code."""
    with patch.object(sys, "argv", ["cja_auto_sdr", *argv_tail]):
        with pytest.raises(SystemExit) as exc_info:
            _main_impl()
    return int(exc_info.value.code)


def test_smoke_sdr_mode_dry_run_dispatches_successfully() -> None:
    """SDR mode should route through dry-run path and exit successfully."""
    with (
        patch("cja_auto_sdr.generator.resolve_data_view_names", return_value=(["dv_123"], {})),
        patch("cja_auto_sdr.generator.setup_logging", return_value=logging.getLogger("test")),
        patch("cja_auto_sdr.generator.run_dry_run", return_value=True) as mock_run_dry_run,
    ):
        exit_code = _run_main_impl(["dv_123", "--dry-run"])

    assert exit_code == 0
    mock_run_dry_run.assert_called_once()


def test_smoke_diff_mode_dispatches_successfully() -> None:
    """Diff mode should route to handle_diff_command and return expected exit code."""
    with (
        patch(
            "cja_auto_sdr.generator.resolve_data_view_names",
            side_effect=[(["dv_source"], {}), (["dv_target"], {})],
        ),
        patch("cja_auto_sdr.generator.handle_diff_command", return_value=(True, False, None)) as mock_diff,
    ):
        exit_code = _run_main_impl(["--diff", "dv_source", "dv_target"])

    assert exit_code == 0
    mock_diff.assert_called_once()


def test_smoke_discovery_mode_dispatches_successfully() -> None:
    """Discovery mode should route to list_dataviews."""
    with patch("cja_auto_sdr.generator.list_dataviews", return_value=True) as mock_list:
        exit_code = _run_main_impl(["--list-dataviews", "--format", "json"])

    assert exit_code == 0
    mock_list.assert_called_once()


def test_smoke_org_report_mode_dispatches_successfully() -> None:
    """Org-report mode should route to run_org_report."""
    with patch("cja_auto_sdr.generator.run_org_report", return_value=(True, False)) as mock_org_report:
        exit_code = _run_main_impl(["--org-report", "--format", "json", "--quiet"])

    assert exit_code == 0
    mock_org_report.assert_called_once()


def test_smoke_profile_mode_dispatches_successfully() -> None:
    """Profile command group should route to list_profiles."""
    with patch("cja_auto_sdr.generator.list_profiles", return_value=True) as mock_list_profiles:
        exit_code = _run_main_impl(["--profile-list", "--format", "json"])

    assert exit_code == 0
    mock_list_profiles.assert_called_once()


def test_fast_path_version_does_not_fall_through_to_generator_main() -> None:
    """Fast-path --version should not dispatch to generator.main()."""
    from cja_auto_sdr import __main__ as entrypoint

    with (
        patch.object(sys, "argv", ["cja_auto_sdr", "--version"]),
        patch("cja_auto_sdr.__main__._print_version") as mock_print_version,
        patch("cja_auto_sdr.generator.main") as mock_generator_main,
    ):
        with pytest.raises(SystemExit) as exc_info:
            entrypoint.main()

    assert int(exc_info.value.code) == 0
    mock_print_version.assert_called_once()
    mock_generator_main.assert_not_called()


def test_fast_path_exit_codes_does_not_fall_through_to_generator_main() -> None:
    """Fast-path --exit-codes should not dispatch to generator.main()."""
    from cja_auto_sdr import __main__ as entrypoint

    with (
        patch.object(sys, "argv", ["cja_auto_sdr", "--exit-codes"]),
        patch("cja_auto_sdr.__main__._print_exit_codes") as mock_print_exit_codes,
        patch("cja_auto_sdr.generator.main") as mock_generator_main,
    ):
        with pytest.raises(SystemExit) as exc_info:
            entrypoint.main()

    assert int(exc_info.value.code) == 0
    mock_print_exit_codes.assert_called_once()
    mock_generator_main.assert_not_called()
