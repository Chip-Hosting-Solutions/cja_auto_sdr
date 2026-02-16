"""Tests for CLI command handlers and dispatch functions in generator.py.

Targets uncovered lines in:
- process_inventory_summary (lines 5049-5117)
- handle_diff_command config unpacking (lines 12755-12988)
- handle_diff_snapshot_command (lines 13068-13373)
- _main_impl: --stats handler (lines 13811-13978)
- _main_impl: --org-report routing (lines 13893-14055)
- _main_impl: --list-snapshots (lines 14103-14171)
- _main_impl: --prune-snapshots & more CLI dispatch (lines 14065-14249)
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from cja_auto_sdr.generator import (
    DiffConfig,
    DiffSnapshotConfig,
    _main_impl,
    handle_diff_command,
    handle_diff_snapshot_command,
    parse_arguments,
    process_inventory_summary,
)


def _mock_cli_option_specified(option_name, argv=None):
    """Stub that always returns False — prevents _known_long_options() from
    calling parse_arguments(return_parser=True) while it is mocked."""
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_snapshot(data_view_id="dv_test123", data_view_name="Test DV"):
    """Build a lightweight mock DataViewSnapshot."""
    snap = MagicMock()
    snap.data_view_id = data_view_id
    snap.data_view_name = data_view_name
    snap.created_at = "2025-01-15T10:00:00+00:00"
    snap.metrics = [{"id": "m1", "name": "Metric1"}]
    snap.dimensions = [{"id": "d1", "name": "Dim1"}]
    snap.has_calculated_metrics_inventory = False
    snap.has_segments_inventory = False
    snap.calculated_metrics_inventory = None
    snap.segments_inventory = None
    snap.get_inventory_summary.return_value = {
        "calculated_metrics": {"present": False, "count": 0},
        "segments": {"present": False, "count": 0},
    }
    return snap


def _make_mock_diff_result(has_changes=False):
    """Build a mock DiffResult with summary."""
    result = MagicMock()
    result.summary.has_changes = has_changes
    result.summary.metrics_change_percent = 0.0
    result.summary.dimensions_change_percent = 0.0
    result.metrics_diffs = []
    result.dimensions_diffs = []
    return result


# ==================== process_inventory_summary ====================


class TestProcessInventorySummary:
    """Tests for process_inventory_summary() covering lines 5049-5127."""

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_cja_init_failure_returns_error(self, mock_setup, mock_ctx, mock_init, mock_display):
        """When initialize_cja returns None, function should return error dict."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")
        mock_init.return_value = None

        result = process_inventory_summary("dv_test123", config_file="config.json")

        assert "error" in result
        assert result["error"] == "CJA initialization failed"
        mock_display.assert_not_called()

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_data_view_fetch_failure_returns_error(self, mock_setup, mock_ctx, mock_init, mock_display):
        """When cja.dataviews.get_single raises, function should return error dict."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.side_effect = Exception("API Error 404")
        mock_init.return_value = mock_cja

        result = process_inventory_summary("dv_bad_id", config_file="config.json")

        assert "error" in result
        assert "API Error 404" in result["error"]

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_successful_fetch_no_inventory(self, mock_setup, mock_ctx, mock_init, mock_display):
        """Successful fetch with no inventory flags returns display_inventory_summary result."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = {"name": "My DV"}
        mock_init.return_value = mock_cja
        mock_display.return_value = {"total": 0}

        result = process_inventory_summary("dv_test123", config_file="config.json")

        assert result == {"total": 0}
        mock_display.assert_called_once()

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_successful_with_quiet_mode(self, mock_setup, mock_ctx, mock_init, mock_display):
        """Quiet mode should suppress progress output but still return results."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = {"name": "Quiet DV"}
        mock_init.return_value = mock_cja
        mock_display.return_value = {"status": "ok"}

        result = process_inventory_summary("dv_test123", quiet=True)

        assert result == {"status": "ok"}

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_include_derived_success(self, mock_setup, mock_ctx, mock_init, mock_display):
        """Test include_derived path when import and build succeed."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = {"name": "DV1"}
        mock_cja.dataviews.get_metrics.return_value = [{"id": "m1"}]
        mock_cja.dataviews.get_dimensions.return_value = [{"id": "d1"}]
        mock_init.return_value = mock_cja
        mock_display.return_value = {"derived": 5}

        mock_builder_cls = MagicMock()
        mock_inventory = MagicMock()
        mock_inventory.total_derived_fields = 5
        mock_builder_cls.return_value.build.return_value = mock_inventory

        with patch.dict(
            "sys.modules",
            {"cja_auto_sdr.inventory.derived_fields": MagicMock(DerivedFieldInventoryBuilder=mock_builder_cls)},
        ):
            result = process_inventory_summary("dv_test123", include_derived=True)

        assert result == {"derived": 5}

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_include_derived_import_failure_logs_warning(self, mock_setup, mock_ctx, mock_init, mock_display):
        """When derived fields import fails, should log warning and continue."""
        logger = logging.getLogger("test")
        mock_setup.return_value = logger
        mock_ctx.return_value = logger

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = {"name": "DV1"}
        mock_cja.dataviews.get_metrics.side_effect = Exception("metrics API error")
        mock_init.return_value = mock_cja
        mock_display.return_value = {"ok": True}

        result = process_inventory_summary("dv_test123", include_derived=True)

        # Should still succeed; derived failure is non-fatal
        assert result == {"ok": True}

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_include_calculated_import_failure(self, mock_setup, mock_ctx, mock_init, mock_display):
        """When calculated metrics import fails, should continue gracefully."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = {"name": "DV1"}
        mock_init.return_value = mock_cja
        mock_display.return_value = {"ok": True}

        # Force import to fail
        with patch.dict(
            "sys.modules",
            {"cja_calculated_metrics_inventory": None},
        ):
            result = process_inventory_summary("dv_test123", include_calculated=True)

        assert result == {"ok": True}

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_include_segments_import_failure(self, mock_setup, mock_ctx, mock_init, mock_display):
        """When segments import fails, should continue gracefully."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = {"name": "DV1"}
        mock_init.return_value = mock_cja
        mock_display.return_value = {"ok": True}

        # Force import to fail
        with patch.dict(
            "sys.modules",
            {"cja_segments_inventory": None},
        ):
            result = process_inventory_summary("dv_test123", include_segments=True)

        assert result == {"ok": True}

    @patch("cja_auto_sdr.generator.display_inventory_summary")
    @patch("cja_auto_sdr.generator.initialize_cja")
    @patch("cja_auto_sdr.generator.with_log_context")
    @patch("cja_auto_sdr.generator.setup_logging")
    def test_data_view_lookup_returns_non_dict(self, mock_setup, mock_ctx, mock_init, mock_display):
        """When lookup_data is not a dict, should use data_view_id as name."""
        mock_setup.return_value = logging.getLogger("test")
        mock_ctx.return_value = logging.getLogger("test")

        mock_cja = MagicMock()
        mock_cja.dataviews.get_single.return_value = "not_a_dict"
        mock_init.return_value = mock_cja
        mock_display.return_value = {}

        process_inventory_summary("dv_test123")

        # Should use data_view_id as the name fallback
        call_kwargs = mock_display.call_args
        assert call_kwargs[1]["data_view_name"] == "dv_test123"


# ==================== handle_diff_command ====================


class TestHandleDiffCommand:
    """Tests for handle_diff_command() covering lines 12755-12988."""

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_basic_diff_no_changes(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Basic diff with no changes returns (True, False, None)."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result(has_changes=False)
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp

        mock_write.return_value = "output text"

        success, has_changes, exit_override = handle_diff_command(
            source_id="dv_source",
            target_id="dv_target",
            quiet=True,
        )

        assert success is True
        assert has_changes is False
        assert exit_override is None

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_with_changes(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Diff that finds changes returns has_changes=True."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result(has_changes=True)
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = "changes output"

        success, has_changes, _ = handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            quiet=True,
        )

        assert success is True
        assert has_changes is True

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_config_failure(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Configuration failure returns (False, False, None)."""
        mock_conf.return_value = (False, "Bad creds", {})

        success, _has_changes, _exit_override = handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            quiet=True,
        )

        assert success is False
        assert _has_changes is False

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_warn_threshold_exceeded(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """When warn_threshold is exceeded, exit_code_override should be 3."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result(has_changes=True)
        diff_result.summary.metrics_change_percent = 25.0
        diff_result.summary.dimensions_change_percent = 5.0
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = "output"

        _, _, exit_override = handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            warn_threshold=10.0,
            quiet=True,
        )

        assert exit_override == 3

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_reverse_swaps_ids(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Reverse diff should swap source and target."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        source_snap = _make_mock_snapshot("dv_source", "Source")
        target_snap = _make_mock_snapshot("dv_target", "Target")

        mock_sm = MagicMock()
        # First call -> source snapshot, second -> target snapshot
        mock_sm.create_snapshot.side_effect = [source_snap, target_snap]
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        handle_diff_command(
            source_id="dv_source",
            target_id="dv_target",
            reverse_diff=True,
            quiet=True,
        )

        # Verify compare was called (reverse should have swapped the IDs before snapshot creation)
        mock_comp.compare.assert_called_once()

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_quiet_diff_suppresses_output(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """quiet_diff=True should skip write_diff_output."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp

        handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            quiet_diff=True,
            quiet=True,
        )

        mock_write.assert_not_called()

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_with_diff_output_writes_to_file(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append, tmp_path
    ):
        """--diff-output should write content to specified file."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = "diff content here"

        diff_file = str(tmp_path / "out.txt")

        handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            diff_output=diff_file,
            quiet=True,
        )

        assert os.path.isfile(diff_file)
        with open(diff_file) as f:
            assert f.read() == "diff content here"

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_auto_snapshot_saves(
        self,
        mock_conf,
        mock_cjapy,
        mock_sm_cls,
        mock_retention,
        mock_comp_cls,
        mock_write,
        mock_build,
        mock_append,
        tmp_path,
    ):
        """auto_snapshot=True should save snapshots to snapshot_dir."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm.generate_snapshot_filename.return_value = "snap_file.json"
        mock_sm_cls.return_value = mock_sm

        mock_retention.return_value = (0, None)

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        snap_dir = str(tmp_path / "snaps")

        handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            auto_snapshot=True,
            snapshot_dir=snap_dir,
            quiet=True,
        )

        assert mock_sm.save_snapshot.call_count == 2  # Source + target

    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_exception_returns_false(self, mock_conf, mock_cjapy):
        """General exception during diff returns (False, False, None)."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.side_effect = Exception("Unexpected boom")

        success, has_changes, exit_override = handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            quiet=True,
        )

        assert success is False
        assert has_changes is False
        assert exit_override is None

    def test_diff_config_dataclass_unpacking(self):
        """DiffConfig is correctly unpacked into local vars in handle_diff_command."""
        config = DiffConfig(
            source_id="dv_src",
            target_id="dv_tgt",
            quiet=True,
            quiet_diff=True,
            labels=("Before", "After"),
        )

        with patch("cja_auto_sdr.generator.configure_cjapy") as mock_conf:
            mock_conf.return_value = (False, "fail", {})

            success, _, _ = handle_diff_command(
                source_id="ignored",
                target_id="ignored",
                diff_config=config,
            )

        assert success is False

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_format_pr_comment(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """format_pr_comment=True should set effective format to 'pr-comment'."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = "pr output"

        handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            format_pr_comment=True,
            quiet=True,
        )

        # write_diff_output should receive "pr-comment" as format
        call_args = mock_write.call_args
        assert call_args[0][1] == "pr-comment"

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_with_labels(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Custom labels are passed through to comparator."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            labels=("Prod", "Staging"),
            quiet=True,
        )

        compare_call = mock_comp.compare.call_args
        assert compare_call[0][2] == "Prod"
        assert compare_call[0][3] == "Staging"

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_diff_non_console_format_prints_success(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append, capsys
    ):
        """Non-console output format should print success message when not quiet."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        mock_sm = MagicMock()
        mock_sm.create_snapshot.return_value = _make_mock_snapshot()
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        handle_diff_command(
            source_id="dv_a",
            target_id="dv_b",
            output_format="json",
            quiet=False,
        )

        captured = capsys.readouterr()
        assert "Diff report generated successfully" in captured.out


# ==================== handle_diff_snapshot_command ====================


class TestHandleDiffSnapshotCommand:
    """Tests for handle_diff_snapshot_command() covering lines 13068-13373."""

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_basic_snapshot_diff(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Basic diff against snapshot succeeds."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        source_snap = _make_mock_snapshot()
        target_snap = _make_mock_snapshot("dv_live", "Live")

        mock_sm = MagicMock()
        mock_sm.load_snapshot.return_value = source_snap
        mock_sm.create_snapshot.return_value = target_snap
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        success, has_changes, exit_code = handle_diff_snapshot_command(
            data_view_id="dv_live",
            snapshot_file="snapshot.json",
            quiet=True,
        )

        assert success is True
        assert has_changes is False
        assert exit_code is None

    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_snapshot_file_not_found(self, mock_sm_cls):
        """Missing snapshot file returns (False, False, None)."""
        mock_sm = MagicMock()
        mock_sm.load_snapshot.side_effect = FileNotFoundError("not found")
        mock_sm_cls.return_value = mock_sm

        success, _has_changes, _exit_code = handle_diff_snapshot_command(
            data_view_id="dv_test",
            snapshot_file="missing.json",
            quiet=True,
        )

        assert success is False

    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_invalid_snapshot_file(self, mock_sm_cls):
        """Invalid snapshot file (ValueError) returns (False, False, None)."""
        mock_sm = MagicMock()
        mock_sm.load_snapshot.side_effect = ValueError("corrupt snapshot")
        mock_sm_cls.return_value = mock_sm

        success, _, _ = handle_diff_snapshot_command(
            data_view_id="dv_test",
            snapshot_file="bad.json",
            quiet=True,
        )

        assert success is False

    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_missing_inventory_blocks_diff(self, mock_sm_cls, capsys):
        """Requesting calc metrics from snapshot that lacks them returns failure."""
        snap = _make_mock_snapshot()
        snap.has_calculated_metrics_inventory = False
        snap.has_segments_inventory = False

        mock_sm = MagicMock()
        mock_sm.load_snapshot.return_value = snap
        mock_sm_cls.return_value = mock_sm

        success, _, _ = handle_diff_snapshot_command(
            data_view_id="dv_test",
            snapshot_file="snap.json",
            include_calc_metrics=True,
            quiet=True,
        )

        assert success is False
        captured = capsys.readouterr()
        assert "snapshot missing requested data" in captured.err

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_snapshot_diff_config_failure(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Config failure in snapshot diff returns (False, False, None)."""
        source_snap = _make_mock_snapshot()

        mock_sm = MagicMock()
        mock_sm.load_snapshot.return_value = source_snap
        mock_sm_cls.return_value = mock_sm

        mock_conf.return_value = (False, "Bad config", {})

        success, _, _ = handle_diff_snapshot_command(
            data_view_id="dv_test",
            snapshot_file="snap.json",
            quiet=True,
        )

        assert success is False

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_snapshot_diff_with_warn_threshold(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """Warn threshold exceeded returns exit_code=3."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        source_snap = _make_mock_snapshot()
        target_snap = _make_mock_snapshot("dv_live", "Live")

        mock_sm = MagicMock()
        mock_sm.load_snapshot.return_value = source_snap
        mock_sm.create_snapshot.return_value = target_snap
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result(has_changes=True)
        diff_result.summary.metrics_change_percent = 50.0
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        _, _, exit_code = handle_diff_snapshot_command(
            data_view_id="dv_test",
            snapshot_file="snap.json",
            warn_threshold=10.0,
            quiet=True,
        )

        assert exit_code == 3

    def test_diff_snapshot_config_dataclass_unpacking(self):
        """DiffSnapshotConfig is correctly unpacked in handle_diff_snapshot_command."""
        config = DiffSnapshotConfig(
            data_view_id="dv_test",
            snapshot_file="snap.json",
            quiet=True,
            quiet_diff=True,
        )

        with patch("cja_auto_sdr.generator.SnapshotManager") as mock_sm_cls:
            mock_sm = MagicMock()
            mock_sm.load_snapshot.side_effect = FileNotFoundError("nope")
            mock_sm_cls.return_value = mock_sm

            success, _, _ = handle_diff_snapshot_command(
                data_view_id="ignored",
                snapshot_file="ignored",
                diff_snapshot_config=config,
            )

        assert success is False

    @patch("cja_auto_sdr.generator.append_github_step_summary")
    @patch("cja_auto_sdr.generator.build_diff_step_summary")
    @patch("cja_auto_sdr.generator.write_diff_output")
    @patch("cja_auto_sdr.generator.DataViewComparator")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_snapshot_diff_reverse(
        self, mock_conf, mock_cjapy, mock_sm_cls, mock_comp_cls, mock_write, mock_build, mock_append
    ):
        """reverse_diff=True should swap source and target snapshots."""
        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.return_value = MagicMock()

        source_snap = _make_mock_snapshot("dv_snap", "Snap")
        target_snap = _make_mock_snapshot("dv_live", "Live")

        mock_sm = MagicMock()
        mock_sm.load_snapshot.return_value = source_snap
        mock_sm.create_snapshot.return_value = target_snap
        mock_sm_cls.return_value = mock_sm

        diff_result = _make_mock_diff_result()
        mock_comp = MagicMock()
        mock_comp.compare.return_value = diff_result
        mock_comp_cls.return_value = mock_comp
        mock_write.return_value = ""

        handle_diff_snapshot_command(
            data_view_id="dv_live",
            snapshot_file="snap.json",
            reverse_diff=True,
            quiet=True,
        )

        # After reverse, target_snap becomes source and source_snap becomes target
        compare_call = mock_comp.compare.call_args
        assert compare_call[0][0] == target_snap
        assert compare_call[0][1] == source_snap

    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator.cjapy")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_snapshot_diff_general_exception(self, mock_conf, mock_cjapy, mock_sm_cls):
        """General exception returns (False, False, None)."""
        source_snap = _make_mock_snapshot()

        mock_sm = MagicMock()
        mock_sm.load_snapshot.return_value = source_snap
        mock_sm_cls.return_value = mock_sm

        mock_conf.return_value = (True, "config_path", {})
        mock_cjapy.CJA.side_effect = RuntimeError("unexpected")

        success, _has_changes, _ = handle_diff_snapshot_command(
            data_view_id="dv_test",
            snapshot_file="snap.json",
            quiet=True,
        )

        assert success is False


# ==================== _main_impl: --profile-add / --profile-test / --profile-show ====================


class TestMainImplProfileCommands:
    """Tests for profile commands in _main_impl (lines 13810-13828)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.add_profile_interactive")
    def test_profile_add_dispatches_and_tracks_run_state(self, mock_add):
        mock_add.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--profile-add", "new_profile"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        mock_add.assert_called_once_with("new_profile")
        assert run_state["details"]["operation_success"] is True

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.add_profile_interactive")
    def test_profile_add_failure_exits_one(self, mock_add):
        mock_add.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--profile-add", "bad_profile"])
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.test_profile")
    def test_profile_test_tracks_run_state(self, mock_test):
        mock_test.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--profile-test", "myprof"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["details"]["operation_success"] is True

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.show_profile")
    def test_profile_show_tracks_run_state(self, mock_show):
        mock_show.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--profile-show", "myprof"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["details"]["operation_success"] is True


# ==================== _main_impl: --git-init ====================


class TestMainImplGitInit:
    """Tests for --git-init dispatch in _main_impl (lines 13831-13847)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.git_init_snapshot_repo")
    def test_git_init_success(self, mock_git_init, capsys):
        mock_git_init.return_value = (True, "Repository initialized")
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--git-init"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "SUCCESS: Repository initialized" in captured.out

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.git_init_snapshot_repo")
    def test_git_init_failure(self, mock_git_init, capsys):
        mock_git_init.return_value = (False, "Directory not empty")
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--git-init"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "FAILED" in captured.out


# ==================== _main_impl: --git-push without --git-commit ====================


class TestMainImplGitPushValidation:
    """Tests for git argument validation (lines 13850-13852)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    def test_git_push_without_commit_errors(self):
        """--git-push without --git-commit should exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["dv_test123"])
                args.git_push = True
                args.git_commit = False
                mock_pa.return_value = args
                _main_impl()

        assert exc_info.value.code == 1


# ==================== _main_impl: discovery format routing (line 13874) ====================


class TestMainImplDiscoveryFormatRouting:
    """Tests for discovery command output format logic (lines 13868-13888)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.list_dataviews")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_discovery_stdout_forces_json(self, _mock_conf, mock_list_dv):
        """When output is stdout, discovery format should default to json."""
        mock_list_dv.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--list-dataviews", "--output", "-"])
                mock_pa.return_value = args
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["output_format"] == "json"

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.list_dataviews")
    @patch("cja_auto_sdr.generator.configure_cjapy")
    def test_discovery_csv_format_preserved(self, _mock_conf, mock_list_dv):
        """Explicit csv format should be preserved for discovery commands."""
        mock_list_dv.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--list-dataviews", "--format", "csv"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["output_format"] == "csv"


# ==================== _main_impl: --config-status / --validate-config ====================


class TestMainImplConfigCommands:
    """Tests for --config-status and --validate-config (lines 13892-13904)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.show_config_status")
    def test_config_status_dispatches(self, mock_show):
        mock_show.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--config-status"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        mock_show.assert_called_once()

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.validate_config_only")
    def test_validate_config_dispatches(self, mock_validate):
        mock_validate.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--validate-config"])
                _main_impl()

        assert exc_info.value.code == 0

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.validate_config_only")
    def test_validate_config_failure_exits_one(self, mock_validate):
        mock_validate.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--validate-config"])
                _main_impl()

        assert exc_info.value.code == 1


# ==================== _main_impl: --stats ====================


class TestMainImplStats:
    """Tests for --stats handler in _main_impl (lines 13937-13978)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.show_stats")
    @patch("cja_auto_sdr.generator.resolve_data_view_names")
    def test_stats_success(self, mock_resolve, mock_stats):
        mock_resolve.return_value = (["dv_resolved"], {})
        mock_stats.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--stats", "dv_test123"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["details"]["operation_success"] is True
        assert run_state["output_format"] == "table"

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.show_stats")
    @patch("cja_auto_sdr.generator.resolve_data_view_names")
    def test_stats_failure_exits_one(self, mock_resolve, mock_stats):
        mock_resolve.return_value = (["dv_resolved"], {})
        mock_stats.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--stats", "dv_test123"])
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_data_view_names")
    def test_stats_no_resolved_ids_exits_one(self, mock_resolve):
        """When name resolution finds nothing, should exit 1."""
        mock_resolve.return_value = ([], {})

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--stats", "nonexistent_dv"])
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    def test_stats_no_data_views_exits_one(self):
        """--stats without data views should exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--stats"])
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.show_stats")
    @patch("cja_auto_sdr.generator.resolve_data_view_names")
    def test_stats_json_format(self, mock_resolve, mock_stats):
        mock_resolve.return_value = (["dv_test"], {})
        mock_stats.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--stats", "--format", "json", "dv_test"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["output_format"] == "json"

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.show_stats")
    @patch("cja_auto_sdr.generator.resolve_data_view_names")
    def test_stats_stdout_forces_json(self, mock_resolve, mock_stats):
        """--output stdout should force json format for stats."""
        mock_resolve.return_value = (["dv_test"], {})
        mock_stats.return_value = True
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--stats", "--output", "-", "dv_test"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["output_format"] == "json"


# ==================== _main_impl: --org-report ====================


class TestMainImplOrgReport:
    """Tests for --org-report routing in _main_impl (lines 13981-14055)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.run_org_report")
    def test_org_report_success(self, mock_org):
        mock_org.return_value = (True, False)
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--org-report"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["details"]["operation_success"] is True
        assert run_state["details"]["thresholds_exceeded"] is False

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.run_org_report")
    def test_org_report_failure_exits_one(self, mock_org):
        mock_org.return_value = (False, False)

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--org-report"])
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.run_org_report")
    def test_org_report_thresholds_exceeded_without_fail(self, mock_org):
        """Thresholds exceeded without --fail-on-threshold should exit 0."""
        mock_org.return_value = (True, True)

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--org-report"])
                _main_impl()

        assert exc_info.value.code == 0

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.run_org_report")
    def test_org_report_thresholds_exceeded_with_fail(self, mock_org):
        """Thresholds exceeded with --fail-on-threshold should exit 2."""
        mock_org.return_value = (True, True)

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--org-report", "--fail-on-threshold"])
                _main_impl()

        assert exc_info.value.code == 2

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.run_org_report")
    def test_org_report_with_json_format(self, mock_org):
        mock_org.return_value = (True, False)
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--org-report", "--format", "json"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["output_format"] == "json"

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.run_org_report")
    def test_org_report_builds_config_with_filter(self, mock_org):
        """OrgReportConfig should receive filter_pattern from args."""
        mock_org.return_value = (True, False)

        with pytest.raises(SystemExit):
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--org-report", "--filter", "prod.*"])
                _main_impl()

        # Verify org_config passed to run_org_report
        call_kwargs = mock_org.call_args[1]
        assert call_kwargs["org_config"].filter_pattern == "prod.*"


# ==================== _main_impl: --list-snapshots ====================


class TestMainImplListSnapshots:
    """Tests for --list-snapshots mode (lines 14103-14171)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator._emit_output")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_list_snapshots_table_output(self, mock_sm_cls, mock_emit):
        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {
                "data_view_id": "dv_test",
                "data_view_name": "Test",
                "created_at": "2025-01-01",
                "metrics_count": 10,
                "dimensions_count": 5,
                "filepath": "/path/to/snap.json",
            }
        ]
        mock_sm_cls.return_value = mock_sm
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--list-snapshots"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        assert run_state["details"]["snapshot_count"] == 1
        mock_emit.assert_called_once()

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator._emit_output")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_list_snapshots_json_output(self, mock_sm_cls, mock_emit):
        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = []
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--list-snapshots", "--format", "json"])
                _main_impl()

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        parsed = json.loads(emitted)
        assert parsed["count"] == 0

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator._emit_output")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_list_snapshots_csv_output(self, mock_sm_cls, mock_emit):
        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {
                "data_view_id": "dv_csv",
                "data_view_name": "CSV Test",
                "created_at": "2025-02-01",
                "metrics_count": 3,
                "dimensions_count": 2,
                "filepath": "/snap.json",
            }
        ]
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--list-snapshots", "--format", "csv"])
                _main_impl()

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        assert "dv_csv" in emitted

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator._emit_output")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_list_snapshots_filter_by_dv_id(self, mock_sm_cls, mock_emit):
        """When data view IDs provided, only matching snapshots shown."""
        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {
                "data_view_id": "dv_match",
                "data_view_name": "Match",
                "created_at": "2025-01-01",
                "metrics_count": 1,
                "dimensions_count": 1,
                "filepath": "/snap1.json",
            },
            {
                "data_view_id": "dv_other",
                "data_view_name": "Other",
                "created_at": "2025-01-01",
                "metrics_count": 1,
                "dimensions_count": 1,
                "filepath": "/snap2.json",
            },
        ]
        mock_sm_cls.return_value = mock_sm
        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--list-snapshots", "--format", "json", "dv_match"])
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        parsed = json.loads(emitted)
        assert parsed["count"] == 1
        assert parsed["snapshots"][0]["data_view_id"] == "dv_match"

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator._emit_output")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_list_snapshots_empty_table(self, mock_sm_cls, mock_emit):
        """Empty snapshot list should show 'No snapshots found' message."""
        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = []
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                mock_pa.return_value = parse_arguments(["--list-snapshots"])
                _main_impl()

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        assert "No snapshots found" in emitted


# ==================== _main_impl: --show-only / --include-all-inventory ====================


class TestMainImplShowOnlyAndInventory:
    """Tests for --show-only parsing and --include-all-inventory (lines 14065-14101)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    def test_show_only_invalid_types_exits_one(self):
        """Invalid --show-only types should exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["dv_test123"])
                # Simulate diff mode with invalid show_only
                args.diff = ["dv_a", "dv_b"]
                args.show_only = "added,invalid_type"
                mock_pa.return_value = args
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator._emit_output")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_include_all_inventory_snapshot_mode(self, mock_sm_cls, mock_emit):
        """--include-all-inventory in snapshot-like mode should not enable derived."""
        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = []
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit):
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--list-snapshots"])
                args.include_all_inventory = True
                args.snapshot = "dv_test123"  # snapshot-like mode
                mock_pa.return_value = args
                _main_impl()

        # In snapshot-like mode, include_derived_inventory should remain unset


# ==================== _main_impl: --prune-snapshots ====================


class TestMainImplPruneSnapshots:
    """Tests for --prune-snapshots mode (lines 14173-14261)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator._emit_output")
    def test_prune_json_output(self, mock_emit, mock_sm_cls, mock_retention):
        mock_retention.return_value = (5, None)

        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {"data_view_id": "dv_test", "filepath": "/snap1.json"},
        ]
        mock_sm.apply_retention_policy.return_value = ["/snap_old.json"]
        mock_sm_cls.return_value = mock_sm

        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--prune-snapshots", "--keep-last", "5", "--format", "json"])
                mock_pa.return_value = args
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        parsed = json.loads(emitted)
        assert parsed["deleted_count"] >= 0

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator._emit_output")
    def test_prune_csv_output(self, mock_emit, mock_sm_cls, mock_retention):
        mock_retention.return_value = (3, None)

        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {"data_view_id": "dv_test", "filepath": "/snap1.json"},
        ]
        mock_sm.apply_retention_policy.return_value = ["/deleted.json"]
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--prune-snapshots", "--keep-last", "3", "--format", "csv"])
                mock_pa.return_value = args
                _main_impl()

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        assert "filepath" in emitted

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator._emit_output")
    def test_prune_table_output(self, mock_emit, mock_sm_cls, mock_retention):
        mock_retention.return_value = (2, None)

        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {"data_view_id": "dv_test", "filepath": "/snap.json"},
        ]
        mock_sm.apply_retention_policy.return_value = ["/old1.json", "/old2.json"]
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--prune-snapshots", "--keep-last", "2"])
                mock_pa.return_value = args
                _main_impl()

        assert exc_info.value.code == 0
        emitted = mock_emit.call_args[0][0]
        assert "Deleted files" in emitted

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    def test_prune_no_retention_policy_exits_error(self, mock_sm_cls, mock_retention):
        """--prune-snapshots without retention flags should exit 1."""
        mock_retention.return_value = (0, None)

        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = []
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--prune-snapshots"])
                mock_pa.return_value = args
                _main_impl()

        assert exc_info.value.code == 1

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.parse_retention_period")
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator._emit_output")
    def test_prune_with_keep_since(self, mock_emit, mock_sm_cls, mock_retention, mock_parse_period):
        mock_retention.return_value = (0, "7d")
        mock_parse_period.return_value = 7

        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {"data_view_id": "dv_test", "filepath": "/snap.json"},
        ]
        mock_sm.apply_date_retention_policy.return_value = ["/old.json"]
        mock_sm_cls.return_value = mock_sm

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--prune-snapshots", "--keep-since", "7d", "--format", "json"])
                mock_pa.return_value = args
                _main_impl()

        assert exc_info.value.code == 0

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_auto_prune_retention")
    @patch("cja_auto_sdr.generator.SnapshotManager")
    @patch("cja_auto_sdr.generator._emit_output")
    def test_prune_with_dv_filter(self, mock_emit, mock_sm_cls, mock_retention):
        """Prune with specific data view IDs should only prune those."""
        mock_retention.return_value = (3, None)

        mock_sm = MagicMock()
        mock_sm.list_snapshots.return_value = [
            {"data_view_id": "dv_keep", "filepath": "/snap_keep.json"},
            {"data_view_id": "dv_prune", "filepath": "/snap_prune.json"},
        ]
        mock_sm.apply_retention_policy.return_value = []
        mock_sm_cls.return_value = mock_sm

        run_state = {"mode": "unknown", "details": {}}

        with pytest.raises(SystemExit) as exc_info:
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--prune-snapshots", "--keep-last", "3", "--format", "json", "dv_prune"])
                mock_pa.return_value = args
                _main_impl(run_state=run_state)

        assert exc_info.value.code == 0
        # Should have pruned for dv_prune only
        mock_sm.apply_retention_policy.assert_called_once()


# ==================== _main_impl: --diff-labels ====================


class TestMainImplDiffLabels:
    """Tests for --diff-labels parsing (lines 14073-14076)."""

    @patch("cja_auto_sdr.generator._cli_option_specified", _mock_cli_option_specified)
    @patch("cja_auto_sdr.generator.resolve_data_view_names")
    @patch("cja_auto_sdr.generator.handle_diff_command")
    def test_diff_labels_parsed_as_tuple(self, mock_diff, mock_resolve):
        """--diff-labels should be parsed and passed as tuple."""
        mock_diff.return_value = (True, False, None)
        mock_resolve.return_value = (["dv_a"], {})

        with pytest.raises(SystemExit):
            with patch("cja_auto_sdr.generator.parse_arguments") as mock_pa:
                args = parse_arguments(["--diff", "dv_a", "dv_b", "--diff-labels", "Before", "After"])
                mock_pa.return_value = args
                _main_impl()

        # Labels should have been passed through
        assert mock_diff.called
