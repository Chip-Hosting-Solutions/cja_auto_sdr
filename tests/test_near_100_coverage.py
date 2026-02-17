"""Tests for near-100% coverage modules.

Covers remaining gaps in:
- api/tuning.py: sample window truncation (line 93)
- core/logging.py: duplicate handler dedup in flush (line 408)
- diff/git.py: push success path (line 287)
"""

import logging
import subprocess
from unittest.mock import MagicMock, patch

from cja_auto_sdr.generator import (
    APITuningConfig,
    APIWorkerTuner,
    flush_logging_handlers,
    git_commit_snapshot,
)


class TestTuningSampleWindowTruncation:
    """Test that response_times list is truncated when exceeding sample_window."""

    def test_truncates_response_times_beyond_sample_window(self):
        """When more responses are recorded than sample_window, the list is truncated."""
        config = APITuningConfig(
            sample_window=3,
            cooldown_seconds=0,
            scale_up_threshold_ms=100,
            scale_down_threshold_ms=2000,
            min_workers=1,
            max_workers=10,
        )
        tuner = APIWorkerTuner(config=config, initial_workers=3)

        # Use response times in the "no adjustment" range (between thresholds)
        # so the tuner doesn't clear the window after adjusting.
        # Record more responses than sample_window to trigger truncation.
        for _ in range(5):
            tuner.record_response_time(500.0)  # 100 < 500 < 2000: no adjustment

        # Internal list should be at most sample_window length
        assert len(tuner._response_times) <= config.sample_window


class TestFlushLoggingHandlersDuplicateDedup:
    """Test that flush_logging_handlers deduplicates handlers seen via propagation."""

    def test_deduplicates_shared_handler(self):
        """When a handler appears in both child and parent, it should only flush once."""
        shared_handler = logging.StreamHandler()
        shared_handler.flush = MagicMock()

        # Create parent and child loggers sharing the same handler
        parent = logging.getLogger("test_flush_dedup_parent")
        parent.handlers = [shared_handler]
        parent.propagate = False

        child = logging.getLogger("test_flush_dedup_parent.child")
        child.handlers = [shared_handler]  # same handler instance
        child.propagate = True

        flush_logging_handlers(child)

        # Handler should be flushed exactly once despite appearing in both loggers
        shared_handler.flush.assert_called_once()

    def test_flushes_distinct_handlers(self):
        """Distinct handlers should each be flushed."""
        handler_a = logging.StreamHandler()
        handler_a.flush = MagicMock()
        handler_b = logging.StreamHandler()
        handler_b.flush = MagicMock()

        parent = logging.getLogger("test_flush_distinct_parent")
        parent.handlers = [handler_a]
        parent.propagate = False

        child = logging.getLogger("test_flush_distinct_parent.child")
        child.handlers = [handler_b]
        child.propagate = True

        flush_logging_handlers(child)

        handler_a.flush.assert_called_once()
        handler_b.flush.assert_called_once()


class TestGitPushSuccess:
    """Test git_commit_snapshot push success path."""

    def test_push_success_logs_and_returns_sha(self, tmp_path):
        """When push=True and push succeeds, returns True with commit SHA."""
        # Initialize a real git repo
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp_path), capture_output=True)

        # Create a snapshot-style directory
        dv_dir = tmp_path / "Test_DV_dv_push"
        dv_dir.mkdir()
        (dv_dir / "metadata.json").write_text('{"test": "push"}')

        # Mock only the push subprocess call (let git init/add/commit work normally)
        original_run = subprocess.run

        def mock_run(cmd, *args, **kwargs):
            if cmd == ["git", "push"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return original_run(cmd, *args, **kwargs)

        with patch("subprocess.run", side_effect=mock_run):
            success, result = git_commit_snapshot(
                snapshot_dir=tmp_path,
                data_view_id="dv_push",
                data_view_name="Push Test",
                metrics_count=1,
                dimensions_count=1,
                push=True,
            )

        assert success is True
        assert len(result) == 8  # Short SHA, no push failure annotation
