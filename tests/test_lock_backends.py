"""Tests for lock backend and manager behavior."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from cja_auto_sdr.core.locks.backends import LeaseFileLockBackend, LockInfo
from cja_auto_sdr.core.locks.manager import LockManager


def _write_lock_info(lock_path: Path, lock_id: str) -> None:
    now = datetime.now(UTC).isoformat()
    info = LockInfo(
        lock_id=lock_id,
        pid=os.getpid(),
        host="test-host",
        owner="test-owner",
        started_at=now,
        updated_at=now,
        backend="lease",
        version=1,
    )
    lock_path.write_text(json.dumps(info.to_dict()) + "\n", encoding="utf-8")


def test_lease_backend_acquire_writes_bootstrap_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lease.lock"
        backend = LeaseFileLockBackend()

        handle = backend.acquire(lock_path, stale_threshold_seconds=3600)

        assert handle is not None
        info = backend.read_info(lock_path)
        assert info is not None
        assert info.lock_id == handle.lock_id
        assert info.pid == os.getpid()
        assert info.backend == "lease"

        backend.release(handle)
        assert not lock_path.exists()


def test_lease_backend_waits_for_transient_unreadable_lock_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lease.lock"
        lock_path.write_text("", encoding="utf-8")
        backend = LeaseFileLockBackend()

        lock_id = "delayed-lock-id"

        def _delayed_write() -> None:
            time.sleep(backend.unreadable_retry_sleep_seconds * 2)
            _write_lock_info(lock_path, lock_id)

        writer = threading.Thread(target=_delayed_write, daemon=True)
        writer.start()
        try:
            handle = backend.acquire(lock_path, stale_threshold_seconds=3600)
        finally:
            writer.join(timeout=2)

        # Acquisition should fail because metadata became readable and fresh.
        assert handle is None
        info = backend.read_info(lock_path)
        assert info is not None
        assert info.lock_id == lock_id


def test_lease_backend_reclaims_persistently_unreadable_lock_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lease.lock"
        lock_path.write_text("{bad json", encoding="utf-8")
        backend = LeaseFileLockBackend()

        start = time.monotonic()
        handle = backend.acquire(lock_path, stale_threshold_seconds=3600)
        elapsed = time.monotonic() - start

        assert handle is not None
        # Recovery should be bounded by retry window, not stale threshold.
        assert elapsed < 2.0
        backend.release(handle)


def test_lock_manager_heartbeat_updates_lease_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lease.lock"
        manager = LockManager(
            lock_path=lock_path,
            owner="test-owner",
            stale_threshold_seconds=1,
            backend_name="lease",
        )

        assert manager.acquire() is True
        first_info = manager.read_info()
        assert first_info is not None
        first_updated_at = first_info["updated_at"]

        refreshed = False
        deadline = time.time() + 3
        while time.time() < deadline:
            current_info = manager.read_info()
            if current_info is not None and current_info["updated_at"] != first_updated_at:
                refreshed = True
                break
            time.sleep(0.1)

        manager.release()
        assert refreshed is True
