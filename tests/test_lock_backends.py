"""Tests for lock backend and manager behavior."""

from __future__ import annotations

import errno
import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import cja_auto_sdr.core.locks.backends as backends_module
from cja_auto_sdr.core.locks.backends import (
    FcntlFileLockBackend,
    LeaseFileLockBackend,
    LockInfo,
)
from cja_auto_sdr.core.locks.manager import LockManager


def _build_lock_info(
    lock_id: str,
    owner: str = "test-owner",
    *,
    pid: int | None = None,
    host: str | None = None,
    started_at: str | None = None,
    updated_at: str | None = None,
    backend: str = "lease",
) -> LockInfo:
    now = datetime.now(UTC).isoformat()
    return LockInfo(
        lock_id=lock_id,
        pid=os.getpid() if pid is None else pid,
        host=socket.gethostname() if host is None else host,
        owner=owner,
        started_at=now if started_at is None else started_at,
        updated_at=now if updated_at is None else updated_at,
        backend=backend,
        version=1,
    )


def _write_lock_info(lock_path: Path, lock_id: str) -> None:
    info = _build_lock_info(lock_id)
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


def test_lock_manager_falls_back_to_lease_when_flock_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    if backends_module.fcntl is None:
        pytest.skip("fcntl not available on this platform")

    def _unsupported_flock(fd: int, operation: int) -> None:
        del fd, operation
        raise OSError(errno.EOPNOTSUPP, "flock unsupported")

    monkeypatch.setattr(backends_module.fcntl, "flock", _unsupported_flock)

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lock.lock"
        manager = LockManager(
            lock_path=lock_path,
            owner="test-owner",
            stale_threshold_seconds=10,
            backend_name="auto",
        )
        assert isinstance(manager.backend, FcntlFileLockBackend)

        assert manager.acquire() is True
        assert isinstance(manager.backend, LeaseFileLockBackend)
        info = manager.read_info()
        assert info is not None
        assert info["backend"] == "lease"
        manager.release()


def test_lock_manager_surfaces_non_contention_flock_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    if backends_module.fcntl is None:
        pytest.skip("fcntl not available on this platform")

    def _broken_flock(fd: int, operation: int) -> None:
        del fd, operation
        raise OSError(errno.EIO, "flock I/O error")

    monkeypatch.setattr(backends_module.fcntl, "flock", _broken_flock)

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lock.lock"
        manager = LockManager(
            lock_path=lock_path,
            owner="test-owner",
            stale_threshold_seconds=10,
            backend_name="auto",
        )
        with pytest.raises(OSError, match="flock I/O error"):
            manager.acquire()
        assert isinstance(manager.backend, FcntlFileLockBackend)


def test_fcntl_backend_blocks_when_active_lease_lock_exists() -> None:
    if backends_module.fcntl is None:
        pytest.skip("fcntl not available on this platform")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lock.lock"
        lease_manager = LockManager(
            lock_path=lock_path,
            owner="lease-owner",
            stale_threshold_seconds=3600,
            backend_name="lease",
        )
        assert lease_manager.acquire() is True

        fcntl_manager = LockManager(
            lock_path=lock_path,
            owner="fcntl-owner",
            stale_threshold_seconds=3600,
            backend_name="fcntl",
        )
        try:
            assert fcntl_manager.acquire() is False
        finally:
            lease_manager.release()

        assert fcntl_manager.acquire() is True
        fcntl_manager.release()


def test_fcntl_backend_blocks_when_lease_lock_created_after_path_check() -> None:
    if backends_module.fcntl is None:
        pytest.skip("fcntl not available on this platform")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lock.lock"
        lease_manager = LockManager(
            lock_path=lock_path,
            owner="lease-owner",
            stale_threshold_seconds=3600,
            backend_name="lease",
        )
        assert lease_manager.acquire() is True

        # Simulate the race condition by forcing the fcntl open path to report
        # "not created exclusively", which requires metadata validation.
        preopened_fd = os.open(lock_path, os.O_RDWR)
        try:
            with patch.object(FcntlFileLockBackend, "_open_lock_file", return_value=(preopened_fd, False)):
                fcntl_manager = LockManager(
                    lock_path=lock_path,
                    owner="fcntl-owner",
                    stale_threshold_seconds=3600,
                    backend_name="fcntl",
                )
                assert fcntl_manager.acquire() is False
        finally:
            try:
                os.close(preopened_fd)
            except OSError:
                pass

        lease_manager.release()


def test_lease_backend_does_not_expire_local_live_pid_even_if_old() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lease.lock"
        old = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
        info = _build_lock_info(
            "live-local-pid",
            owner="live-owner",
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=old,
            updated_at=old,
            backend="fcntl",
        )
        lock_path.write_text(json.dumps(info.to_dict()) + "\n", encoding="utf-8")

        backend = LeaseFileLockBackend()
        handle = backend.acquire(lock_path, stale_threshold_seconds=1)
        assert handle is None

        current = backend.read_info(lock_path)
        assert current is not None
        assert current.lock_id == "live-local-pid"


def test_metadata_write_failure_does_not_leave_fresh_false_lockout() -> None:
    if backends_module.fcntl is None:
        pytest.skip("fcntl not available on this platform")

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lock.lock"
        manager = LockManager(
            lock_path=lock_path,
            owner="test-owner",
            stale_threshold_seconds=3600,
            backend_name="fcntl",
        )

        write_calls = {"count": 0}
        original_write = FcntlFileLockBackend.write_info

        def _fail_first_write(self: FcntlFileLockBackend, handle, info):  # type: ignore[no-untyped-def]
            write_calls["count"] += 1
            if write_calls["count"] == 1:
                raise OSError(errno.EIO, "simulated write failure")
            return original_write(self, handle, info)

        with patch.object(FcntlFileLockBackend, "write_info", _fail_first_write):
            assert manager.acquire() is False

        # Cleanup should prevent stale fresh-file lockout; second attempt should succeed.
        assert manager.acquire() is True
        manager.release()


def test_lease_backend_stale_holder_cannot_overwrite_new_owner_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "lease.lock"
        backend = LeaseFileLockBackend()

        stale_handle = backend.acquire(lock_path, stale_threshold_seconds=3600)
        assert stale_handle is not None
        stale_info = _build_lock_info(stale_handle.lock_id, owner="stale-owner")
        backend.write_info(stale_handle, stale_info)

        # Simulate lease takeover: stale lock path is removed and recreated by a new owner.
        assert backend._safe_unlink(lock_path) is True

        active_handle = backend.acquire(lock_path, stale_threshold_seconds=3600)
        assert active_handle is not None
        active_info = _build_lock_info(active_handle.lock_id, owner="active-owner")
        backend.write_info(active_handle, active_info)

        # Late heartbeat/write from stale holder must not clobber active owner metadata.
        backend.write_info(stale_handle, replace(stale_info, updated_at=datetime.now(UTC).isoformat()))
        current = backend.read_info(lock_path)
        assert current is not None
        assert current.lock_id == active_handle.lock_id
        assert current.owner == "active-owner"

        backend.release(active_handle)
        backend.release(stale_handle)
