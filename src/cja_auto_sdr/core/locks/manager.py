"""Lock manager orchestrating backend selection and lifecycle."""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from cja_auto_sdr.core.locks.backends import (
    FcntlFileLockBackend,
    LeaseFileLockBackend,
    LockBackend,
    LockBackendUnavailableError,
    LockHandle,
    LockInfo,
)

DEFAULT_LOCK_BACKEND_ENV = "CJA_LOCK_BACKEND"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def create_lock_backend(
    backend_name: str | None = None,
    *,
    logger: logging.Logger | None = None,
) -> LockBackend:
    """Create lock backend from explicit value or environment override."""
    log = logger or logging.getLogger(__name__)
    requested = (backend_name or os.environ.get(DEFAULT_LOCK_BACKEND_ENV, "auto")).strip().lower()

    if requested == "auto":
        if FcntlFileLockBackend.is_supported():
            return FcntlFileLockBackend()
        log.warning("fcntl locks unavailable; using lease lock backend")
        return LeaseFileLockBackend()

    if requested == "fcntl":
        if FcntlFileLockBackend.is_supported():
            return FcntlFileLockBackend()
        log.warning("Requested fcntl backend is unavailable; falling back to lease backend")
        return LeaseFileLockBackend()

    if requested == "lease":
        return LeaseFileLockBackend()

    log.warning("Unknown lock backend '%s'; falling back to auto selection", requested)
    return create_lock_backend("auto", logger=log)


class LockManager:
    """Backend-agnostic non-blocking lock manager."""

    def __init__(
        self,
        *,
        lock_path: Path,
        owner: str,
        stale_threshold_seconds: int = 3600,
        backend_name: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.lock_path = lock_path
        self.owner = owner
        self.stale_threshold_seconds = max(1, stale_threshold_seconds)
        self.logger = logger or logging.getLogger(__name__)
        self.backend = create_lock_backend(backend_name, logger=self.logger)

        self._handle: LockHandle | None = None
        self._lock_info: LockInfo | None = None

        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Attempt lock acquisition without blocking."""
        if self._handle is not None:
            return True

        handle: LockHandle | None
        try:
            handle = self.backend.acquire(self.lock_path, self.stale_threshold_seconds)
        except LockBackendUnavailableError as e:
            if isinstance(self.backend, FcntlFileLockBackend):
                self.logger.warning(
                    "fcntl backend unavailable for '%s' (%s); falling back to lease backend",
                    self.lock_path,
                    e,
                )
                self.backend = LeaseFileLockBackend()
                handle = self.backend.acquire(self.lock_path, self.stale_threshold_seconds)
            else:
                raise
        if handle is None:
            return False

        lock_info = LockInfo(
            lock_id=getattr(handle, "lock_id", str(uuid.uuid4())),
            pid=os.getpid(),
            host=socket.gethostname(),
            owner=self.owner,
            started_at=_utcnow_iso(),
            updated_at=_utcnow_iso(),
            backend=self.backend.name,
            version=1,
        )

        try:
            self.backend.write_info(handle, lock_info)
        except OSError:
            # If metadata write fails, release lock and report acquisition failure.
            self.backend.release(handle)
            self._cleanup_failed_metadata_write()
            return False

        self._handle = handle
        self._lock_info = lock_info
        self._start_heartbeat_if_needed()
        return True

    def release(self) -> None:
        """Release lock if held."""
        self._stop_heartbeat()
        if self._handle is None:
            return
        self.backend.release(self._handle)
        self._handle = None
        self._lock_info = None

    def read_info(self) -> dict | None:
        """Read lock metadata for diagnostics."""
        lock_info = self.backend.read_info(self.lock_path)
        if lock_info is None:
            return None
        return lock_info.to_dict()

    def _start_heartbeat_if_needed(self) -> None:
        if not self.backend.requires_heartbeat or self._handle is None or self._lock_info is None:
            return
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        interval_seconds = min(30.0, max(1.0, self.stale_threshold_seconds / 3))
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval_seconds,),
            daemon=True,
            name=f"lock-heartbeat-{self.lock_path.name}",
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_thread = None

    def _heartbeat_loop(self, interval_seconds: float) -> None:
        while not self._heartbeat_stop.wait(interval_seconds):
            if self._handle is None or self._lock_info is None:
                return

            self._lock_info = replace(self._lock_info, updated_at=_utcnow_iso())
            try:
                self.backend.write_info(self._handle, self._lock_info)
            except OSError:
                # Metadata heartbeat failure should not automatically drop lock ownership.
                self.logger.warning("Failed to refresh lock metadata heartbeat for %s", self.lock_path)

    def _cleanup_failed_metadata_write(self) -> None:
        """Best-effort cleanup to avoid false lockouts after write_info failures."""
        # Lease backend handles are path-bound via O_EXCL and release removes lock file
        # when ownership matches. For fcntl, a failed metadata write can leave a fresh,
        # unreadable file that contenders may interpret as active contention.
        with suppress(OSError):
            self.lock_path.unlink()
