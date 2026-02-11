"""Lock backend implementations.

Design principles:
- Ownership is defined by backend lock state (OS lock or lease ownership).
- Metadata is informational and must not be treated as lock truth.
- Metadata writes are best-effort observability and should never make
  a held lock appear free.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on non-POSIX only
    fcntl = None

_FLOCK_UNSUPPORTED_ERRNOS = {
    err_no
    for err_no in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOSYS", None),
    )
    if err_no is not None
}


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_all(fd: int, payload: bytes) -> None:
    """Write complete payload to fd, handling short writes."""
    total_written = 0
    while total_written < len(payload):
        written = os.write(fd, payload[total_written:])
        if written <= 0:
            raise OSError("short write while persisting lock metadata")
        total_written += written


def _write_info_fd(fd: int, info: LockInfo) -> None:
    payload = (json.dumps(info.to_dict(), sort_keys=True) + "\n").encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    _write_all(fd, payload)
    os.fsync(fd)


class LockBackendUnavailableError(OSError):
    """Raised when a backend exists but is unusable for the target lock path."""


@dataclass
class LockInfo:
    """Serializable lock metadata for diagnostics."""

    lock_id: str
    pid: int
    host: str
    owner: str
    started_at: str
    updated_at: str
    backend: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LockInfo | None:
        try:
            return cls(
                lock_id=str(data["lock_id"]),
                pid=int(data["pid"]),
                host=str(data["host"]),
                owner=str(data.get("owner", "")),
                started_at=str(data["started_at"]),
                updated_at=str(data.get("updated_at", data["started_at"])),
                backend=str(data.get("backend", "")),
                version=int(data.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError):
            return None


class LockHandle(Protocol):
    """Opaque backend-specific lock handle."""

    lock_path: Path
    lock_id: str


class LockBackend(Protocol):
    """Backend abstraction for lock acquisition and metadata operations."""

    name: str
    requires_heartbeat: bool

    def acquire(self, lock_path: Path, stale_threshold_seconds: int) -> LockHandle | None:
        """Try acquiring lock non-blocking. Returns handle if acquired."""

    def release(self, handle: LockHandle) -> None:
        """Release lock held by handle."""

    def write_info(self, handle: LockHandle, info: LockInfo) -> None:
        """Persist metadata for the currently held lock."""

    def read_info(self, lock_path: Path) -> LockInfo | None:
        """Read metadata for diagnostics, if available."""


@dataclass
class _FcntlLockHandle:
    lock_path: Path
    fd: int
    lock_id: str
    closed: bool = False


class FcntlFileLockBackend:
    """POSIX advisory locking backend backed by `fcntl.flock`."""

    name = "fcntl"
    requires_heartbeat = False

    @staticmethod
    def is_supported() -> bool:
        return fcntl is not None

    def acquire(self, lock_path: Path, stale_threshold_seconds: int) -> _FcntlLockHandle | None:
        del stale_threshold_seconds  # Not needed with OS-managed lock lifetime.
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return None

        try:
            assert fcntl is not None  # For type checkers.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return None
            if e.errno in _FLOCK_UNSUPPORTED_ERRNOS:
                raise LockBackendUnavailableError(
                    f"flock is unsupported for lock path '{lock_path}'"
                ) from e
            raise

        return _FcntlLockHandle(lock_path=lock_path, fd=fd, lock_id=str(uuid.uuid4()))

    def release(self, handle: _FcntlLockHandle) -> None:
        if handle.closed:
            return
        try:
            assert fcntl is not None  # For type checkers.
            fcntl.flock(handle.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                os.close(handle.fd)
            handle.closed = True

    def write_info(self, handle: _FcntlLockHandle, info: LockInfo) -> None:
        _write_info_fd(handle.fd, info)

    def read_info(self, lock_path: Path) -> LockInfo | None:
        if not lock_path.exists():
            return None
        try:
            with open(lock_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(data, dict):
            return None
        return LockInfo.from_dict(data)


@dataclass
class _LeaseLockHandle:
    lock_path: Path
    fd: int
    lock_id: str
    closed: bool = False


class LeaseFileLockBackend:
    """File-lease fallback backend for environments without `fcntl`.

    The lock file itself acts as lease marker. Stale/corrupt lease files
    are reclaimed during acquisition attempts.
    """

    name = "lease"
    requires_heartbeat = True
    acquire_attempts = 3
    unreadable_retry_attempts = 10
    unreadable_retry_sleep_seconds = 0.05

    def acquire(self, lock_path: Path, stale_threshold_seconds: int) -> _LeaseLockHandle | None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_id = str(uuid.uuid4())

        for _ in range(self.acquire_attempts):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                try:
                    bootstrap_info = self._build_bootstrap_info(lock_id)
                    _write_info_fd(fd, bootstrap_info)
                except OSError:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    self._safe_unlink(lock_path)
                    return None
                return _LeaseLockHandle(lock_path=lock_path, fd=fd, lock_id=lock_id)
            except FileExistsError:
                lock_info = self._read_info_with_retries(lock_path)
                if lock_info is None:
                    # Corrupt/unreadable lease metadata: reclaim after bounded retries.
                    if not self._safe_unlink(lock_path):
                        return None
                    continue

                if self._is_stale(lock_info, stale_threshold_seconds):
                    if not self._safe_unlink(lock_path):
                        return None
                    continue

                return None
            except OSError:
                return None

        return None

    @staticmethod
    def _build_bootstrap_info(lock_id: str) -> LockInfo:
        now = _utcnow_iso()
        return LockInfo(
            lock_id=lock_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            owner="",
            started_at=now,
            updated_at=now,
            backend="lease",
            version=1,
        )

    def _read_info_with_retries(self, lock_path: Path) -> LockInfo | None:
        for attempt in range(self.unreadable_retry_attempts + 1):
            info = self.read_info(lock_path)
            if info is not None:
                return info
            if not lock_path.exists():
                return None
            if attempt < self.unreadable_retry_attempts:
                time.sleep(self.unreadable_retry_sleep_seconds)
        return None

    def release(self, handle: _LeaseLockHandle) -> None:
        if handle.closed:
            return
        try:
            lock_info = self.read_info(handle.lock_path)
            if lock_info is None:
                return
            if lock_info.lock_id != handle.lock_id:
                return
            self._safe_unlink(handle.lock_path)
        finally:
            with contextlib.suppress(OSError):
                os.close(handle.fd)
            handle.closed = True

    def write_info(self, handle: _LeaseLockHandle, info: LockInfo) -> None:
        if handle.closed:
            raise OSError("lock handle is closed")
        _write_info_fd(handle.fd, info)

    def read_info(self, lock_path: Path) -> LockInfo | None:
        if not lock_path.exists():
            return None
        try:
            with open(lock_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(data, dict):
            return None
        return LockInfo.from_dict(data)

    @staticmethod
    def _safe_unlink(lock_path: Path) -> bool:
        try:
            lock_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _is_stale(self, info: LockInfo, stale_threshold_seconds: int) -> bool:
        if info.host == socket.gethostname() and not self._is_process_running(info.pid):
            return True

        reference = self._parse_iso(info.updated_at) or self._parse_iso(info.started_at)
        if reference is None:
            return True

        now = datetime.now(UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)

        return (now - reference).total_seconds() > max(1, stale_threshold_seconds)

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as e:
            return e.errno == errno.EPERM
