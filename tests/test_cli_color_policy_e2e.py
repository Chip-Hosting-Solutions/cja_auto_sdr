"""End-to-end CLI tests for color policy behavior in validate-config mode."""

from __future__ import annotations

import os
import select
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANSI_ESCAPE_PREFIX = "\x1b["
CREDENTIAL_ENV_KEYS = ("ORG_ID", "CLIENT_ID", "SECRET", "SCOPES", "SANDBOX")


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in CREDENTIAL_ENV_KEYS:
        env.pop(key, None)
    env.pop("FORCE_COLOR", None)
    env.pop("NO_COLOR", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _validate_config_command(config_path: Path, *, no_color: bool = False) -> list[str]:
    command = [
        "uv",
        "run",
        "cja_auto_sdr",
        "--validate-config",
        "--config-file",
        str(config_path),
    ]
    if no_color:
        command.append("--no-color")
    return command


def _run_validate_config_redirected(
    tmp_path: Path,
    *,
    no_color: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    config_path = tmp_path / "missing_config.json"
    env = _base_env()
    if env_overrides:
        env.update(env_overrides)

    return subprocess.run(
        _validate_config_command(config_path, no_color=no_color),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_validate_config_redirected_default_has_no_ansi(tmp_path: Path) -> None:
    """Redirected output should disable ANSI by default."""
    result = _run_validate_config_redirected(tmp_path)

    assert result.returncode == 1
    assert ANSI_ESCAPE_PREFIX not in (result.stdout + result.stderr)


def test_validate_config_redirected_force_color_emits_ansi(tmp_path: Path) -> None:
    """FORCE_COLOR=1 should emit ANSI even when output is redirected."""
    result = _run_validate_config_redirected(tmp_path, env_overrides={"FORCE_COLOR": "1"})

    assert result.returncode == 1
    assert ANSI_ESCAPE_PREFIX in (result.stdout + result.stderr)


def test_validate_config_no_color_flag_overrides_force_color_in_redirected_mode(tmp_path: Path) -> None:
    """--no-color should override FORCE_COLOR in redirected mode."""
    result = _run_validate_config_redirected(
        tmp_path,
        no_color=True,
        env_overrides={"FORCE_COLOR": "1"},
    )

    assert result.returncode == 1
    assert ANSI_ESCAPE_PREFIX not in (result.stdout + result.stderr)


@pytest.mark.skipif(os.name == "nt", reason="Pseudo-TTY tests require POSIX")
def test_validate_config_tty_defaults_to_ansi(tmp_path: Path) -> None:
    """TTY output should enable ANSI by default when no color flags/env overrides are set."""
    import pty

    config_path = tmp_path / "missing_config.json"
    env = _base_env()

    master_fd, slave_fd = pty.openpty()
    try:
        process = subprocess.Popen(
            _validate_config_command(config_path),
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=REPO_ROOT,
            env=env,
        )
    finally:
        os.close(slave_fd)

    chunks: list[bytes] = []
    deadline = time.monotonic() + 30
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                process.kill()
                raise AssertionError("validate-config TTY test timed out")

            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                chunk = os.read(master_fd, 4096)
                if not chunk:
                    break
                chunks.append(chunk)
                continue

            if process.poll() is not None:
                break

        while True:
            chunk = os.read(master_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError:
        # PTY can raise after process exit/FD teardown; collected data is sufficient.
        pass
    finally:
        os.close(master_fd)

    return_code = process.wait(timeout=10)
    output = b"".join(chunks).decode("utf-8", errors="replace")

    assert return_code == 1
    assert ANSI_ESCAPE_PREFIX in output
