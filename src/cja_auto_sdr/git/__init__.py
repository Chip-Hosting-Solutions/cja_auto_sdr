"""Git module - Git integration for snapshots."""

from __future__ import annotations

from cja_auto_sdr.diff.git import (
    generate_git_commit_message,
    git_commit_snapshot,
    git_get_user_info,
    git_init_snapshot_repo,
    is_git_repository,
    save_git_friendly_snapshot,
)

__all__ = [
    "generate_git_commit_message",
    "git_commit_snapshot",
    "git_get_user_info",
    "git_init_snapshot_repo",
    "is_git_repository",
    "save_git_friendly_snapshot",
]


def __getattr__(name):
    """Lazy import from generator for backwards compatibility."""
    from cja_auto_sdr import generator

    if hasattr(generator, name):
        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
