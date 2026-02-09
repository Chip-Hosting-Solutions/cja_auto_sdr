"""Dry-run mode (current implementation lives in generator)."""

from __future__ import annotations

__all__ = ["run_dry_run"]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
