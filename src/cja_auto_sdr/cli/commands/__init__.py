"""CLI command handlers (current implementation lives in generator)."""

from __future__ import annotations

__all__ = [
    "generate_sample_config",
    "list_dataviews",
    "show_config_status",
    "validate_config_only",
]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
