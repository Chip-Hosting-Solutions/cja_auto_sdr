# ruff: noqa: F822
"""CLI argument parsing (current implementation lives in generator)."""

__all__ = ["parse_arguments"]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
