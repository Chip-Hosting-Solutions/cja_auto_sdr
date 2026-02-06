"""
CJA Auto SDR - Customer Journey Analytics Solution Design Reference Generator

A tool for generating Solution Design Reference (SDR) documentation
from Adobe Customer Journey Analytics data views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["__version__", "main"]

if TYPE_CHECKING:
    from cja_auto_sdr.generator import __version__, main


def __getattr__(name: str) -> Any:
    if name in __all__:
        from cja_auto_sdr import generator

        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
