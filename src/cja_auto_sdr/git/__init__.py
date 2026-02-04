"""Git module - Git integration for snapshots.

This module provides future-facing imports for the modular structure.
Currently, all symbols should be imported from cja_auto_sdr.generator
to avoid circular import issues.
"""

__all__ = []


def __getattr__(name):
    """Lazy import from generator for backwards compatibility."""
    from cja_auto_sdr import generator
    if hasattr(generator, name):
        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
