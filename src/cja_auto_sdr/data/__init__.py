"""Data module - Data structures and comparison logic.

This module provides future-facing imports for the modular structure.
Currently, all symbols should be imported from cja_auto_sdr.generator
to avoid circular import issues.

Example usage:
    # Current (use this)
    from cja_auto_sdr.generator import ProcessingResult, DiffSummary

    # Future (when extraction is complete)
    from cja_auto_sdr.data import ProcessingResult, DiffSummary
"""

# Deferred imports to avoid circular import with generator.py
# Will be populated when code is fully extracted

__all__ = []


def __getattr__(name):
    """Lazy import from generator for backwards compatibility."""
    from cja_auto_sdr import generator
    if hasattr(generator, name):
        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
