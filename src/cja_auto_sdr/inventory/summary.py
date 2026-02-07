"""Inventory summary mode (current implementation lives in generator)."""

__all__ = ["display_inventory_summary"]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
