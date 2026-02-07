"""Interactive CLI flows (current implementation lives in generator)."""

__all__ = [
    "interactive_select_dataviews",
    "interactive_wizard",
    "prompt_for_selection",
]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
