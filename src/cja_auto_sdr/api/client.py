"""CJA client initialization (current implementation lives in generator)."""

__all__ = ["configure_cjapy", "initialize_cja"]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
