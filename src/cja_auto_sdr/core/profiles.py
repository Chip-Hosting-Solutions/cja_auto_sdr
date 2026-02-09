"""Profile management helpers (current implementation lives in generator)."""

from __future__ import annotations

__all__ = [
    "add_profile_interactive",
    "get_cja_home",
    "get_profile_path",
    "get_profiles_dir",
    "list_profiles",
    "load_profile_credentials",
    "resolve_active_profile",
    "show_profile",
    "test_profile",
    "validate_profile_name",
]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
