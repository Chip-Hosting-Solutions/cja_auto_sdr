"""Small helper for lazy attribute resolution to avoid import cycles."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable


def make_getattr(
    module_name: str,
    export_names: Iterable[str],
    *,
    target_module: str | None = None,
    mapping: dict[str, str] | None = None,
) -> Callable[[str], object]:
    """
    Create a __getattr__ that lazily resolves exports from target modules.

    Args:
        module_name: Name of the current module (for error messages).
        export_names: Iterable of export names.
        target_module: Single module path that contains all exports.
        mapping: Explicit mapping of name -> module path (overrides target_module).
    """
    export_set = set(export_names)
    if mapping is None:
        if not target_module:
            raise ValueError("target_module or mapping is required")
        mapping = {name: target_module for name in export_set}

    def __getattr__(name: str) -> object:
        if name in mapping:
            module = importlib.import_module(mapping[name])
            return getattr(module, name)
        if name in export_set:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r} (lazy target missing)")
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
