"""CLI module - Command-line interface components."""

from cja_auto_sdr.cli.commands import (
    generate_sample_config,
    list_dataviews,
    show_config_status,
    validate_config_only,
)
from cja_auto_sdr.cli.interactive import (
    interactive_select_dataviews,
    interactive_wizard,
    prompt_for_selection,
)
from cja_auto_sdr.cli.main import main
from cja_auto_sdr.cli.parser import parse_arguments

__all__ = [
    "generate_sample_config",
    "interactive_select_dataviews",
    "interactive_wizard",
    "list_dataviews",
    "main",
    "parse_arguments",
    "prompt_for_selection",
    "show_config_status",
    "validate_config_only",
]


def __getattr__(name):
    """Lazy import from generator for backwards compatibility."""
    from cja_auto_sdr import generator
    if hasattr(generator, name):
        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
