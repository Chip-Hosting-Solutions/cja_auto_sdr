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


from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, __all__, target_module="cja_auto_sdr.generator")
