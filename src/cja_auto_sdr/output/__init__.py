"""Output module - Output format writers and registries."""

from cja_auto_sdr.output.excel import ExcelFormatCache, apply_excel_formatting
from cja_auto_sdr.output.protocols import OutputWriter
from cja_auto_sdr.output.registry import WRITER_REGISTRY, get_writer
from cja_auto_sdr.output.writers import (
    write_csv_output,
    write_excel_output,
    write_html_output,
    write_json_output,
    write_markdown_output,
)

__all__ = [
    "ExcelFormatCache",
    "OutputWriter",
    "WRITER_REGISTRY",
    "apply_excel_formatting",
    "get_writer",
    "write_csv_output",
    "write_excel_output",
    "write_html_output",
    "write_json_output",
    "write_markdown_output",
]


def __getattr__(name):
    """Lazy import from generator for backwards compatibility."""
    from cja_auto_sdr import generator
    if hasattr(generator, name):
        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
