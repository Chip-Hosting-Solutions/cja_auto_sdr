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
    "WRITER_REGISTRY",
    "ExcelFormatCache",
    "OutputWriter",
    "apply_excel_formatting",
    "get_writer",
    "write_csv_output",
    "write_excel_output",
    "write_html_output",
    "write_json_output",
    "write_markdown_output",
]


_LAZY_EXPORTS = [
    "format_output",
    "generate_output_files",
    "OUTPUT_FORMATS",
]

from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(__name__, _LAZY_EXPORTS, target_module="cja_auto_sdr.generator")
