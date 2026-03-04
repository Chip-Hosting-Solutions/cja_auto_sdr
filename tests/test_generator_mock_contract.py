"""Compatibility contract for generator symbols used by tests and mocks.

This file intentionally freezes a focused set of symbols that test suites patch
via ``cja_auto_sdr.generator.<symbol>``. Keeping this contract stable reduces
breakage risk during internal modularization.
"""

from cja_auto_sdr import generator

KEY_GENERATOR_MOCK_SYMBOLS: tuple[str, ...] = (
    "aggregate_quality_issues",
    "append_github_step_summary",
    "build_org_report_json_data",
    "apply_excel_formatting",
    "build_diff_step_summary",
    "build_org_step_summary",
    "build_quality_step_summary",
    "cjapy",
    "configure_cjapy",
    "DataQualityChecker",
    "DataViewComparator",
    "describe_dataview",
    "ExcelFormatCache",
    "generate_sample_config",
    "get_cached_data_views",
    "handle_compare_snapshots_command",
    "handle_diff_command",
    "handle_diff_snapshot_command",
    "handle_snapshot_command",
    "import_profile_non_interactive",
    "initialize_cja",
    "list_connections",
    "list_datasets",
    "list_dataviews",
    "list_profiles",
    "parse_arguments",
    "ParallelAPIFetcher",
    "process_single_dataview",
    "prompt_for_selection",
    "resolve_active_profile",
    "resolve_data_view_names",
    "run_org_report",
    "setup_logging",
    "show_config_status",
    "show_profile",
    "show_stats",
    "SnapshotManager",
    "test_profile",
    "validate_config_only",
    "validate_data_view",
    "write_csv_output",
    "write_diff_console_output",
    "write_diff_csv_output",
    "write_diff_excel_output",
    "write_diff_html_output",
    "write_diff_json_output",
    "write_diff_markdown_output",
    "write_diff_output",
    "write_diff_pr_comment_output",
    "write_excel_output",
    "write_html_output",
    "write_json_output",
    "write_markdown_output",
    "write_org_report_csv",
    "write_org_report_excel",
    "write_org_report_html",
    "write_org_report_json",
    "write_org_report_markdown",
    "_cli_option_specified",
    "_cli_option_value",
    "_emit_output",
    "_validate_org_report_output_request",
)


def test_generator_mock_contract_symbols_exist() -> None:
    """All frozen symbols must remain importable from cja_auto_sdr.generator."""
    missing = [symbol for symbol in KEY_GENERATOR_MOCK_SYMBOLS if not hasattr(generator, symbol)]
    assert not missing, f"Missing generator symbols used by tests/mocks: {missing}"


def test_generator_mock_contract_symbols_are_not_none() -> None:
    """Frozen symbols should resolve to concrete objects for patching."""
    unresolved = [symbol for symbol in KEY_GENERATOR_MOCK_SYMBOLS if getattr(generator, symbol) is None]
    assert not unresolved, f"Generator symbols unexpectedly resolved to None: {unresolved}"
