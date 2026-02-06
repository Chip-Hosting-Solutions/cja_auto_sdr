"""
Test backwards compatibility for cja_auto_sdr.generator module.

This test ensures all 133+ public symbols remain importable from
cja_auto_sdr.generator after the modular refactor (v3.2.0).

All existing code that imports from cja_auto_sdr.generator must continue
to work without modification.
"""

import pytest


class TestBackwardsCompatibilityImports:
    """Test that all public symbols can be imported from cja_auto_sdr.generator."""

    # ===========================================
    # Version
    # ===========================================

    def test_version_importable(self):
        """__version__ must be importable."""
        from cja_auto_sdr.generator import __version__
        assert isinstance(__version__, str)
        assert __version__  # Not empty

    # ===========================================
    # Exception Classes
    # ===========================================

    def test_exception_classes_importable(self):
        """All exception classes must be importable."""
        from cja_auto_sdr.generator import (
            CJASDRError,
            ConfigurationError,
            APIError,
            ValidationError,
            OutputError,
            ProfileError,
            ProfileNotFoundError,
            ProfileConfigError,
            CredentialSourceError,
            CircuitBreakerOpen,
            RetryableHTTPError,
        )
        # Verify they are exception classes
        assert issubclass(CJASDRError, Exception)
        assert issubclass(ConfigurationError, CJASDRError)
        assert issubclass(APIError, CJASDRError)
        assert issubclass(ValidationError, CJASDRError)
        assert issubclass(OutputError, CJASDRError)
        assert issubclass(ProfileError, CJASDRError)
        assert issubclass(ProfileNotFoundError, ProfileError)
        assert issubclass(ProfileConfigError, ProfileError)
        assert issubclass(CredentialSourceError, CJASDRError)
        assert issubclass(CircuitBreakerOpen, Exception)
        assert issubclass(RetryableHTTPError, Exception)

    # ===========================================
    # Configuration Dataclasses
    # ===========================================

    def test_config_dataclasses_importable(self):
        """All configuration dataclasses must be importable."""
        from cja_auto_sdr.generator import (
            RetryConfig,
            CacheConfig,
            LogConfig,
            WorkerConfig,
            APITuningConfig,
            CircuitBreakerConfig,
            SDRConfig,
        )
        # Verify they can be instantiated with defaults
        assert RetryConfig()
        assert CacheConfig()
        assert LogConfig()
        assert WorkerConfig()
        assert APITuningConfig()
        assert CircuitBreakerConfig()
        assert SDRConfig()

    # ===========================================
    # Org-Wide Dataclasses
    # ===========================================

    def test_org_dataclasses_importable(self):
        """All org-wide analysis dataclasses must be importable."""
        from cja_auto_sdr.generator import (
            OrgReportConfig,
            ComponentInfo,
            DataViewSummary,
            SimilarityPair,
            DataViewCluster,
            ComponentDistribution,
            OrgReportResult,
            OrgReportComparison,
        )
        # Verify OrgReportConfig can be instantiated
        assert OrgReportConfig()

    # ===========================================
    # Enums
    # ===========================================

    def test_enums_importable(self):
        """All enums must be importable."""
        from cja_auto_sdr.generator import (
            CircuitState,
            ChangeType,
        )
        # Verify enum values
        assert CircuitState.CLOSED
        assert CircuitState.OPEN
        assert CircuitState.HALF_OPEN
        assert ChangeType.ADDED
        assert ChangeType.REMOVED
        assert ChangeType.MODIFIED
        assert ChangeType.UNCHANGED

    # ===========================================
    # Data Classes (Diff, Snapshot, Processing)
    # ===========================================

    def test_data_classes_importable(self):
        """All data-related classes must be importable."""
        from cja_auto_sdr.generator import (
            ProcessingResult,
            ComponentDiff,
            MetadataDiff,
            DiffSummary,
            InventoryItemDiff,
            DiffResult,
            DataViewSnapshot,
            SnapshotManager,
            DataViewComparator,
            WizardConfig,
        )
        # Verify they are classes
        assert ProcessingResult
        assert ComponentDiff
        assert MetadataDiff
        assert DiffSummary
        assert InventoryItemDiff
        assert DiffResult
        assert DataViewSnapshot
        assert SnapshotManager
        assert DataViewComparator
        assert WizardConfig

    # ===========================================
    # API Classes
    # ===========================================

    def test_api_classes_importable(self):
        """All API-related classes must be importable."""
        from cja_auto_sdr.generator import (
            CircuitBreaker,
            ParallelAPIFetcher,
            PerformanceTracker,
            APIWorkerTuner,
            ErrorMessageHelper,
            ConfigValidator,
            CredentialLoader,
            JsonFileCredentialLoader,
            DotenvCredentialLoader,
            EnvironmentCredentialLoader,
            CredentialResolver,
        )
        assert CircuitBreaker
        assert ParallelAPIFetcher
        assert PerformanceTracker
        assert APIWorkerTuner
        assert ErrorMessageHelper
        assert ConfigValidator
        assert CredentialLoader
        assert JsonFileCredentialLoader
        assert DotenvCredentialLoader
        assert EnvironmentCredentialLoader
        assert CredentialResolver

    # ===========================================
    # Cache Classes
    # ===========================================

    def test_cache_classes_importable(self):
        """All cache classes must be importable."""
        from cja_auto_sdr.generator import (
            ValidationCache,
            SharedValidationCache,
            OrgReportCache,
            DataViewCache,
            ExcelFormatCache,
        )
        assert ValidationCache
        assert SharedValidationCache
        assert OrgReportCache
        assert DataViewCache
        assert ExcelFormatCache

    # ===========================================
    # Output/Formatting Classes
    # ===========================================

    def test_output_classes_importable(self):
        """All output-related classes must be importable."""
        from cja_auto_sdr.generator import (
            ConsoleColors,
            ANSIColors,
            OutputWriter,
            JSONFormatter,
            DataQualityChecker,
            BatchProcessor,
            OrgComponentAnalyzer,
        )
        assert ConsoleColors
        assert ANSIColors
        assert OutputWriter
        assert JSONFormatter
        assert DataQualityChecker
        assert BatchProcessor
        assert OrgComponentAnalyzer

    # ===========================================
    # Modular Output API
    # ===========================================

    def test_output_module_importable(self):
        """The modular output package must import without errors."""
        from cja_auto_sdr import output

        assert output
        assert callable(output.write_excel_output)

    def test_output_registry_excel_writer_available(self):
        """Excel writer must be registered in the modular output API."""
        from cja_auto_sdr.output import get_writer

        excel_writer = get_writer("excel")
        xlsx_writer = get_writer("xlsx")

        assert callable(excel_writer)
        assert callable(xlsx_writer)

    # ===========================================
    # Constants
    # ===========================================

    def test_constants_importable(self):
        """All module-level constants must be importable."""
        from cja_auto_sdr.generator import (
            FORMAT_ALIASES,
            EXTENSION_TO_FORMAT,
            DEFAULT_API_FETCH_WORKERS,
            DEFAULT_VALIDATION_WORKERS,
            DEFAULT_BATCH_WORKERS,
            MAX_BATCH_WORKERS,
            AUTO_WORKERS_SENTINEL,
            DEFAULT_CACHE_SIZE,
            DEFAULT_CACHE_TTL,
            LOG_FILE_MAX_BYTES,
            LOG_FILE_BACKUP_COUNT,
            DEFAULT_RETRY,
            DEFAULT_CACHE,
            DEFAULT_LOG,
            DEFAULT_WORKERS,
            DEFAULT_RETRY_CONFIG,
            VALIDATION_SCHEMA,
            RETRYABLE_EXCEPTIONS,
            RETRYABLE_STATUS_CODES,
            CONFIG_SCHEMA,
            JWT_DEPRECATED_FIELDS,
            ENV_VAR_MAPPING,
            CREDENTIAL_FIELDS,
        )
        # Verify constants have expected types
        assert isinstance(FORMAT_ALIASES, dict)
        assert isinstance(EXTENSION_TO_FORMAT, dict)
        assert isinstance(DEFAULT_API_FETCH_WORKERS, int)
        assert isinstance(DEFAULT_VALIDATION_WORKERS, int)
        assert isinstance(DEFAULT_BATCH_WORKERS, int)
        assert isinstance(MAX_BATCH_WORKERS, int)
        assert isinstance(AUTO_WORKERS_SENTINEL, int)
        assert isinstance(DEFAULT_CACHE_SIZE, int)
        assert isinstance(DEFAULT_CACHE_TTL, int)
        assert isinstance(LOG_FILE_MAX_BYTES, int)
        assert isinstance(LOG_FILE_BACKUP_COUNT, int)
        assert isinstance(DEFAULT_RETRY_CONFIG, dict)
        assert isinstance(VALIDATION_SCHEMA, dict)
        assert isinstance(RETRYABLE_STATUS_CODES, set)
        assert isinstance(CONFIG_SCHEMA, dict)
        assert isinstance(JWT_DEPRECATED_FIELDS, dict)
        assert isinstance(ENV_VAR_MAPPING, dict)
        assert isinstance(CREDENTIAL_FIELDS, dict)

    # ===========================================
    # Core Functions
    # ===========================================

    def test_core_functions_importable(self):
        """All core utility functions must be importable."""
        from cja_auto_sdr.generator import (
            infer_format_from_path,
            should_generate_format,
            auto_detect_workers,
            format_file_size,
            open_file_in_default_app,
            retry_with_backoff,
            make_api_call_with_retry,
            setup_logging,
        )
        assert callable(infer_format_from_path)
        assert callable(should_generate_format)
        assert callable(auto_detect_workers)
        assert callable(format_file_size)
        assert callable(open_file_in_default_app)
        assert callable(retry_with_backoff)
        assert callable(make_api_call_with_retry)
        assert callable(setup_logging)

    # ===========================================
    # Profile Functions
    # ===========================================

    def test_profile_functions_importable(self):
        """All profile-related functions must be importable."""
        from cja_auto_sdr.generator import (
            get_cja_home,
            get_profiles_dir,
            get_profile_path,
            validate_profile_name,
            load_profile_config_json,
            load_profile_dotenv,
            load_profile_credentials,
            resolve_active_profile,
            list_profiles,
            add_profile_interactive,
            mask_sensitive_value,
            show_profile,
            test_profile,
        )
        assert callable(get_cja_home)
        assert callable(get_profiles_dir)
        assert callable(get_profile_path)
        assert callable(validate_profile_name)
        assert callable(load_profile_config_json)
        assert callable(load_profile_dotenv)
        assert callable(load_profile_credentials)
        assert callable(resolve_active_profile)
        assert callable(list_profiles)
        assert callable(add_profile_interactive)
        assert callable(mask_sensitive_value)
        assert callable(show_profile)
        assert callable(test_profile)

    # ===========================================
    # Credential Functions
    # ===========================================

    def test_credential_functions_importable(self):
        """All credential-related functions must be importable."""
        from cja_auto_sdr.generator import (
            validate_credentials,
            normalize_credential_value,
            filter_credentials,
            load_credentials_from_env,
            validate_env_credentials,
            configure_cjapy,
            validate_config_file,
            initialize_cja,
            validate_data_view,
        )
        assert callable(validate_credentials)
        assert callable(normalize_credential_value)
        assert callable(filter_credentials)
        assert callable(load_credentials_from_env)
        assert callable(validate_env_credentials)
        assert callable(configure_cjapy)
        assert callable(validate_config_file)
        assert callable(initialize_cja)
        assert callable(validate_data_view)

    # ===========================================
    # Git Functions
    # ===========================================

    def test_git_functions_importable(self):
        """All git-related functions must be importable."""
        from cja_auto_sdr.generator import (
            parse_retention_period,
            is_git_repository,
            git_get_user_info,
            save_git_friendly_snapshot,
            generate_git_commit_message,
            git_commit_snapshot,
            git_init_snapshot_repo,
        )
        assert callable(parse_retention_period)
        assert callable(is_git_repository)
        assert callable(git_get_user_info)
        assert callable(save_git_friendly_snapshot)
        assert callable(generate_git_commit_message)
        assert callable(git_commit_snapshot)
        assert callable(git_init_snapshot_repo)

    # ===========================================
    # Output Writer Functions
    # ===========================================

    def test_output_writer_functions_importable(self):
        """All output writer functions must be importable."""
        from cja_auto_sdr.generator import (
            apply_excel_formatting,
            write_csv_output,
            write_json_output,
            write_html_output,
            write_markdown_output,
        )
        assert callable(apply_excel_formatting)
        assert callable(write_csv_output)
        assert callable(write_json_output)
        assert callable(write_html_output)
        assert callable(write_markdown_output)

    # ===========================================
    # Diff Writer Functions
    # ===========================================

    def test_diff_writer_functions_importable(self):
        """All diff writer functions must be importable."""
        from cja_auto_sdr.generator import (
            write_diff_console_output,
            write_diff_grouped_by_field_output,
            write_diff_pr_comment_output,
            detect_breaking_changes,
            write_diff_json_output,
            write_diff_markdown_output,
            write_diff_html_output,
            write_diff_excel_output,
            write_diff_csv_output,
            write_diff_output,
            _format_side_by_side,
            _format_markdown_side_by_side,
        )
        assert callable(write_diff_console_output)
        assert callable(write_diff_grouped_by_field_output)
        assert callable(write_diff_pr_comment_output)
        assert callable(detect_breaking_changes)
        assert callable(write_diff_json_output)
        assert callable(write_diff_markdown_output)
        assert callable(write_diff_html_output)
        assert callable(write_diff_excel_output)
        assert callable(write_diff_csv_output)
        assert callable(write_diff_output)
        assert callable(_format_side_by_side)
        assert callable(_format_markdown_side_by_side)

    # ===========================================
    # Processing Functions
    # ===========================================

    def test_processing_functions_importable(self):
        """All processing functions must be importable."""
        from cja_auto_sdr.generator import (
            display_inventory_summary,
            process_inventory_summary,
            process_single_dataview,
            process_single_dataview_worker,
            run_dry_run,
        )
        assert callable(display_inventory_summary)
        assert callable(process_inventory_summary)
        assert callable(process_single_dataview)
        assert callable(process_single_dataview_worker)
        assert callable(run_dry_run)

    # ===========================================
    # CLI Functions
    # ===========================================

    def test_cli_functions_importable(self):
        """All CLI functions must be importable."""
        from cja_auto_sdr.generator import (
            parse_arguments,
            is_data_view_id,
            levenshtein_distance,
            find_similar_names,
            get_cached_data_views,
            prompt_for_selection,
            resolve_data_view_names,
            list_dataviews,
            interactive_select_dataviews,
            interactive_wizard,
            generate_sample_config,
            show_config_status,
            validate_config_only,
            show_stats,
        )
        assert callable(parse_arguments)
        assert callable(is_data_view_id)
        assert callable(levenshtein_distance)
        assert callable(find_similar_names)
        assert callable(get_cached_data_views)
        assert callable(prompt_for_selection)
        assert callable(resolve_data_view_names)
        assert callable(list_dataviews)
        assert callable(interactive_select_dataviews)
        assert callable(interactive_wizard)
        assert callable(generate_sample_config)
        assert callable(show_config_status)
        assert callable(validate_config_only)
        assert callable(show_stats)

    # ===========================================
    # Org Report Functions
    # ===========================================

    def test_org_report_functions_importable(self):
        """All org report functions must be importable."""
        from cja_auto_sdr.generator import (
            compare_org_reports,
            write_org_report_console,
            write_org_report_stats_only,
            write_org_report_comparison_console,
            build_org_report_json_data,
            write_org_report_json,
            write_org_report_excel,
            write_org_report_markdown,
            write_org_report_html,
            write_org_report_csv,
            run_org_report,
        )
        assert callable(compare_org_reports)
        assert callable(write_org_report_console)
        assert callable(write_org_report_stats_only)
        assert callable(write_org_report_comparison_console)
        assert callable(build_org_report_json_data)
        assert callable(write_org_report_json)
        assert callable(write_org_report_excel)
        assert callable(write_org_report_markdown)
        assert callable(write_org_report_html)
        assert callable(write_org_report_csv)
        assert callable(run_org_report)

    # ===========================================
    # Command Handler Functions
    # ===========================================

    def test_command_handler_functions_importable(self):
        """All command handler functions must be importable."""
        from cja_auto_sdr.generator import (
            handle_snapshot_command,
            handle_diff_command,
            handle_diff_snapshot_command,
            handle_compare_snapshots_command,
            main,
        )
        assert callable(handle_snapshot_command)
        assert callable(handle_diff_command)
        assert callable(handle_diff_snapshot_command)
        assert callable(handle_compare_snapshots_command)
        assert callable(main)

    # ===========================================
    # Private/Internal Symbols Used by Tests
    # ===========================================

    def test_internal_symbols_importable(self):
        """Internal symbols used by tests must remain importable."""
        from cja_auto_sdr.generator import (
            _data_view_cache,
            _format_error_msg,
            _get_credential_fields,
            _config_from_env,
        )
        # These are internal but used in tests
        assert _data_view_cache is not None
        assert callable(_format_error_msg)
        assert callable(_get_credential_fields)
        assert callable(_config_from_env)


class TestBackwardsCompatibilityBehavior:
    """Test that key behaviors work correctly after refactor."""

    def test_infer_format_from_path_works(self):
        """infer_format_from_path should work correctly."""
        from cja_auto_sdr.generator import infer_format_from_path

        assert infer_format_from_path("test.xlsx") == "excel"
        assert infer_format_from_path("test.csv") == "csv"
        assert infer_format_from_path("test.json") == "json"
        assert infer_format_from_path("test.html") == "html"
        assert infer_format_from_path("test.md") == "markdown"
        assert infer_format_from_path("test.txt") is None
        assert infer_format_from_path("-") is None
        assert infer_format_from_path("stdout") is None

    def test_should_generate_format_works(self):
        """should_generate_format should work correctly."""
        from cja_auto_sdr.generator import should_generate_format

        # Direct format match
        assert should_generate_format("excel", "excel") is True
        assert should_generate_format("csv", "csv") is True

        # 'all' format
        assert should_generate_format("all", "excel") is True
        assert should_generate_format("all", "csv") is True
        assert should_generate_format("all", "json") is True

        # Aliases
        assert should_generate_format("reports", "excel") is True
        assert should_generate_format("reports", "markdown") is True
        assert should_generate_format("reports", "csv") is False

        assert should_generate_format("data", "csv") is True
        assert should_generate_format("data", "json") is True
        assert should_generate_format("data", "excel") is False

    def test_console_colors_works(self):
        """ConsoleColors should have all expected methods."""
        from cja_auto_sdr.generator import ConsoleColors

        # Test class methods exist and return strings
        assert isinstance(ConsoleColors.success("test"), str)
        assert isinstance(ConsoleColors.error("test"), str)
        assert isinstance(ConsoleColors.warning("test"), str)
        assert isinstance(ConsoleColors.info("test"), str)
        assert isinstance(ConsoleColors.bold("test"), str)
        assert isinstance(ConsoleColors.dim("test"), str)

        # Test aliases
        assert isinstance(ConsoleColors.green("test"), str)
        assert isinstance(ConsoleColors.red("test"), str)
        assert isinstance(ConsoleColors.yellow("test"), str)
        assert isinstance(ConsoleColors.cyan("test"), str)

        # Test diff colors (theme-aware)
        assert isinstance(ConsoleColors.diff_added("test"), str)
        assert isinstance(ConsoleColors.diff_removed("test"), str)
        assert isinstance(ConsoleColors.diff_modified("test"), str)

    def test_retry_config_to_dict_works(self):
        """RetryConfig.to_dict() should work correctly."""
        from cja_auto_sdr.generator import RetryConfig

        config = RetryConfig(max_retries=5, base_delay=2.0)
        d = config.to_dict()

        assert d["max_retries"] == 5
        assert d["base_delay"] == 2.0

    def test_default_retry_config_structure(self):
        """DEFAULT_RETRY_CONFIG should have expected structure."""
        from cja_auto_sdr.generator import DEFAULT_RETRY_CONFIG

        assert "max_retries" in DEFAULT_RETRY_CONFIG
        assert "base_delay" in DEFAULT_RETRY_CONFIG
        assert "max_delay" in DEFAULT_RETRY_CONFIG
        assert "exponential_base" in DEFAULT_RETRY_CONFIG
        assert "jitter" in DEFAULT_RETRY_CONFIG

    def test_parse_arguments_basic(self):
        """parse_arguments should work with basic args."""
        from cja_auto_sdr.generator import parse_arguments
        import sys

        # Save original argv
        original_argv = sys.argv

        try:
            # Test with --help would exit, so test --version
            sys.argv = ["cja_auto_sdr", "--version"]
            # This will raise SystemExit(0) which is expected
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

    def test_processing_result_dataclass(self):
        """ProcessingResult should be usable as a dataclass."""
        from cja_auto_sdr.generator import ProcessingResult

        result = ProcessingResult(
            data_view_id="dv_test",
            data_view_name="Test View",
            success=True,
            duration=1.5,
        )

        assert result.data_view_id == "dv_test"
        assert result.data_view_name == "Test View"
        assert result.success is True
        assert result.duration == 1.5

    def test_diff_summary_dataclass(self):
        """DiffSummary should be usable as a dataclass."""
        from cja_auto_sdr.generator import DiffSummary

        summary = DiffSummary()

        # Check default values
        assert summary.metrics_added == 0
        assert summary.metrics_removed == 0
        assert summary.metrics_modified == 0
        assert summary.dimensions_added == 0
        assert summary.dimensions_removed == 0
        assert summary.dimensions_modified == 0

    def test_circuit_state_enum_values(self):
        """CircuitState enum should have expected values."""
        from cja_auto_sdr.generator import CircuitState

        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_change_type_enum_values(self):
        """ChangeType enum should have expected values."""
        from cja_auto_sdr.generator import ChangeType

        assert ChangeType.ADDED.value == "added"
        assert ChangeType.REMOVED.value == "removed"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.UNCHANGED.value == "unchanged"


class TestAllSymbolsCount:
    """Verify total symbol count matches expectations."""

    def test_total_symbol_count(self):
        """Verify we have at least 133 importable symbols."""
        from cja_auto_sdr import generator

        # Get all public symbols (not starting with _)
        public_symbols = [
            name for name in dir(generator)
            if not name.startswith('_') or name == '__version__'
        ]

        # Also include specific private symbols used by tests
        private_symbols_used = [
            '_data_view_cache',
            '_format_error_msg',
            '_format_side_by_side',
            '_format_markdown_side_by_side',
            '_get_credential_fields',
            '_config_from_env',
        ]

        for sym in private_symbols_used:
            if hasattr(generator, sym) and sym not in public_symbols:
                public_symbols.append(sym)

        # We expect at least 133 symbols based on the plan
        assert len(public_symbols) >= 133, (
            f"Expected at least 133 symbols, found {len(public_symbols)}: "
            f"{sorted(public_symbols)[:20]}..."
        )
