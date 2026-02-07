"""Tests for command-line interface"""
import pytest
import sys
import os
import json
import subprocess
import tempfile
from unittest.mock import patch, MagicMock
import argparse


# Import the function we're testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cja_auto_sdr.generator import parse_arguments, generate_sample_config, _extract_dataset_info, list_connections, list_datasets, list_dataviews, _emit_output


class TestCLIArguments:
    """Test command-line argument parsing"""

    def test_parse_single_data_view(self):
        """Test parsing a single data view ID"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.data_views == ['dv_12345']
            assert args.batch is False
            assert args.workers == 'auto'  # Default is now 'auto' for automatic detection

    def test_parse_multiple_data_views(self):
        """Test parsing multiple data view IDs"""
        test_args = ['cja_sdr_generator.py', 'dv_12345', 'dv_67890', 'dv_abcde']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.data_views == ['dv_12345', 'dv_67890', 'dv_abcde']
            assert len(args.data_views) == 3

    def test_parse_batch_flag(self):
        """Test parsing with --batch flag"""
        test_args = ['cja_sdr_generator.py', '--batch', 'dv_12345', 'dv_67890']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.batch is True
            assert args.data_views == ['dv_12345', 'dv_67890']

    def test_parse_custom_workers(self):
        """Test parsing with custom worker count"""
        test_args = ['cja_sdr_generator.py', '--workers', '8', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.workers == '8'  # Now a string, parsed to int in main()

    def test_parse_output_dir(self):
        """Test parsing with custom output directory"""
        test_args = ['cja_sdr_generator.py', '--output-dir', './reports', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.output_dir == './reports'

    def test_parse_continue_on_error(self):
        """Test parsing with --continue-on-error flag"""
        test_args = ['cja_sdr_generator.py', '--continue-on-error', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.continue_on_error is True

    def test_parse_log_level(self):
        """Test parsing with custom log level"""
        test_args = ['cja_sdr_generator.py', '--log-level', 'DEBUG', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.log_level == 'DEBUG'

    def test_parse_missing_data_view(self):
        """Test that missing data view ID returns empty list (validated in main)"""
        test_args = ['cja_sdr_generator.py']
        with patch.object(sys, 'argv', test_args):
            # With nargs='*', empty data_views is allowed at parse time
            # Validation is done in main() to support --version flag
            args = parse_arguments()
            assert args.data_views == []

    def test_parse_config_file(self):
        """Test parsing with custom config file"""
        test_args = ['cja_sdr_generator.py', '--config-file', 'custom_config.json', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.config_file == 'custom_config.json'

    def test_default_values(self):
        """Test that default values are set correctly"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.workers == 'auto'  # Default is now 'auto' for automatic detection
            assert args.output_dir == '.'
            assert args.config_file == 'config.json'
            assert args.continue_on_error is False
            assert args.log_level == 'INFO'
            assert args.production is False

    def test_production_flag(self):
        """Test parsing with --production flag"""
        test_args = ['cja_sdr_generator.py', '--production', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.production is True

    def test_production_with_log_level(self):
        """Test that production and log-level can be specified together"""
        test_args = ['cja_sdr_generator.py', '--production', '--log-level', 'DEBUG', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.production is True
            assert args.log_level == 'DEBUG'  # Both parsed, main() decides priority

    def test_dry_run_flag(self):
        """Test parsing with --dry-run flag"""
        test_args = ['cja_sdr_generator.py', '--dry-run', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.dry_run is True

    def test_dry_run_default_false(self):
        """Test that dry-run is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.dry_run is False

    def test_dry_run_with_multiple_data_views(self):
        """Test dry-run with multiple data views"""
        test_args = ['cja_sdr_generator.py', '--dry-run', 'dv_12345', 'dv_67890']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.dry_run is True
            assert args.data_views == ['dv_12345', 'dv_67890']

    def test_quiet_flag(self):
        """Test parsing with --quiet flag"""
        test_args = ['cja_sdr_generator.py', '--quiet', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.quiet is True

    def test_quiet_short_flag(self):
        """Test parsing with -q short flag"""
        test_args = ['cja_sdr_generator.py', '-q', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.quiet is True

    def test_quiet_default_false(self):
        """Test that quiet is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.quiet is False

    def test_version_flag_exits(self):
        """Test that --version flag causes SystemExit"""
        test_args = ['cja_sdr_generator.py', '--version']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == 0  # Clean exit

    def test_list_dataviews_flag(self):
        """Test parsing with --list-dataviews flag"""
        test_args = ['cja_sdr_generator.py', '--list-dataviews']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_dataviews is True

    def test_list_dataviews_default_false(self):
        """Test that list-dataviews is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_dataviews is False

    def test_skip_validation_flag(self):
        """Test parsing with --skip-validation flag"""
        test_args = ['cja_sdr_generator.py', '--skip-validation', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.skip_validation is True

    def test_skip_validation_default_false(self):
        """Test that skip-validation is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.skip_validation is False

    def test_skip_validation_with_batch(self):
        """Test skip-validation with batch mode"""
        test_args = ['cja_sdr_generator.py', '--batch', '--skip-validation', 'dv_12345', 'dv_67890']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.skip_validation is True
            assert args.batch is True
            assert args.data_views == ['dv_12345', 'dv_67890']

    def test_sample_config_flag(self):
        """Test parsing with --sample-config flag"""
        test_args = ['cja_sdr_generator.py', '--sample-config']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.sample_config is True

    def test_sample_config_default_false(self):
        """Test that sample-config is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.sample_config is False


class TestSampleConfig:
    """Test sample configuration file generation"""

    def test_generate_sample_config_creates_file(self):
        """Test that generate_sample_config creates a file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_config.json')
            result = generate_sample_config(output_path)
            assert result is True
            assert os.path.exists(output_path)

    def test_generate_sample_config_valid_json(self):
        """Test that generated config is valid JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_config.json')
            generate_sample_config(output_path)
            with open(output_path) as f:
                config = json.load(f)
            assert 'org_id' in config
            assert 'client_id' in config
            assert 'secret' in config

    def test_generate_sample_config_has_oauth_fields(self):
        """Test that generated config has OAuth S2S fields"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_config.json')
            generate_sample_config(output_path)
            with open(output_path) as f:
                config = json.load(f)
            # OAuth S2S fields
            assert 'org_id' in config
            assert 'client_id' in config
            assert 'secret' in config
            assert 'scopes' in config


class TestUXImprovements:
    """Test UX improvement features"""

    def test_validate_only_flag(self):
        """Test parsing with --validate-only flag (alias for --dry-run)"""
        test_args = ['cja_sdr_generator.py', '--validate-only', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.dry_run is True

    def test_max_issues_flag(self):
        """Test parsing with --max-issues flag"""
        test_args = ['cja_sdr_generator.py', '--max-issues', '10', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.max_issues == 10

    def test_max_issues_default_zero(self):
        """Test that max-issues defaults to 0 (show all)"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.max_issues == 0

    def test_max_issues_with_skip_validation(self):
        """Test max-issues with skip-validation"""
        test_args = ['cja_sdr_generator.py', '--max-issues', '5', '--skip-validation', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.max_issues == 5
            assert args.skip_validation is True


class TestProcessingResult:
    """Test ProcessingResult dataclass"""

    def test_file_size_formatted_bytes(self):
        """Test file size formatting for bytes"""
        from cja_auto_sdr.generator import ProcessingResult
        result = ProcessingResult(
            data_view_id='dv_test',
            data_view_name='Test',
            success=True,
            duration=1.0,
            file_size_bytes=500
        )
        assert result.file_size_formatted == '500 B'

    def test_file_size_formatted_kilobytes(self):
        """Test file size formatting for kilobytes"""
        from cja_auto_sdr.generator import ProcessingResult
        result = ProcessingResult(
            data_view_id='dv_test',
            data_view_name='Test',
            success=True,
            duration=1.0,
            file_size_bytes=2048
        )
        assert result.file_size_formatted == '2.0 KB'

    def test_file_size_formatted_megabytes(self):
        """Test file size formatting for megabytes"""
        from cja_auto_sdr.generator import ProcessingResult
        result = ProcessingResult(
            data_view_id='dv_test',
            data_view_name='Test',
            success=True,
            duration=1.0,
            file_size_bytes=1048576
        )
        assert result.file_size_formatted == '1.0 MB'

    def test_file_size_formatted_zero(self):
        """Test file size formatting for zero bytes"""
        from cja_auto_sdr.generator import ProcessingResult
        result = ProcessingResult(
            data_view_id='dv_test',
            data_view_name='Test',
            success=True,
            duration=1.0,
            file_size_bytes=0
        )
        assert result.file_size_formatted == '0 B'


class TestCacheFlags:
    """Test cache-related CLI flags"""

    def test_enable_cache_flag(self):
        """Test parsing with --enable-cache flag"""
        test_args = ['cja_sdr_generator.py', '--enable-cache', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.enable_cache is True

    def test_enable_cache_default_false(self):
        """Test that enable-cache defaults to False"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.enable_cache is False

    def test_clear_cache_flag(self):
        """Test parsing with --clear-cache flag"""
        test_args = ['cja_sdr_generator.py', '--enable-cache', '--clear-cache', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.clear_cache is True
            assert args.enable_cache is True

    def test_clear_cache_default_false(self):
        """Test that clear-cache defaults to False"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.clear_cache is False

    def test_cache_size_flag(self):
        """Test parsing with --cache-size flag"""
        test_args = ['cja_sdr_generator.py', '--cache-size', '5000', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.cache_size == 5000

    def test_cache_size_default(self):
        """Test that cache-size defaults to 1000"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.cache_size == 1000

    def test_cache_ttl_flag(self):
        """Test parsing with --cache-ttl flag"""
        test_args = ['cja_sdr_generator.py', '--cache-ttl', '7200', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.cache_ttl == 7200

    def test_cache_ttl_default(self):
        """Test that cache-ttl defaults to 3600"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.cache_ttl == 3600

    def test_all_cache_flags_combined(self):
        """Test all cache flags together"""
        test_args = [
            'cja_sdr_generator.py',
            '--enable-cache', '--clear-cache',
            '--cache-size', '2000', '--cache-ttl', '1800',
            'dv_12345'
        ]
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.enable_cache is True
            assert args.clear_cache is True
            assert args.cache_size == 2000
            assert args.cache_ttl == 1800


class TestConstants:
    """Test that constants are properly used in defaults"""

    def test_workers_default_uses_auto(self):
        """Test that workers default is 'auto' for automatic detection"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.workers == 'auto'  # Default is now 'auto' for automatic detection

    def test_cache_size_default_uses_constant(self):
        """Test that cache_size default matches DEFAULT_CACHE_SIZE constant"""
        from cja_auto_sdr.generator import DEFAULT_CACHE_SIZE
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.cache_size == DEFAULT_CACHE_SIZE

    def test_cache_ttl_default_uses_constant(self):
        """Test that cache_ttl default matches DEFAULT_CACHE_TTL constant"""
        from cja_auto_sdr.generator import DEFAULT_CACHE_TTL
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.cache_ttl == DEFAULT_CACHE_TTL


class TestConsoleScriptEntryPoints:
    """Test console script entry point configuration"""

    def test_main_function_is_importable(self):
        """Test that main function can be imported from cja_sdr_generator"""
        from cja_auto_sdr.generator import main
        assert callable(main)

    def test_main_function_with_version_flag(self):
        """Test that main function handles --version flag correctly"""
        from cja_auto_sdr.generator import main
        test_args = ['cja_auto_sdr', '--version']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_function_with_help_flag(self):
        """Test that main function handles --help flag correctly"""
        from cja_auto_sdr.generator import main
        test_args = ['cja_auto_sdr', '--help']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_function_with_sample_config_flag(self):
        """Test that main function handles --sample-config flag"""
        from cja_auto_sdr.generator import main
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'sample_config.json')
            test_args = ['cja_auto_sdr', '--sample-config']
            with patch.object(sys, 'argv', test_args):
                with patch('cja_auto_sdr.generator.generate_sample_config') as mock_gen:
                    mock_gen.return_value = True
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0
                    mock_gen.assert_called_once()

    def test_entry_point_defined_in_pyproject(self):
        """Test that console script entry points are defined in pyproject.toml"""
        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'pyproject.toml'
        )
        with open(pyproject_path, 'r') as f:
            content = f.read()

        # Verify both entry point variants exist with correct targets
        assert 'cja_auto_sdr = "cja_auto_sdr.generator:main"' in content
        assert 'cja-auto-sdr = "cja_auto_sdr.generator:main"' in content

        # Verify [project.scripts] section exists
        assert '[project.scripts]' in content

    def test_entry_point_builds_correctly(self):
        """Test that the package build system is configured"""
        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'pyproject.toml'
        )
        with open(pyproject_path, 'r') as f:
            content = f.read()

        # Check build system is defined
        assert '[build-system]' in content
        assert 'hatchling' in content
        assert 'build-backend = "hatchling.build"' in content

    def test_parse_arguments_works_with_console_script_name(self):
        """Test that argument parsing works with console script names"""
        # Test with underscore variant
        test_args = ['cja_auto_sdr', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.data_views == ['dv_12345']

        # Test with hyphen variant
        test_args = ['cja-auto-sdr', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.data_views == ['dv_12345']

    def test_version_output_format(self):
        """Test that version output follows expected format"""
        from cja_auto_sdr.generator import __version__
        import subprocess
        result = subprocess.run(
            ['uv', 'run', 'cja_auto_sdr', '--version'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        assert result.returncode == 0
        assert __version__ in result.stdout


class TestRetryArguments:
    """Test retry-related CLI arguments"""

    def test_max_retries_flag(self):
        """Test parsing with --max-retries flag"""
        test_args = ['cja_sdr_generator.py', '--max-retries', '5', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.max_retries == 5

    def test_max_retries_default(self):
        """Test that max-retries uses default from DEFAULT_RETRY_CONFIG"""
        from cja_auto_sdr.generator import DEFAULT_RETRY_CONFIG
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.max_retries == DEFAULT_RETRY_CONFIG['max_retries']

    def test_retry_base_delay_flag(self):
        """Test parsing with --retry-base-delay flag"""
        test_args = ['cja_sdr_generator.py', '--retry-base-delay', '2.5', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.retry_base_delay == 2.5

    def test_retry_base_delay_default(self):
        """Test that retry-base-delay uses default from DEFAULT_RETRY_CONFIG"""
        from cja_auto_sdr.generator import DEFAULT_RETRY_CONFIG
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.retry_base_delay == DEFAULT_RETRY_CONFIG['base_delay']

    def test_retry_max_delay_flag(self):
        """Test parsing with --retry-max-delay flag"""
        test_args = ['cja_sdr_generator.py', '--retry-max-delay', '60.0', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.retry_max_delay == 60.0

    def test_retry_max_delay_default(self):
        """Test that retry-max-delay uses default from DEFAULT_RETRY_CONFIG"""
        from cja_auto_sdr.generator import DEFAULT_RETRY_CONFIG
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.retry_max_delay == DEFAULT_RETRY_CONFIG['max_delay']

    def test_all_retry_flags_combined(self):
        """Test all retry flags together"""
        test_args = [
            'cja_sdr_generator.py',
            '--max-retries', '10',
            '--retry-base-delay', '0.5',
            '--retry-max-delay', '120',
            'dv_12345'
        ]
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.max_retries == 10
            assert args.retry_base_delay == 0.5
            assert args.retry_max_delay == 120.0

    def test_retry_env_var_max_retries(self):
        """Test that MAX_RETRIES env var sets default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.dict(os.environ, {'MAX_RETRIES': '7'}):
            with patch.object(sys, 'argv', test_args):
                args = parse_arguments()
                assert args.max_retries == 7

    def test_retry_env_var_base_delay(self):
        """Test that RETRY_BASE_DELAY env var sets default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.dict(os.environ, {'RETRY_BASE_DELAY': '3.5'}):
            with patch.object(sys, 'argv', test_args):
                args = parse_arguments()
                assert args.retry_base_delay == 3.5

    def test_retry_env_var_max_delay(self):
        """Test that RETRY_MAX_DELAY env var sets default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.dict(os.environ, {'RETRY_MAX_DELAY': '90.0'}):
            with patch.object(sys, 'argv', test_args):
                args = parse_arguments()
                assert args.retry_max_delay == 90.0

    def test_retry_cli_overrides_env_var(self):
        """Test that CLI arguments override environment variables"""
        test_args = ['cja_sdr_generator.py', '--max-retries', '2', 'dv_12345']
        with patch.dict(os.environ, {'MAX_RETRIES': '10'}):
            with patch.object(sys, 'argv', test_args):
                args = parse_arguments()
                assert args.max_retries == 2


class TestValidateConfigFlag:
    """Test --validate-config flag"""

    def test_validate_config_flag(self):
        """Test parsing with --validate-config flag"""
        test_args = ['cja_sdr_generator.py', '--validate-config']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.validate_config is True
            assert args.data_views == []

    def test_validate_config_default_false(self):
        """Test that validate-config is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.validate_config is False

    def test_validate_config_no_dataview_required(self):
        """Test that --validate-config doesn't require data view argument"""
        test_args = ['cja_sdr_generator.py', '--validate-config']
        with patch.object(sys, 'argv', test_args):
            # Should parse without error even though no data view is provided
            args = parse_arguments()
            assert args.validate_config is True


class TestFormatValidation:
    """Test output format validation"""

    def test_format_console_valid_for_diff(self):
        """Test that console format is accepted for diff mode"""
        test_args = ['cja_sdr_generator.py', '--diff', 'dv_A', 'dv_B', '--format', 'console']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.format == 'console'
            assert args.diff is True

    def test_format_console_parsed_for_sdr(self):
        """Test that console format is parsed (validation happens at runtime)"""
        test_args = ['cja_sdr_generator.py', 'dv_12345', '--format', 'console']
        with patch.object(sys, 'argv', test_args):
            # Argparse allows console as a choice, runtime validation catches it
            args = parse_arguments()
            assert args.format == 'console'

    def test_format_excel_valid_for_sdr(self):
        """Test that excel format is valid for SDR"""
        test_args = ['cja_sdr_generator.py', 'dv_12345', '--format', 'excel']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.format == 'excel'

    def test_format_all_valid_for_sdr(self):
        """Test that all format is valid for SDR"""
        test_args = ['cja_sdr_generator.py', 'dv_12345', '--format', 'all']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.format == 'all'

    def test_format_json_valid_for_both(self):
        """Test that json format is valid for both SDR and diff"""
        # SDR mode
        test_args = ['cja_sdr_generator.py', 'dv_12345', '--format', 'json']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.format == 'json'

        # Diff mode
        test_args = ['cja_sdr_generator.py', '--diff', 'dv_A', 'dv_B', '--format', 'json']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.format == 'json'


class TestListConnectionsArgs:
    """Test --list-connections argument parsing"""

    def test_list_connections_flag(self):
        """Test parsing with --list-connections flag"""
        test_args = ['cja_sdr_generator.py', '--list-connections']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_connections is True

    def test_list_connections_default_false(self):
        """Test that list-connections is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_connections is False

    def test_list_connections_with_format(self):
        """Test --list-connections with --format json"""
        test_args = ['cja_sdr_generator.py', '--list-connections', '--format', 'json']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_connections is True
            assert args.format == 'json'

    def test_list_connections_with_csv_output(self):
        """Test --list-connections with --format csv and --output"""
        test_args = ['cja_sdr_generator.py', '--list-connections', '--format', 'csv', '--output', 'conns.csv']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_connections is True
            assert args.format == 'csv'
            assert args.output == 'conns.csv'


class TestListDatasetsArgs:
    """Test --list-datasets argument parsing"""

    def test_list_datasets_flag(self):
        """Test parsing with --list-datasets flag"""
        test_args = ['cja_sdr_generator.py', '--list-datasets']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_datasets is True

    def test_list_datasets_default_false(self):
        """Test that list-datasets is False by default"""
        test_args = ['cja_sdr_generator.py', 'dv_12345']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_datasets is False

    def test_list_datasets_with_format(self):
        """Test --list-datasets with --format csv"""
        test_args = ['cja_sdr_generator.py', '--list-datasets', '--format', 'csv']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_datasets is True
            assert args.format == 'csv'

    def test_list_datasets_with_profile(self):
        """Test --list-datasets with --profile"""
        test_args = ['cja_sdr_generator.py', '--list-datasets', '--profile', 'client-a']
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.list_datasets is True
            assert args.profile == 'client-a'


class TestDiscoveryMutualExclusivity:
    """Test that discovery commands are mutually exclusive"""

    def test_list_dataviews_and_connections_rejected(self):
        """Test that --list-dataviews and --list-connections cannot be combined"""
        test_args = ['cja_sdr_generator.py', '--list-dataviews', '--list-connections']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_list_dataviews_and_datasets_rejected(self):
        """Test that --list-dataviews and --list-datasets cannot be combined"""
        test_args = ['cja_sdr_generator.py', '--list-dataviews', '--list-datasets']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_list_connections_and_datasets_rejected(self):
        """Test that --list-connections and --list-datasets cannot be combined"""
        test_args = ['cja_sdr_generator.py', '--list-connections', '--list-datasets']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_all_three_rejected(self):
        """Test that all three discovery flags cannot be combined"""
        test_args = ['cja_sdr_generator.py', '--list-dataviews', '--list-connections', '--list-datasets']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestExtractDatasetInfo:
    """Test _extract_dataset_info() helper"""

    def test_standard_fields(self):
        """Test extraction with standard id/name fields"""
        result = _extract_dataset_info({'id': 'ds_123', 'name': 'Web Events'})
        assert result == {'id': 'ds_123', 'name': 'Web Events'}

    def test_alternate_id_field(self):
        """Test extraction with datasetId field"""
        result = _extract_dataset_info({'datasetId': 'ds_456', 'name': 'Mobile Events'})
        assert result == {'id': 'ds_456', 'name': 'Mobile Events'}

    def test_alternate_name_field_title(self):
        """Test extraction with title field"""
        result = _extract_dataset_info({'id': 'ds_789', 'title': 'Product Catalog'})
        assert result == {'id': 'ds_789', 'name': 'Product Catalog'}

    def test_alternate_name_field_datasetName(self):
        """Test extraction with datasetName field"""
        result = _extract_dataset_info({'id': 'ds_111', 'datasetName': 'Test Dataset'})
        assert result == {'id': 'ds_111', 'name': 'Test Dataset'}

    def test_missing_fields(self):
        """Test extraction with empty dict"""
        result = _extract_dataset_info({})
        assert result == {'id': 'N/A', 'name': 'N/A'}

    def test_non_dict_input(self):
        """Test extraction with non-dict input"""
        result = _extract_dataset_info('ds_string_id')
        assert result == {'id': 'ds_string_id', 'name': 'N/A'}

    def test_none_input(self):
        """Test extraction with None input"""
        result = _extract_dataset_info(None)
        assert result == {'id': 'N/A', 'name': 'N/A'}

    def test_dataSetId_field(self):
        """Test extraction with dataSetId (camelCase) field"""
        result = _extract_dataset_info({'dataSetId': 'ds_222', 'dataSetName': 'Events'})
        assert result == {'id': 'ds_222', 'name': 'Events'}

    def test_snake_case_fields(self):
        """Test extraction with snake_case dataset fields"""
        result = _extract_dataset_info({'dataset_id': 'ds_333', 'dataset_name': 'Legacy Events'})
        assert result == {'id': 'ds_333', 'name': 'Legacy Events'}


class TestListDataviewsFunction:
    """Test list_dataviews() function with mocks"""

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_dataviews_null_owner_shows_na(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_dataviews shows N/A when owner is null (not the string 'None')"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'Test View', 'owner': None}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_dataviews(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['dataViews'][0]['owner'] == 'N/A'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_dataviews_null_owner_table(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_dataviews table output shows N/A for null owner"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'Test View', 'owner': None}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_dataviews(output_format='table')

        assert result is True
        output = f.getvalue()
        assert 'N/A' in output
        assert 'Owner: None' not in output

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_dataviews_owner_dict_with_null_name_shows_na(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_dataviews shows N/A when owner is {'name': None}"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'Test View', 'owner': {'name': None}}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_dataviews(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['dataViews'][0]['owner'] == 'N/A'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_dataviews_owner_dict_with_null_name_table(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_dataviews table output shows N/A when owner is {'name': None}"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'Test View', 'owner': {'name': None}}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_dataviews(output_format='table')

        assert result is True
        output = f.getvalue()
        assert 'N/A' in output
        assert 'None' not in output

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_dataviews_missing_owner_shows_na(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_dataviews shows N/A when owner key is absent"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'Test View'}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_dataviews(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['dataViews'][0]['owner'] == 'N/A'


class TestListConnectionsFunction:
    """Test list_connections() function with mocks"""

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_json(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections with JSON output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_123',
                    'name': 'Production Connection',
                    'ownerFullName': 'John Doe',
                    'dataSets': [
                        {'id': 'ds_456', 'name': 'Web Events'},
                        {'id': 'ds_789', 'name': 'Mobile Events'}
                    ]
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['count'] == 1
        assert output['connections'][0]['id'] == 'conn_123'
        assert len(output['connections'][0]['datasets']) == 2

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_empty(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections with no connections and no data views"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = []

        result = list_connections(output_format='table')
        assert result is True

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_empty_json_prints_stdout(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections prints empty JSON payload to stdout when genuinely empty"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = []

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output == {"connections": [], "count": 0}

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_empty_csv_prints_stdout(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections prints empty CSV payload to stdout when genuinely empty"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = []

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='csv')

        assert result is True
        assert f.getvalue() == "connection_id,connection_name,owner,dataset_id,dataset_name\n"

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_csv(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections with CSV output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': 'Owner',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset One'}]
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='csv')

        assert result is True
        lines = f.getvalue().strip().split('\n')
        assert lines[0] == 'connection_id,connection_name,owner,dataset_id,dataset_name'
        assert 'conn_1' in lines[1]
        assert 'ds_1' in lines[1]

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_null_owner_shows_na(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections shows N/A when ownerFullName is null (not the string 'None')"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': None,
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset One'}]
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['connections'][0]['owner'] == 'N/A'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_null_owner_table(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections table output shows N/A for null ownerFullName"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': None,
                    'dataSets': []
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='table')

        assert result is True
        output = f.getvalue()
        assert 'Owner: N/A' in output
        assert 'Owner: None' not in output

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_missing_owner_shows_na(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections shows N/A when owner key is absent"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'dataSets': []
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['connections'][0]['owner'] == 'N/A'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_owner_dict_with_null_name_shows_na(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections shows N/A when ownerFullName is empty string"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': '',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset One'}]
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['connections'][0]['owner'] == 'N/A'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_owner_dict_with_null_name_table(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_connections table output shows N/A when ownerFullName is empty"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': '',
                    'dataSets': []
                }
            ]
        }

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='table')

        assert result is True
        output = f.getvalue()
        assert 'Owner: N/A' in output
        assert 'Owner: None' not in output

    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_config_failure(self, mock_profile, mock_configure):
        """Test list_connections when configuration fails"""
        mock_configure.return_value = (False, 'Missing credentials', None)

        import io
        from contextlib import redirect_stderr
        f = io.StringIO()
        with redirect_stderr(f):
            result = list_connections(output_format='json')

        assert result is False
        error = json.loads(f.getvalue())
        assert error == {"error": "Configuration error: Missing credentials"}

    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_config_failure_csv_emits_json_error(self, mock_profile, mock_configure):
        """Test list_connections emits machine-readable error in CSV mode on config failure"""
        mock_configure.return_value = (False, 'Missing credentials', None)

        import io
        from contextlib import redirect_stderr
        f = io.StringIO()
        with redirect_stderr(f):
            result = list_connections(output_format='csv')

        assert result is False
        error = json.loads(f.getvalue())
        assert error == {"error": "Configuration error: Missing credentials"}


class TestListDatasetsFunction:
    """Test list_datasets() function with mocks"""

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_json(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets with JSON output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_456',
                    'name': 'Production Connection',
                    'dataSets': [
                        {'id': 'ds_789', 'name': 'Web Events'}
                    ]
                }
            ]
        }
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_123', 'name': 'Web Data View', 'parentDataGroupId': 'conn_456'}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['count'] == 1
        assert output['dataViews'][0]['connection']['id'] == 'conn_456'
        assert len(output['dataViews'][0]['datasets']) == 1
        mock_cja_instance.getDataView.assert_not_called()

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_unknown_connection(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets when data view has no parentDataGroupId"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_orphan', 'name': 'Orphan View'}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output['dataViews'][0]['connection']['name'] is None
        assert output['dataViews'][0]['connection']['id'] == 'N/A'
        mock_cja_instance.getDataView.assert_not_called()

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_nan_parent_connection_id_json(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets normalizes NaN parentDataGroupId in JSON output"""
        import pandas as pd

        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = pd.DataFrame([
            {'id': 'dv_orphan_nan', 'name': 'Orphan NaN', 'parentDataGroupId': float('nan')}
        ])

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='json')

        assert result is True
        output_text = f.getvalue()
        assert '"id": NaN' not in output_text
        output = json.loads(output_text)
        assert output['dataViews'][0]['connection']['id'] == 'N/A'
        mock_cja_instance.getDataView.assert_not_called()

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_csv(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets with CSV output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Conn One',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset'}]
                }
            ]
        }
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View One', 'parentDataGroupId': 'conn_1'}
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='csv')

        assert result is True
        lines = f.getvalue().strip().split('\n')
        assert lines[0] == 'dataview_id,dataview_name,connection_id,connection_name,dataset_id,dataset_name'
        assert 'dv_1' in lines[1]
        assert 'conn_1' in lines[1]
        assert 'ds_1' in lines[1]
        mock_cja_instance.getDataView.assert_not_called()

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_empty_dataviews(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets with no data views"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = []

        result = list_datasets(output_format='table')
        assert result is True

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_empty_json_prints_stdout(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets prints empty JSON payload to stdout"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = []

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert output == {"dataViews": [], "count": 0}
        mock_cja_instance.getDataView.assert_not_called()

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_empty_csv_prints_stdout(self, mock_profile, mock_configure, mock_cjapy):
        """Test list_datasets prints empty CSV payload to stdout"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = []

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='csv')

        assert result is True
        assert f.getvalue() == "dataview_id,dataview_name,connection_id,connection_name,dataset_id,dataset_name\n"
        mock_cja_instance.getDataView.assert_not_called()

    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_config_failure_json_emits_json_error(self, mock_profile, mock_configure):
        """Test list_datasets emits machine-readable error in JSON mode on config failure"""
        mock_configure.return_value = (False, 'Missing credentials', None)

        import io
        from contextlib import redirect_stderr
        f = io.StringIO()
        with redirect_stderr(f):
            result = list_datasets(output_format='json')

        assert result is False
        error = json.loads(f.getvalue())
        assert error == {"error": "Configuration error: Missing credentials"}

    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_config_failure_csv_emits_json_error(self, mock_profile, mock_configure):
        """Test list_datasets emits machine-readable error in CSV mode on config failure"""
        mock_configure.return_value = (False, 'Missing credentials', None)

        import io
        from contextlib import redirect_stderr
        f = io.StringIO()
        with redirect_stderr(f):
            result = list_datasets(output_format='csv')

        assert result is False
        error = json.loads(f.getvalue())
        assert error == {"error": "Configuration error: Missing credentials"}


class TestFileOutput:
    """Test that list commands write output to disk via --output."""

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_json_file(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test list_connections writes JSON to a file via --output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': 'Owner',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset One'}]
                }
            ]
        }

        out_file = str(tmp_path / "connections.json")
        result = list_connections(output_format='json', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            output = json.load(f)
        assert output['count'] == 1
        assert output['connections'][0]['id'] == 'conn_1'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_csv_file(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test list_connections writes CSV to a file via --output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': 'Owner',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset One'}]
                }
            ]
        }

        out_file = str(tmp_path / "connections.csv")
        result = list_connections(output_format='csv', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            content = f.read()
        lines = content.strip().split('\n')
        assert lines[0] == 'connection_id,connection_name,owner,dataset_id,dataset_name'
        assert 'conn_1' in lines[1]

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_json_file(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test list_datasets writes JSON to a file via --output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Conn One',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset'}]
                }
            ]
        }
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View One', 'parentDataGroupId': 'conn_1'}
        ]

        out_file = str(tmp_path / "datasets.json")
        result = list_datasets(output_format='json', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            output = json.load(f)
        assert output['count'] == 1
        assert output['dataViews'][0]['connection']['id'] == 'conn_1'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_dataviews_json_file(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test list_dataviews writes JSON to a file via --output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'My View', 'owner': {'name': 'Test User'}}
        ]

        out_file = str(tmp_path / "dataviews.json")
        result = list_dataviews(output_format='json', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            output = json.load(f)
        assert output['count'] == 1
        assert output['dataViews'][0]['id'] == 'dv_1'

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_table_file(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test list_connections writes table output to a file via --output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test Conn',
                    'ownerFullName': 'Owner',
                    'dataSets': [{'id': 'ds_1', 'name': 'Dataset One'}]
                }
            ]
        }

        out_file = str(tmp_path / "connections.txt")
        result = list_connections(output_format='table', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            content = f.read()
        assert 'conn_1' in content
        assert 'Test Conn' in content

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_creates_parent_dirs(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test that --output creates parent directories if needed"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Test',
                    'ownerFullName': 'Owner',
                    'dataSets': []
                }
            ]
        }

        out_file = str(tmp_path / "subdir" / "nested" / "connections.json")
        result = list_connections(output_format='json', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            output = json.load(f)
        assert output['count'] == 1

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_csv_file(self, mock_profile, mock_configure, mock_cjapy, tmp_path):
        """Test list_datasets writes CSV (6-column join) to a file via --output"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {
            'content': [
                {
                    'id': 'conn_1',
                    'name': 'Conn One',
                    'dataSets': [
                        {'id': 'ds_1', 'name': 'Dataset A'},
                        {'id': 'ds_2', 'name': 'Dataset B'}
                    ]
                }
            ]
        }
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View One', 'parentDataGroupId': 'conn_1'}
        ]

        out_file = str(tmp_path / "datasets.csv")
        result = list_datasets(output_format='csv', output_file=out_file)

        assert result is True
        with open(out_file) as f:
            content = f.read()
        lines = content.strip().split('\n')
        assert lines[0] == 'dataview_id,dataview_name,connection_id,connection_name,dataset_id,dataset_name'
        # One row per dataset (2 datasets under the same data view)
        assert len(lines) == 3
        assert 'ds_1' in lines[1]
        assert 'ds_2' in lines[2]
        assert 'dv_1' in lines[1]
        assert 'conn_1' in lines[1]


class TestEmitOutputPager:
    """Test _emit_output auto-pager behaviour for long console output."""

    def test_uses_pager_when_output_exceeds_terminal(self):
        """Test that _emit_output invokes a pager when output is taller than terminal"""
        long_text = '\n'.join(f'line {i}' for i in range(200))
        mock_proc = MagicMock()
        mock_proc.communicate = MagicMock()

        with patch('sys.stdout') as mock_stdout, \
             patch('os.get_terminal_size', return_value=os.terminal_size((80, 24))), \
             patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            mock_stdout.isatty.return_value = True
            _emit_output(long_text, None, False)

        mock_popen.assert_called_once_with(['less', '-R'], stdin=subprocess.PIPE)
        mock_proc.communicate.assert_called_once_with(long_text.rstrip('\n').encode())

    def test_no_pager_when_output_fits_terminal(self):
        """Test that _emit_output prints normally when output fits in terminal"""
        short_text = '\n'.join(f'line {i}' for i in range(5))

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()

        with patch('os.get_terminal_size', return_value=os.terminal_size((80, 24))), \
             redirect_stdout(f):
            _emit_output(short_text, None, False)

        assert 'line 0' in f.getvalue()
        assert 'line 4' in f.getvalue()

    def test_no_pager_when_not_tty(self):
        """Test that _emit_output does not page when stdout is not a TTY (piped)"""
        long_text = '\n'.join(f'line {i}' for i in range(200))

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()

        with redirect_stdout(f):
            # StringIO.isatty() returns False, so no pager should be used
            _emit_output(long_text, None, False)

        assert 'line 0' in f.getvalue()
        assert 'line 199' in f.getvalue()

    def test_no_pager_for_stdout_pipe_mode(self):
        """Test that _emit_output does not page when is_stdout=True (--output -)"""
        long_text = '\n'.join(f'line {i}' for i in range(200))

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()

        with redirect_stdout(f):
            _emit_output(long_text, None, True)

        assert 'line 0' in f.getvalue()

    def test_pager_respects_PAGER_env(self):
        """Test that _emit_output uses $PAGER when set"""
        long_text = '\n'.join(f'line {i}' for i in range(200))
        mock_proc = MagicMock()
        mock_proc.communicate = MagicMock()

        with patch('sys.stdout') as mock_stdout, \
             patch('os.get_terminal_size', return_value=os.terminal_size((80, 24))), \
             patch.dict(os.environ, {'PAGER': 'more'}), \
             patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            mock_stdout.isatty.return_value = True
            _emit_output(long_text, None, False)

        mock_popen.assert_called_once_with(['more'], stdin=subprocess.PIPE)

    def test_pager_fallback_on_error(self):
        """Test that _emit_output falls back to print when pager is unavailable"""
        long_text = '\n'.join(f'line {i}' for i in range(200))

        import io
        from contextlib import redirect_stdout

        with patch('sys.stdout') as mock_stdout, \
             patch('os.get_terminal_size', return_value=os.terminal_size((80, 24))), \
             patch('subprocess.Popen', side_effect=FileNotFoundError):
            mock_stdout.isatty.return_value = True
            # When pager fails, should fall through to print() — we need
            # to verify it doesn't raise.  Since stdout is mocked, check
            # that write was called as a fallback.
            _emit_output(long_text, None, False)
            mock_stdout.write.assert_called()


class TestConnectionsPermissionsFallback:
    """Test fallback behaviour when getConnections returns empty due to missing admin privileges."""

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_fallback_from_dataviews(self, mock_profile, mock_configure, mock_cjapy):
        """getConnections empty + data views with parentDataGroupId → derived IDs + warning"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View A', 'parentDataGroupId': 'dg_abc'},
            {'id': 'dv_2', 'name': 'View B', 'parentDataGroupId': 'dg_abc'},
            {'id': 'dv_3', 'name': 'View C', 'parentDataGroupId': 'dg_xyz'},
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='table')

        assert result is True
        output = f.getvalue()
        assert 'product-admin' in output.lower() or 'product-admin' in output
        assert 'dg_abc' in output
        assert 'dg_xyz' in output
        assert '2 data view(s)' in output   # dg_abc has 2
        assert '1 data view(s)' in output   # dg_xyz has 1

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_fallback_json(self, mock_profile, mock_configure, mock_cjapy):
        """getConnections empty + data views → JSON includes warning and derived connections"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View A', 'parentDataGroupId': 'dg_abc'},
            {'id': 'dv_2', 'name': 'View B', 'parentDataGroupId': 'dg_xyz'},
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert 'warning' in output
        assert output['count'] == 2
        ids = [c['id'] for c in output['connections']]
        assert 'dg_abc' in ids
        assert 'dg_xyz' in ids
        # name should be null for derived connections
        for conn in output['connections']:
            assert conn['name'] is None

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_connections_fallback_csv(self, mock_profile, mock_configure, mock_cjapy):
        """getConnections empty + data views → CSV includes derived connections with dataview_count"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View A', 'parentDataGroupId': 'dg_abc'},
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_connections(output_format='csv')

        assert result is True
        lines = f.getvalue().strip().split('\n')
        assert 'dataview_count' in lines[0]
        assert 'dg_abc' in lines[1]


class TestDatasetsPermissionsFallback:
    """Test list_datasets behaviour when connection details are unavailable."""

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_no_connection_details_table(self, mock_profile, mock_configure, mock_cjapy):
        """conn_map empty + data views with parentDataGroupId → show ID without 'Unknown'"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View A', 'parentDataGroupId': 'dg_abc'},
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='table')

        assert result is True
        output = f.getvalue()
        assert 'Connection: dg_abc' in output
        assert 'Unknown' not in output
        assert 'Datasets: (none)' not in output
        assert 'product-admin' in output.lower() or 'product-admin' in output

    @patch('cja_auto_sdr.generator.cjapy')
    @patch('cja_auto_sdr.generator.configure_cjapy')
    @patch('cja_auto_sdr.generator.resolve_active_profile', return_value=None)
    def test_list_datasets_no_connection_details_json(self, mock_profile, mock_configure, mock_cjapy):
        """conn_map empty + data views with parentDataGroupId → JSON uses null for connection_name"""
        mock_configure.return_value = (True, 'config', None)
        mock_cja_instance = mock_cjapy.CJA.return_value
        mock_cja_instance.getConnections.return_value = {'content': []}
        mock_cja_instance.getDataViews.return_value = [
            {'id': 'dv_1', 'name': 'View A', 'parentDataGroupId': 'dg_abc'},
        ]

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = list_datasets(output_format='json')

        assert result is True
        output = json.loads(f.getvalue())
        assert 'warning' in output
        assert output['dataViews'][0]['connection']['name'] is None
        assert output['dataViews'][0]['connection']['id'] == 'dg_abc'
        assert output['dataViews'][0]['datasets'] == []
