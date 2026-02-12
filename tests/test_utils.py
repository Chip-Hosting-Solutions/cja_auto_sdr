"""Tests for utility functions."""

import io
import json
import logging
import os
import sys

# Import the functions and classes we're testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cja_auto_sdr.core.logging import SensitiveDataFilter, flush_logging_handlers, with_log_context
from cja_auto_sdr.generator import (
    VALIDATION_SCHEMA,
    JSONFormatter,
    PerformanceTracker,
    _format_error_msg,
    setup_logging,
    validate_config_file,
)


class TestLoggingSetup:
    """Test logging configuration"""

    def test_json_formatter_includes_custom_record_fields(self):
        """JSONFormatter should include custom fields from logging extra."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.batch_id = "batch-123"
        record.extra_fields = {"mode": "batch"}

        payload = json.loads(formatter.format(record))

        assert payload["message"] == "hello"
        assert payload["mode"] == "batch"
        assert payload["batch_id"] == "batch-123"
        assert "process" in payload
        assert "thread" in payload

    def test_flush_logging_handlers_flushes_propagated_root_handlers(self):
        """flush_logging_handlers should flush root handlers when logger has none."""

        class _FlushTrackingHandler(logging.StreamHandler):
            def __init__(self):
                super().__init__(stream=io.StringIO())
                self.flush_count = 0

            def flush(self):
                self.flush_count += 1
                super().flush()

        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)

        for handler in original_handlers:
            root_logger.removeHandler(handler)

        tracking_handler = _FlushTrackingHandler()
        root_logger.addHandler(tracking_handler)

        try:
            test_logger = logging.getLogger("test.flush")
            test_logger.handlers.clear()
            flush_logging_handlers(test_logger)
            assert tracking_handler.flush_count >= 1
        finally:
            root_logger.removeHandler(tracking_handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)

    def test_with_log_context_includes_fields_in_json_output(self):
        """with_log_context should emit adapter context as JSON fields."""
        stream = io.StringIO()
        logger = logging.getLogger("test.context.adapter")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        original_handlers = list(logger.handlers)
        logger.handlers.clear()

        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            contextual_logger = with_log_context(logger, batch_id="batch-ctx", run_mode="batch")
            contextual_logger.info("context message", extra={"data_view_id": "dv_ctx"})
            payload = json.loads(stream.getvalue().strip())
            assert payload["batch_id"] == "batch-ctx"
            assert payload["run_mode"] == "batch"
            assert payload["data_view_id"] == "dv_ctx"
        finally:
            logger.removeHandler(handler)
            for existing in original_handlers:
                logger.addHandler(existing)

    def test_json_formatter_redacts_sensitive_fields_and_message(self):
        """JSONFormatter should redact sensitive values in messages and extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.redaction.json",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Auth failed token=abc123 password:hunter2 Bearer very-secret-token",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {
            "client_secret": "top-secret",
            "nested": {"access_token": "token-123", "note": "token=def456"},
        }
        record.authorization = "Bearer another-secret-token"
        record.session_token = "session-abc"

        payload = json.loads(formatter.format(record))

        assert "[REDACTED]" in payload["message"]
        assert "abc123" not in payload["message"]
        assert "hunter2" not in payload["message"]
        assert payload["client_secret"] == "[REDACTED]"
        assert payload["nested"]["access_token"] == "[REDACTED]"
        assert payload["nested"]["note"].endswith("[REDACTED]")
        assert payload["authorization"] == "[REDACTED]"
        assert payload["session_token"] == "[REDACTED]"

    def test_json_formatter_redacts_authorization_bearer_header(self):
        """Authorization: Bearer <token> should redact the full credential."""
        formatter = JSONFormatter()
        secret = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.signature"
        record = logging.LogRecord(
            name="test.redaction.authorization",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg=f"Request failed Authorization: Bearer {secret} and authorization=Bearer {secret}",
            args=(),
            exc_info=None,
        )

        payload = json.loads(formatter.format(record))

        assert secret not in payload["message"]
        assert "Authorization: Bearer [REDACTED]" in payload["message"]
        assert "authorization=Bearer [REDACTED]" in payload["message"]

    def test_json_formatter_redacts_camelcase_sensitive_fields(self):
        """camelCase token/secret keys should be treated as sensitive."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.redaction.camelcase",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="payload includes accessToken=abc123",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {
            "clientSecret": "top-secret",
            "accessToken": "access-123",
            "refreshToken": "refresh-456",
            "nested": {"authHeader": "Bearer nested-secret", "apiKey": "api-xyz"},
        }

        payload = json.loads(formatter.format(record))

        assert payload["clientSecret"] == "[REDACTED]"
        assert payload["accessToken"] == "[REDACTED]"
        assert payload["refreshToken"] == "[REDACTED]"
        assert payload["nested"]["authHeader"] == "[REDACTED]"
        assert payload["nested"]["apiKey"] == "[REDACTED]"
        assert "abc123" not in payload["message"]

    def test_sensitive_data_filter_redacts_text_logs(self):
        """SensitiveDataFilter should redact sensitive values for text logs."""
        stream = io.StringIO()
        logger = logging.getLogger("test.redaction.text")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        original_handlers = list(logger.handlers)
        logger.handlers.clear()

        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s | %(client_secret)s | %(token)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

        try:
            logger.info(
                "Processing credentials token=abc123 password:hunter2",
                extra={"client_secret": "secret-value", "token": "token-value"},
            )
            output = stream.getvalue().strip()
            assert "[REDACTED]" in output
            assert "abc123" not in output
            assert "hunter2" not in output
            assert "secret-value" not in output
            assert "token-value" not in output
        finally:
            logger.removeHandler(handler)
            for existing in original_handlers:
                logger.addHandler(existing)

    def test_sensitive_data_filter_redacts_camelcase_extras(self):
        """SensitiveDataFilter should redact camelCase credential fields."""
        stream = io.StringIO()
        logger = logging.getLogger("test.redaction.text.camelcase")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        original_handlers = list(logger.handlers)
        logger.handlers.clear()

        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s | %(clientSecret)s | %(accessToken)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

        try:
            logger.info(
                "Authorization: Bearer should-redact",
                extra={"clientSecret": "secret-value", "accessToken": "token-value"},
            )
            output = stream.getvalue().strip()
            assert "should-redact" not in output
            assert "secret-value" not in output
            assert "token-value" not in output
            assert "Authorization: Bearer [REDACTED]" in output
        finally:
            logger.removeHandler(handler)
            for existing in original_handlers:
                logger.addHandler(existing)

    def test_logging_creates_log_directory(self, tmp_path, monkeypatch):
        """Test that logging creates the logs directory"""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        setup_logging("dv_test_12345", batch_mode=False, log_level="INFO")

        # Check that logs directory was created
        log_dir = tmp_path / "logs"
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_logging_creates_log_file(self, tmp_path, monkeypatch):
        """Test that logging creates a log file"""
        monkeypatch.chdir(tmp_path)

        setup_logging("dv_test_12345", batch_mode=False, log_level="INFO")

        # Check that a log file was created
        log_dir = tmp_path / "logs"
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0

    def test_batch_mode_log_filename(self, tmp_path, monkeypatch):
        """Test that batch mode creates correctly named log file"""
        monkeypatch.chdir(tmp_path)

        setup_logging(batch_mode=True, log_level="INFO")

        log_dir = tmp_path / "logs"
        log_files = list(log_dir.glob("SDR_Batch_Generation_*.log"))
        assert len(log_files) > 0

    def test_log_level_configuration(self, tmp_path, monkeypatch):
        """Test that log level is configured correctly"""
        monkeypatch.chdir(tmp_path)

        logger = setup_logging("dv_test", batch_mode=False, log_level="DEBUG")

        assert logger.level == logging.DEBUG or logging.root.level == logging.DEBUG


class TestConfigValidation:
    """Test configuration file validation"""

    def test_valid_config_file(self, mock_config_file):
        """Test validation of valid config file"""
        logger = logging.getLogger("test")
        # Should not raise an exception
        result = validate_config_file(mock_config_file, logger)
        assert result is True

    def test_missing_config_file(self):
        """Test validation with missing config file"""
        logger = logging.getLogger("test")
        result = validate_config_file("nonexistent_config.json", logger)
        # Should return False for missing file
        assert result is False

    def test_invalid_json_config(self, tmp_path):
        """Test validation with invalid JSON"""
        logger = logging.getLogger("test")
        invalid_config = tmp_path / "invalid_config.json"
        invalid_config.write_text("{ invalid json }")

        result = validate_config_file(str(invalid_config), logger)
        # Should return False for invalid JSON
        assert result is False

    def test_missing_required_fields(self, tmp_path):
        """Test validation with missing required fields"""
        logger = logging.getLogger("test")
        incomplete_config = tmp_path / "incomplete_config.json"
        incomplete_config.write_text(
            json.dumps(
                {
                    "org_id": "test_org",
                    "client_id": "test_client",
                    # Missing other required fields
                }
            )
        )

        # Should fail when required fields are missing
        result = validate_config_file(str(incomplete_config), logger)
        assert result is False


class TestPerformanceTracker:
    """Test performance tracking functionality"""

    def test_performance_tracker_tracks_operations(self):
        """Test that performance tracker records operations"""
        logger = logging.getLogger("test")
        tracker = PerformanceTracker(logger)

        tracker.start("test_operation")
        tracker.end("test_operation")

        assert "test_operation" in tracker.metrics
        assert tracker.metrics["test_operation"] >= 0  # Allow 0 for very fast operations

    def test_performance_tracker_multiple_operations(self):
        """Test tracking multiple operations"""
        logger = logging.getLogger("test")
        tracker = PerformanceTracker(logger)

        operations = ["op1", "op2", "op3"]
        for op in operations:
            tracker.start(op)
            tracker.end(op)

        assert len(tracker.metrics) == 3
        for op in operations:
            assert op in tracker.metrics

    def test_performance_tracker_summary(self):
        """Test that summary is generated correctly"""
        logger = logging.getLogger("test")
        tracker = PerformanceTracker(logger)

        tracker.start("test_op")
        tracker.end("test_op")

        summary = tracker.get_summary()
        assert "PERFORMANCE SUMMARY" in summary
        assert "test_op" in summary

    def test_performance_tracker_no_metrics(self):
        """Test summary with no metrics collected"""
        logger = logging.getLogger("test")
        tracker = PerformanceTracker(logger)

        summary = tracker.get_summary()
        assert "No performance metrics collected" in summary

    def test_performance_tracker_timing_accuracy(self):
        """Test that tracker measures time accurately"""
        import time

        logger = logging.getLogger("test")
        tracker = PerformanceTracker(logger)

        tracker.start("timed_op")
        time.sleep(0.1)  # Sleep for 100ms
        tracker.end("timed_op")

        # Should be at least 0.1 seconds
        assert tracker.metrics["timed_op"] >= 0.1

    def test_performance_tracker_nested_operations(self):
        """Test tracking nested/overlapping operations"""
        logger = logging.getLogger("test")
        tracker = PerformanceTracker(logger)

        tracker.start("outer_op")
        tracker.start("inner_op")
        tracker.end("inner_op")
        tracker.end("outer_op")

        assert "outer_op" in tracker.metrics
        assert "inner_op" in tracker.metrics
        # Outer operation should take longer
        assert tracker.metrics["outer_op"] >= tracker.metrics["inner_op"]


class TestErrorMessageFormatting:
    """Test error message formatting helper"""

    def test_format_error_msg_operation_only(self):
        """Test formatting with just operation"""
        msg = _format_error_msg("creating file")
        assert msg == "Error creating file"

    def test_format_error_msg_with_item_type(self):
        """Test formatting with operation and item_type"""
        msg = _format_error_msg("checking duplicates", "Metrics")
        assert msg == "Error checking duplicates for Metrics"

    def test_format_error_msg_with_error(self):
        """Test formatting with operation and error"""
        err = ValueError("test error")
        msg = _format_error_msg("processing data", error=err)
        assert msg == "Error processing data: test error"

    def test_format_error_msg_full(self):
        """Test formatting with all parameters"""
        err = RuntimeError("connection failed")
        msg = _format_error_msg("fetching API data", "Dimensions", err)
        assert msg == "Error fetching API data for Dimensions: connection failed"

    def test_format_error_msg_none_values(self):
        """Test formatting handles None gracefully"""
        msg = _format_error_msg("testing", None, None)
        assert msg == "Error testing"

    def test_format_error_msg_exception_without_message(self):
        """Test formatting with exception that has no message"""
        err = Exception()
        msg = _format_error_msg("processing", error=err)
        assert msg == "Error processing: "

    def test_format_error_msg_special_characters(self):
        """Test formatting with special characters in error"""
        err = ValueError("file 'test.txt' not found: <path>")
        msg = _format_error_msg("reading file", error=err)
        assert "file 'test.txt' not found" in msg
        assert "<path>" in msg


class TestValidationSchema:
    """Test centralized validation schema"""

    def test_validation_schema_has_required_keys(self):
        """Test schema contains all required keys"""
        assert "required_metric_fields" in VALIDATION_SCHEMA
        assert "required_dimension_fields" in VALIDATION_SCHEMA
        assert "critical_fields" in VALIDATION_SCHEMA

    def test_validation_schema_metric_fields(self):
        """Test metric required fields are correct"""
        fields = VALIDATION_SCHEMA["required_metric_fields"]
        assert "id" in fields
        assert "name" in fields
        assert "type" in fields

    def test_validation_schema_dimension_fields(self):
        """Test dimension required fields are correct"""
        fields = VALIDATION_SCHEMA["required_dimension_fields"]
        assert "id" in fields
        assert "name" in fields
        assert "type" in fields

    def test_validation_schema_critical_fields(self):
        """Test critical fields are correct"""
        fields = VALIDATION_SCHEMA["critical_fields"]
        assert "id" in fields
        assert "name" in fields
        assert "description" in fields

    def test_validation_schema_is_immutable_reference(self):
        """Test schema values are lists (can be used directly)"""
        assert isinstance(VALIDATION_SCHEMA["required_metric_fields"], list)
        assert isinstance(VALIDATION_SCHEMA["required_dimension_fields"], list)
        assert isinstance(VALIDATION_SCHEMA["critical_fields"], list)

    def test_validation_schema_integration_with_checker(self):
        """Test VALIDATION_SCHEMA works with DataQualityChecker"""
        import pandas as pd

        from cja_auto_sdr.generator import DataQualityChecker

        logger = logging.getLogger("test")
        checker = DataQualityChecker(logger)

        # Create test DataFrame with all required fields
        df = pd.DataFrame(
            {
                "id": ["m1", "m2"],
                "name": ["Metric 1", "Metric 2"],
                "type": ["int", "currency"],
                "description": ["Desc 1", "Desc 2"],
            }
        )

        # Use VALIDATION_SCHEMA values directly
        checker.check_all_quality_issues_optimized(
            df, "Metrics", VALIDATION_SCHEMA["required_metric_fields"], VALIDATION_SCHEMA["critical_fields"]
        )

        # Should have no critical issues for valid data
        critical_issues = [i for i in checker.issues if i["severity"] == "CRITICAL"]
        assert len(critical_issues) == 0
