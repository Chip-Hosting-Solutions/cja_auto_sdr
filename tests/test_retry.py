"""
Tests for retry with exponential backoff functionality
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, ".")

from cja_auto_sdr.api.resilience import _effective_retry_config
from cja_auto_sdr.generator import (
    DEFAULT_RETRY_CONFIG,
    RETRYABLE_EXCEPTIONS,
    make_api_call_with_retry,
    retry_with_backoff,
)


class TestRetryDecorator:
    """Test the retry_with_backoff decorator"""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, jitter=False)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_connection_error(self):
        """Test that ConnectionError triggers retry"""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3  # 1 initial + 2 retries

    def test_retry_on_timeout_error(self):
        """Test that TimeoutError triggers retry"""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Request timed out")
            return "success"

        result = timeout_func()
        assert result == "success"
        assert call_count == 2

    def test_no_retry_on_value_error(self):
        """Test that non-retryable exceptions are not retried"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def value_error_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid value")

        with pytest.raises(ValueError):
            value_error_func()
        assert call_count == 1  # No retry for ValueError

    def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries"""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()
        assert call_count == 3  # 1 initial + 2 retries

    def test_exponential_backoff_delays(self):
        """Test that delays increase exponentially"""
        delays = []
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.1, exponential_base=2, jitter=False)
        def track_delay_func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ConnectionError("Fail")
            return "success"

        with patch("time.sleep") as mock_sleep:
            track_delay_func()
            # Check delays: 0.1, 0.2, 0.4 (base * 2^attempt)
            assert mock_sleep.call_count == 3
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert abs(delays[0] - 0.1) < 0.01  # First delay
            assert abs(delays[1] - 0.2) < 0.01  # Second delay
            assert abs(delays[2] - 0.4) < 0.01  # Third delay

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay"""

        @retry_with_backoff(max_retries=5, base_delay=10, max_delay=15, exponential_base=2, jitter=False)
        def capped_delay_func():
            raise ConnectionError("Fail")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ConnectionError):
                capped_delay_func()
            # All delays should be capped at 15
            for call in mock_sleep.call_args_list:
                assert call.args[0] <= 15

    def test_jitter_adds_randomization(self):
        """Test that jitter adds randomization to delays"""
        delays_with_jitter = []

        @retry_with_backoff(max_retries=10, base_delay=1, jitter=True)
        def jitter_func():
            raise ConnectionError("Fail")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ConnectionError):
                jitter_func()
            delays_with_jitter = [call.args[0] for call in mock_sleep.call_args_list]

        # With jitter, delays should vary (not all identical)
        # Due to random.uniform(0.5, 1.5), values should differ
        unique_delays = set(round(d, 2) for d in delays_with_jitter)
        # Should have some variation (not all exactly the same)
        assert len(unique_delays) > 1 or len(delays_with_jitter) <= 1

    def test_custom_retryable_exceptions(self):
        """Test custom exception types for retry"""
        call_count = 0

        class CustomError(Exception):
            pass

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False, retryable_exceptions=(CustomError,))
        def custom_error_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise CustomError("Custom retry error")
            return "success"

        result = custom_error_func()
        assert result == "success"
        assert call_count == 3


class TestMakeApiCallWithRetry:
    """Test the make_api_call_with_retry function"""

    def test_successful_api_call(self):
        """Test successful API call returns result"""
        mock_api = Mock(return_value={"data": "test"})

        result = make_api_call_with_retry(mock_api, "arg1", kwarg1="value1", operation_name="test_api")

        assert result == {"data": "test"}
        mock_api.assert_called_once_with("arg1", kwarg1="value1")

    def test_api_call_with_retry(self):
        """Test API call retries on network error"""
        mock_api = Mock(
            side_effect=[ConnectionError("Network error"), ConnectionError("Network error"), {"data": "success"}]
        )

        with patch("time.sleep"):  # Skip actual delays
            result = make_api_call_with_retry(mock_api, "arg1", operation_name="test_api")

        assert result == {"data": "success"}
        assert mock_api.call_count == 3

    def test_api_call_max_retries_exceeded(self):
        """Test API call raises after max retries"""
        mock_api = Mock(side_effect=ConnectionError("Always fails"))

        with patch("time.sleep"):  # Skip actual delays
            with pytest.raises(ConnectionError):
                make_api_call_with_retry(mock_api, "arg1", operation_name="test_api")

        # 1 initial + 3 retries (default max_retries)
        assert mock_api.call_count == DEFAULT_RETRY_CONFIG["max_retries"] + 1

    def test_api_call_logs_warnings(self):
        """Test that retry attempts are logged"""
        mock_api = Mock(side_effect=[ConnectionError("Error 1"), {"data": "success"}])
        mock_logger = Mock()

        with patch("time.sleep"):
            result = make_api_call_with_retry(mock_api, logger=mock_logger, operation_name="test_api")

        assert result == {"data": "success"}
        # Should have logged a warning for the failed attempt
        mock_logger.warning.assert_called()


class TestDefaultConfig:
    """Test default configuration values"""

    def test_default_max_retries(self):
        """Test default max_retries value"""
        assert DEFAULT_RETRY_CONFIG["max_retries"] == 3

    def test_default_base_delay(self):
        """Test default base_delay value"""
        assert DEFAULT_RETRY_CONFIG["base_delay"] == 1.0

    def test_default_max_delay(self):
        """Test default max_delay value"""
        assert DEFAULT_RETRY_CONFIG["max_delay"] == 30.0

    def test_default_exponential_base(self):
        """Test default exponential_base value"""
        assert DEFAULT_RETRY_CONFIG["exponential_base"] == 2

    def test_default_jitter(self):
        """Test default jitter value"""
        assert DEFAULT_RETRY_CONFIG["jitter"] is True

    def test_retryable_exceptions(self):
        """Test retryable exception types"""
        assert ConnectionError in RETRYABLE_EXCEPTIONS
        assert TimeoutError in RETRYABLE_EXCEPTIONS
        assert OSError in RETRYABLE_EXCEPTIONS


class TestRetryIntegration:
    """Integration tests for retry functionality"""

    def test_retry_preserves_function_metadata(self):
        """Test that decorator preserves function metadata"""

        @retry_with_backoff()
        def documented_function():
            """This is a documented function"""
            return True

        assert documented_function.__name__ == "documented_function"
        assert "documented" in documented_function.__doc__

    def test_retry_works_with_class_methods(self):
        """Test retry works with class methods via make_api_call_with_retry"""

        class ApiClient:
            def __init__(self):
                self.call_count = 0

            def get_data(self, data_id):
                self.call_count += 1
                if self.call_count < 2:
                    raise ConnectionError("Network error")
                return {"id": data_id, "data": "test"}

        client = ApiClient()

        with patch("time.sleep"):
            result = make_api_call_with_retry(client.get_data, "test_id", operation_name="get_data")

        assert result == {"id": "test_id", "data": "test"}
        assert client.call_count == 2


class TestRetryEnvOverrides:
    """Tests for environment-driven retry configuration."""

    def test_effective_retry_config_ignores_invalid_env(self):
        """Invalid env values should not raise and should keep defaults."""
        with patch.dict(
            os.environ,
            {"MAX_RETRIES": "bad", "RETRY_BASE_DELAY": "nope", "RETRY_MAX_DELAY": "nanx"},
            clear=False,
        ):
            cfg = _effective_retry_config()

        assert cfg["max_retries"] == DEFAULT_RETRY_CONFIG["max_retries"]
        assert cfg["base_delay"] == DEFAULT_RETRY_CONFIG["base_delay"]
        assert cfg["max_delay"] == DEFAULT_RETRY_CONFIG["max_delay"]
