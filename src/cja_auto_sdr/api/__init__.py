"""API module - CJA API integration components.

This module provides:
- Error message helpers with actionable suggestions
- Circuit breaker pattern for API resilience
- Retry logic with exponential backoff

Note: Additional API components (PerformanceTracker, APIWorkerTuner, credentials, etc.)
should be imported directly from cja_auto_sdr.generator until they are extracted
into this module. This avoids circular import issues.
"""

from cja_auto_sdr.api.resilience import (
    ErrorMessageHelper,
    CircuitBreaker,
    RETRYABLE_EXCEPTIONS,
    retry_with_backoff,
    make_api_call_with_retry,
)

__all__ = [
    # Resilience (from api/resilience.py)
    'ErrorMessageHelper',
    'CircuitBreaker',
    'RETRYABLE_EXCEPTIONS',
    'retry_with_backoff',
    'make_api_call_with_retry',
]
