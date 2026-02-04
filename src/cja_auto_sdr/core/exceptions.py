"""Custom exceptions for CJA Auto SDR.

All exception classes are designed to provide clear, actionable error messages
with context about what went wrong and how to fix it.
"""

from typing import Optional


class CJASDRError(Exception):
    """Base exception for all CJA SDR errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(CJASDRError):
    """Exception raised for configuration-related errors.

    Examples:
        - Missing config file
        - Invalid JSON in config file
        - Missing required credentials
        - Invalid credential format
    """

    def __init__(self, message: str, config_file: Optional[str] = None,
                 field: Optional[str] = None, details: Optional[str] = None):
        self.config_file = config_file
        self.field = field
        super().__init__(message, details)


class APIError(CJASDRError):
    """Exception raised for API communication failures.

    Wraps HTTP errors and network failures with context about
    the operation that failed.
    """

    def __init__(self, message: str, status_code: Optional[int] = None,
                 operation: Optional[str] = None, details: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.status_code = status_code
        self.operation = operation
        self.original_error = original_error
        super().__init__(message, details)

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        if self.operation:
            parts.append(f"during {self.operation}")
        if self.details:
            parts.append(self.details)
        return " - ".join(parts)


class ValidationError(CJASDRError):
    """Exception raised for data quality validation failures.

    Used when validation encounters critical issues that prevent
    further processing.
    """

    def __init__(self, message: str, item_type: Optional[str] = None,
                 issue_count: int = 0, details: Optional[str] = None):
        self.item_type = item_type
        self.issue_count = issue_count
        super().__init__(message, details)


class OutputError(CJASDRError):
    """Exception raised for file writing failures.

    Examples:
        - Permission denied
        - Disk full
        - Invalid path
        - Serialization error
    """

    def __init__(self, message: str, output_path: Optional[str] = None,
                 output_format: Optional[str] = None, details: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.output_path = output_path
        self.output_format = output_format
        self.original_error = original_error
        super().__init__(message, details)


class ProfileError(CJASDRError):
    """Base exception for profile-related errors.

    Used when operations involving credential profiles fail.
    """

    def __init__(self, message: str, profile_name: Optional[str] = None,
                 details: Optional[str] = None):
        self.profile_name = profile_name
        super().__init__(message, details)


class ProfileNotFoundError(ProfileError):
    """Raised when a profile directory doesn't exist.

    Examples:
        - Profile directory not found in ~/.cja/orgs/
        - Neither config.json nor .env exists in profile directory
    """
    pass


class ProfileConfigError(ProfileError):
    """Raised when a profile has invalid configuration.

    Examples:
        - Invalid JSON in config.json
        - Missing required credentials
        - Invalid profile name format
    """
    pass


class CredentialSourceError(CJASDRError):
    """Exception raised when credential loading fails from any source.

    Provides consistent error handling across all credential sources
    (profiles, environment variables, config files).

    Attributes:
        source: Name of the credential source (e.g., "profile", "env", "config_file")
        reason: Why the credential loading failed
    """

    def __init__(self, message: str, source: str,
                 reason: Optional[str] = None, details: Optional[str] = None):
        self.source = source
        self.reason = reason
        super().__init__(message, details)

    def __str__(self) -> str:
        parts = [f"[{self.source}] {self.message}"]
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        if self.details:
            parts.append(self.details)
        return " - ".join(parts)


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open and request is rejected."""

    def __init__(self, message: str = "Circuit breaker is open", time_until_retry: float = 0):
        self.message = message
        self.time_until_retry = time_until_retry
        super().__init__(self.message)


class RetryableHTTPError(Exception):
    """Exception raised when API returns a retryable HTTP status code."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}" if message else f"HTTP {status_code}")
