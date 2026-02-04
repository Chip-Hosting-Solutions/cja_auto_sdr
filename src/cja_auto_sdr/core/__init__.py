"""Core module - Foundation components with no internal dependencies.

This module provides the basic building blocks used throughout the application:
- Version information
- Custom exceptions
- Configuration dataclasses
- Constants and defaults
- Console colors and formatting utilities
"""

from cja_auto_sdr.core.version import __version__

from cja_auto_sdr.core.exceptions import (
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

from cja_auto_sdr.core.config import (
    RetryConfig,
    CacheConfig,
    LogConfig,
    WorkerConfig,
    APITuningConfig,
    CircuitState,
    CircuitBreakerConfig,
    SDRConfig,
    WizardConfig,
)

from cja_auto_sdr.core.constants import (
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
    RETRYABLE_STATUS_CODES,
    CONFIG_SCHEMA,
    JWT_DEPRECATED_FIELDS,
    ENV_VAR_MAPPING,
    CREDENTIAL_FIELDS,
    auto_detect_workers,
    infer_format_from_path,
    should_generate_format,
    _get_credential_fields,
)

from cja_auto_sdr.core.colors import (
    ConsoleColors,
    ANSIColors,
    format_file_size,
    open_file_in_default_app,
    _format_error_msg,
)

__all__ = [
    # Version
    '__version__',
    # Exceptions
    'CJASDRError',
    'ConfigurationError',
    'APIError',
    'ValidationError',
    'OutputError',
    'ProfileError',
    'ProfileNotFoundError',
    'ProfileConfigError',
    'CredentialSourceError',
    'CircuitBreakerOpen',
    'RetryableHTTPError',
    # Config dataclasses
    'RetryConfig',
    'CacheConfig',
    'LogConfig',
    'WorkerConfig',
    'APITuningConfig',
    'CircuitState',
    'CircuitBreakerConfig',
    'SDRConfig',
    'WizardConfig',
    # Constants
    'FORMAT_ALIASES',
    'EXTENSION_TO_FORMAT',
    'DEFAULT_API_FETCH_WORKERS',
    'DEFAULT_VALIDATION_WORKERS',
    'DEFAULT_BATCH_WORKERS',
    'MAX_BATCH_WORKERS',
    'AUTO_WORKERS_SENTINEL',
    'DEFAULT_CACHE_SIZE',
    'DEFAULT_CACHE_TTL',
    'LOG_FILE_MAX_BYTES',
    'LOG_FILE_BACKUP_COUNT',
    'DEFAULT_RETRY',
    'DEFAULT_CACHE',
    'DEFAULT_LOG',
    'DEFAULT_WORKERS',
    'DEFAULT_RETRY_CONFIG',
    'VALIDATION_SCHEMA',
    'RETRYABLE_STATUS_CODES',
    'CONFIG_SCHEMA',
    'JWT_DEPRECATED_FIELDS',
    'ENV_VAR_MAPPING',
    'CREDENTIAL_FIELDS',
    'auto_detect_workers',
    'infer_format_from_path',
    'should_generate_format',
    '_get_credential_fields',
    # Colors
    'ConsoleColors',
    'ANSIColors',
    'format_file_size',
    'open_file_in_default_app',
    '_format_error_msg',
]
