"""Logging helpers for CJA Auto SDR."""

import atexit
import contextlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from cja_auto_sdr.core.constants import LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES

_LOG_RECORD_RESERVED_FIELDS = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime", "extra_fields"}


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging output.

    Produces JSON lines suitable for log aggregation systems (Splunk, ELK, CloudWatch).
    Each log record is a single JSON object on one line.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "process_name": record.processName,
            "thread": record.thread,
            "thread_name": record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any explicit extra fields passed to the logger.
        extra_fields = {}
        record_extra_fields = getattr(record, "extra_fields", None)
        if isinstance(record_extra_fields, dict):
            extra_fields.update(record_extra_fields)

        # Also include custom LogRecord attributes set via logging's `extra`.
        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_RESERVED_FIELDS or key.startswith("_"):
                continue
            extra_fields.setdefault(key, value)

        if extra_fields:
            log_entry.update(extra_fields)

        return json.dumps(log_entry, default=str)


# Module-level tracking to prevent duplicate logger initialization
_logging_initialized = False
_current_log_file = None
_atexit_registered = False


class ContextLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges contextual fields into record extras."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.get("extra")
        merged_extra = dict(self.extra)
        if isinstance(extra, dict):
            merged_extra.update(extra)
        kwargs["extra"] = merged_extra
        return msg, kwargs


def _unwrap_logger(logger: logging.Logger | logging.LoggerAdapter | None) -> logging.Logger | None:
    current = logger
    while isinstance(current, logging.LoggerAdapter):
        current = current.logger
    if isinstance(current, logging.Logger):
        return current
    return None


def with_log_context(
    logger: logging.Logger | logging.LoggerAdapter | object, **context: object
) -> logging.Logger | logging.LoggerAdapter | object:
    """Return a logger enriched with persistent contextual fields."""
    if not isinstance(logger, (logging.Logger, logging.LoggerAdapter)):
        # Preserve test doubles/mocks that may not satisfy logging interfaces.
        return logger

    base_logger = _unwrap_logger(logger)
    if base_logger is None:
        return logger

    normalized_context = {k: v for k, v in context.items() if v is not None}
    existing_context = {}
    if isinstance(logger, logging.LoggerAdapter):
        existing_context = dict(getattr(logger, "extra", {}))

    existing_context.update(normalized_context)
    return ContextLoggerAdapter(base_logger, existing_context)


def flush_logging_handlers(logger: logging.Logger | logging.LoggerAdapter | None = None) -> None:
    """Flush logger handlers, including propagated root handlers."""
    handlers: list[logging.Handler] = []
    seen: set[int] = set()

    unwrapped_logger = _unwrap_logger(logger)

    if unwrapped_logger is not None:
        current: logging.Logger | None = unwrapped_logger
        while current is not None:
            handlers.extend(current.handlers)
            if not current.propagate:
                break
            current = current.parent

    if not handlers:
        handlers.extend(logging.root.handlers)

    for handler in handlers:
        handler_id = id(handler)
        if handler_id in seen:
            continue
        seen.add(handler_id)
        with contextlib.suppress(Exception):
            handler.flush()


def setup_logging(
    data_view_id: str | None = None, batch_mode: bool = False, log_level: str | None = None, log_format: str = "text"
) -> logging.Logger:
    """Setup logging to both file and console.

    Args:
        data_view_id: Data view ID for log file naming
        batch_mode: Whether running in batch mode
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format - "text" (default) or "json" for structured logging

    Returns:
        Configured logger instance

    Priority: 1) Passed parameter, 2) Environment variable LOG_LEVEL, 3) Default INFO
    """
    global _logging_initialized, _current_log_file, _atexit_registered

    # Register atexit handler once to ensure logs are flushed on exit
    if not _atexit_registered:
        atexit.register(logging.shutdown)
        _atexit_registered = True

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
    except PermissionError:
        print("Warning: Cannot create logs directory (permission denied). Logging to console only.", file=sys.stderr)
        log_dir = None
    except OSError as e:
        print(f"Warning: Cannot create logs directory: {e}. Logging to console only.", file=sys.stderr)
        log_dir = None

    # Create log filename with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if log_dir is not None:
        if batch_mode:
            log_file = log_dir / f"SDR_Batch_Generation_{timestamp}.log"
        else:
            log_file = log_dir / f"SDR_Generation_{data_view_id}_{timestamp}.log"
    else:
        log_file = None

    # Determine log level with priority: parameter > env var > default
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO")

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        print(f"Warning: Invalid log level '{log_level}', using INFO", file=sys.stderr)
        log_level = "INFO"

    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any existing handlers from root logger
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # Configure logging handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        # Use RotatingFileHandler to prevent unbounded log growth
        handlers.append(RotatingFileHandler(log_file, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT))

    # Select formatter based on log_format
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Apply formatter and level to all handlers, then add to root logger
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(numeric_level)
        logging.root.addHandler(handler)

    # Set root logger level explicitly
    logging.root.setLevel(numeric_level)

    # Get the module logger
    logger = logging.getLogger("cja_auto_sdr.generator")
    # Ensure it propagates to root and doesn't have its own restrictive level
    logger.propagate = True
    logger.setLevel(logging.NOTSET)

    # Track initialization state to prevent duplicates
    _logging_initialized = True
    _current_log_file = log_file

    if log_file is not None:
        logger.info(f"Logging initialized. Log file: {log_file}")
    else:
        logger.info("Logging initialized. Console output only.")

    # Flush handlers to ensure log file is not empty even on early exit
    for handler in logging.root.handlers:
        handler.flush()

    return logger
