"""Logging helpers for CJA Auto SDR."""

import atexit
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from cja_auto_sdr.core.constants import LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging output.

    Produces JSON lines suitable for log aggregation systems (Splunk, ELK, CloudWatch).
    Each log record is a single JSON object on one line.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed to the logger
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


# Module-level tracking to prevent duplicate logger initialization
_logging_initialized = False
_current_log_file = None
_atexit_registered = False


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
