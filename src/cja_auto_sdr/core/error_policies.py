"""Shared exception policies for defensive command boundaries.

This module centralizes "best-effort" exception tuples used by helpers that
must degrade gracefully instead of aborting the main flow.
"""

from __future__ import annotations

import subprocess

# Best-effort boundaries should catch all Exception subclasses while still
# allowing BaseException control-flow signals (KeyboardInterrupt/SystemExit) to
# propagate.
RECOVERABLE_BEST_EFFORT_EXCEPTIONS: tuple[type[Exception], ...] = (Exception,)

# Best-effort API connectivity checks during initialization.
RECOVERABLE_CONNECTION_TEST_EXCEPTIONS: tuple[type[Exception], ...] = RECOVERABLE_BEST_EFFORT_EXCEPTIONS

# Optional dotenv bootstrap should never abort initialization.
RECOVERABLE_DOTENV_BOOTSTRAP_EXCEPTIONS: tuple[type[Exception], ...] = RECOVERABLE_BEST_EFFORT_EXCEPTIONS

# Best-effort optional inventory enrichment in snapshot and summary flows.
RECOVERABLE_OPTIONAL_ENRICHMENT_EXCEPTIONS: tuple[type[Exception], ...] = RECOVERABLE_BEST_EFFORT_EXCEPTIONS

# Best-effort "open in default app" helpers should never raise.
RECOVERABLE_OPEN_FILE_EXCEPTIONS: tuple[type[Exception], ...] = RECOVERABLE_BEST_EFFORT_EXCEPTIONS

# Lock metadata is advisory/diagnostic only; parse failures must never abort
# acquisition/recovery paths.
RECOVERABLE_LOCK_METADATA_PARSE_EXCEPTIONS: tuple[type[Exception], ...] = RECOVERABLE_BEST_EFFORT_EXCEPTIONS

# Git helpers expose tuple-return contracts and run subprocesses with
# text decoding enabled in several paths. Include ValueError so
# UnicodeDecodeError remains non-fatal.
RECOVERABLE_GIT_SUBPROCESS_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    subprocess.SubprocessError,
)
