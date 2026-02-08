import argparse
import contextlib
import csv
import html
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypeVar, runtime_checkable

import cjapy
import pandas as pd
from tqdm import tqdm

# Attempt to load python-dotenv if available (optional dependency)
_DOTENV_AVAILABLE = False
_DOTENV_LOADED = False
try:
    from dotenv import load_dotenv

    _DOTENV_LOADED = load_dotenv()  # Returns True if .env file was found and loaded
    _DOTENV_AVAILABLE = True
except ImportError:
    pass  # python-dotenv not installed

# Attempt to load argcomplete for shell tab-completion (optional dependency)
_ARGCOMPLETE_AVAILABLE = False
try:
    import argcomplete

    _ARGCOMPLETE_AVAILABLE = True
except ImportError:
    pass  # argcomplete not installed

# ==================== IMPORTS FROM MODULAR SUBPACKAGES ====================
# These imports are from the new modular structure introduced in v3.2.0
# They are re-exported here for backwards compatibility

from cja_auto_sdr.api.resilience import (
    RETRYABLE_EXCEPTIONS,
    CircuitBreaker,
    ErrorMessageHelper,
    make_api_call_with_retry,
    retry_with_backoff,
)
from cja_auto_sdr.core.colors import (
    ConsoleColors,
    _format_error_msg,
    format_file_size,
    open_file_in_default_app,
)
from cja_auto_sdr.core.config import (
    APITuningConfig,
    CacheConfig,
    CircuitBreakerConfig,
    CircuitState,
    LogConfig,
    RetryConfig,
    SDRConfig,
    WorkerConfig,
)
from cja_auto_sdr.core.constants import (
    AUTO_WORKERS_SENTINEL,
    BANNER_WIDTH,
    CONFIG_SCHEMA,
    CREDENTIAL_FIELDS,
    DEFAULT_API_FETCH_WORKERS,
    DEFAULT_BATCH_WORKERS,
    DEFAULT_CACHE,
    DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_TTL,
    DEFAULT_LOG,
    DEFAULT_RETRY,
    DEFAULT_RETRY_CONFIG,
    DEFAULT_VALIDATION_WORKERS,
    DEFAULT_WORKERS,
    ENV_VAR_MAPPING,
    EXTENSION_TO_FORMAT,
    FORMAT_ALIASES,
    JWT_DEPRECATED_FIELDS,
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_MAX_BYTES,
    MAX_BATCH_WORKERS,
    RETRYABLE_STATUS_CODES,
    VALIDATION_SCHEMA,
    _get_credential_fields,
    auto_detect_workers,
    infer_format_from_path,
    should_generate_format,
)
from cja_auto_sdr.core.exceptions import (
    APIError,
    CircuitBreakerOpen,
    CJASDRError,
    ConfigurationError,
    CredentialSourceError,
    OutputError,
    ProfileConfigError,
    ProfileError,
    ProfileNotFoundError,
    RetryableHTTPError,
    ValidationError,
)
from cja_auto_sdr.core.version import __version__
from cja_auto_sdr.org.analyzer import OrgComponentAnalyzer
from cja_auto_sdr.org.cache import OrgReportCache

# ==================== LEGACY DEFINITIONS REMOVED ====================
# The following sections have been moved to cja_auto_sdr.core:
# - Version (__version__) -> core/version.py
# - Exceptions (CJASDRError, etc.) -> core/exceptions.py
# - Config dataclasses (RetryConfig, etc.) -> core/config.py
# - Constants (FORMAT_ALIASES, etc.) -> core/constants.py
# - Colors (ConsoleColors, etc.) -> core/colors.py
#
# The following sections have been moved to cja_auto_sdr.api:
# - ErrorMessageHelper -> api/resilience.py
# - CircuitBreaker -> api/resilience.py
# - RETRYABLE_EXCEPTIONS -> api/resilience.py
# - retry_with_backoff -> api/resilience.py
# - make_api_call_with_retry -> api/resilience.py
#
# They are imported above for backwards compatibility.
# ==================== ORG-WIDE ANALYSIS IMPORTS ====================
# These were moved to cja_auto_sdr.org in v3.2.0
# They are re-exported here for backwards compatibility
from cja_auto_sdr.org.models import (
    ComponentDistribution,
    ComponentInfo,
    DataViewCluster,
    DataViewSummary,
    OrgReportComparison,
    OrgReportConfig,
    OrgReportResult,
    SimilarityPair,
)

# ==================== LEGACY ORG-WIDE DEFINITIONS (REMOVED) ====================
# The following classes have been moved to cja_auto_sdr.org:
# - OrgReportConfig -> org/models.py
# - ComponentInfo -> org/models.py
# - DataViewSummary -> org/models.py
# - SimilarityPair -> org/models.py
# - DataViewCluster -> org/models.py
# - ComponentDistribution -> org/models.py
# - OrgReportResult -> org/models.py
# - OrgReportComparison -> org/models.py
# - OrgReportCache -> org/cache.py
# - OrgComponentAnalyzer -> org/analyzer.py


TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"

# ==================== OUTPUT WRITER PROTOCOL ====================


@runtime_checkable
class OutputWriter(Protocol):
    """Protocol defining the interface for output format writers.

    All output writers should implement this interface to ensure
    consistent behavior and enable easy addition of new formats.

    Example implementation:
        class CSVWriter:
            def write(
                self,
                metrics_df: pd.DataFrame,
                dimensions_df: pd.DataFrame,
                dataview_info: dict,
                output_path: Path,
                quality_results: Optional[List[Dict]] = None
            ) -> str:
                # Write CSV files and return output path
                ...
    """

    def write(
        self,
        metrics_df: pd.DataFrame,
        dimensions_df: pd.DataFrame,
        dataview_info: dict[str, Any],
        output_path: Path,
        quality_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """Write output in the implemented format.

        Args:
            metrics_df: DataFrame containing metrics data
            dimensions_df: DataFrame containing dimensions data
            dataview_info: Dictionary with data view metadata
            output_path: Base path for output files
            quality_results: Optional list of data quality issues

        Returns:
            Path to the created output file(s)

        Raises:
            OutputError: If writing fails
        """
        ...


# Type alias for validation issues
ValidationIssue = dict[str, Any]

# Type variable for generic return types
T = TypeVar("T")


# Note: Constants, error formatting, and ConsoleColors have been moved to
# cja_auto_sdr.core and are imported above.
# Note: ErrorMessageHelper, CircuitBreaker, retry_with_backoff, and
# make_api_call_with_retry have been moved to cja_auto_sdr.api.resilience

# ==================== PLACEHOLDER FOR DELETED API RESILIENCE CODE ====================
# The following large section (ErrorMessageHelper, RetryableHTTPError, RETRYABLE_EXCEPTIONS,
# CircuitBreaker, retry_with_backoff, make_api_call_with_retry) has been moved to
# cja_auto_sdr.api.resilience and imported above.
# ==================== DATA STRUCTURES ====================


@dataclass
class ProcessingResult:
    """Result of processing a single data view"""

    data_view_id: str
    data_view_name: str
    success: bool
    duration: float
    metrics_count: int = 0
    dimensions_count: int = 0
    dq_issues_count: int = 0
    dq_issues: list[ValidationIssue] = field(default_factory=list)
    dq_severity_counts: dict[str, int] = field(default_factory=dict)
    output_file: str = ""
    error_message: str = ""
    file_size_bytes: int = 0
    # Inventory statistics (populated when inventory options are used)
    segments_count: int = 0
    segments_high_complexity: int = 0
    calculated_metrics_count: int = 0
    calculated_metrics_high_complexity: int = 0
    derived_fields_count: int = 0
    derived_fields_high_complexity: int = 0

    @property
    def file_size_formatted(self) -> str:
        """Return human-readable file size (e.g., '1.5 MB', '256 KB')."""
        return format_file_size(self.file_size_bytes)

    @property
    def has_inventory(self) -> bool:
        """Check if any inventory data was collected."""
        return self.segments_count > 0 or self.calculated_metrics_count > 0 or self.derived_fields_count > 0

    @property
    def total_high_complexity(self) -> int:
        """Total count of high-complexity items across all inventories."""
        return (
            self.segments_high_complexity
            + self.calculated_metrics_high_complexity
            + self.derived_fields_high_complexity
        )


@dataclass
class WorkerArgs:
    """Arguments for process_single_dataview_worker (replaces opaque tuple)."""

    data_view_id: str
    config_file: str = "config.json"
    output_dir: str = "."
    log_level: str = "INFO"
    log_format: str = "text"
    output_format: str = "excel"
    enable_cache: bool = False
    cache_size: int = 1000
    cache_ttl: int = 3600
    quiet: bool = False
    skip_validation: bool = False
    max_issues: int = 0
    clear_cache: bool = False
    show_timings: bool = False
    metrics_only: bool = False
    dimensions_only: bool = False
    profile: str | None = None
    shared_cache: Any = None
    api_tuning_config: Any = None
    circuit_breaker_config: Any = None
    include_derived_inventory: bool = False
    include_calculated_metrics: bool = False
    include_segments_inventory: bool = False
    inventory_only: bool = False
    inventory_order: str | None = None
    quality_report_only: bool = False


def _exit_error(msg: str) -> NoReturn:
    """Print a coloured error message to stderr and exit with code 1."""
    print(ConsoleColors.error(f"ERROR: {msg}"), file=sys.stderr)
    sys.exit(1)


def _cli_option_specified(option_name: str, argv: list[str] | None = None) -> bool:
    """Return True if an option was explicitly provided via --flag or --flag=value."""
    tokens = argv if argv is not None else sys.argv[1:]
    return any(token == option_name or token.startswith(f"{option_name}=") for token in tokens)


QUALITY_SEVERITY_ORDER: tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
QUALITY_SEVERITY_RANK = {severity: index for index, severity in enumerate(QUALITY_SEVERITY_ORDER)}
DEFAULT_AUTO_PRUNE_KEEP_LAST = 20
DEFAULT_AUTO_PRUNE_KEEP_SINCE = "30d"
QUALITY_REPORT_PREFERRED_COLUMNS: tuple[str, ...] = (
    "Data View ID",
    "Data View Name",
    "Severity",
    "Category",
    "Type",
    "Item Name",
    "Issue",
    "Details",
)


def normalize_quality_severity(severity: str) -> str:
    """Normalize severity input and validate against supported values."""
    normalized = severity.upper()
    if normalized not in QUALITY_SEVERITY_RANK:
        raise ValueError(f"Invalid quality severity: {severity}")
    return normalized


def count_quality_issues_by_severity(issues: list[dict[str, Any]]) -> dict[str, int]:
    """Count quality issues by severity in canonical order."""
    counts = {severity: 0 for severity in QUALITY_SEVERITY_ORDER}
    for issue in issues:
        severity = str(issue.get("Severity", "")).upper()
        if severity in counts:
            counts[severity] += 1
    return {severity: count for severity, count in counts.items() if count > 0}


def has_quality_issues_at_or_above(issues: list[dict[str, Any]], threshold: str) -> bool:
    """Return True if at least one issue meets/exceeds the configured severity."""
    threshold_rank = QUALITY_SEVERITY_RANK[normalize_quality_severity(threshold)]
    for issue in issues:
        severity = str(issue.get("Severity", "")).upper()
        if severity in QUALITY_SEVERITY_RANK and QUALITY_SEVERITY_RANK[severity] <= threshold_rank:
            return True
    return False


def aggregate_quality_issues(results: list[ProcessingResult]) -> list[dict[str, Any]]:
    """Flatten quality issues across processing results with data view context."""
    issues: list[dict[str, Any]] = []
    for result in results:
        for issue in result.dq_issues:
            issue_with_context = dict(issue)
            issue_with_context.setdefault("Data View ID", result.data_view_id)
            issue_with_context.setdefault("Data View Name", result.data_view_name)
            issues.append(issue_with_context)
    return issues


def resolve_auto_prune_retention(
    keep_last: int,
    keep_since: str | None,
    auto_prune: bool,
    keep_last_specified: bool = False,
    keep_since_specified: bool = False,
) -> tuple[int, str | None]:
    """Resolve effective retention settings for auto-prune defaults.

    Defaults are applied only when both retention flags were omitted.
    Explicit values (including --keep-last 0) are preserved.
    """
    effective_keep_last = keep_last
    effective_keep_since = keep_since

    if auto_prune and not keep_last_specified and not keep_since_specified:
        if effective_keep_last <= 0:
            effective_keep_last = DEFAULT_AUTO_PRUNE_KEEP_LAST
        if not effective_keep_since:
            effective_keep_since = DEFAULT_AUTO_PRUNE_KEEP_SINCE

    return effective_keep_last, effective_keep_since


def _build_quality_report_dataframe(issues: list[dict[str, Any]]) -> pd.DataFrame:
    """Build quality report dataframe with stable columns for empty/non-empty output."""
    if not issues:
        return pd.DataFrame(columns=list(QUALITY_REPORT_PREFERRED_COLUMNS))

    df = pd.DataFrame(issues)
    preferred_cols = [col for col in QUALITY_REPORT_PREFERRED_COLUMNS if col in df.columns]
    other_cols = [col for col in df.columns if col not in preferred_cols]
    return df[preferred_cols + other_cols]


def write_quality_report_output(
    issues: list[dict[str, Any]],
    report_format: str,
    output: str | None,
    output_dir: str | Path,
) -> str:
    """Write standalone quality report in JSON or CSV format."""
    output_to_stdout = output in ("-", "stdout")
    report_format = report_format.lower()

    if report_format not in ("json", "csv"):
        raise ValueError(f"Unsupported quality report format: {report_format}")

    issues_df = _build_quality_report_dataframe(issues)

    if output_to_stdout:
        if report_format == "json":
            json.dump(issues, sys.stdout, indent=2, ensure_ascii=False)
            print()
        else:
            issues_df.to_csv(sys.stdout, index=False)
        return "stdout"

    if output:
        output_path = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"quality_report_{timestamp}.{report_format}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if report_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
    else:
        issues_df.to_csv(output_path, index=False)

    return str(output_path)


def append_github_step_summary(markdown: str, logger: logging.Logger | None = None) -> bool:
    """Append markdown to GitHub Actions job summary when available."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return False

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown.rstrip() + "\n\n")
        return True
    except OSError as e:
        if logger is not None:
            logger.warning(f"Failed to write GitHub step summary: {e}")
        return False


def build_quality_step_summary(results: list[ProcessingResult]) -> str:
    """Build markdown summary table for quality issues."""
    total_views = len(results)
    successful_views = sum(1 for r in results if r.success)
    all_issues = aggregate_quality_issues(results)
    severity_counts = count_quality_issues_by_severity(all_issues)

    lines = [
        "### Data Quality Summary",
        "",
        f"- Data views processed: {successful_views}/{total_views}",
        f"- Total quality issues: {len(all_issues)}",
        "",
    ]

    if severity_counts:
        lines.extend(["| Severity | Count |", "|---|---:|"])
        for severity in QUALITY_SEVERITY_ORDER:
            count = severity_counts.get(severity, 0)
            if count > 0:
                lines.append(f"| {severity} | {count} |")
        lines.append("")

    lines.extend(["| Data View | ID | Issues | Highest Severity |", "|---|---|---:|---|"])
    for result in results:
        highest = "NONE"
        for severity in QUALITY_SEVERITY_ORDER:
            if result.dq_severity_counts.get(severity, 0) > 0:
                highest = severity
                break
        lines.append(
            f"| {result.data_view_name or '-'} | `{result.data_view_id}` | {result.dq_issues_count} | {highest} |"
        )

    return "\n".join(lines)


def build_diff_step_summary(diff_result: "DiffResult") -> str:
    """Build markdown summary table for diff output."""
    summary = diff_result.summary
    total_changes = summary.total_changes + summary.calc_metrics_changed + summary.segments_changed
    lines = [
        "### Diff Summary",
        "",
        f"- Source: `{diff_result.metadata_diff.source_id}` ({diff_result.metadata_diff.source_name})",
        f"- Target: `{diff_result.metadata_diff.target_id}` ({diff_result.metadata_diff.target_name})",
        f"- Total changes: {total_changes}",
        "",
        "| Type | Source | Target | Added | Removed | Modified | Unchanged |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Metrics | {summary.source_metrics_count} | {summary.target_metrics_count} | "
            f"{summary.metrics_added} | {summary.metrics_removed} | {summary.metrics_modified} | {summary.metrics_unchanged} |"
        ),
        (
            f"| Dimensions | {summary.source_dimensions_count} | {summary.target_dimensions_count} | "
            f"{summary.dimensions_added} | {summary.dimensions_removed} | {summary.dimensions_modified} | {summary.dimensions_unchanged} |"
        ),
    ]

    if summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0:
        lines.append(
            (
                f"| Calc Metrics | {summary.source_calc_metrics_count} | {summary.target_calc_metrics_count} | "
                f"{summary.calc_metrics_added} | {summary.calc_metrics_removed} | {summary.calc_metrics_modified} | {summary.calc_metrics_unchanged} |"
            )
        )
    if summary.source_segments_count > 0 or summary.target_segments_count > 0:
        lines.append(
            (
                f"| Segments | {summary.source_segments_count} | {summary.target_segments_count} | "
                f"{summary.segments_added} | {summary.segments_removed} | {summary.segments_modified} | {summary.segments_unchanged} |"
            )
        )

    return "\n".join(lines)


def build_org_step_summary(result: "OrgReportResult") -> str:
    """Build markdown summary table for org-report output."""
    dist = result.distribution
    lines = [
        "### Org Report Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Data Views Analyzed | {result.successful_data_views} / {result.total_data_views} |",
        f"| Total Unique Components | {result.total_unique_components} |",
        f"| Total Unique Metrics | {result.total_unique_metrics} |",
        f"| Total Unique Dimensions | {result.total_unique_dimensions} |",
        f"| Core Components | {dist.total_core} |",
        f"| Common Components | {dist.total_common} |",
        f"| Limited Components | {dist.total_limited} |",
        f"| Isolated Components | {dist.total_isolated} |",
        f"| Recommendations | {len(result.recommendations)} |",
        f"| Governance Violations | {len(result.governance_violations or [])} |",
        f"| Duration (s) | {result.duration:.2f} |",
    ]
    return "\n".join(lines)


# ==================== DIFF COMPARISON ====================

from cja_auto_sdr.api.cache import SharedValidationCache, ValidationCache

# ==================== API WORKER TUNER (moved to api/tuning.py) ====================
from cja_auto_sdr.api.tuning import APIWorkerTuner

# ==================== LOGGING SETUP ====================
# ==================== LOGGING (moved to core/logging.py) ====================
from cja_auto_sdr.core.logging import JSONFormatter, setup_logging

# ==================== PERFORMANCE TRACKING (moved to core/perf.py) ====================
from cja_auto_sdr.core.perf import PerformanceTracker
from cja_auto_sdr.diff.comparator import DataViewComparator
from cja_auto_sdr.diff.git import (
    generate_git_commit_message,
    git_commit_snapshot,
    git_get_user_info,
    git_init_snapshot_repo,
    is_git_repository,
    save_git_friendly_snapshot,
)
from cja_auto_sdr.diff.models import (
    ChangeType,
    ComponentDiff,
    DataViewSnapshot,
    DiffResult,
    DiffSummary,
    InventoryItemDiff,
    MetadataDiff,
)
from cja_auto_sdr.diff.snapshot import SnapshotManager, parse_retention_period

# ==================== PROFILE MANAGEMENT ====================


def get_cja_home() -> Path:
    """Get CJA home directory (~/.cja or $CJA_HOME).

    Returns:
        Path to CJA home directory
    """
    cja_home = os.environ.get("CJA_HOME")
    if cja_home:
        return Path(cja_home).expanduser()
    return Path.home() / ".cja"


def get_profiles_dir() -> Path:
    """Get profiles directory (~/.cja/orgs/).

    Returns:
        Path to profiles directory
    """
    return get_cja_home() / "orgs"


def get_profile_path(profile_name: str) -> Path:
    """Get path to a specific profile directory.

    Args:
        profile_name: Name of the profile

    Returns:
        Path to profile directory
    """
    return get_profiles_dir() / profile_name


def validate_profile_name(name: str) -> tuple[bool, str | None]:
    """Validate profile name (alphanumeric, dashes, underscores only).

    Args:
        name: Profile name to validate

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not name:
        return False, "Profile name cannot be empty"

    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", name):
        return False, (
            f"Profile name '{name}' is invalid. "
            "Must start with alphanumeric and contain only letters, numbers, dashes, and underscores."
        )

    if len(name) > 64:
        return False, f"Profile name '{name}' is too long (max 64 characters)"

    return True, None


def load_profile_config_json(profile_path: Path) -> dict[str, str] | None:
    """Load credentials from profile's config.json.

    Args:
        profile_path: Path to profile directory

    Returns:
        Dictionary with credentials if config.json exists and is valid, None otherwise
    """
    config_file = profile_path / "config.json"
    if not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            config = json.load(f)
        if isinstance(config, dict):
            return {k: str(v).strip() for k, v in config.items() if v}
        return None
    except OSError, json.JSONDecodeError:
        return None


def load_profile_dotenv(profile_path: Path) -> dict[str, str] | None:
    """Load credentials from profile's .env file.

    Args:
        profile_path: Path to profile directory

    Returns:
        Dictionary with credentials if .env exists and is valid, None otherwise
    """
    env_file = profile_path / ".env"
    if not env_file.exists():
        return None

    credentials = {}
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        # Map env var names to config field names
                        config_key = key.lower()
                        # Use CREDENTIAL_FIELDS for allowed fields (single source of truth)
                        if config_key in CREDENTIAL_FIELDS["all"]:
                            credentials[config_key] = value
    except OSError:
        return None

    return credentials if credentials else None


def load_profile_credentials(profile_name: str, logger: logging.Logger) -> dict[str, str] | None:
    """Load and merge credentials from profile (config.json + .env).

    Precedence within profile:
    1. .env values override config.json values
    2. Matches existing behavior (env vars > config file)

    Args:
        profile_name: Name of the profile to load
        logger: Logger instance

    Returns:
        Dictionary with merged credentials, or None if profile not found

    Raises:
        ProfileNotFoundError: If profile directory doesn't exist
        ProfileConfigError: If profile has no valid configuration
    """
    # Validate profile name
    is_valid, error_msg = validate_profile_name(profile_name)
    if not is_valid:
        raise ProfileConfigError(error_msg, profile_name=profile_name)

    profile_path = get_profile_path(profile_name)

    if not profile_path.exists():
        raise ProfileNotFoundError(
            f"Profile '{profile_name}' not found",
            profile_name=profile_name,
            details=f"Expected directory: {profile_path}",
        )

    if not profile_path.is_dir():
        raise ProfileConfigError(
            "Profile path is not a directory", profile_name=profile_name, details=str(profile_path)
        )

    # Load config.json first
    credentials = load_profile_config_json(profile_path) or {}
    json_source = bool(credentials)

    # Override with .env values (skip empty values to avoid overwriting valid config)
    env_credentials = load_profile_dotenv(profile_path)
    if env_credentials:
        credentials.update({k: v for k, v in env_credentials.items() if v})

    if not credentials:
        raise ProfileConfigError(
            f"Profile '{profile_name}' has no configuration",
            profile_name=profile_name,
            details=f"Expected config.json or .env in {profile_path}",
        )

    if json_source and env_credentials:
        logger.debug(f"Profile '{profile_name}': merged config.json with .env overrides")
    elif json_source:
        logger.debug(f"Profile '{profile_name}': loaded from config.json")
    else:
        logger.debug(f"Profile '{profile_name}': loaded from .env")

    return credentials


def resolve_active_profile(cli_profile: str | None = None) -> str | None:
    """Resolve active profile: --profile > CJA_PROFILE > None.

    Args:
        cli_profile: Profile name from --profile CLI argument

    Returns:
        Active profile name, or None if no profile is active
    """
    if cli_profile:
        return cli_profile
    return os.environ.get("CJA_PROFILE")


def list_profiles(output_format: str = "table") -> bool:
    """List all profiles with status indicators.

    Args:
        output_format: Output format - "table" (default) or "json"

    Returns:
        True if successful, False otherwise
    """
    profiles_dir = get_profiles_dir()

    if not profiles_dir.exists():
        if output_format == "json":
            print(json.dumps({"profiles": [], "count": 0}, indent=2))
        else:
            print()
            print("No profiles directory found.")
            print(f"Expected: {profiles_dir}")
            print()
            print("To create profiles, run:")
            print("  cja_auto_sdr --profile-add <profile-name>")
            print()
        return True

    profiles = []
    active_profile = os.environ.get("CJA_PROFILE")

    for item in sorted(profiles_dir.iterdir()):
        if not item.is_dir():
            continue

        profile_name = item.name
        has_config_json = (item / "config.json").exists()
        has_env = (item / ".env").exists()
        is_active = profile_name == active_profile

        if has_config_json or has_env:
            config_source = []
            if has_config_json:
                config_source.append("config.json")
            if has_env:
                config_source.append(".env")

            profiles.append({"name": profile_name, "active": is_active, "sources": config_source, "path": str(item)})

    if output_format == "json":
        print(json.dumps({"profiles": profiles, "count": len(profiles), "active": active_profile}, indent=2))
    else:
        print()
        print("=" * BANNER_WIDTH)
        print("AVAILABLE PROFILES")
        print("=" * BANNER_WIDTH)
        print()

        if not profiles:
            print("No profiles found.")
            print()
            print("To create a profile, run:")
            print("  cja_auto_sdr --profile-add <profile-name>")
        else:
            print(f"{'Profile':<25} {'Sources':<20} {'Status'}")
            print("-" * 60)
            for p in profiles:
                status = "[active]" if p["active"] else ""
                sources = ", ".join(p["sources"])
                marker = "●" if p["active"] else " "
                print(f"{marker} {p['name']:<23} {sources:<20} {status}")

            print()
            print(f"Total: {len(profiles)} profile(s)")
            print()
            print("Usage:")
            print("  cja_auto_sdr --profile <name> --list-dataviews")
            print("  export CJA_PROFILE=<name>")

        print()

    return True


def add_profile_interactive(profile_name: str) -> bool:
    """Interactively create a new profile.

    Args:
        profile_name: Name for the new profile

    Returns:
        True if profile created successfully, False otherwise
    """
    # Validate profile name
    is_valid, error_msg = validate_profile_name(profile_name)
    if not is_valid:
        print(f"Error: {error_msg}")
        return False

    profile_path = get_profile_path(profile_name)

    if profile_path.exists():
        print(f"Profile '{profile_name}' already exists at: {profile_path}")
        print()
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return False

    # Create profile directory
    try:
        profile_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating profile directory: {e}")
        return False

    print()
    print("=" * BANNER_WIDTH)
    print(f"CREATING PROFILE: {profile_name}")
    print("=" * BANNER_WIDTH)
    print()
    print("Enter your Adobe OAuth credentials.")
    print("(Get these from Adobe Developer Console → Project → Credentials)")
    print()

    try:
        org_id = input("Organization ID (ends with @AdobeOrg): ").strip()
        if not org_id:
            print("Error: Organization ID is required")
            return False

        client_id = input("Client ID: ").strip()
        if not client_id:
            print("Error: Client ID is required")
            return False

        # Use getpass for secret if available
        try:
            import getpass

            secret = getpass.getpass("Client Secret: ").strip()
        except ImportError, getpass.GetPassWarning:
            secret = input("Client Secret: ").strip()

        if not secret:
            print("Error: Client Secret is required")
            return False

        scopes = input("OAuth Scopes (from Developer Console): ").strip()
        if not scopes:
            print("Error: OAuth Scopes are required")
            return False

    except KeyboardInterrupt, EOFError:
        print("\nAborted.")
        return False

    # Create config.json
    config = {"org_id": org_id, "client_id": client_id, "secret": secret, "scopes": scopes}

    config_file = profile_path / "config.json"
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        # Set restrictive permissions
        config_file.chmod(0o600)
    except OSError as e:
        print(f"Error writing config file: {e}")
        return False

    print()
    print(f"Profile '{profile_name}' created successfully!")
    print(f"  Location: {profile_path}")
    print()
    print("Test your profile:")
    print(f"  cja_auto_sdr --profile-test {profile_name}")
    print()
    print("Use your profile:")
    print(f"  cja_auto_sdr --profile {profile_name} --list-dataviews")
    print()

    return True


def mask_sensitive_value(value: str, show_chars: int = 4) -> str:
    """Mask sensitive values for display.

    Args:
        value: The value to mask
        show_chars: Number of characters to show at start and end

    Returns:
        Masked value string
    """
    if not value:
        return "(empty)"

    if len(value) <= show_chars * 2:
        return "*" * len(value)

    return f"{value[:show_chars]}{'*' * (len(value) - show_chars * 2)}{value[-show_chars:]}"


def show_profile(profile_name: str) -> bool:
    """Display profile config with masked secrets.

    Args:
        profile_name: Name of the profile to show

    Returns:
        True if successful, False otherwise
    """
    try:
        logger = logging.getLogger("profile_show")
        logger.setLevel(logging.WARNING)
        credentials = load_profile_credentials(profile_name, logger)
    except ProfileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except ProfileConfigError as e:
        print(f"Error: {e}")
        return False

    profile_path = get_profile_path(profile_name)

    print()
    print("=" * BANNER_WIDTH)
    print(f"PROFILE: {profile_name}")
    print("=" * BANNER_WIDTH)
    print()
    print(f"Location: {profile_path}")
    print()

    # Show sources
    sources = []
    if (profile_path / "config.json").exists():
        sources.append("config.json")
    if (profile_path / ".env").exists():
        sources.append(".env")
    print(f"Sources: {', '.join(sources)}")
    print()

    # Show credentials with masked values
    print("Credentials:")
    print("-" * 40)

    sensitive_fields = {"secret", "client_id"}

    for key in ["org_id", "client_id", "secret", "scopes", "sandbox"]:
        if key in credentials:
            value = credentials[key]
            if key in sensitive_fields:
                display_value = mask_sensitive_value(value)
            else:
                display_value = value
            print(f"  {key}: {display_value}")

    print()
    return True


def test_profile(profile_name: str) -> bool:
    """Test profile credentials and API connectivity.

    Args:
        profile_name: Name of the profile to test

    Returns:
        True if test successful, False otherwise
    """
    print()
    print("=" * BANNER_WIDTH)
    print(f"TESTING PROFILE: {profile_name}")
    print("=" * BANNER_WIDTH)
    print()

    # Load credentials
    try:
        logger = logging.getLogger("profile_test")
        logger.setLevel(logging.WARNING)
        credentials = load_profile_credentials(profile_name, logger)
    except ProfileNotFoundError as e:
        print(f"FAILED: {e}")
        return False
    except ProfileConfigError as e:
        print(f"FAILED: {e}")
        return False

    print("1. Profile found and loaded")

    # Validate credentials
    issues = ConfigValidator.validate_all(credentials, logger)
    if issues:
        print("2. Credential validation: WARNINGS")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("2. Credential validation: OK")

    # Test API connectivity
    print("3. Testing API connectivity...")

    try:
        # Create temp config file
        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="cja_profile_test_")
        json.dump(credentials, temp_config)
        temp_config.close()

        try:
            cjapy.importConfigFile(temp_config.name)
            cja = cjapy.CJA()
            dataviews = cja.getDataViews()

            if dataviews is not None:
                count = len(dataviews) if hasattr(dataviews, "__len__") else 0
                print("   API connection: SUCCESS")
                print(f"   Data views accessible: {count}")
                print()
                print("Profile test: PASSED")
                print()
                return True
            else:
                print("   API connection: OK (no data views found)")
                print()
                print("Profile test: PASSED")
                print()
                return True

        finally:
            os.unlink(temp_config.name)

    except Exception as e:
        print("   API connection: FAILED")
        print(f"   Error: {e}")
        print()
        print("Profile test: FAILED")
        print()
        print("Common issues:")
        print("  - Invalid client_id or secret")
        print("  - Incorrect OAuth scopes")
        print("  - API project not provisioned for CJA")
        print()
        return False


# ==================== CONFIG VALIDATION (moved to core/config_validation.py) ====================
# ==================== CJA CLIENT (moved to api/client.py) ====================
from cja_auto_sdr.api.client import _config_from_env, configure_cjapy, initialize_cja
from cja_auto_sdr.core.config_validation import (
    ConfigValidator,
    validate_config_file,
    validate_credentials,
)

# ==================== CREDENTIAL LOADING (moved to core/credentials.py) ====================
from cja_auto_sdr.core.credentials import (
    CredentialLoader,
    CredentialResolver,
    DotenvCredentialLoader,
    EnvironmentCredentialLoader,
    JsonFileCredentialLoader,
    filter_credentials,
    load_credentials_from_env,
    normalize_credential_value,
    validate_env_credentials,
)

# ==================== DATA VIEW VALIDATION ====================


def validate_data_view(cja: cjapy.CJA, data_view_id: str, logger: logging.Logger) -> bool:
    """Validate that the data view exists and is accessible.

    Args:
        cja: Initialized CJA instance
        data_view_id: Data view ID to validate
        logger: Logger instance

    Returns:
        True if data view is valid and accessible, False otherwise
    """
    try:
        logger.info("=" * BANNER_WIDTH)
        logger.info("VALIDATING DATA VIEW")
        logger.info("=" * BANNER_WIDTH)
        logger.info(f"Data View ID: {data_view_id}")

        # Basic format validation
        if not data_view_id or not isinstance(data_view_id, str):
            logger.error("Invalid data view ID format")
            logger.error("Data view ID must be a non-empty string")
            return False

        if not data_view_id.startswith("dv_"):
            logger.warning(f"Data view ID '{data_view_id}' does not follow standard format (dv_...)")
            logger.warning("This may still be valid, but unusual")

        # Attempt to fetch data view info
        logger.info("Fetching data view information from API...")
        try:
            dv_info = cja.getDataView(data_view_id)
        except AttributeError as e:
            logger.error("API method 'getDataView' not available")
            logger.error("Possible causes:")
            logger.error("  1. Outdated cjapy version - try: pip install --upgrade cjapy")
            logger.error("  2. CJA connection not properly initialized")
            logger.error("  3. Authentication failed silently")
            logger.debug(f"AttributeError details: {e}")
            return False
        except Exception as api_error:
            logger.error(f"API call failed: {api_error!s}")
            logger.error("Possible reasons:")
            logger.error("  1. Data view does not exist")
            logger.error("  2. You don't have permission to access this data view")
            logger.error("  3. Network connectivity issues")
            logger.error("  4. API authentication has expired")
            return False

        # Validate response
        if not dv_info:
            # Try to list available data views to provide context
            available_count = None
            try:
                available_dvs = cja.getDataViews()
                available_count = len(available_dvs) if available_dvs else 0

                if available_count > 0:
                    logger.info(f"You have access to {available_count} data view(s):")
                    for i, dv in enumerate(available_dvs[:10]):  # Show first 10
                        dv_id = dv.get("id", "unknown")
                        dv_name = dv.get("name", "unknown")
                        logger.info(f"  {i + 1}. {dv_name} (ID: {dv_id})")
                    if available_count > 10:
                        logger.info(f"  ... and {available_count - 10} more")
                    logger.info("")
            except Exception as list_error:
                logger.debug(f"Could not list available data views: {list_error!s}")

            # Show enhanced error message
            error_msg = ErrorMessageHelper.get_data_view_error_message(data_view_id, available_count=available_count)
            logger.error("\n" + error_msg)
            return False

        # Extract and validate data view details
        dv_name = dv_info.get("name", "Unknown")
        dv_description = dv_info.get("description", "No description")
        dv_owner = dv_info.get("owner", {}).get("name", "Unknown")

        logger.info("✓ Data view validated successfully!")
        logger.info(f"  Name: {dv_name}")
        logger.info(f"  ID: {data_view_id}")
        logger.info(f"  Owner: {dv_owner}")
        if dv_description and dv_description != "No description":
            logger.info(f"  Description: {dv_description[:100]}{'...' if len(dv_description) > 100 else ''}")

        # Additional validation checks
        warnings = []

        if "components" in dv_info:
            components = dv_info.get("components", {})
            if not components.get("dimensions") and not components.get("metrics"):
                warnings.append("Data view appears to have no components defined")

        if warnings:
            logger.warning("Data view validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        return True

    except Exception as e:
        logger.error("=" * BANNER_WIDTH)
        logger.error("DATA VIEW VALIDATION ERROR")
        logger.error("=" * BANNER_WIDTH)
        logger.error(f"Unexpected error during validation: {e!s}")
        logger.exception("Full error details:")
        logger.error("")
        logger.error("Please verify:")
        logger.error("  1. The data view ID is correct")
        logger.error("  2. You have access to this data view")
        logger.error("  3. Your API credentials are valid")
        return False


# ==================== OPTIMIZED API DATA FETCHING (moved to api/fetch.py) ====================
from cja_auto_sdr.api.fetch import ParallelAPIFetcher

# ==================== DATA QUALITY VALIDATION (moved to api/quality.py) ====================
from cja_auto_sdr.api.quality import DataQualityChecker

# ==================== EXCEL GENERATION ====================


class ExcelFormatCache:
    """Cache for Excel format objects to avoid recreating identical formats.

    xlsxwriter creates a new format object for each add_format() call, even if
    the properties are identical. This class caches formats by their properties
    to reuse them across multiple sheets, improving performance by 15-25% for
    workbooks with multiple sheets.

    Usage:
        cache = ExcelFormatCache(workbook)
        header_fmt = cache.get_format({'bold': True, 'bg_color': '#366092'})
    """

    def __init__(self, workbook):
        self.workbook = workbook
        self._cache: dict[tuple, Any] = {}

    def get_format(self, properties: dict[str, Any]) -> Any:
        """Get or create a format with the given properties.

        Args:
            properties: Dictionary of format properties (e.g., {'bold': True})

        Returns:
            xlsxwriter Format object
        """
        # Convert dict to a hashable key (sorted tuple of items)
        # Use repr() for nested values to avoid collisions (e.g. [1,2] vs '[1, 2]')
        cache_key = tuple(sorted((k, repr(v)) for k, v in properties.items()))

        if cache_key not in self._cache:
            self._cache[cache_key] = self.workbook.add_format(properties)

        return self._cache[cache_key]


def apply_excel_formatting(
    writer, df, sheet_name, logger: logging.Logger, format_cache: ExcelFormatCache | None = None
):
    """Apply formatting to Excel sheets with error handling.

    Args:
        writer: pandas ExcelWriter object
        df: DataFrame to format
        sheet_name: Name of the sheet
        logger: Logger instance
        format_cache: Optional ExcelFormatCache for format reuse across sheets
    """
    try:
        logger.info(f"Formatting sheet: {sheet_name}")

        # Calculate row offset for Data Quality sheet (summary section at top)
        summary_rows = 0
        if sheet_name == "Data Quality" and "Severity" in df.columns:
            summary_rows = 7  # Title + header + 5 severity levels + blank row

        # Reorder columns for component sheets (name first for readability)
        if sheet_name in ("Metrics", "Dimensions", "Derived Fields", "Calculated Metrics") and "name" in df.columns:
            preferred_order = ["name", "type", "id", "description", "title"]
            existing_cols = [col for col in preferred_order if col in df.columns]
            other_cols = [col for col in df.columns if col not in preferred_order]
            df = df[existing_cols + other_cols]

        # Write dataframe to sheet with offset for summary
        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=summary_rows)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Use format cache if provided, otherwise create formats directly
        # Format cache improves performance by 15-25% when formatting multiple sheets
        cache = format_cache if format_cache else ExcelFormatCache(workbook)

        # Add summary section for Data Quality sheet
        if sheet_name == "Data Quality" and "Severity" in df.columns:
            # Calculate severity counts
            severity_counts = df["Severity"].value_counts()

            # Summary formats (using cache for reuse)
            title_format = cache.get_format({"bold": True, "font_size": 14, "font_color": "#366092", "bottom": 2})
            summary_header = cache.get_format({"bold": True, "bg_color": "#D9E1F2", "border": 1, "align": "center"})
            summary_cell = cache.get_format({"border": 1, "align": "center"})

            # Write summary title
            worksheet.write(0, 0, "Issue Summary", title_format)
            worksheet.merge_range(0, 0, 0, 1, "Issue Summary", title_format)

            # Write summary headers
            worksheet.write(1, 0, "Severity", summary_header)
            worksheet.write(1, 1, "Count", summary_header)

            # Write severity counts in order
            row = 2
            total_count = 0
            for sev in DataQualityChecker.SEVERITY_ORDER:
                count = severity_counts.get(sev, 0)
                if count > 0 or sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:  # Always show main levels
                    worksheet.write(row, 0, sev, summary_cell)
                    worksheet.write(row, 1, int(count), summary_cell)
                    total_count += count
                    row += 1

            # Set column widths for summary
            worksheet.set_column(0, 0, 12)
            worksheet.set_column(1, 1, 8)

        # Common format definitions (cached for reuse across sheets)
        header_format = cache.get_format(
            {
                "bold": True,
                "bg_color": "#366092",
                "font_color": "white",
                "border": 1,
                "align": "center",
                "text_wrap": True,
            }
        )

        grey_format = cache.get_format(
            {"bg_color": "#F2F2F2", "border": 1, "text_wrap": True, "align": "top", "valign": "top"}
        )

        white_format = cache.get_format(
            {"bg_color": "#FFFFFF", "border": 1, "text_wrap": True, "align": "top", "valign": "top"}
        )

        # Bold formats for Name column in Metrics/Dimensions sheets
        name_bold_grey = cache.get_format(
            {"bg_color": "#F2F2F2", "border": 1, "text_wrap": True, "align": "top", "valign": "top", "bold": True}
        )

        name_bold_white = cache.get_format(
            {"bg_color": "#FFFFFF", "border": 1, "text_wrap": True, "align": "top", "valign": "top", "bold": True}
        )

        # Special formats for Data Quality sheet
        if sheet_name == "Data Quality":
            # Severity icons for visual indicators (Excel only)
            severity_icons = {
                "CRITICAL": "\u25cf",  # ● filled circle
                "HIGH": "\u25b2",  # ▲ triangle up
                "MEDIUM": "\u25a0",  # ■ filled square
                "LOW": "\u25cb",  # ○ empty circle
                "INFO": "\u2139",  # ℹ info symbol
            }

            # Row formats (for non-severity columns) - using cache
            critical_format = cache.get_format(
                {
                    "bg_color": "#FFC7CE",
                    "font_color": "#9C0006",
                    "border": 1,
                    "text_wrap": True,
                    "align": "top",
                    "valign": "top",
                }
            )

            high_format = cache.get_format(
                {
                    "bg_color": "#FFEB9C",
                    "font_color": "#9C6500",
                    "border": 1,
                    "text_wrap": True,
                    "align": "top",
                    "valign": "top",
                }
            )

            medium_format = cache.get_format(
                {
                    "bg_color": "#C6EFCE",
                    "font_color": "#006100",
                    "border": 1,
                    "text_wrap": True,
                    "align": "top",
                    "valign": "top",
                }
            )

            low_format = cache.get_format(
                {
                    "bg_color": "#DDEBF7",
                    "font_color": "#1F4E78",
                    "border": 1,
                    "text_wrap": True,
                    "align": "top",
                    "valign": "top",
                }
            )

            info_format = cache.get_format(
                {
                    "bg_color": "#E2EFDA",
                    "font_color": "#375623",
                    "border": 1,
                    "text_wrap": True,
                    "align": "top",
                    "valign": "top",
                }
            )

            # Bold formats for Severity column (emphasize priority) - using cache
            critical_bold = cache.get_format(
                {
                    "bg_color": "#FFC7CE",
                    "font_color": "#9C0006",
                    "bold": True,
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )

            high_bold = cache.get_format(
                {
                    "bg_color": "#FFEB9C",
                    "font_color": "#9C6500",
                    "bold": True,
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )

            medium_bold = cache.get_format(
                {
                    "bg_color": "#C6EFCE",
                    "font_color": "#006100",
                    "bold": True,
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )

            low_bold = cache.get_format(
                {
                    "bg_color": "#DDEBF7",
                    "font_color": "#1F4E78",
                    "bold": True,
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )

            info_bold = cache.get_format(
                {
                    "bg_color": "#E2EFDA",
                    "font_color": "#375623",
                    "bold": True,
                    "border": 1,
                    "align": "center",
                    "valign": "vcenter",
                }
            )

            # Map severity to formats
            severity_formats = {
                "CRITICAL": (critical_format, critical_bold),
                "HIGH": (high_format, high_bold),
                "MEDIUM": (medium_format, medium_bold),
                "LOW": (low_format, low_bold),
                "INFO": (info_format, info_bold),
            }

        # Format header row (offset by summary rows if present)
        header_row = summary_rows
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(header_row, col_num, value, header_format)

        # Column width caps - tighter limits for Metrics/Dimensions sheets
        if sheet_name in ("Metrics", "Dimensions"):
            # Specific column width limits for better readability
            column_width_caps = {
                "name": 40,
                "type": 20,
                "id": 35,
                "title": 40,
                "description": 55,  # Narrower than default, relies on text wrap
            }
            default_cap = 50  # Narrower default for other columns
        else:
            column_width_caps = {}
            default_cap = 100

        # Set column widths with appropriate caps
        for idx, col in enumerate(df.columns):
            series = df[col]
            col_lower = col.lower()
            max_cap = column_width_caps.get(col_lower, default_cap)
            max_len = min(
                max(
                    max(len(str(val).split("\n")[0]) for val in series) if len(series) > 0 else 0, len(str(series.name))
                )
                + 2,
                max_cap,
            )
            worksheet.set_column(idx, idx, max_len)

        # Apply row formatting (offset by summary rows)
        data_start_row = summary_rows + 1  # +1 for header row

        # Cache column indices outside the loop for performance (avoids repeated hash lookups)
        severity_col_idx = df.columns.get_loc("Severity") if "Severity" in df.columns else -1
        name_col_idx = df.columns.get_loc("name") if "name" in df.columns else -1
        is_data_quality_sheet = sheet_name == "Data Quality" and severity_col_idx >= 0
        is_component_sheet = sheet_name in ("Metrics", "Dimensions") and name_col_idx >= 0

        for idx in range(len(df)):
            max_lines = max((str(val).count("\n") for val in df.iloc[idx]), default=0) + 1
            row_height = min(max_lines * 15, 400)
            excel_row = data_start_row + idx

            # Apply severity-based formatting for Data Quality sheet
            if is_data_quality_sheet:
                severity = str(df.iloc[idx]["Severity"])
                row_format, bold_format = severity_formats.get(severity, (low_format, low_bold))

                # Set row height and default format
                worksheet.set_row(excel_row, row_height, row_format)

                # Write Severity column with icon and bold format
                icon = severity_icons.get(severity, "")
                worksheet.write(excel_row, severity_col_idx, f"{icon} {severity}", bold_format)
            else:
                row_format = grey_format if idx % 2 == 0 else white_format
                worksheet.set_row(excel_row, row_height, row_format)

                # Apply bold Name column for Metrics/Dimensions sheets
                if is_component_sheet:
                    name_format = name_bold_grey if idx % 2 == 0 else name_bold_white
                    worksheet.write(excel_row, name_col_idx, df.iloc[idx]["name"], name_format)

        # Add autofilter to data table (offset by summary rows)
        worksheet.autofilter(summary_rows, 0, summary_rows + len(df), len(df.columns) - 1)

        # Freeze header row (summary + data header visible when scrolling)
        worksheet.freeze_panes(summary_rows + 1, 0)

        logger.info(f"Successfully formatted sheet: {sheet_name}")

    except Exception as e:
        logger.error(_format_error_msg(f"formatting sheet {sheet_name}", error=e))
        raise


# ==================== OUTPUT FORMAT WRITERS ====================


def write_excel_output(
    data_dict: dict[str, pd.DataFrame], base_filename: str, output_dir: str | Path, logger: logging.Logger
) -> str:
    """
    Write data to a formatted Excel workbook.

    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance

    Returns:
        Path to Excel output file
    """
    try:
        logger.info("Generating Excel output...")

        excel_file = os.path.join(output_dir, f"{base_filename}.xlsx")
        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            format_cache = ExcelFormatCache(writer.book)
            for sheet_name, df in data_dict.items():
                if df.empty:
                    placeholder_df = pd.DataFrame({"Note": [f"No data available for {sheet_name}"]})
                    apply_excel_formatting(writer, placeholder_df, sheet_name, logger, format_cache)
                else:
                    apply_excel_formatting(writer, df, sheet_name, logger, format_cache)

        logger.info(f"Excel file created: {excel_file}")
        return excel_file

    except PermissionError as e:
        logger.error(f"Permission denied creating Excel file: {e}")
        logger.error("Check write permissions for the output directory")
        raise
    except OSError as e:
        logger.error(f"OS error creating Excel file: {e}")
        logger.error("Check disk space and path validity")
        raise
    except Exception as e:
        logger.error(_format_error_msg("creating Excel file", error=e))
        raise


def write_csv_output(
    data_dict: dict[str, pd.DataFrame], base_filename: str, output_dir: str | Path, logger: logging.Logger
) -> str:
    """
    Write data to CSV files (one per sheet)

    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance

    Returns:
        Path to output directory containing CSV files
    """
    try:
        logger.info("Generating CSV output...")

        # Create subdirectory for CSV files
        csv_dir = os.path.join(output_dir, f"{base_filename}_csv")
        os.makedirs(csv_dir, exist_ok=True)

        # Write each DataFrame to a separate CSV file
        for sheet_name, df in data_dict.items():
            csv_file = os.path.join(csv_dir, f"{sheet_name.replace(' ', '_').lower()}.csv")
            df.to_csv(csv_file, index=False, encoding="utf-8")
            logger.info(f"  ✓ Created CSV: {os.path.basename(csv_file)}")

        logger.info(f"CSV files created in: {csv_dir}")
        return csv_dir

    except PermissionError as e:
        logger.error(f"Permission denied creating CSV files: {e}")
        logger.error("Check write permissions for the output directory")
        raise
    except OSError as e:
        logger.error(f"OS error creating CSV files: {e}")
        logger.error("Check disk space and path validity")
        raise
    except Exception as e:
        logger.error(_format_error_msg("creating CSV files", error=e))
        raise


def write_json_output(
    data_dict: dict[str, pd.DataFrame],
    metadata_dict: dict[str, Any],
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    inventory_objects: dict[str, Any] | None = None,
) -> str:
    """
    Write data to JSON format with hierarchical structure

    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        metadata_dict: Metadata information
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        inventory_objects: Optional dict with 'derived' and 'calculated' inventory objects
                          for detailed JSON output using to_json()

    Returns:
        Path to JSON output file
    """
    try:
        logger.info("Generating JSON output...")

        # Build JSON structure
        json_data = {
            "metadata": metadata_dict,
            "data_view": {},
            "metrics": [],
            "dimensions": [],
            "data_quality": [],
            "derived_fields": {},
            "calculated_metrics": {},
            "segments": {},
        }

        inventory_objects = inventory_objects or {}

        # Convert DataFrames to JSON-serializable format
        for sheet_name, df in data_dict.items():
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient="records")

            # Map to appropriate section
            if sheet_name == "Data Quality":
                json_data["data_quality"] = records
            elif sheet_name == "Metrics":
                json_data["metrics"] = records
            elif sheet_name == "Dimensions":
                json_data["dimensions"] = records
            elif sheet_name == "DataView Details":
                # For single-record sheets, store as object not array
                json_data["data_view"] = records[0] if records else {}
            elif sheet_name == "Derived Fields":
                # Use inventory object's to_json() for detailed output if available
                if inventory_objects.get("derived"):
                    json_data["derived_fields"] = inventory_objects["derived"].to_json()
                else:
                    json_data["derived_fields"] = {"fields": records}
            elif sheet_name == "Calculated Metrics":
                # Use inventory object's to_json() for detailed output if available
                if inventory_objects.get("calculated"):
                    json_data["calculated_metrics"] = inventory_objects["calculated"].to_json()
                else:
                    json_data["calculated_metrics"] = {"metrics": records}
            elif sheet_name == "Segments":
                # Use inventory object's to_json() for detailed output if available
                if inventory_objects.get("segments"):
                    json_data["segments"] = inventory_objects["segments"].to_json()
                else:
                    json_data["segments"] = {"segments": records}

        # Write JSON file
        json_file = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ JSON file created: {json_file}")
        return json_file

    except PermissionError as e:
        logger.error(f"Permission denied creating JSON file: {e}")
        logger.error("Check write permissions for the output directory")
        raise
    except OSError as e:
        logger.error(f"OS error creating JSON file: {e}")
        logger.error("Check disk space and path validity")
        raise
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization error: {e}")
        logger.error("Data contains non-serializable values")
        raise
    except Exception as e:
        logger.error(_format_error_msg("creating JSON file", error=e))
        raise


def write_html_output(
    data_dict: dict[str, pd.DataFrame],
    metadata_dict: dict[str, Any],
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
) -> str:
    """
    Write data to HTML format with professional styling

    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        metadata_dict: Metadata information
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance

    Returns:
        Path to HTML output file
    """
    try:
        logger.info("Generating HTML output...")

        # Build HTML content
        html_parts = []

        # HTML header with CSS
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CJA Solution Design Reference</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .metadata-item {
            margin: 8px 0;
            display: flex;
            align-items: baseline;
        }
        .metadata-label {
            font-weight: bold;
            min-width: 200px;
            color: #2c3e50;
        }
        .metadata-value {
            color: #555;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        thead {
            background-color: #3498db;
            color: white;
        }
        th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tbody tr:hover {
            background-color: #e8f4f8;
        }
        .severity-CRITICAL {
            background-color: #e74c3c !important;
            color: white;
            font-weight: bold;
        }
        .severity-HIGH {
            background-color: #e67e22 !important;
            color: white;
        }
        .severity-MEDIUM {
            background-color: #f39c12 !important;
            color: white;
        }
        .severity-LOW {
            background-color: #95a5a6 !important;
            color: white;
        }
        .severity-INFO {
            background-color: #3498db !important;
            color: white;
        }
        .section {
            margin-bottom: 50px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }
        @media print {
            body {
                background-color: white;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 CJA Solution Design Reference</h1>
        """)

        # Metadata section
        html_parts.append('<div class="metadata">')
        html_parts.append("<h2>📋 Metadata</h2>")
        for key, value in metadata_dict.items():
            safe_value = str(value).replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"""
            <div class="metadata-item">
                <span class="metadata-label">{key}:</span>
                <span class="metadata-value">{safe_value}</span>
            </div>
            """)
        html_parts.append("</div>")

        # Data sections
        section_icons = {
            "Data Quality": "🔍",
            "DataView Details": "📊",
            "Metrics": "📈",
            "Dimensions": "📐",
            "Derived Fields": "🔧",
            "Calculated Metrics": "🧮",
            "Segments": "🎯",
        }

        for sheet_name, df in data_dict.items():
            if df.empty:
                continue

            icon = section_icons.get(sheet_name, "📄")
            html_parts.append('<div class="section">')
            html_parts.append(f"<h2>{icon} {sheet_name}</h2>")

            # Convert DataFrame to HTML with custom styling
            df_html = df.to_html(index=False, escape=False, classes="data-table")

            # Add severity-based row classes for Data Quality sheet
            if sheet_name == "Data Quality" and "Severity" in df.columns:
                rows = df_html.split("<tr>")
                styled_rows = [rows[0]]  # Keep header

                for i, row in enumerate(rows[1:], 1):
                    if i <= len(df):
                        severity = df.iloc[i - 1]["Severity"] if i - 1 < len(df) else ""
                        if severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                            row = f'<tr class="severity-{severity}">' + row.split(">", 1)[1]
                        else:
                            row = "<tr>" + row
                        styled_rows.append(row)

                df_html = "".join(styled_rows)

            html_parts.append(df_html)
            html_parts.append("</div>")

        # Footer
        html_parts.append(f"""
        <div class="footer">
            <p>Generated by CJA SDR Generator v{__version__}</p>
            <p>Generated at {metadata_dict.get("Generated At", "N/A")}</p>
        </div>
    </div>
</body>
</html>
        """)

        # Write HTML file
        html_file = os.path.join(output_dir, f"{base_filename}.html")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        logger.info(f"✓ HTML file created: {html_file}")
        return html_file

    except PermissionError as e:
        logger.error(f"Permission denied creating HTML file: {e}")
        logger.error("Check write permissions for the output directory")
        raise
    except OSError as e:
        logger.error(f"OS error creating HTML file: {e}")
        logger.error("Check disk space and path validity")
        raise
    except Exception as e:
        logger.error(_format_error_msg("creating HTML file", error=e))
        raise


def write_markdown_output(
    data_dict: dict[str, pd.DataFrame],
    metadata_dict: dict[str, Any],
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
) -> str:
    """
    Write data to Markdown format for GitHub, Confluence, and other platforms

    Features:
    - GitHub-flavored markdown tables
    - Table of contents with section links
    - Collapsible sections for large tables
    - Proper escaping of special characters
    - Issue summary for Data Quality

    Args:
        data_dict: Dictionary mapping sheet names to DataFrames
        metadata_dict: Metadata information
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance

    Returns:
        Path to Markdown output file
    """
    try:
        logger.info("Generating Markdown output...")

        def escape_markdown(text: str) -> str:
            """Escape special markdown characters in table cells"""
            if pd.isna(text) or text is None:
                return ""
            text = str(text)
            # Escape pipe characters that would break tables
            text = text.replace("|", "\\|")
            # Escape backticks
            text = text.replace("`", "\\`")
            # Replace newlines with spaces in table cells
            text = text.replace("\n", " ")
            text = text.replace("\r", " ")
            return text.strip()

        def df_to_markdown_table(df: pd.DataFrame, sheet_name: str) -> str:
            """Convert DataFrame to markdown table format.

            Uses vectorized operations instead of iterrows() for better performance
            on large DataFrames (20-40% faster for datasets with 100+ rows).
            """
            if df.empty:
                return f"\n*No {sheet_name.lower()} found.*\n"

            # Header row
            headers = [escape_markdown(col) for col in df.columns]
            header_row = "| " + " | ".join(headers) + " |"

            # Separator row with left alignment
            separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

            # Data rows - vectorized approach using apply() instead of iterrows()
            # This avoids the overhead of creating Series objects for each row
            def format_row(row: pd.Series) -> str:
                cells = [escape_markdown(row[col]) for col in df.columns]
                return "| " + " | ".join(cells) + " |"

            data_rows = df.apply(format_row, axis=1).tolist()

            return "\n".join([header_row, separator_row, *data_rows])

        md_parts = []

        # Title
        md_parts.append("# 📊 CJA Solution Design Reference\n")

        # Metadata section
        md_parts.append("## 📋 Metadata\n")
        if metadata_dict:
            for key, value in metadata_dict.items():
                md_parts.append(f"**{key}:** {escape_markdown(str(value))}")
            md_parts.append("")

        # Table of contents
        md_parts.append("## 📑 Table of Contents\n")
        toc_items = []
        for sheet_name in data_dict:
            # Create anchor-safe links
            anchor = sheet_name.lower().replace(" ", "-").replace("_", "-")
            toc_items.append(f"- [{sheet_name}](#{anchor})")
        md_parts.append("\n".join(toc_items))
        md_parts.append("\n---\n")

        # Process each sheet
        for sheet_name, df in data_dict.items():
            md_parts.append(f"## {sheet_name}\n")

            # Add special handling for Data Quality sheet
            if sheet_name == "Data Quality" and not df.empty and "Severity" in df.columns:
                # Add issue summary
                severity_counts = df["Severity"].value_counts()
                md_parts.append("### Issue Summary\n")
                md_parts.append("| Severity | Count |")
                md_parts.append("| --- | --- |")

                severity_emojis = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪", "INFO": "🔵"}
                for sev in DataQualityChecker.SEVERITY_ORDER:
                    count = severity_counts.get(sev, 0)
                    if count > 0 or sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                        emoji = severity_emojis.get(sev, "")
                        md_parts.append(f"| {emoji} {sev} | {count} |")
                md_parts.append("")

            # For large tables (>50 rows), use collapsible sections
            if len(df) > 50:
                md_parts.append("<details>")
                md_parts.append(f"<summary>View {len(df)} rows (click to expand)</summary>\n")
                md_parts.append(df_to_markdown_table(df, sheet_name))
                md_parts.append("\n</details>\n")
            else:
                # For smaller tables, show directly
                md_parts.append(df_to_markdown_table(df, sheet_name))
                md_parts.append("")

            # Add counts
            md_parts.append(f"*Total {sheet_name}: {len(df)} items*\n")
            md_parts.append("---\n")

        # Footer
        md_parts.append("---")
        md_parts.append("*Generated by CJA Auto SDR Generator*")

        # Write to file
        markdown_file = os.path.join(output_dir, f"{base_filename}.md")
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_parts))

        logger.info(f"✓ Markdown file created: {markdown_file}")
        return markdown_file

    except PermissionError as e:
        logger.error(f"Permission denied creating Markdown file: {e}")
        logger.error("Check write permissions for the output directory")
        raise
    except OSError as e:
        logger.error(f"OS error creating Markdown file: {e}")
        logger.error("Check disk space and path validity")
        raise
    except Exception as e:
        logger.error(_format_error_msg("creating Markdown file", error=e))
        raise


# ==================== DIFF COMPARISON OUTPUT WRITERS ====================


# ANSIColors is an alias for diff output compatibility
# It delegates to ConsoleColors but accepts an 'enabled' parameter for explicit control
class ANSIColors:
    """ANSI escape codes for colored terminal output.

    This class provides the same functionality as ConsoleColors but with an
    explicit 'enabled' parameter for cases where color control is needed
    independent of TTY detection.
    """

    # Re-export constants from ConsoleColors
    GREEN = ConsoleColors.GREEN
    RED = ConsoleColors.RED
    YELLOW = ConsoleColors.YELLOW
    CYAN = ConsoleColors.CYAN
    BOLD = ConsoleColors.BOLD
    RESET = ConsoleColors.RESET
    ANSI_ESCAPE = ConsoleColors.ANSI_ESCAPE

    @classmethod
    def green(cls, text: str, enabled: bool = True) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}" if enabled else text

    @classmethod
    def red(cls, text: str, enabled: bool = True) -> str:
        return f"{cls.RED}{text}{cls.RESET}" if enabled else text

    @classmethod
    def yellow(cls, text: str, enabled: bool = True) -> str:
        return f"{cls.YELLOW}{text}{cls.RESET}" if enabled else text

    @classmethod
    def cyan(cls, text: str, enabled: bool = True) -> str:
        return f"{cls.CYAN}{text}{cls.RESET}" if enabled else text

    @classmethod
    def bold(cls, text: str, enabled: bool = True) -> str:
        return f"{cls.BOLD}{text}{cls.RESET}" if enabled else text

    # Delegate utility methods to ConsoleColors
    visible_len = ConsoleColors.visible_len
    rjust = ConsoleColors.rjust
    ljust = ConsoleColors.ljust


def write_diff_console_output(
    diff_result: DiffResult,
    changes_only: bool = False,
    summary_only: bool = False,
    side_by_side: bool = False,
    use_color: bool = True,
) -> str:
    """
    Write diff comparison to console with color-coded output.

    Args:
        diff_result: The DiffResult to output
        changes_only: Only show changed items (hide unchanged)
        summary_only: Only show summary statistics
        side_by_side: Show side-by-side comparison for modified items
        use_color: Use ANSI color codes in output (default: True)

    Returns:
        Formatted string for console output
    """
    lines = []
    summary = diff_result.summary
    meta = diff_result.metadata_diff
    c = use_color  # Shorthand for color enabled flag

    # Header
    lines.append("=" * 80)
    lines.append(ANSIColors.bold("DATA VIEW COMPARISON REPORT", c))
    lines.append("=" * 80)
    lines.append(f"{diff_result.source_label}: {meta.source_name} ({meta.source_id})")
    lines.append(f"{diff_result.target_label}: {meta.target_name} ({meta.target_id})")
    lines.append(f"Generated: {diff_result.generated_at}")
    lines.append("=" * 80)

    # Summary table with percentages
    lines.append("")
    lines.append(ANSIColors.bold("SUMMARY", c))

    # Build full header labels with data view name and ID
    src_header = f"{diff_result.source_label}: {meta.source_name} ({meta.source_id})"
    tgt_header = f"{diff_result.target_label}: {meta.target_name} ({meta.target_id})"

    # Calculate dynamic column widths based on full header lengths
    src_width = max(8, len(src_header))
    tgt_width = max(8, len(tgt_header))
    total_width = 20 + src_width + tgt_width + 10 + 10 + 10 + 12 + 12 + 7  # +7 for spacing

    lines.append(
        f"{'':20s} {src_header:>{src_width}s} {tgt_header:>{tgt_width}s} {'Added':>10s} {'Removed':>10s} {'Modified':>10s} {'Unchanged':>12s} {'Changed':>12s}"
    )
    lines.append("-" * total_width)

    # Metrics row with percentage (using ANSI-aware padding for colored strings)
    metrics_pct = f"({summary.metrics_change_percent:.1f}%)"
    added_str = (
        ANSIColors.green(f"+{summary.metrics_added}", c) if summary.metrics_added else f"+{summary.metrics_added}"
    )
    removed_str = (
        ANSIColors.red(f"-{summary.metrics_removed}", c) if summary.metrics_removed else f"-{summary.metrics_removed}"
    )
    modified_str = (
        ANSIColors.yellow(f"~{summary.metrics_modified}", c)
        if summary.metrics_modified
        else f"~{summary.metrics_modified}"
    )
    lines.append(
        f"{'Metrics':20s} {summary.source_metrics_count:{src_width}d} {summary.target_metrics_count:{tgt_width}d} "
        f"{ANSIColors.rjust(added_str, 10)} {ANSIColors.rjust(removed_str, 10)} {ANSIColors.rjust(modified_str, 10)} {summary.metrics_unchanged:>12d} {metrics_pct:>12s}"
    )

    # Dimensions row with percentage (using ANSI-aware padding for colored strings)
    dims_pct = f"({summary.dimensions_change_percent:.1f}%)"
    added_str = (
        ANSIColors.green(f"+{summary.dimensions_added}", c)
        if summary.dimensions_added
        else f"+{summary.dimensions_added}"
    )
    removed_str = (
        ANSIColors.red(f"-{summary.dimensions_removed}", c)
        if summary.dimensions_removed
        else f"-{summary.dimensions_removed}"
    )
    modified_str = (
        ANSIColors.yellow(f"~{summary.dimensions_modified}", c)
        if summary.dimensions_modified
        else f"~{summary.dimensions_modified}"
    )
    lines.append(
        f"{'Dimensions':20s} {summary.source_dimensions_count:{src_width}d} {summary.target_dimensions_count:{tgt_width}d} "
        f"{ANSIColors.rjust(added_str, 10)} {ANSIColors.rjust(removed_str, 10)} {ANSIColors.rjust(modified_str, 10)} {summary.dimensions_unchanged:>12d} {dims_pct:>12s}"
    )

    # Inventory rows (if present)
    if summary.has_inventory_changes or (
        summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0
    ):
        lines.append("-" * total_width)
        lines.append(ANSIColors.bold("INVENTORY", c))

        # Calculated metrics row
        if summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0:
            added_str = (
                ANSIColors.green(f"+{summary.calc_metrics_added}", c)
                if summary.calc_metrics_added
                else f"+{summary.calc_metrics_added}"
            )
            removed_str = (
                ANSIColors.red(f"-{summary.calc_metrics_removed}", c)
                if summary.calc_metrics_removed
                else f"-{summary.calc_metrics_removed}"
            )
            modified_str = (
                ANSIColors.yellow(f"~{summary.calc_metrics_modified}", c)
                if summary.calc_metrics_modified
                else f"~{summary.calc_metrics_modified}"
            )
            lines.append(
                f"{'Calc Metrics':20s} {summary.source_calc_metrics_count:{src_width}d} {summary.target_calc_metrics_count:{tgt_width}d} "
                f"{ANSIColors.rjust(added_str, 10)} {ANSIColors.rjust(removed_str, 10)} {ANSIColors.rjust(modified_str, 10)} {summary.calc_metrics_unchanged:>12d} {'':>12s}"
            )

        # Segments row
        if summary.source_segments_count > 0 or summary.target_segments_count > 0:
            added_str = (
                ANSIColors.green(f"+{summary.segments_added}", c)
                if summary.segments_added
                else f"+{summary.segments_added}"
            )
            removed_str = (
                ANSIColors.red(f"-{summary.segments_removed}", c)
                if summary.segments_removed
                else f"-{summary.segments_removed}"
            )
            modified_str = (
                ANSIColors.yellow(f"~{summary.segments_modified}", c)
                if summary.segments_modified
                else f"~{summary.segments_modified}"
            )
            lines.append(
                f"{'Segments':20s} {summary.source_segments_count:{src_width}d} {summary.target_segments_count:{tgt_width}d} "
                f"{ANSIColors.rjust(added_str, 10)} {ANSIColors.rjust(removed_str, 10)} {ANSIColors.rjust(modified_str, 10)} {summary.segments_unchanged:>12d} {'':>12s}"
            )

    lines.append("-" * total_width)

    if summary_only:
        lines.append("")
        if summary.has_changes:
            lines.append(f"Total changes: {summary.total_changes}")
            lines.append(f"Summary: {summary.natural_language_summary}")
        else:
            lines.append(ANSIColors.green("No differences found.", c))
        lines.append("=" * 80)
        return "\n".join(lines)

    # Get changes for both metrics and dimensions
    metric_changes = [d for d in diff_result.metric_diffs if d.change_type != ChangeType.UNCHANGED]
    dim_changes = [d for d in diff_result.dimension_diffs if d.change_type != ChangeType.UNCHANGED]

    # Calculate global max ID width for consistent alignment across both sections
    all_changes = metric_changes + dim_changes
    global_max_id_len = max((len(d.id) for d in all_changes), default=0)

    # Metrics changes
    if metric_changes or not changes_only:
        lines.append("")
        change_count = len(metric_changes)
        lines.append(ANSIColors.bold(f"METRICS CHANGES ({change_count})", c))
        if metric_changes:
            for diff in metric_changes:
                colored_symbol = _get_colored_symbol(diff.change_type, c)
                lines.append(f'  [{colored_symbol}] {diff.id:{global_max_id_len}s} "{diff.name}"')
                if side_by_side and diff.change_type == ChangeType.MODIFIED:
                    # Side-by-side view for modified items
                    sbs_lines = _format_side_by_side(diff, diff_result.source_label, diff_result.target_label)
                    lines.extend(sbs_lines)
                else:
                    detail = _get_change_detail(diff)
                    if detail:
                        lines.append(f"      {detail}")
        else:
            lines.append("  No changes")

    # Dimensions changes
    if dim_changes or not changes_only:
        lines.append("")
        change_count = len(dim_changes)
        lines.append(ANSIColors.bold(f"DIMENSIONS CHANGES ({change_count})", c))
        if dim_changes:
            for diff in dim_changes:
                colored_symbol = _get_colored_symbol(diff.change_type, c)
                lines.append(f'  [{colored_symbol}] {diff.id:{global_max_id_len}s} "{diff.name}"')
                if side_by_side and diff.change_type == ChangeType.MODIFIED:
                    # Side-by-side view for modified items
                    sbs_lines = _format_side_by_side(diff, diff_result.source_label, diff_result.target_label)
                    lines.extend(sbs_lines)
                else:
                    detail = _get_change_detail(diff)
                    if detail:
                        lines.append(f"      {detail}")
        else:
            lines.append("  No changes")

    # Inventory changes (if present)
    if diff_result.has_inventory_diffs:
        lines.append("")
        lines.append("-" * 80)
        lines.append(ANSIColors.bold("INVENTORY CHANGES", c))
        lines.append("-" * 80)

        # Calculated metrics inventory changes
        if diff_result.calc_metrics_diffs is not None:
            calc_changes = [d for d in diff_result.calc_metrics_diffs if d.change_type != ChangeType.UNCHANGED]
            lines.append("")
            lines.append(ANSIColors.bold(f"CALCULATED METRICS ({len(calc_changes)} changes)", c))
            if calc_changes:
                for diff in calc_changes:
                    colored_symbol = _get_colored_symbol(diff.change_type, c)
                    lines.append(f'  [{colored_symbol}] {diff.id} "{diff.name}"')
                    if diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
                        detail = _get_inventory_change_detail(diff)
                        if detail:
                            lines.append(f"      {detail}")
            else:
                lines.append("  No changes")

        # Segments inventory changes
        if diff_result.segments_diffs is not None:
            seg_changes = [d for d in diff_result.segments_diffs if d.change_type != ChangeType.UNCHANGED]
            lines.append("")
            lines.append(ANSIColors.bold(f"SEGMENTS ({len(seg_changes)} changes)", c))
            if seg_changes:
                for diff in seg_changes:
                    colored_symbol = _get_colored_symbol(diff.change_type, c)
                    lines.append(f'  [{colored_symbol}] {diff.id} "{diff.name}"')
                    if diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
                        detail = _get_inventory_change_detail(diff)
                        if detail:
                            lines.append(f"      {detail}")
            else:
                lines.append("  No changes")

    # Footer with total summary
    lines.append("")
    lines.append("=" * 80)
    if summary.has_changes:
        # Build color-coded total summary line
        total_parts = []
        if summary.total_added:
            total_parts.append(ANSIColors.green(f"{summary.total_added} added", c))
        if summary.total_removed:
            total_parts.append(ANSIColors.red(f"{summary.total_removed} removed", c))
        if summary.total_modified:
            total_parts.append(ANSIColors.yellow(f"{summary.total_modified} modified", c))
        total_line = ", ".join(total_parts)
        lines.append(ANSIColors.bold(f"Total: {total_line}", c))
        lines.append(
            f"  Metrics: {summary.metrics_added} added, {summary.metrics_removed} removed, {summary.metrics_modified} modified"
        )
        lines.append(
            f"  Dimensions: {summary.dimensions_added} added, {summary.dimensions_removed} removed, {summary.dimensions_modified} modified"
        )
        # Add inventory summary lines if present
        if summary.has_inventory_changes:
            if summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0:
                lines.append(
                    f"  Calc Metrics: {summary.calc_metrics_added} added, {summary.calc_metrics_removed} removed, {summary.calc_metrics_modified} modified"
                )
            if summary.source_segments_count > 0 or summary.target_segments_count > 0:
                lines.append(
                    f"  Segments: {summary.segments_added} added, {summary.segments_removed} removed, {summary.segments_modified} modified"
                )
    else:
        lines.append(ANSIColors.green("✓ No differences found", c))
    lines.append("=" * 80)

    return "\n".join(lines)


def _get_change_symbol(change_type: ChangeType) -> str:
    """Get symbol for change type"""
    symbols = {ChangeType.ADDED: "+", ChangeType.REMOVED: "-", ChangeType.MODIFIED: "~", ChangeType.UNCHANGED: " "}
    return symbols.get(change_type, "?")


def _get_colored_symbol(change_type: ChangeType, use_color: bool = True) -> str:
    """Get color-coded symbol for change type"""
    symbol = _get_change_symbol(change_type)
    if not use_color:
        return symbol
    if change_type == ChangeType.ADDED:
        return ANSIColors.green(symbol, use_color)
    elif change_type == ChangeType.REMOVED:
        return ANSIColors.red(symbol, use_color)
    elif change_type == ChangeType.MODIFIED:
        return ANSIColors.yellow(symbol, use_color)
    return symbol


def _format_diff_value(val: Any, truncate: bool = True, max_len: int = 30) -> str:
    """Format a value for diff display, handling None and NaN."""
    if val is None:
        return "(empty)"
    try:
        if pd.isna(val):
            return "(empty)"
    except TypeError, ValueError:
        pass
    result = str(val)
    if truncate and len(result) > max_len:
        result = result[:max_len]
    return result


def _get_change_detail(diff: ComponentDiff, truncate: bool = True) -> str:
    """Get detail string for a component diff"""
    if diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
        changes = []
        for field, (old_val, new_val) in diff.changed_fields.items():
            old_str = _format_diff_value(old_val, truncate)
            new_str = _format_diff_value(new_val, truncate)
            changes.append(f"{field}: '{old_str}' -> '{new_str}'")
        return "; ".join(changes)
    return ""


def _get_inventory_change_detail(diff: InventoryItemDiff, truncate: bool = True) -> str:
    """Get detail string for an inventory item diff"""
    if diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
        changes = []
        for field, (old_val, new_val) in diff.changed_fields.items():
            old_str = _format_diff_value(old_val, truncate)
            new_str = _format_diff_value(new_val, truncate)
            changes.append(f"{field}: '{old_str}' -> '{new_str}'")
        return "; ".join(changes)
    return ""


def _format_side_by_side(
    diff: ComponentDiff, source_label: str, target_label: str, col_width: int = 35, max_col_width: int = 60
) -> list[str]:
    """
    Format a component diff as a side-by-side comparison table.

    Args:
        diff: The ComponentDiff to format
        source_label: Label for source side
        target_label: Label for target side
        col_width: Base width of each column
        max_col_width: Maximum width of each column (text will wrap)

    Returns:
        List of formatted lines for the side-by-side view
    """
    lines = []
    if diff.change_type != ChangeType.MODIFIED or not diff.changed_fields:
        return lines

    # Pre-compute all display strings
    field_displays = []
    for field, (old_val, new_val) in diff.changed_fields.items():
        old_str = _format_diff_value(old_val, truncate=False)
        new_str = _format_diff_value(new_val, truncate=False)
        old_display = f"{field}: {old_str}"
        new_display = f"{field}: {new_str}"
        field_displays.append((old_display, new_display))

    # Calculate column width: expand to fit content but cap at max_col_width
    col_width = max(col_width, len(source_label) + 2, len(target_label) + 2)
    for old_display, new_display in field_displays:
        col_width = max(col_width, min(len(old_display), max_col_width), min(len(new_display), max_col_width))
    col_width = min(col_width, max_col_width)

    # Header for this component
    lines.append(f"    ┌{'─' * (col_width + 2)}┬{'─' * (col_width + 2)}┐")
    lines.append(f"    │ {source_label:<{col_width}} │ {target_label:<{col_width}} │")
    lines.append(f"    ├{'─' * (col_width + 2)}┼{'─' * (col_width + 2)}┤")

    # Changed fields with text wrapping
    for old_display, new_display in field_displays:
        # Wrap each side independently
        old_wrapped = textwrap.wrap(old_display, width=col_width) or [""]
        new_wrapped = textwrap.wrap(new_display, width=col_width) or [""]

        # Pad to same number of lines
        max_lines = max(len(old_wrapped), len(new_wrapped))
        old_wrapped.extend([""] * (max_lines - len(old_wrapped)))
        new_wrapped.extend([""] * (max_lines - len(new_wrapped)))

        # Output each line
        for old_line, new_line in zip(old_wrapped, new_wrapped, strict=True):
            lines.append(f"    │ {old_line:<{col_width}} │ {new_line:<{col_width}} │")

    lines.append(f"    └{'─' * (col_width + 2)}┴{'─' * (col_width + 2)}┘")

    return lines


def write_diff_grouped_by_field_output(diff_result: DiffResult, use_color: bool = True, limit: int = 10) -> str:
    """
    Write diff output grouped by changed field instead of by component.

    Args:
        diff_result: The DiffResult to output
        use_color: Use ANSI color codes
        limit: Max items per section (0 = unlimited)

    Returns:
        Formatted string for console output
    """
    lines = []
    summary = diff_result.summary
    meta = diff_result.metadata_diff
    c = use_color

    # Header
    lines.append("=" * 80)
    lines.append(ANSIColors.bold("DATA VIEW COMPARISON - GROUPED BY FIELD", c))
    lines.append("=" * 80)
    lines.append(f"{diff_result.source_label}: {meta.source_name}")
    lines.append(f"{diff_result.target_label}: {meta.target_name}")
    lines.append(f"Generated: {diff_result.generated_at}")
    lines.append("=" * 80)

    # Collect all changed fields across all components
    field_changes: dict[str, list[tuple[str, str, Any, Any]]] = {}  # field -> [(id, name, old, new), ...]

    # Also track breaking changes (type or schemaPath changes)
    breaking_changes = []

    all_diffs = diff_result.metric_diffs + diff_result.dimension_diffs
    for diff in all_diffs:
        if diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
            for field, (old_val, new_val) in diff.changed_fields.items():
                if field not in field_changes:
                    field_changes[field] = []
                field_changes[field].append((diff.id, diff.name, old_val, new_val))

                # Track breaking changes
                if field in ("type", "schemaPath"):
                    breaking_changes.append((diff.id, diff.name, field, old_val, new_val))

    # Summary
    lines.append("")
    lines.append(ANSIColors.bold("SUMMARY", c))
    lines.append(f"Total components changed: {summary.total_changes}")
    lines.append(f"  Added: {ANSIColors.green(str(summary.metrics_added + summary.dimensions_added), c)}")
    lines.append(f"  Removed: {ANSIColors.red(str(summary.metrics_removed + summary.dimensions_removed), c)}")
    lines.append(f"  Modified: {ANSIColors.yellow(str(summary.metrics_modified + summary.dimensions_modified), c)}")
    lines.append(f"Fields with changes: {len(field_changes)}")

    # Breaking changes warning
    if breaking_changes:
        lines.append("")
        lines.append(ANSIColors.red("⚠  BREAKING CHANGES DETECTED", c))
        lines.append("-" * 40)
        for comp_id, _comp_name, field, old_val, new_val in breaking_changes:
            lines.append(f"  {comp_id}: {field} changed")
            lines.append(
                f"    '{_format_diff_value(old_val, truncate=False)}' → '{_format_diff_value(new_val, truncate=False)}'"
            )

    # Group by field
    lines.append("")
    lines.append(ANSIColors.bold("CHANGES BY FIELD", c))
    lines.append("-" * 80)

    for field in sorted(field_changes.keys()):
        changes = field_changes[field]
        lines.append("")
        lines.append(f"{ANSIColors.cyan(field, c)} ({len(changes)} component{'s' if len(changes) != 1 else ''}):")

        items_to_show = changes if limit == 0 else changes[:limit]
        for comp_id, _comp_name, old_val, new_val in items_to_show:
            old_str = _format_diff_value(old_val, truncate=True)
            new_str = _format_diff_value(new_val, truncate=True)
            lines.append(f"  {comp_id}: '{old_str}' → '{new_str}'")

        if limit > 0 and len(changes) > limit:
            lines.append(f"  ... and {len(changes) - limit} more")

    # Added/removed summary
    added = [d for d in all_diffs if d.change_type == ChangeType.ADDED]
    removed = [d for d in all_diffs if d.change_type == ChangeType.REMOVED]

    if added:
        lines.append("")
        lines.append(ANSIColors.green(f"ADDED ({len(added)})", c))
        added_to_show = added if limit == 0 else added[:limit]
        lines.extend(f"  [+] {diff.id}" for diff in added_to_show)
        if limit > 0 and len(added) > limit:
            lines.append(f"  ... and {len(added) - limit} more")

    if removed:
        lines.append("")
        lines.append(ANSIColors.red(f"REMOVED ({len(removed)})", c))
        removed_to_show = removed if limit == 0 else removed[:limit]
        lines.extend(f"  [-] {diff.id}" for diff in removed_to_show)
        if limit > 0 and len(removed) > limit:
            lines.append(f"  ... and {len(removed) - limit} more")

    lines.append("")
    lines.append("=" * 80)
    lines.append(ANSIColors.cyan(f"Summary: {summary.natural_language_summary}", c))
    lines.append("=" * 80)

    return "\n".join(lines)


def write_diff_pr_comment_output(diff_result: DiffResult, changes_only: bool = False) -> str:
    """
    Write diff output in GitHub/GitLab PR comment format with collapsible details.

    Args:
        diff_result: The DiffResult to output
        changes_only: Only include changed items

    Returns:
        Markdown formatted string optimized for PR comments
    """
    lines = []
    summary = diff_result.summary

    # Header
    lines.append("### 📊 Data View Comparison")
    lines.append("")
    lines.append(f"**{diff_result.source_label}** → **{diff_result.target_label}**")
    lines.append("")

    # Summary table
    lines.append("| Component | Source | Target | Added | Removed | Modified | Unchanged | Changed |")
    lines.append("|-----------|-------:|-------:|------:|--------:|---------:|----------:|--------:|")
    lines.append(
        f"| Metrics | {summary.source_metrics_count} | {summary.target_metrics_count} | "
        f"+{summary.metrics_added} | -{summary.metrics_removed} | ~{summary.metrics_modified} | "
        f"{summary.metrics_unchanged} | {summary.metrics_change_percent:.1f}% |"
    )
    lines.append(
        f"| Dimensions | {summary.source_dimensions_count} | {summary.target_dimensions_count} | "
        f"+{summary.dimensions_added} | -{summary.dimensions_removed} | ~{summary.dimensions_modified} | "
        f"{summary.dimensions_unchanged} | {summary.dimensions_change_percent:.1f}% |"
    )
    lines.append("")

    # Breaking changes warning
    breaking_changes = []
    all_diffs = diff_result.metric_diffs + diff_result.dimension_diffs
    for diff in all_diffs:
        if diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
            for field in diff.changed_fields:
                if field in ("type", "schemaPath"):
                    old_val, new_val = diff.changed_fields[field]
                    breaking_changes.append((diff.id, field, old_val, new_val))

    if breaking_changes:
        lines.append("#### ⚠ Breaking Changes Detected")
        lines.append("")
        lines.append("| Component | Field | Before | After |")
        lines.append("|-----------|-------|--------|-------|")
        for comp_id, field, old_val, new_val in breaking_changes[:10]:
            lines.append(
                f"| `{comp_id}` | {field} | `{_format_diff_value(old_val, truncate=False)}` | `{_format_diff_value(new_val, truncate=False)}` |"
            )
        if len(breaking_changes) > 10:
            lines.append(f"| ... | | | +{len(breaking_changes) - 10} more |")
        lines.append("")

    # Natural language summary
    lines.append(f"**Summary:** {summary.natural_language_summary}")
    lines.append("")

    # Collapsible details
    metric_changes = [d for d in diff_result.metric_diffs if d.change_type != ChangeType.UNCHANGED]
    dim_changes = [d for d in diff_result.dimension_diffs if d.change_type != ChangeType.UNCHANGED]

    if metric_changes:
        lines.append("<details>")
        lines.append(f"<summary>📈 Metrics Changes ({len(metric_changes)})</summary>")
        lines.append("")
        lines.append("| Change | ID | Name |")
        lines.append("|--------|----|----- |")
        for diff in metric_changes[:25]:
            symbol = {ChangeType.ADDED: "➕", ChangeType.REMOVED: "➖", ChangeType.MODIFIED: "✏️"}.get(
                diff.change_type, ""
            )
            lines.append(f"| {symbol} | `{diff.id}` | {diff.name} |")
        if len(metric_changes) > 25:
            lines.append(f"| ... | | +{len(metric_changes) - 25} more |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if dim_changes:
        lines.append("<details>")
        lines.append(f"<summary>📏 Dimensions Changes ({len(dim_changes)})</summary>")
        lines.append("")
        lines.append("| Change | ID | Name |")
        lines.append("|--------|----|----- |")
        for diff in dim_changes[:25]:
            symbol = {ChangeType.ADDED: "➕", ChangeType.REMOVED: "➖", ChangeType.MODIFIED: "✏️"}.get(
                diff.change_type, ""
            )
            lines.append(f"| {symbol} | `{diff.id}` | {diff.name} |")
        if len(dim_changes) > 25:
            lines.append(f"| ... | | +{len(dim_changes) - 25} more |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by CJA SDR Generator v{diff_result.tool_version}*")

    return "\n".join(lines)


def detect_breaking_changes(diff_result: DiffResult) -> list[dict[str, Any]]:
    """
    Detect breaking changes in a diff result.

    Breaking changes include:
    - Changes to 'type' field (data type changes)
    - Changes to 'schemaPath' field (schema mapping changes)
    - Removal of existing components

    Args:
        diff_result: The DiffResult to analyze

    Returns:
        List of breaking change dictionaries with details
    """
    breaking_changes = []

    all_diffs = diff_result.metric_diffs + diff_result.dimension_diffs

    for diff in all_diffs:
        # Removed components are breaking
        if diff.change_type == ChangeType.REMOVED:
            breaking_changes.append(
                {
                    "component_id": diff.id,
                    "component_name": diff.name,
                    "change_type": "removed",
                    "severity": "high",
                    "description": f"Component '{diff.name}' was removed",
                }
            )

        # Check for type or schema changes
        elif diff.change_type == ChangeType.MODIFIED and diff.changed_fields:
            for field, (old_val, new_val) in diff.changed_fields.items():
                if field == "type":
                    breaking_changes.append(
                        {
                            "component_id": diff.id,
                            "component_name": diff.name,
                            "change_type": "type_changed",
                            "field": field,
                            "old_value": old_val,
                            "new_value": new_val,
                            "severity": "high",
                            "description": f"Data type changed from '{_format_diff_value(old_val, truncate=False)}' to '{_format_diff_value(new_val, truncate=False)}'",
                        }
                    )
                elif field == "schemaPath":
                    breaking_changes.append(
                        {
                            "component_id": diff.id,
                            "component_name": diff.name,
                            "change_type": "schema_changed",
                            "field": field,
                            "old_value": old_val,
                            "new_value": new_val,
                            "severity": "medium",
                            "description": f"Schema path changed from '{_format_diff_value(old_val, truncate=False)}' to '{_format_diff_value(new_val, truncate=False)}'",
                        }
                    )

    return breaking_changes


def write_diff_json_output(
    diff_result: DiffResult,
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    changes_only: bool = False,
) -> str:
    """
    Write diff comparison to JSON format.

    Args:
        diff_result: The DiffResult to output
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        changes_only: Only include changed items

    Returns:
        Path to JSON output file
    """
    try:
        logger.info("Generating diff JSON output...")

        summary = diff_result.summary
        meta = diff_result.metadata_diff

        def serialize_component_diff(d: ComponentDiff) -> dict:
            return {
                "id": d.id,
                "name": d.name,
                "change_type": d.change_type.value,
                "changed_fields": {k: {"source": v[0], "target": v[1]} for k, v in (d.changed_fields or {}).items()},
                "source_data": d.source_data,
                "target_data": d.target_data,
            }

        def serialize_inventory_diff(d: InventoryItemDiff) -> dict:
            return {
                "id": d.id,
                "name": d.name,
                "change_type": d.change_type.value,
                "inventory_type": d.inventory_type,
                "changed_fields": {k: {"source": v[0], "target": v[1]} for k, v in (d.changed_fields or {}).items()},
                "source_data": d.source_data,
                "target_data": d.target_data,
            }

        # Filter diffs if changes_only
        metric_diffs = diff_result.metric_diffs
        dimension_diffs = diff_result.dimension_diffs
        if changes_only:
            metric_diffs = [d for d in metric_diffs if d.change_type != ChangeType.UNCHANGED]
            dimension_diffs = [d for d in dimension_diffs if d.change_type != ChangeType.UNCHANGED]

        json_data = {
            "metadata": {
                "generated_at": diff_result.generated_at,
                "tool_version": diff_result.tool_version,
                "source_label": diff_result.source_label,
                "target_label": diff_result.target_label,
            },
            "source": {
                "id": meta.source_id,
                "name": meta.source_name,
                "owner": meta.source_owner,
                "description": meta.source_description,
            },
            "target": {
                "id": meta.target_id,
                "name": meta.target_name,
                "owner": meta.target_owner,
                "description": meta.target_description,
            },
            "summary": {
                "source_metrics_count": summary.source_metrics_count,
                "target_metrics_count": summary.target_metrics_count,
                "source_dimensions_count": summary.source_dimensions_count,
                "target_dimensions_count": summary.target_dimensions_count,
                "metrics_added": summary.metrics_added,
                "metrics_removed": summary.metrics_removed,
                "metrics_modified": summary.metrics_modified,
                "metrics_unchanged": summary.metrics_unchanged,
                "metrics_change_percent": summary.metrics_change_percent,
                "dimensions_added": summary.dimensions_added,
                "dimensions_removed": summary.dimensions_removed,
                "dimensions_modified": summary.dimensions_modified,
                "dimensions_unchanged": summary.dimensions_unchanged,
                "dimensions_change_percent": summary.dimensions_change_percent,
                "has_changes": summary.has_changes,
                "total_changes": summary.total_changes,
                "total_added": summary.total_added,
                "total_removed": summary.total_removed,
                "total_modified": summary.total_modified,
                "total_summary": summary.total_summary,
            },
            "metric_diffs": [serialize_component_diff(d) for d in metric_diffs],
            "dimension_diffs": [serialize_component_diff(d) for d in dimension_diffs],
        }

        # Add inventory diffs if present
        if diff_result.has_inventory_diffs:
            inventory_summary = {}
            if summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0:
                inventory_summary["calculated_metrics"] = {
                    "source_count": summary.source_calc_metrics_count,
                    "target_count": summary.target_calc_metrics_count,
                    "added": summary.calc_metrics_added,
                    "removed": summary.calc_metrics_removed,
                    "modified": summary.calc_metrics_modified,
                    "unchanged": summary.calc_metrics_unchanged,
                }
            if summary.source_segments_count > 0 or summary.target_segments_count > 0:
                inventory_summary["segments"] = {
                    "source_count": summary.source_segments_count,
                    "target_count": summary.target_segments_count,
                    "added": summary.segments_added,
                    "removed": summary.segments_removed,
                    "modified": summary.segments_modified,
                    "unchanged": summary.segments_unchanged,
                }
            json_data["inventory_summary"] = inventory_summary

            if diff_result.calc_metrics_diffs is not None:
                calc_diffs = diff_result.calc_metrics_diffs
                if changes_only:
                    calc_diffs = [d for d in calc_diffs if d.change_type != ChangeType.UNCHANGED]
                json_data["calculated_metrics_diffs"] = [serialize_inventory_diff(d) for d in calc_diffs]

            if diff_result.segments_diffs is not None:
                seg_diffs = diff_result.segments_diffs
                if changes_only:
                    seg_diffs = [d for d in seg_diffs if d.change_type != ChangeType.UNCHANGED]
                json_data["segments_diffs"] = [serialize_inventory_diff(d) for d in seg_diffs]

        json_file = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Diff JSON file created: {json_file}")
        return json_file

    except Exception as e:
        logger.error(_format_error_msg("creating diff JSON file", error=e))
        raise


def write_diff_markdown_output(
    diff_result: DiffResult,
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    changes_only: bool = False,
    side_by_side: bool = False,
) -> str:
    """
    Write diff comparison to Markdown format.

    Args:
        diff_result: The DiffResult to output
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        changes_only: Only include changed items
        side_by_side: Show side-by-side comparison for modified items

    Returns:
        Path to Markdown output file
    """
    try:
        logger.info("Generating diff Markdown output...")

        summary = diff_result.summary
        meta = diff_result.metadata_diff
        md_parts = []

        # Title
        md_parts.append("# Data View Comparison Report\n")

        # Metadata
        md_parts.append("## Comparison Details\n")
        md_parts.append(f"**{diff_result.source_label}:** {meta.source_name} (`{meta.source_id}`)")
        md_parts.append(f"**{diff_result.target_label}:** {meta.target_name} (`{meta.target_id}`)")
        md_parts.append(f"**Generated:** {diff_result.generated_at}")
        md_parts.append(f"**Tool Version:** {diff_result.tool_version}\n")

        # Summary table
        md_parts.append("## Summary\n")
        md_parts.append(
            f"| Component | {diff_result.source_label} | {diff_result.target_label} | Added | Removed | Modified | Unchanged | Changed |"
        )
        md_parts.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        md_parts.append(
            f"| Metrics | {summary.source_metrics_count} | {summary.target_metrics_count} | "
            f"+{summary.metrics_added} | -{summary.metrics_removed} | ~{summary.metrics_modified} | "
            f"{summary.metrics_unchanged} | {summary.metrics_change_percent:.1f}% |"
        )
        md_parts.append(
            f"| Dimensions | {summary.source_dimensions_count} | {summary.target_dimensions_count} | "
            f"+{summary.dimensions_added} | -{summary.dimensions_removed} | ~{summary.dimensions_modified} | "
            f"{summary.dimensions_unchanged} | {summary.dimensions_change_percent:.1f}% |"
        )

        # Add inventory rows to summary if present
        if diff_result.has_inventory_diffs:
            if summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0:
                md_parts.append(
                    f"| **Calc Metrics** | {summary.source_calc_metrics_count} | {summary.target_calc_metrics_count} | "
                    f"+{summary.calc_metrics_added} | -{summary.calc_metrics_removed} | ~{summary.calc_metrics_modified} | "
                    f"{summary.calc_metrics_unchanged} | - |"
                )
            if summary.source_segments_count > 0 or summary.target_segments_count > 0:
                md_parts.append(
                    f"| **Segments** | {summary.source_segments_count} | {summary.target_segments_count} | "
                    f"+{summary.segments_added} | -{summary.segments_removed} | ~{summary.segments_modified} | "
                    f"{summary.segments_unchanged} | - |"
                )
        md_parts.append("")

        if not summary.has_changes:
            md_parts.append("**✓ No differences found.**\n")
        else:
            md_parts.append(f"**Total: {summary.total_summary}**\n")

        # Metrics changes
        metric_changes = [d for d in diff_result.metric_diffs if d.change_type != ChangeType.UNCHANGED]
        if metric_changes or not changes_only:
            md_parts.append("## Metrics Changes\n")
            if metric_changes:
                md_parts.append("| Status | ID | Name | Details |")
                md_parts.append("| --- | --- | --- | --- |")
                for diff in metric_changes:
                    symbol = _get_change_emoji(diff.change_type)
                    detail = _get_change_detail(diff).replace("|", "\\|")
                    md_parts.append(f"| {symbol} | `{diff.id}` | {diff.name} | {detail} |")

                # Add side-by-side detail for modified items
                if side_by_side:
                    modified = [d for d in metric_changes if d.change_type == ChangeType.MODIFIED]
                    if modified:
                        md_parts.append("\n### Modified Metrics - Side by Side\n")
                        for diff in modified:
                            md_parts.extend(
                                _format_markdown_side_by_side(diff, diff_result.source_label, diff_result.target_label)
                            )
            else:
                md_parts.append("*No changes*")
            md_parts.append("")

        # Dimensions changes
        dim_changes = [d for d in diff_result.dimension_diffs if d.change_type != ChangeType.UNCHANGED]
        if dim_changes or not changes_only:
            md_parts.append("## Dimensions Changes\n")
            if dim_changes:
                md_parts.append("| Status | ID | Name | Details |")
                md_parts.append("| --- | --- | --- | --- |")
                for diff in dim_changes:
                    symbol = _get_change_emoji(diff.change_type)
                    detail = _get_change_detail(diff).replace("|", "\\|")
                    md_parts.append(f"| {symbol} | `{diff.id}` | {diff.name} | {detail} |")

                # Add side-by-side detail for modified items
                if side_by_side:
                    modified = [d for d in dim_changes if d.change_type == ChangeType.MODIFIED]
                    if modified:
                        md_parts.append("\n### Modified Dimensions - Side by Side\n")
                        for diff in modified:
                            md_parts.extend(
                                _format_markdown_side_by_side(diff, diff_result.source_label, diff_result.target_label)
                            )
            else:
                md_parts.append("*No changes*")
            md_parts.append("")

        # Inventory changes (if present)
        if diff_result.has_inventory_diffs:
            md_parts.append("---\n")
            md_parts.append("# Inventory Changes\n")

            # Calculated metrics inventory changes
            if diff_result.calc_metrics_diffs is not None:
                calc_changes = [d for d in diff_result.calc_metrics_diffs if d.change_type != ChangeType.UNCHANGED]
                md_parts.append(f"## Calculated Metrics Changes ({len(calc_changes)})\n")
                if calc_changes:
                    md_parts.append("| Status | ID | Name | Details |")
                    md_parts.append("| --- | --- | --- | --- |")
                    for diff in calc_changes:
                        symbol = _get_change_emoji(diff.change_type)
                        detail = _get_inventory_change_detail(diff).replace("|", "\\|")
                        md_parts.append(f"| {symbol} | `{diff.id}` | {diff.name} | {detail} |")
                else:
                    md_parts.append("*No changes*")
                md_parts.append("")

            # Segments inventory changes
            if diff_result.segments_diffs is not None:
                seg_changes = [d for d in diff_result.segments_diffs if d.change_type != ChangeType.UNCHANGED]
                md_parts.append(f"## Segments Changes ({len(seg_changes)})\n")
                if seg_changes:
                    md_parts.append("| Status | ID | Name | Details |")
                    md_parts.append("| --- | --- | --- | --- |")
                    for diff in seg_changes:
                        symbol = _get_change_emoji(diff.change_type)
                        detail = _get_inventory_change_detail(diff).replace("|", "\\|")
                        md_parts.append(f"| {symbol} | `{diff.id}` | {diff.name} | {detail} |")
                else:
                    md_parts.append("*No changes*")
                md_parts.append("")

        md_parts.append("---")
        md_parts.append("*Generated by CJA Auto SDR Generator*")

        markdown_file = os.path.join(output_dir, f"{base_filename}.md")
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_parts))

        logger.info(f"Diff Markdown file created: {markdown_file}")
        return markdown_file

    except Exception as e:
        logger.error(_format_error_msg("creating diff Markdown file", error=e))
        raise


def _get_change_emoji(change_type: ChangeType) -> str:
    """Get emoji for change type"""
    emojis = {ChangeType.ADDED: "+", ChangeType.REMOVED: "-", ChangeType.MODIFIED: "~", ChangeType.UNCHANGED: ""}
    return emojis.get(change_type, "")


def _format_markdown_side_by_side(diff: ComponentDiff, source_label: str, target_label: str) -> list[str]:
    """
    Format a component diff as a side-by-side markdown table.

    Args:
        diff: The ComponentDiff to format
        source_label: Label for source side
        target_label: Label for target side

    Returns:
        List of markdown lines for the side-by-side view
    """
    lines = []
    if diff.change_type != ChangeType.MODIFIED or not diff.changed_fields:
        return lines

    # Component header
    lines.append(f"\n**`{diff.id}`** - {diff.name}\n")

    # Side-by-side table
    lines.append(f"| Field | {source_label} | {target_label} |")
    lines.append("| --- | --- | --- |")

    for field, (old_val, new_val) in diff.changed_fields.items():
        old_formatted = _format_diff_value(old_val, truncate=False)
        new_formatted = _format_diff_value(new_val, truncate=False)
        # Use italic for empty values in markdown
        old_str = "*(empty)*" if old_formatted == "(empty)" else old_formatted.replace("|", "\\|")
        new_str = "*(empty)*" if new_formatted == "(empty)" else new_formatted.replace("|", "\\|")

        # Truncate very long values
        if len(old_str) > 50:
            old_str = old_str[:47] + "..."
        if len(new_str) > 50:
            new_str = new_str[:47] + "..."

        lines.append(f"| `{field}` | {old_str} | {new_str} |")

    lines.append("")
    return lines


def write_diff_html_output(
    diff_result: DiffResult,
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    changes_only: bool = False,
) -> str:
    """
    Write diff comparison to HTML format with professional styling.

    Args:
        diff_result: The DiffResult to output
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        changes_only: Only include changed items

    Returns:
        Path to HTML output file
    """
    try:
        logger.info("Generating diff HTML output...")

        summary = diff_result.summary
        meta = diff_result.metadata_diff
        html_parts = []

        # HTML header with CSS
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data View Comparison Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .metadata {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .summary-table th, .summary-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .summary-table th {
            background-color: #3498db;
            color: white;
        }
        .diff-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        .diff-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .diff-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        .row-added {
            background-color: #d4edda !important;
        }
        .row-removed {
            background-color: #f8d7da !important;
        }
        .row-modified {
            background-color: #fff3cd !important;
        }
        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }
        .badge-added { background-color: #28a745; color: white; }
        .badge-removed { background-color: #dc3545; color: white; }
        .badge-modified { background-color: #ffc107; color: black; }
        .no-changes {
            color: #28a745;
            font-weight: bold;
            font-size: 18px;
            text-align: center;
            padding: 20px;
        }
        .total-changes {
            font-size: 18px;
            font-weight: bold;
            margin: 20px 0;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Data View Comparison Report</h1>
""")

        # Metadata section
        html_parts.append(f"""
        <div class="metadata">
            <p><strong>{diff_result.source_label}:</strong> {meta.source_name} (<code>{meta.source_id}</code>)</p>
            <p><strong>{diff_result.target_label}:</strong> {meta.target_name} (<code>{meta.target_id}</code>)</p>
            <p><strong>Generated:</strong> {diff_result.generated_at}</p>
        </div>
""")

        # Summary table
        html_parts.append(f"""
        <h2>Summary</h2>
        <table class="summary-table">
            <tr>
                <th>Component</th>
                <th>{diff_result.source_label}</th>
                <th>{diff_result.target_label}</th>
                <th>Added</th>
                <th>Removed</th>
                <th>Modified</th>
                <th>Unchanged</th>
                <th>Changed</th>
            </tr>
            <tr>
                <td>Metrics</td>
                <td>{summary.source_metrics_count}</td>
                <td>{summary.target_metrics_count}</td>
                <td><span class="badge badge-added">+{summary.metrics_added}</span></td>
                <td><span class="badge badge-removed">-{summary.metrics_removed}</span></td>
                <td><span class="badge badge-modified">~{summary.metrics_modified}</span></td>
                <td>{summary.metrics_unchanged}</td>
                <td>{summary.metrics_change_percent:.1f}%</td>
            </tr>
            <tr>
                <td>Dimensions</td>
                <td>{summary.source_dimensions_count}</td>
                <td>{summary.target_dimensions_count}</td>
                <td><span class="badge badge-added">+{summary.dimensions_added}</span></td>
                <td><span class="badge badge-removed">-{summary.dimensions_removed}</span></td>
                <td><span class="badge badge-modified">~{summary.dimensions_modified}</span></td>
                <td>{summary.dimensions_unchanged}</td>
                <td>{summary.dimensions_change_percent:.1f}%</td>
            </tr>
        </table>
""")

        # Add inventory rows to summary if present
        if diff_result.has_inventory_diffs:
            inv_rows = []
            if summary.source_calc_metrics_count > 0 or summary.target_calc_metrics_count > 0:
                inv_rows.append(f"""
            <tr>
                <td><strong>Calc Metrics</strong></td>
                <td>{summary.source_calc_metrics_count}</td>
                <td>{summary.target_calc_metrics_count}</td>
                <td><span class="badge badge-added">+{summary.calc_metrics_added}</span></td>
                <td><span class="badge badge-removed">-{summary.calc_metrics_removed}</span></td>
                <td><span class="badge badge-modified">~{summary.calc_metrics_modified}</span></td>
                <td>{summary.calc_metrics_unchanged}</td>
                <td>-</td>
            </tr>""")
            if summary.source_segments_count > 0 or summary.target_segments_count > 0:
                inv_rows.append(f"""
            <tr>
                <td><strong>Segments</strong></td>
                <td>{summary.source_segments_count}</td>
                <td>{summary.target_segments_count}</td>
                <td><span class="badge badge-added">+{summary.segments_added}</span></td>
                <td><span class="badge badge-removed">-{summary.segments_removed}</span></td>
                <td><span class="badge badge-modified">~{summary.segments_modified}</span></td>
                <td>{summary.segments_unchanged}</td>
                <td>-</td>
            </tr>""")
            if inv_rows:
                # Insert inventory rows before the closing </table> tag
                html_parts[-1] = html_parts[-1].replace("</table>", "".join(inv_rows) + "\n        </table>")

        if not summary.has_changes:
            html_parts.append('<p class="no-changes">No differences found.</p>')
        else:
            html_parts.append(f'<p class="total-changes">Total changes: {summary.total_changes}</p>')

        # Helper function to generate diff table
        def generate_diff_table(diffs: list[ComponentDiff], title: str):
            changes = [d for d in diffs if d.change_type != ChangeType.UNCHANGED]
            if not changes and changes_only:
                return ""

            html = f"<h2>{title}</h2>\n"
            if not changes:
                html += "<p><em>No changes</em></p>\n"
                return html

            html += """<table class="diff-table">
                <tr>
                    <th>Status</th>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Details</th>
                </tr>"""

            for diff in changes:
                row_class = f"row-{diff.change_type.value}"
                badge_class = f"badge-{diff.change_type.value}"
                badge_text = diff.change_type.value.upper()
                detail = _get_change_detail(diff)
                detail_escaped = detail.replace("<", "&lt;").replace(">", "&gt;")

                html += f'''
                <tr class="{row_class}">
                    <td><span class="badge {badge_class}">{badge_text}</span></td>
                    <td><code>{diff.id}</code></td>
                    <td>{diff.name}</td>
                    <td>{detail_escaped}</td>
                </tr>'''

            html += "</table>\n"
            return html

        html_parts.append(generate_diff_table(diff_result.metric_diffs, "Metrics Changes"))
        html_parts.append(generate_diff_table(diff_result.dimension_diffs, "Dimensions Changes"))

        # Inventory diff sections (if present)
        if diff_result.has_inventory_diffs:

            def generate_inventory_diff_table(diffs: list[InventoryItemDiff] | None, title: str):
                if diffs is None:
                    return ""
                changes = [d for d in diffs if d.change_type != ChangeType.UNCHANGED]
                if not changes and changes_only:
                    return ""

                html = f"<h2>{title}</h2>\n"
                if not changes:
                    html += "<p><em>No changes</em></p>\n"
                    return html

                html += """<table class="diff-table">
                    <tr>
                        <th>Status</th>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Details</th>
                    </tr>"""

                for diff in changes:
                    row_class = f"row-{diff.change_type.value}"
                    badge_class = f"badge-{diff.change_type.value}"
                    badge_text = diff.change_type.value.upper()
                    detail = _get_inventory_change_detail(diff)
                    detail_escaped = detail.replace("<", "&lt;").replace(">", "&gt;")

                    html += f'''
                    <tr class="{row_class}">
                        <td><span class="badge {badge_class}">{badge_text}</span></td>
                        <td><code>{diff.id}</code></td>
                        <td>{diff.name}</td>
                        <td>{detail_escaped}</td>
                    </tr>'''

                html += "</table>\n"
                return html

            html_parts.append(
                "<h2 style='border-top: 2px solid #3498db; padding-top: 20px; margin-top: 30px;'>Inventory Changes</h2>"
            )
            html_parts.append(
                generate_inventory_diff_table(diff_result.calc_metrics_diffs, "Calculated Metrics Changes")
            )
            html_parts.append(generate_inventory_diff_table(diff_result.segments_diffs, "Segments Changes"))

        # Footer
        html_parts.append(f"""
        <div class="footer">
            <p>Generated by CJA SDR Generator v{diff_result.tool_version}</p>
        </div>
    </div>
</body>
</html>
""")

        html_file = os.path.join(output_dir, f"{base_filename}.html")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        logger.info(f"Diff HTML file created: {html_file}")
        return html_file

    except Exception as e:
        logger.error(_format_error_msg("creating diff HTML file", error=e))
        raise


def write_diff_excel_output(
    diff_result: DiffResult,
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    changes_only: bool = False,
) -> str:
    """
    Write diff comparison to Excel format with color-coded rows.

    Args:
        diff_result: The DiffResult to output
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        changes_only: Only include changed items

    Returns:
        Path to Excel output file
    """
    try:
        logger.info("Generating diff Excel output...")

        summary = diff_result.summary
        meta = diff_result.metadata_diff
        excel_file = os.path.join(output_dir, f"{base_filename}.xlsx")

        with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
            workbook = writer.book

            # Define formats
            added_format = workbook.add_format({"bg_color": "#d4edda", "border": 1})
            removed_format = workbook.add_format({"bg_color": "#f8d7da", "border": 1})
            modified_format = workbook.add_format({"bg_color": "#fff3cd", "border": 1})
            normal_format = workbook.add_format({"border": 1})

            # Summary sheet - build rows dynamically
            summary_rows = [
                {
                    "Component": "Metrics",
                    diff_result.source_label: summary.source_metrics_count,
                    diff_result.target_label: summary.target_metrics_count,
                    "Added": summary.metrics_added,
                    "Removed": summary.metrics_removed,
                    "Modified": summary.metrics_modified,
                    "Unchanged": summary.metrics_unchanged,
                    "Changed %": f"{summary.metrics_change_percent:.1f}%",
                },
                {
                    "Component": "Dimensions",
                    diff_result.source_label: summary.source_dimensions_count,
                    diff_result.target_label: summary.target_dimensions_count,
                    "Added": summary.dimensions_added,
                    "Removed": summary.dimensions_removed,
                    "Modified": summary.dimensions_modified,
                    "Unchanged": summary.dimensions_unchanged,
                    "Changed %": f"{summary.dimensions_change_percent:.1f}%",
                },
            ]

            # Add inventory rows if present (check for actual inventory diffs)
            if diff_result.calc_metrics_diffs is not None:
                summary_rows.append(
                    {
                        "Component": "Calc Metrics",
                        diff_result.source_label: summary.source_calc_metrics_count,
                        diff_result.target_label: summary.target_calc_metrics_count,
                        "Added": summary.calc_metrics_added,
                        "Removed": summary.calc_metrics_removed,
                        "Modified": summary.calc_metrics_modified,
                        "Unchanged": summary.calc_metrics_unchanged,
                        "Changed %": f"{summary.calc_metrics_change_percent:.1f}%",
                    }
                )
            if diff_result.segments_diffs is not None:
                summary_rows.append(
                    {
                        "Component": "Segments",
                        diff_result.source_label: summary.source_segments_count,
                        diff_result.target_label: summary.target_segments_count,
                        "Added": summary.segments_added,
                        "Removed": summary.segments_removed,
                        "Modified": summary.segments_modified,
                        "Unchanged": summary.segments_unchanged,
                        "Changed %": f"{summary.segments_change_percent:.1f}%",
                    }
                )

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Metadata sheet
            metadata_data = {
                "Property": [
                    "Source ID",
                    "Source Name",
                    "Target ID",
                    "Target Name",
                    "Generated At",
                    "Has Changes",
                    "Total Changes",
                ],
                "Value": [
                    meta.source_id,
                    meta.source_name,
                    meta.target_id,
                    meta.target_name,
                    diff_result.generated_at,
                    str(summary.has_changes),
                    summary.total_changes,
                ],
            }
            metadata_df = pd.DataFrame(metadata_data)
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

            # Helper function to write diff sheet
            def write_diff_sheet(diffs: list[ComponentDiff], sheet_name: str):
                if changes_only:
                    diffs = [d for d in diffs if d.change_type != ChangeType.UNCHANGED]

                if not diffs:
                    df = pd.DataFrame({"Message": ["No changes"]})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    return

                rows = [
                    {
                        "Status": diff.change_type.value.upper(),
                        "ID": diff.id,
                        "Name": diff.name,
                        "Details": _get_change_detail(diff),
                    }
                    for diff in diffs
                ]

                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Apply color formatting
                worksheet = writer.sheets[sheet_name]
                for row_idx, diff in enumerate(diffs, start=1):
                    if diff.change_type == ChangeType.ADDED:
                        fmt = added_format
                    elif diff.change_type == ChangeType.REMOVED:
                        fmt = removed_format
                    elif diff.change_type == ChangeType.MODIFIED:
                        fmt = modified_format
                    else:
                        fmt = normal_format

                    for col_idx in range(len(df.columns)):
                        worksheet.write(row_idx, col_idx, df.iloc[row_idx - 1, col_idx], fmt)

            write_diff_sheet(diff_result.metric_diffs, "Metrics Diff")
            write_diff_sheet(diff_result.dimension_diffs, "Dimensions Diff")

            # Helper function to write inventory diff sheet
            def write_inventory_diff_sheet(diffs: list[InventoryItemDiff] | None, sheet_name: str):
                if diffs is None:
                    return

                if changes_only:
                    diffs = [d for d in diffs if d.change_type != ChangeType.UNCHANGED]

                if not diffs:
                    df = pd.DataFrame({"Message": ["No changes"]})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    return

                rows = [
                    {
                        "Status": diff.change_type.value.upper(),
                        "ID": diff.id,
                        "Name": diff.name,
                        "Details": _get_inventory_change_detail(diff),
                    }
                    for diff in diffs
                ]

                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Apply color formatting
                worksheet = writer.sheets[sheet_name]
                for row_idx, diff in enumerate(diffs, start=1):
                    if diff.change_type == ChangeType.ADDED:
                        fmt = added_format
                    elif diff.change_type == ChangeType.REMOVED:
                        fmt = removed_format
                    elif diff.change_type == ChangeType.MODIFIED:
                        fmt = modified_format
                    else:
                        fmt = normal_format

                    for col_idx in range(len(df.columns)):
                        worksheet.write(row_idx, col_idx, df.iloc[row_idx - 1, col_idx], fmt)

            # Write inventory diff sheets if present
            if diff_result.calc_metrics_diffs is not None:
                write_inventory_diff_sheet(diff_result.calc_metrics_diffs, "Calc Metrics Diff")
            if diff_result.segments_diffs is not None:
                write_inventory_diff_sheet(diff_result.segments_diffs, "Segments Diff")

        logger.info(f"Diff Excel file created: {excel_file}")
        return excel_file

    except Exception as e:
        logger.error(_format_error_msg("creating diff Excel file", error=e))
        raise


def write_diff_csv_output(
    diff_result: DiffResult,
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    changes_only: bool = False,
) -> str:
    """
    Write diff comparison to CSV files.

    Args:
        diff_result: The DiffResult to output
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        changes_only: Only include changed items

    Returns:
        Path to output directory containing CSV files
    """
    try:
        logger.info("Generating diff CSV output...")

        summary = diff_result.summary
        meta = diff_result.metadata_diff

        # Create subdirectory for CSV files
        csv_dir = os.path.join(output_dir, f"{base_filename}_csv")
        os.makedirs(csv_dir, exist_ok=True)

        # Summary CSV - build rows dynamically
        summary_rows = [
            {
                "Component": "Metrics",
                "Source_Count": summary.source_metrics_count,
                "Target_Count": summary.target_metrics_count,
                "Added": summary.metrics_added,
                "Removed": summary.metrics_removed,
                "Modified": summary.metrics_modified,
                "Unchanged": summary.metrics_unchanged,
                "Changed_Percent": summary.metrics_change_percent,
            },
            {
                "Component": "Dimensions",
                "Source_Count": summary.source_dimensions_count,
                "Target_Count": summary.target_dimensions_count,
                "Added": summary.dimensions_added,
                "Removed": summary.dimensions_removed,
                "Modified": summary.dimensions_modified,
                "Unchanged": summary.dimensions_unchanged,
                "Changed_Percent": summary.dimensions_change_percent,
            },
        ]

        # Add inventory rows if present (check for actual inventory diffs)
        if diff_result.calc_metrics_diffs is not None:
            summary_rows.append(
                {
                    "Component": "Calc_Metrics",
                    "Source_Count": summary.source_calc_metrics_count,
                    "Target_Count": summary.target_calc_metrics_count,
                    "Added": summary.calc_metrics_added,
                    "Removed": summary.calc_metrics_removed,
                    "Modified": summary.calc_metrics_modified,
                    "Unchanged": summary.calc_metrics_unchanged,
                    "Changed_Percent": summary.calc_metrics_change_percent,
                }
            )
        if diff_result.segments_diffs is not None:
            summary_rows.append(
                {
                    "Component": "Segments",
                    "Source_Count": summary.source_segments_count,
                    "Target_Count": summary.target_segments_count,
                    "Added": summary.segments_added,
                    "Removed": summary.segments_removed,
                    "Modified": summary.segments_modified,
                    "Unchanged": summary.segments_unchanged,
                    "Changed_Percent": summary.segments_change_percent,
                }
            )

        pd.DataFrame(summary_rows).to_csv(os.path.join(csv_dir, "summary.csv"), index=False)
        logger.info("  Created: summary.csv")

        # Metadata CSV
        metadata_data = {
            "Property": [
                "source_id",
                "source_name",
                "target_id",
                "target_name",
                "generated_at",
                "has_changes",
                "total_changes",
            ],
            "Value": [
                meta.source_id,
                meta.source_name,
                meta.target_id,
                meta.target_name,
                diff_result.generated_at,
                str(summary.has_changes),
                summary.total_changes,
            ],
        }
        pd.DataFrame(metadata_data).to_csv(os.path.join(csv_dir, "metadata.csv"), index=False)
        logger.info("  Created: metadata.csv")

        # Helper function to write diff CSV
        def write_diff_csv(diffs: list[ComponentDiff], filename: str):
            if changes_only:
                diffs = [d for d in diffs if d.change_type != ChangeType.UNCHANGED]

            rows = [
                {
                    "status": diff.change_type.value,
                    "id": diff.id,
                    "name": diff.name,
                    "details": _get_change_detail(diff),
                }
                for diff in diffs
            ]

            pd.DataFrame(rows).to_csv(os.path.join(csv_dir, filename), index=False)
            logger.info(f"  Created: {filename}")

        write_diff_csv(diff_result.metric_diffs, "metrics_diff.csv")
        write_diff_csv(diff_result.dimension_diffs, "dimensions_diff.csv")

        # Helper function to write inventory diff CSV
        def write_inventory_diff_csv(diffs: list[InventoryItemDiff] | None, filename: str):
            if diffs is None:
                return

            if changes_only:
                diffs = [d for d in diffs if d.change_type != ChangeType.UNCHANGED]

            rows = [
                {
                    "status": diff.change_type.value,
                    "id": diff.id,
                    "name": diff.name,
                    "details": _get_inventory_change_detail(diff),
                }
                for diff in diffs
            ]

            pd.DataFrame(rows).to_csv(os.path.join(csv_dir, filename), index=False)
            logger.info(f"  Created: {filename}")

        # Write inventory diff CSVs if present
        if diff_result.calc_metrics_diffs is not None:
            write_inventory_diff_csv(diff_result.calc_metrics_diffs, "calc_metrics_diff.csv")
        if diff_result.segments_diffs is not None:
            write_inventory_diff_csv(diff_result.segments_diffs, "segments_diff.csv")

        logger.info(f"Diff CSV files created in: {csv_dir}")
        return csv_dir

    except Exception as e:
        logger.error(_format_error_msg("creating diff CSV files", error=e))
        raise


def write_diff_output(
    diff_result: DiffResult,
    output_format: str,
    base_filename: str,
    output_dir: str | Path,
    logger: logging.Logger,
    changes_only: bool = False,
    summary_only: bool = False,
    side_by_side: bool = False,
    use_color: bool = True,
    group_by_field: bool = False,
    group_by_field_limit: int = 10,
) -> str | None:
    """
    Write diff comparison output in specified format(s).

    Args:
        diff_result: The DiffResult to output
        output_format: Output format ('console', 'json', 'markdown', 'html', 'excel', 'csv', 'all', 'pr-comment')
        base_filename: Base filename without extension
        output_dir: Output directory path
        logger: Logger instance
        changes_only: Only include changed items
        summary_only: Only show summary (console only)
        side_by_side: Show side-by-side comparison for modified items
        use_color: Use ANSI color codes in console output
        group_by_field: Group changes by field name instead of component
        group_by_field_limit: Max items per section in group-by-field output (0 = unlimited)

    Returns:
        Console output string (for console/pr-comment format) or None
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    console_output = None

    # Handle group-by-field output mode
    if group_by_field and should_generate_format(output_format, "console"):
        console_output = write_diff_grouped_by_field_output(diff_result, use_color, group_by_field_limit)
        print(console_output)
        if output_format == "console":
            return console_output

    # Handle PR comment format
    if output_format == "pr-comment":
        console_output = write_diff_pr_comment_output(diff_result, changes_only)
        print(console_output)
        return console_output

    if should_generate_format(output_format, "console") and not group_by_field:
        console_output = write_diff_console_output(diff_result, changes_only, summary_only, side_by_side, use_color)
        print(console_output)

    if output_format == "console":
        return console_output

    if should_generate_format(output_format, "json"):
        output_files.append(write_diff_json_output(diff_result, base_filename, output_dir, logger, changes_only))

    if should_generate_format(output_format, "markdown"):
        output_files.append(
            write_diff_markdown_output(diff_result, base_filename, output_dir, logger, changes_only, side_by_side)
        )

    if should_generate_format(output_format, "html"):
        output_files.append(write_diff_html_output(diff_result, base_filename, output_dir, logger, changes_only))

    if should_generate_format(output_format, "excel"):
        output_files.append(write_diff_excel_output(diff_result, base_filename, output_dir, logger, changes_only))

    if should_generate_format(output_format, "csv"):
        output_files.append(write_diff_csv_output(diff_result, base_filename, output_dir, logger, changes_only))

    return console_output


# ==================== INVENTORY SUMMARY MODE ====================


def display_inventory_summary(
    data_view_id: str,
    data_view_name: str,
    derived_inventory: Any | None = None,
    calculated_inventory: Any | None = None,
    segments_inventory: Any | None = None,
    output_format: str = "console",
    output_dir: str | Path = ".",
    quiet: bool = False,
    inventory_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Display inventory summary statistics without generating full inventory sheets.

    Args:
        data_view_id: The data view ID
        data_view_name: Human-readable data view name
        derived_inventory: DerivedFieldInventory object (optional)
        calculated_inventory: CalculatedMetricsInventory object (optional)
        segments_inventory: SegmentsInventory object (optional)
        output_format: Output format - "console" or "json"
        output_dir: Directory for JSON output
        quiet: Suppress console output
        inventory_order: Order of inventory types as specified in CLI (default: ['segments', 'calculated', 'derived'])

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "data_view_id": data_view_id,
        "data_view_name": data_view_name,
        "timestamp": datetime.now().isoformat(),
        "inventories": {},
    }

    high_complexity_items = []

    # Process derived fields inventory
    if derived_inventory:
        derived_summary = derived_inventory.get_summary()
        summary["inventories"]["derived_fields"] = derived_summary

        # Collect high-complexity items (>=70)
        high_complexity_items.extend(
            {
                "type": "Derived Field",
                "name": field.component_name,
                "complexity": field.complexity_score,
                "summary": field.logic_summary[:60] + "..." if len(field.logic_summary) > 60 else field.logic_summary,
            }
            for field in derived_inventory.fields
            if field.complexity_score >= 70
        )

    # Process calculated metrics inventory
    if calculated_inventory:
        calc_summary = calculated_inventory.get_summary()
        summary["inventories"]["calculated_metrics"] = calc_summary

        # Collect high-complexity items (>=70)
        high_complexity_items.extend(
            {
                "type": "Calculated Metric",
                "name": metric.metric_name,
                "complexity": metric.complexity_score,
                "summary": metric.formula_summary[:60] + "..."
                if len(metric.formula_summary) > 60
                else metric.formula_summary,
            }
            for metric in calculated_inventory.metrics
            if metric.complexity_score >= 70
        )

    # Process segments inventory
    if segments_inventory:
        seg_summary = segments_inventory.get_summary()
        summary["inventories"]["segments"] = seg_summary

        # Collect high-complexity items (>=70)
        high_complexity_items.extend(
            {
                "type": "Segment",
                "name": segment.segment_name,
                "complexity": segment.complexity_score,
                "summary": segment.definition_summary[:60] + "..."
                if len(segment.definition_summary) > 60
                else segment.definition_summary,
            }
            for segment in segments_inventory.segments
            if segment.complexity_score >= 70
        )

    # Sort high-complexity items by score descending
    high_complexity_items.sort(key=lambda x: x["complexity"], reverse=True)
    summary["high_complexity_items"] = high_complexity_items

    # Console output
    if not quiet and output_format in ("console", "all"):
        print()
        print(ConsoleColors.bold(f"Inventory Summary: {data_view_name}"))
        print(ConsoleColors.dim(f"Data View ID: {data_view_id}"))
        print()

        # Determine display order (default: segments, calculated, derived)
        display_order = inventory_order if inventory_order else ["segments", "calculated", "derived"]

        # Helper functions for displaying each inventory type
        def display_segments():
            if "segments" not in summary["inventories"]:
                return
            ss = summary["inventories"]["segments"]
            print(ConsoleColors.cyan("Segments"))
            print(f"  Total:       {ss['total_segments']}")
            gov = ss.get("governance", {})
            if gov:
                print(f"  Approved:    {gov.get('approved_count', 0)}")
                print(f"  Shared:      {gov.get('shared_count', 0)}")
                print(f"  Tagged:      {gov.get('tagged_count', 0)}")
            containers = ss.get("container_types", {})
            if containers:
                container_str = ", ".join(f"{k}: {v}" for k, v in containers.items())
                print(f"  Containers:  {container_str}")
            print(f"  Complexity:  avg={ss['complexity']['average']:.1f}, max={ss['complexity']['max']:.1f}")
            if ss["complexity"]["high_complexity_count"] > 0:
                print(ConsoleColors.warning(f"  High (>=75): {ss['complexity']['high_complexity_count']}"))
            if ss["complexity"]["elevated_complexity_count"] > 0:
                print(ConsoleColors.dim(f"  Elevated (50-74): {ss['complexity']['elevated_complexity_count']}"))
            print()

        def display_calculated():
            if "calculated_metrics" not in summary["inventories"]:
                return
            cs = summary["inventories"]["calculated_metrics"]
            print(ConsoleColors.cyan("Calculated Metrics"))
            print(f"  Total:       {cs['total_calculated_metrics']}")
            gov = cs.get("governance", {})
            if gov:
                print(f"  Approved:    {gov.get('approved_count', 0)}")
                print(f"  Shared:      {gov.get('shared_count', 0)}")
                print(f"  Tagged:      {gov.get('tagged_count', 0)}")
            print(f"  Complexity:  avg={cs['complexity']['average']:.1f}, max={cs['complexity']['max']:.1f}")
            if cs["complexity"]["high_complexity_count"] > 0:
                print(ConsoleColors.warning(f"  High (>=75): {cs['complexity']['high_complexity_count']}"))
            if cs["complexity"]["elevated_complexity_count"] > 0:
                print(ConsoleColors.dim(f"  Elevated (50-74): {cs['complexity']['elevated_complexity_count']}"))
            print()

        def display_derived():
            if "derived_fields" not in summary["inventories"]:
                return
            ds = summary["inventories"]["derived_fields"]
            print(ConsoleColors.cyan("Derived Fields"))
            print(f"  Total:       {ds['total_derived_fields']}")
            print(f"  Metrics:     {ds['metrics_count']}")
            print(f"  Dimensions:  {ds['dimensions_count']}")
            print(f"  Complexity:  avg={ds['complexity']['average']:.1f}, max={ds['complexity']['max']:.1f}")
            if ds["complexity"]["high_complexity_count"] > 0:
                print(ConsoleColors.warning(f"  High (>=75): {ds['complexity']['high_complexity_count']}"))
            if ds["complexity"]["elevated_complexity_count"] > 0:
                print(ConsoleColors.dim(f"  Elevated (50-74): {ds['complexity']['elevated_complexity_count']}"))
            print()

        # Display in specified order
        display_funcs = {
            "segments": display_segments,
            "calculated": display_calculated,
            "derived": display_derived,
        }
        for inv_type in display_order:
            if inv_type in display_funcs:
                display_funcs[inv_type]()

        # High complexity warnings
        if high_complexity_items:
            print(ConsoleColors.warning(f"High-Complexity Items ({len(high_complexity_items)}):"))
            for item in high_complexity_items[:10]:  # Show top 10
                complexity = item["complexity"]
                print(f"  {ConsoleColors.bold(f'{complexity:3}')} {item['type']:18} {item['name']}")
                if item["summary"]:
                    print(f"       {ConsoleColors.dim(item['summary'])}")
            if len(high_complexity_items) > 10:
                print(ConsoleColors.dim(f"  ... and {len(high_complexity_items) - 10} more"))
            print()

    # JSON output
    if output_format in ("json", "all"):
        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r"[^\w\-]", "_", data_view_name)[:50]
        json_path = Path(output_dir) / f"{safe_name}_inventory_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        if not quiet:
            print(f"Summary saved to: {json_path}")

    return summary


def process_inventory_summary(
    data_view_id: str,
    config_file: str = "config.json",
    output_dir: str | Path = ".",
    log_level: str = "INFO",
    output_format: str = "console",
    quiet: bool = False,
    profile: str | None = None,
    include_derived: bool = False,
    include_calculated: bool = False,
    include_segments: bool = False,
    inventory_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Process inventory summary mode - fetch inventory data and display statistics.

    Args:
        data_view_id: The data view ID to process
        config_file: Path to CJA config file
        output_dir: Directory for JSON output
        log_level: Logging level
        output_format: Output format - "console" or "json"
        quiet: Suppress console output
        profile: Config profile name
        include_derived: Include derived fields inventory
        include_calculated: Include calculated metrics inventory
        include_segments: Include segments inventory
        inventory_order: Order of inventory types as specified in CLI

    Returns:
        Dictionary with summary statistics
    """
    logger = setup_logging(data_view_id, batch_mode=False, log_level=log_level)

    # Initialize CJA
    cja = initialize_cja(config_file, logger, profile=profile)
    if cja is None:
        print(ConsoleColors.error("ERROR: Failed to initialize CJA connection"), file=sys.stderr)
        return {"error": "CJA initialization failed"}

    # Get data view info
    try:
        lookup_data = cja.dataviews.get_single(data_view_id)
        dv_name = lookup_data.get("name", data_view_id) if isinstance(lookup_data, dict) else data_view_id
    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to fetch data view: {e}"), file=sys.stderr)
        return {"error": str(e)}

    if not quiet:
        print(ConsoleColors.info(f"Fetching inventory data for: {dv_name}"))

    derived_inventory = None
    calculated_inventory = None
    segments_inventory = None

    # Fetch derived fields inventory
    if include_derived:
        try:
            from cja_auto_sdr.inventory.derived_fields import DerivedFieldInventoryBuilder

            # Need metrics and dimensions for derived fields
            metrics_data = cja.dataviews.get_metrics(data_view_id)
            dimensions_data = cja.dataviews.get_dimensions(data_view_id)

            metrics_df = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame()
            dimensions_df = pd.DataFrame(dimensions_data) if dimensions_data else pd.DataFrame()

            builder = DerivedFieldInventoryBuilder(logger=logger)
            derived_inventory = builder.build(metrics_df, dimensions_df, data_view_id, dv_name)
            if not quiet:
                print(ConsoleColors.dim(f"  Derived fields: {derived_inventory.total_derived_fields}"))
        except Exception as e:
            logger.warning(f"Failed to build derived fields inventory: {e}")

    # Fetch calculated metrics inventory
    if include_calculated:
        try:
            from cja_calculated_metrics_inventory import CalculatedMetricsInventoryBuilder

            builder = CalculatedMetricsInventoryBuilder(logger=logger)
            calculated_inventory = builder.build(cja, data_view_id, dv_name)
            if not quiet:
                print(ConsoleColors.dim(f"  Calculated metrics: {calculated_inventory.total_calculated_metrics}"))
        except Exception as e:
            logger.warning(f"Failed to build calculated metrics inventory: {e}")

    # Fetch segments inventory
    if include_segments:
        try:
            from cja_segments_inventory import SegmentsInventoryBuilder

            builder = SegmentsInventoryBuilder(logger=logger)
            segments_inventory = builder.build(cja, data_view_id, dv_name)
            if not quiet:
                print(ConsoleColors.dim(f"  Segments: {segments_inventory.total_segments}"))
        except Exception as e:
            logger.warning(f"Failed to build segments inventory: {e}")

    # Display summary
    return display_inventory_summary(
        data_view_id=data_view_id,
        data_view_name=dv_name,
        derived_inventory=derived_inventory,
        calculated_inventory=calculated_inventory,
        segments_inventory=segments_inventory,
        output_format=output_format,
        output_dir=output_dir,
        quiet=quiet,
        inventory_order=inventory_order,
    )


# ==================== REFACTORED SINGLE DATAVIEW PROCESSING ====================


def process_single_dataview(
    data_view_id: str,
    config_file: str = "config.json",
    output_dir: str | Path = ".",
    log_level: str = "INFO",
    log_format: str = "text",
    output_format: str = "excel",
    enable_cache: bool = False,
    cache_size: int = 1000,
    cache_ttl: int = 3600,
    quiet: bool = False,
    skip_validation: bool = False,
    max_issues: int = 0,
    clear_cache: bool = False,
    show_timings: bool = False,
    metrics_only: bool = False,
    dimensions_only: bool = False,
    profile: str | None = None,
    shared_cache: SharedValidationCache | None = None,
    api_tuning_config: APITuningConfig | None = None,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    include_derived_inventory: bool = False,
    include_calculated_metrics: bool = False,
    include_segments_inventory: bool = False,
    inventory_only: bool = False,
    inventory_order: list[str] | None = None,
    quality_report_only: bool = False,
) -> ProcessingResult:
    """
    Process a single data view and generate SDR in specified format(s)

    Args:
        data_view_id: The data view ID to process (must start with 'dv_')
        config_file: Path to CJA config file (default: 'config.json')
        output_dir: Directory to save output files (default: current directory)
        log_level: Logging level - one of DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
        log_format: Log output format - "text" (default) or "json" for structured logging
        output_format: Output format - one of excel, csv, json, html, markdown, all (default: excel)
        enable_cache: Enable validation result caching (default: False)
        cache_size: Maximum cached validation results, >= 1 (default: 1000)
        cache_ttl: Cache time-to-live in seconds, >= 1 (default: 3600)
        quiet: Suppress non-error output (default: False)
        skip_validation: Skip data quality validation for faster processing (default: False)
        max_issues: Limit data quality issues to top N by severity, >= 0; 0 = all (default: 0)
        clear_cache: Clear validation cache before processing (default: False)
        show_timings: Display performance timing breakdown after processing (default: False)
        include_derived_inventory: Include derived field inventory in output (default: False)
        include_calculated_metrics: Include calculated metrics inventory in output (default: False)
        include_segments_inventory: Include segments inventory in output (default: False)
        inventory_only: Output only inventory sheets, skip standard SDR content (default: False)
        inventory_order: Order of inventory sheets as they appear in CLI (default: ['derived', 'calculated', 'segments'])
        quality_report_only: Validate and return quality issues without generating SDR files (default: False)

    Returns:
        ProcessingResult with processing details including success status, metrics/dimensions count,
        output file path, and any error messages
    """
    start_time = time.time()

    # Setup logging for this data view
    logger = setup_logging(data_view_id, batch_mode=False, log_level=log_level, log_format=log_format)
    perf_tracker = PerformanceTracker(logger)

    try:
        # Initialize CJA
        cja = initialize_cja(config_file, logger, profile=profile)
        if cja is None:
            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name="Unknown",
                success=False,
                duration=time.time() - start_time,
                error_message="CJA initialization failed",
            )

        logger.info("✓ CJA connection established successfully")

        # Validate data view
        if not validate_data_view(cja, data_view_id, logger):
            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name="Unknown",
                success=False,
                duration=time.time() - start_time,
                error_message="Data view validation failed",
            )

        logger.info("✓ Data view validation complete - proceeding with data fetch")

        # Fetch data with parallel optimization
        logger.info("=" * BANNER_WIDTH)
        logger.info("Starting optimized data fetch operations")
        logger.info("=" * BANNER_WIDTH)

        # Create circuit breaker if config provided
        circuit_breaker = None
        if circuit_breaker_config is not None:
            circuit_breaker = CircuitBreaker(config=circuit_breaker_config, logger=logger)
            logger.info(f"Circuit breaker enabled (failure_threshold={circuit_breaker_config.failure_threshold})")

        fetcher = ParallelAPIFetcher(
            cja,
            logger,
            perf_tracker,
            max_workers=DEFAULT_API_FETCH_WORKERS,
            quiet=quiet,
            tuning_config=api_tuning_config,
            circuit_breaker=circuit_breaker,
        )
        if api_tuning_config is not None:
            logger.info(
                f"API auto-tuning enabled (min={api_tuning_config.min_workers}, max={api_tuning_config.max_workers})"
            )

        metrics, dimensions, lookup_data = fetcher.fetch_all_data(data_view_id)

        # Log tuner statistics if tuning was enabled
        tuner_stats = fetcher.get_tuner_statistics()
        if tuner_stats is not None and isinstance(tuner_stats, dict):
            logger.info(
                f"API tuner stats: {tuner_stats['scale_ups']} scale-ups, "
                f"{tuner_stats['scale_downs']} scale-downs, "
                f"avg response: {tuner_stats['average_response_ms']:.0f}ms"
            )

        # Check if we have any data to process
        if metrics.empty and dimensions.empty:
            dv_name = lookup_data.get("name", "Unknown") if isinstance(lookup_data, dict) else "Unknown"
            logger.critical("No metrics or dimensions fetched. Cannot generate SDR.")
            logger.critical("Possible causes:")
            logger.critical("  1. Data view has no metrics or dimensions configured")
            logger.critical("  2. Your API credentials lack permission to read components")
            logger.critical("  3. The data view is newly created and not yet populated")
            logger.critical("  4. API rate limiting or temporary service issue")
            logger.critical("")
            logger.critical("Troubleshooting steps:")
            logger.critical("  - Verify the data view has components in the CJA UI")
            logger.critical("  - Check your OAuth scopes include component read permissions")
            logger.critical("  - Try running with --list-dataviews to verify access")
            logger.info("=" * BANNER_WIDTH)
            logger.info("EXECUTION FAILED")
            logger.info("=" * BANNER_WIDTH)
            logger.info(f"Data View: {dv_name} ({data_view_id})")
            logger.info("Error: No metrics or dimensions found")
            logger.info(f"Duration: {time.time() - start_time:.2f}s")
            logger.info("=" * BANNER_WIDTH)
            # Flush handlers to ensure log is written
            for handler in logger.handlers:
                handler.flush()
            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name=dv_name,
                success=False,
                duration=time.time() - start_time,
                error_message="No metrics or dimensions found - data view may be empty or inaccessible",
            )

        logger.info("Data fetch operations completed successfully")

        # Data quality validation (skip if --skip-validation flag is set)
        dq_issues: list[dict[str, Any]] = []
        severity_counts: dict[str, int] = {}
        dq_checker: DataQualityChecker | None = None
        if skip_validation:
            logger.info("=" * BANNER_WIDTH)
            logger.info("Skipping data quality validation (--skip-validation)")
            logger.info("=" * BANNER_WIDTH)
            data_quality_df = pd.DataFrame(columns=["Severity", "Category", "Type", "Item Name", "Issue", "Details"])
        else:
            logger.info("=" * BANNER_WIDTH)
            logger.info("Starting data quality validation (optimized)")
            logger.info("=" * BANNER_WIDTH)

            # Start performance tracking for data quality validation
            perf_tracker.start("Data Quality Validation")

            # Create validation cache if enabled
            # Use shared cache if provided (from batch processing), otherwise create local cache
            validation_cache = None
            if shared_cache is not None:
                validation_cache = shared_cache
                logger.info("Using shared validation cache from batch processor")
            elif enable_cache:
                validation_cache = ValidationCache(max_size=cache_size, ttl_seconds=cache_ttl, logger=logger)
                if clear_cache:
                    validation_cache.clear()
                    logger.info(f"Validation cache cleared and enabled (max_size={cache_size}, ttl={cache_ttl}s)")
                else:
                    logger.info(f"Validation cache enabled (max_size={cache_size}, ttl={cache_ttl}s)")

            dq_checker = DataQualityChecker(logger, validation_cache=validation_cache, quiet=quiet)

            # Run parallel data quality checks (10-15% faster than sequential)
            logger.info("Running parallel data quality checks...")

            try:
                # Parallel validation for metrics and dimensions (10-15% faster)
                dq_checker.check_all_parallel(
                    metrics_df=metrics,
                    dimensions_df=dimensions,
                    metrics_required_fields=VALIDATION_SCHEMA["required_metric_fields"],
                    dimensions_required_fields=VALIDATION_SCHEMA["required_dimension_fields"],
                    critical_fields=VALIDATION_SCHEMA["critical_fields"],
                    max_workers=DEFAULT_VALIDATION_WORKERS,
                )

                # Log aggregated summary instead of individual issue count
                dq_checker.log_summary()

                # Log cache statistics if cache was used
                if validation_cache is not None:
                    perf_tracker.add_cache_statistics(validation_cache)

                # End performance tracking
                perf_tracker.end("Data Quality Validation")

            except Exception as e:
                logger.error(_format_error_msg("during data quality validation", error=e))
                logger.info("Continuing with SDR generation despite validation errors")
                perf_tracker.end("Data Quality Validation")

            # Get data quality issues dataframe (limited if max_issues > 0)
            data_quality_df = dq_checker.get_issues_dataframe(max_issues=max_issues)
            dq_issues = list(dq_checker.issues)
            severity_counts = count_quality_issues_by_severity(dq_issues)

        if quality_report_only:
            dv_name = lookup_data.get("name", "Unknown") if isinstance(lookup_data, dict) else "Unknown"
            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name=dv_name,
                success=True,
                duration=time.time() - start_time,
                metrics_count=len(metrics),
                dimensions_count=len(dimensions),
                dq_issues_count=len(dq_issues),
                dq_issues=dq_issues,
                dq_severity_counts=severity_counts,
            )

        # Derived field inventory (if enabled)
        derived_inventory_df = pd.DataFrame()
        derived_inventory_obj = None  # Store inventory object for JSON output
        if include_derived_inventory:
            logger.info("=" * BANNER_WIDTH)
            logger.info("Building derived field inventory")
            logger.info("=" * BANNER_WIDTH)

            try:
                from cja_auto_sdr.inventory.derived_fields import DerivedFieldInventoryBuilder

                builder = DerivedFieldInventoryBuilder(logger=logger)
                dv_name = lookup_data.get("name", data_view_id) if isinstance(lookup_data, dict) else data_view_id
                derived_inventory_obj = builder.build(metrics, dimensions, data_view_id, dv_name)

                derived_inventory_df = derived_inventory_obj.get_dataframe()

                inv_summary = derived_inventory_obj.get_summary()
                logger.info(
                    f"Derived field inventory: {inv_summary.get('total_derived_fields', 0)} fields "
                    f"({inv_summary.get('metrics_count', 0)} metrics, {inv_summary.get('dimensions_count', 0)} dimensions)"
                )

            except ImportError as e:
                logger.warning(f"Could not import derived field inventory: {e}")
                logger.info("Skipping derived field inventory - module not available")
            except Exception as e:
                logger.error(_format_error_msg("during derived field inventory", error=e))
                logger.info("Continuing with SDR generation despite derived field inventory errors")

        # Calculated metrics inventory (if enabled)
        calculated_metrics_df = pd.DataFrame()
        calculated_inventory_obj = None  # Store inventory object for JSON output
        if include_calculated_metrics:
            logger.info("=" * BANNER_WIDTH)
            logger.info("Building calculated metrics inventory")
            logger.info("=" * BANNER_WIDTH)

            try:
                from cja_auto_sdr.inventory.calculated_metrics import CalculatedMetricsInventoryBuilder

                builder = CalculatedMetricsInventoryBuilder(logger=logger)
                dv_name = lookup_data.get("name", data_view_id) if isinstance(lookup_data, dict) else data_view_id
                calculated_inventory_obj = builder.build(cja, data_view_id, dv_name)

                calculated_metrics_df = calculated_inventory_obj.get_dataframe()

                calc_summary = calculated_inventory_obj.get_summary()
                logger.info(f"Calculated metrics inventory: {calc_summary.get('total_calculated_metrics', 0)} metrics")

            except ImportError as e:
                logger.warning(f"Could not import calculated metrics inventory: {e}")
                logger.info("Skipping calculated metrics inventory - module not available")
            except Exception as e:
                logger.error(_format_error_msg("during calculated metrics inventory", error=e))
                logger.info("Continuing with SDR generation despite calculated metrics inventory errors")

        # Segments inventory (if enabled)
        segments_inventory_df = pd.DataFrame()
        segments_inventory_obj = None  # Store inventory object for JSON output
        if include_segments_inventory:
            logger.info("=" * BANNER_WIDTH)
            logger.info("Building segments inventory")
            logger.info("=" * BANNER_WIDTH)

            try:
                from cja_auto_sdr.inventory.segments import SegmentsInventoryBuilder

                builder = SegmentsInventoryBuilder(logger=logger)
                dv_name = lookup_data.get("name", data_view_id) if isinstance(lookup_data, dict) else data_view_id
                segments_inventory_obj = builder.build(cja, data_view_id, dv_name)

                segments_inventory_df = segments_inventory_obj.get_dataframe()

                seg_summary = segments_inventory_obj.get_summary()
                logger.info(f"Segments inventory: {seg_summary.get('total_segments', 0)} segments")

            except ImportError as e:
                logger.warning(f"Could not import segments inventory: {e}")
                logger.info("Skipping segments inventory - module not available")
            except Exception as e:
                logger.error(_format_error_msg("during segments inventory", error=e))
                logger.info("Continuing with SDR generation despite segments inventory errors")

        # Data processing
        logger.info("=" * BANNER_WIDTH)
        logger.info("Processing data for Excel export")
        logger.info("=" * BANNER_WIDTH)

        try:
            # Process lookup data into DataFrame
            logger.info("Processing data view lookup information...")
            lookup_data_copy = {k: [v] if not isinstance(v, (list, tuple)) else v for k, v in lookup_data.items()}
            max_length = max(len(v) for v in lookup_data_copy.values()) if lookup_data_copy else 1
            lookup_data_copy = {k: v + [None] * (max_length - len(v)) for k, v in lookup_data_copy.items()}
            lookup_df = pd.DataFrame(lookup_data_copy)
            logger.info(f"Processed lookup data with {len(lookup_df)} rows")
        except Exception as e:
            logger.error(_format_error_msg("processing lookup data", error=e))
            lookup_df = pd.DataFrame({"Error": ["Failed to process data view information"]})

        try:
            # Enhanced metadata creation
            logger.info("Creating metadata summary...")
            metric_types = (
                metrics["type"].value_counts().to_dict() if not metrics.empty and "type" in metrics.columns else {}
            )
            metric_summary = [f"{type_}: {count}" for type_, count in metric_types.items()]

            dimension_types = (
                dimensions["type"].value_counts().to_dict()
                if not dimensions.empty and "type" in dimensions.columns
                else {}
            )
            dimension_summary = [f"{type_}: {count}" for type_, count in dimension_types.items()]

            # Get current timezone and formatted timestamp
            local_tz = datetime.now().astimezone().tzinfo
            current_time = datetime.now(local_tz)
            formatted_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

            # Count data quality issues by severity
            dq_summary = [f"{sev}: {count}" for sev, count in severity_counts.items()]

            # Build base metadata properties
            metadata_properties = [
                "Generated Date & timestamp and timezone",
                "Data View ID",
                "Data View Name",
                "Total Metrics",
                "Metrics Breakdown",
                "Total Dimensions",
                "Dimensions Breakdown",
                "Data Quality Issues",
                "Data Quality Summary",
            ]
            metadata_values = [
                formatted_timestamp,
                data_view_id,
                lookup_data.get("name", "Unknown") if isinstance(lookup_data, dict) else "Unknown",
                len(metrics),
                "\n".join(metric_summary) if metric_summary else "No metrics found",
                len(dimensions),
                "\n".join(dimension_summary) if dimension_summary else "No dimensions found",
                len(dq_issues),
                "\n".join(dq_summary) if dq_summary else "No issues",
            ]

            # Add inventory statistics if any inventory was generated
            if segments_inventory_obj or calculated_inventory_obj or derived_inventory_obj:
                metadata_properties.append("--- Inventory Statistics ---")
                metadata_values.append("")

            if segments_inventory_obj:
                seg_summary = segments_inventory_obj.get_summary()
                seg_count = seg_summary.get("total_segments", 0)
                seg_complexity = seg_summary.get("complexity", {})
                seg_high = seg_complexity.get("high_complexity_count", 0)
                seg_elevated = seg_complexity.get("elevated_complexity_count", 0)
                seg_avg = seg_complexity.get("average", 0)
                seg_max = seg_complexity.get("max", 0)

                metadata_properties.extend(
                    [
                        "Segments Count",
                        "Segments Complexity (Avg / Max)",
                        "Segments High Complexity (≥75)",
                        "Segments Elevated Complexity (50-74)",
                    ]
                )
                metadata_values.extend([seg_count, f"{seg_avg:.1f} / {seg_max:.1f}", seg_high, seg_elevated])

            if calculated_inventory_obj:
                calc_summary = calculated_inventory_obj.get_summary()
                calc_count = calc_summary.get("total_calculated_metrics", 0)
                calc_complexity = calc_summary.get("complexity", {})
                calc_high = calc_complexity.get("high_complexity_count", 0)
                calc_elevated = calc_complexity.get("elevated_complexity_count", 0)
                calc_avg = calc_complexity.get("average", 0)
                calc_max = calc_complexity.get("max", 0)

                metadata_properties.extend(
                    [
                        "Calculated Metrics Count",
                        "Calculated Metrics Complexity (Avg / Max)",
                        "Calculated Metrics High Complexity (≥75)",
                        "Calculated Metrics Elevated Complexity (50-74)",
                    ]
                )
                metadata_values.extend([calc_count, f"{calc_avg:.1f} / {calc_max:.1f}", calc_high, calc_elevated])

            if derived_inventory_obj:
                derived_summary = derived_inventory_obj.get_summary()
                derived_count = derived_summary.get("total_derived_fields", 0)
                derived_metrics = derived_summary.get("metrics_count", 0)
                derived_dimensions = derived_summary.get("dimensions_count", 0)
                derived_complexity = derived_summary.get("complexity", {})
                derived_high = derived_complexity.get("high_complexity_count", 0)
                derived_elevated = derived_complexity.get("elevated_complexity_count", 0)
                derived_avg = derived_complexity.get("average", 0)
                derived_max = derived_complexity.get("max", 0)

                metadata_properties.extend(
                    [
                        "Derived Fields Count",
                        "Derived Fields Breakdown",
                        "Derived Fields Complexity (Avg / Max)",
                        "Derived Fields High Complexity (≥75)",
                        "Derived Fields Elevated Complexity (50-74)",
                    ]
                )
                metadata_values.extend(
                    [
                        derived_count,
                        f"Metrics: {derived_metrics}, Dimensions: {derived_dimensions}",
                        f"{derived_avg:.1f} / {derived_max:.1f}",
                        derived_high,
                        derived_elevated,
                    ]
                )

            # Create enhanced metadata DataFrame
            metadata_df = pd.DataFrame({"Property": metadata_properties, "Value": metadata_values})
            logger.info("Metadata created successfully")
        except Exception as e:
            logger.error(_format_error_msg("creating metadata", error=e))
            metadata_df = pd.DataFrame({"Error": ["Failed to create metadata"]})

        # Function to format JSON cells
        def format_json_cell(value):
            """Format JSON objects for Excel display"""
            try:
                if isinstance(value, (dict, list)):
                    return json.dumps(value, indent=2)
                return value
            except Exception as e:
                logger.warning(f"Error formatting JSON cell: {e!s}")
                return str(value)

        try:
            # Apply JSON formatting to all dataframes
            logger.info("Applying JSON formatting to dataframes...")

            for col in lookup_df.columns:
                lookup_df[col] = lookup_df[col].map(format_json_cell)

            if not metrics.empty:
                for col in metrics.columns:
                    metrics[col] = metrics[col].map(format_json_cell)

            if not dimensions.empty:
                for col in dimensions.columns:
                    dimensions[col] = dimensions[col].map(format_json_cell)

            logger.info("JSON formatting applied successfully")
        except Exception as e:
            logger.error(_format_error_msg("applying JSON formatting", error=e))

        # Create Excel file name
        try:
            dv_name = lookup_data.get("name", "Unknown") if isinstance(lookup_data, dict) else "Unknown"
            # Sanitize filename
            dv_name = "".join(c for c in dv_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
            excel_file_name = f"CJA_DataView_{dv_name}_{data_view_id}_SDR.xlsx"

            # Add output directory path
            output_path = Path(output_dir) / excel_file_name
            logger.info(f"Excel file will be saved as: {output_path}")
        except Exception as e:
            logger.error(_format_error_msg("creating filename", error=e))
            excel_file_name = f"CJA_DataView_{data_view_id}_SDR.xlsx"
            output_path = Path(output_dir) / excel_file_name

        # Prepare data for output generation
        logger.info("=" * BANNER_WIDTH)
        logger.info(f"Generating output in format: {output_format}")
        logger.info("=" * BANNER_WIDTH)

        # Prepare data dictionary for all formats
        # In inventory-only mode, skip standard SDR sheets
        if inventory_only:
            data_dict = {}
        else:
            data_dict = {
                "Metadata": metadata_df,
                "Data Quality": data_quality_df,
                "DataView Details": lookup_df,
                "Metrics": metrics,
                "Dimensions": dimensions,
            }

        # Add derived field inventory if available or placeholder if flag was used
        if not derived_inventory_df.empty:
            data_dict["Derived Fields"] = derived_inventory_df
        elif include_derived_inventory:
            data_dict["Derived Fields"] = pd.DataFrame(
                {
                    "Status": ["No derived fields found for this data view"],
                    "Data View ID": [data_view_id],
                    "Note": ["This data view has no derived fields configured"],
                }
            )

        # Add calculated metrics inventory if available or placeholder if flag was used
        if not calculated_metrics_df.empty:
            data_dict["Calculated Metrics"] = calculated_metrics_df
        elif include_calculated_metrics:
            data_dict["Calculated Metrics"] = pd.DataFrame(
                {
                    "Status": ["No calculated metrics found for this data view"],
                    "Data View ID": [data_view_id],
                    "Note": ["This data view has no associated calculated metrics"],
                }
            )

        # Add segments inventory if available or placeholder if flag was used
        if not segments_inventory_df.empty:
            data_dict["Segments"] = segments_inventory_df
        elif include_segments_inventory:
            data_dict["Segments"] = pd.DataFrame(
                {
                    "Status": ["No segments found for this data view"],
                    "Data View ID": [data_view_id],
                    "Note": ["This data view has no associated segments"],
                }
            )

        # Prepare metadata dictionary for JSON/HTML
        metadata_dict = (
            metadata_df.set_index(metadata_df.columns[0])[metadata_df.columns[1]].to_dict()
            if not metadata_df.empty
            else {}
        )

        # Base filename without extension
        base_filename = output_path.stem if isinstance(output_path, Path) else Path(output_path).stem

        # Determine which formats to generate
        if output_format == "all":
            formats_to_generate = ["excel", "csv", "json", "html", "markdown"]
        elif output_format in FORMAT_ALIASES:
            formats_to_generate = FORMAT_ALIASES[output_format]
        else:
            formats_to_generate = [output_format]

        output_files = []

        try:
            for fmt in formats_to_generate:
                if fmt == "excel":
                    logger.info("Generating Excel file...")
                    with pd.ExcelWriter(str(output_path), engine="xlsxwriter") as writer:
                        # Create format cache once for the entire workbook
                        # This improves performance by 15-25% by reusing format objects
                        format_cache = ExcelFormatCache(writer.book)

                        # Write sheets in order, with Data Quality first for visibility
                        sheets_to_write = []

                        # Skip standard sheets in inventory-only mode
                        if not inventory_only:
                            sheets_to_write.extend(
                                [
                                    (metadata_df, "Metadata"),
                                    (data_quality_df, "Data Quality"),
                                ]
                            )
                            sheets_to_write.append((lookup_df, "DataView"))
                            # Add component sheets based on filters
                            if not dimensions_only:
                                sheets_to_write.append((metrics, "Metrics"))
                            if not metrics_only:
                                sheets_to_write.append((dimensions, "Dimensions"))

                        # Add inventory sheets at the end, ordered by CLI argument order
                        inv_order = inventory_order if inventory_order else ["derived", "calculated", "segments"]
                        inventory_sheets = {
                            "derived": (derived_inventory_df, "Derived Fields", include_derived_inventory),
                            "calculated": (calculated_metrics_df, "Calculated Metrics", include_calculated_metrics),
                            "segments": (segments_inventory_df, "Segments", include_segments_inventory),
                        }
                        for inv_type in inv_order:
                            if inv_type in inventory_sheets:
                                df, name, flag_enabled = inventory_sheets[inv_type]
                                if not df.empty:
                                    sheets_to_write.append((df, name))
                                elif flag_enabled:
                                    # Add placeholder when flag was used but no data found
                                    placeholder_df = pd.DataFrame(
                                        {
                                            "Status": [f"No {name.lower()} found for this data view"],
                                            "Data View ID": [data_view_id],
                                            "Note": [
                                                "This data view has no associated " + name.lower().replace(" ", " ")
                                            ],
                                        }
                                    )
                                    sheets_to_write.append((placeholder_df, name))

                        for sheet_data, sheet_name in sheets_to_write:
                            try:
                                if sheet_data.empty:
                                    logger.warning(f"Sheet {sheet_name} is empty, creating placeholder")
                                    placeholder_df = pd.DataFrame({"Note": [f"No data available for {sheet_name}"]})
                                    apply_excel_formatting(writer, placeholder_df, sheet_name, logger, format_cache)
                                else:
                                    apply_excel_formatting(writer, sheet_data, sheet_name, logger, format_cache)
                            except Exception as e:
                                logger.error(f"Failed to write sheet {sheet_name}: {e!s}")
                                continue

                    logger.info(f"✓ Excel file created: {output_path}")
                    output_files.append(str(output_path))

                elif fmt == "csv":
                    csv_output = write_csv_output(data_dict, base_filename, output_dir, logger)
                    output_files.append(csv_output)

                elif fmt == "json":
                    inventory_objects = {
                        "derived": derived_inventory_obj,
                        "calculated": calculated_inventory_obj,
                        "segments": segments_inventory_obj,
                    }
                    json_output = write_json_output(
                        data_dict, metadata_dict, base_filename, output_dir, logger, inventory_objects
                    )
                    output_files.append(json_output)

                elif fmt == "html":
                    html_output = write_html_output(data_dict, metadata_dict, base_filename, output_dir, logger)
                    output_files.append(html_output)

                elif fmt == "markdown":
                    markdown_output = write_markdown_output(data_dict, metadata_dict, base_filename, output_dir, logger)
                    output_files.append(markdown_output)

            if len(output_files) > 1:
                logger.info(f"✓ SDR generation complete! {len(output_files)} files created")
                for file_path in output_files:
                    logger.info(f"  • {file_path}")
            else:
                logger.info(f"✓ SDR generation complete! File saved as: {output_files[0]}")

            # Final summary
            logger.info("=" * BANNER_WIDTH)
            logger.info("EXECUTION SUMMARY")
            logger.info("=" * BANNER_WIDTH)
            logger.info(f"Data View: {dv_name} ({data_view_id})")
            logger.info(f"Metrics: {len(metrics)}")
            logger.info(f"Dimensions: {len(dimensions)}")
            logger.info(f"Data Quality Issues: {len(dq_issues)}")

            if dq_issues:
                logger.info("Data Quality Issues by Severity:")
                for severity, count in severity_counts.items():
                    logger.info(f"  {severity}: {count}")

            logger.info(f"Output file: {output_path}")
            logger.info("=" * BANNER_WIDTH)

            logger.info("Script execution completed successfully")
            logger.info(perf_tracker.get_summary())

            # Display timing summary on stdout if requested
            if show_timings:
                print(perf_tracker.get_summary())

            duration = time.time() - start_time

            # Calculate total file size
            total_size = 0
            for file_path in output_files:
                try:
                    if os.path.isdir(file_path):
                        # For CSV directories, sum all files
                        for root, _dirs, files in os.walk(file_path):
                            for f in files:
                                total_size += os.path.getsize(os.path.join(root, f))
                    else:
                        total_size += os.path.getsize(file_path)
                except OSError:
                    pass

            # Collect inventory statistics for result
            segments_count = 0
            segments_high_complexity = 0
            calculated_metrics_count = 0
            calculated_metrics_high_complexity = 0
            derived_fields_count = 0
            derived_fields_high_complexity = 0

            if segments_inventory_obj:
                seg_summary = segments_inventory_obj.get_summary()
                segments_count = seg_summary.get("total_segments", 0)
                segments_high_complexity = seg_summary.get("complexity", {}).get("high_complexity_count", 0)

            if calculated_inventory_obj:
                calc_summary = calculated_inventory_obj.get_summary()
                calculated_metrics_count = calc_summary.get("total_calculated_metrics", 0)
                calculated_metrics_high_complexity = calc_summary.get("complexity", {}).get("high_complexity_count", 0)

            if derived_inventory_obj:
                derived_summary = derived_inventory_obj.get_summary()
                derived_fields_count = derived_summary.get("total_derived_fields", 0)
                derived_fields_high_complexity = derived_summary.get("complexity", {}).get("high_complexity_count", 0)

            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name=dv_name,
                success=True,
                duration=duration,
                metrics_count=len(metrics),
                dimensions_count=len(dimensions),
                dq_issues_count=len(dq_issues),
                dq_issues=dq_issues,
                dq_severity_counts=severity_counts,
                output_file=str(output_path),
                file_size_bytes=total_size,
                segments_count=segments_count,
                segments_high_complexity=segments_high_complexity,
                calculated_metrics_count=calculated_metrics_count,
                calculated_metrics_high_complexity=calculated_metrics_high_complexity,
                derived_fields_count=derived_fields_count,
                derived_fields_high_complexity=derived_fields_high_complexity,
            )

        except PermissionError as e:
            logger.critical(f"Permission denied writing to {output_path}. File may be open in another program.")
            logger.critical("Please close the file and try again.")
            logger.info("=" * BANNER_WIDTH)
            logger.info("EXECUTION FAILED")
            logger.info("=" * BANNER_WIDTH)
            logger.info(f"Data View: {dv_name} ({data_view_id})")
            logger.info("Error: Permission denied")
            logger.info(f"Duration: {time.time() - start_time:.2f}s")
            logger.info("=" * BANNER_WIDTH)
            for handler in logger.handlers:
                handler.flush()
            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name=dv_name,
                success=False,
                duration=time.time() - start_time,
                error_message=f"Permission denied: {e!s}",
            )
        except Exception as e:
            logger.critical(f"Failed to generate Excel file: {e!s}")
            logger.exception("Full exception details:")
            logger.info("=" * BANNER_WIDTH)
            logger.info("EXECUTION FAILED")
            logger.info("=" * BANNER_WIDTH)
            logger.info(f"Data View: {dv_name} ({data_view_id})")
            logger.info(f"Error: {e!s}")
            logger.info(f"Duration: {time.time() - start_time:.2f}s")
            logger.info("=" * BANNER_WIDTH)
            for handler in logger.handlers:
                handler.flush()
            return ProcessingResult(
                data_view_id=data_view_id,
                data_view_name=dv_name,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e),
            )

    except Exception as e:
        logger.critical(f"Unexpected error processing data view {data_view_id}: {e!s}")
        logger.exception("Full exception details:")
        logger.info("=" * BANNER_WIDTH)
        logger.info("EXECUTION FAILED")
        logger.info("=" * BANNER_WIDTH)
        logger.info(f"Data View ID: {data_view_id}")
        logger.info(f"Error: {e!s}")
        logger.info(f"Duration: {time.time() - start_time:.2f}s")
        logger.info("=" * BANNER_WIDTH)
        for handler in logger.handlers:
            handler.flush()
        return ProcessingResult(
            data_view_id=data_view_id,
            data_view_name="Unknown",
            success=False,
            duration=time.time() - start_time,
            error_message=str(e),
        )


# ==================== WORKER FUNCTION FOR MULTIPROCESSING ====================


def process_single_dataview_worker(args: WorkerArgs) -> ProcessingResult:
    """Worker function for multiprocessing.

    Args:
        args: A WorkerArgs dataclass with all processing parameters.

    Returns:
        ProcessingResult
    """
    # Retry config is propagated via env vars and resolved at call time
    # by _effective_retry_config() in resilience.py — no global mutation needed.

    return process_single_dataview(
        args.data_view_id,
        args.config_file,
        args.output_dir,
        args.log_level,
        args.log_format,
        args.output_format,
        args.enable_cache,
        args.cache_size,
        args.cache_ttl,
        args.quiet,
        args.skip_validation,
        args.max_issues,
        args.clear_cache,
        args.show_timings,
        args.metrics_only,
        args.dimensions_only,
        profile=args.profile,
        shared_cache=args.shared_cache,
        api_tuning_config=args.api_tuning_config,
        circuit_breaker_config=args.circuit_breaker_config,
        include_derived_inventory=args.include_derived_inventory,
        include_calculated_metrics=args.include_calculated_metrics,
        include_segments_inventory=args.include_segments_inventory,
        inventory_only=args.inventory_only,
        inventory_order=args.inventory_order,
        quality_report_only=args.quality_report_only,
    )


# ==================== BATCH PROCESSOR CLASS ====================


class BatchProcessor:
    """
    Process multiple data views in parallel using multiprocessing.

    Provides parallel execution of SDR generation across multiple data views
    with configurable worker count and error handling.

    Args:
        config_file: Path to CJA config file (default: 'config.json')
        output_dir: Directory for output files (default: current directory)
        workers: Number of parallel workers, 1-256 (default: 4)
        continue_on_error: Continue if individual data views fail (default: False)
        log_level: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
        log_format: Log output format - "text" (default) or "json" for structured logging
        output_format: Output format - excel, csv, json, html, markdown, all (default: excel)
        enable_cache: Enable validation result caching (default: False)
        cache_size: Maximum cached validation results, >= 1 (default: 1000)
        cache_ttl: Cache time-to-live in seconds, >= 1 (default: 3600)
        quiet: Suppress non-error output (default: False)
        skip_validation: Skip data quality validation (default: False)
        max_issues: Limit issues to top N by severity, >= 0; 0 = all (default: 0)
        clear_cache: Clear validation cache before processing (default: False)
        show_timings: Display performance timing for each data view (default: False)
        metrics_only: Only include metrics in output (default: False)
        dimensions_only: Only include dimensions in output (default: False)
        profile: Profile name for credentials (default: None)
        shared_cache: Share validation cache across batch workers (default: False)
        api_tuning_config: API worker auto-tuning configuration (default: None)
        circuit_breaker_config: Circuit breaker configuration (default: None)
    """

    def __init__(
        self,
        config_file: str = "config.json",
        output_dir: str = ".",
        workers: int = 4,
        continue_on_error: bool = False,
        log_level: str = "INFO",
        log_format: str = "text",
        output_format: str = "excel",
        enable_cache: bool = False,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        quiet: bool = False,
        skip_validation: bool = False,
        max_issues: int = 0,
        clear_cache: bool = False,
        show_timings: bool = False,
        metrics_only: bool = False,
        dimensions_only: bool = False,
        profile: str | None = None,
        shared_cache: bool = False,
        api_tuning_config: APITuningConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        include_derived_inventory: bool = False,
        include_calculated_metrics: bool = False,
        include_segments_inventory: bool = False,
        inventory_only: bool = False,
        inventory_order: list[str] | None = None,
        quality_report_only: bool = False,
    ):
        self.config_file = config_file
        self.output_dir = output_dir
        self.clear_cache = clear_cache
        self.show_timings = show_timings
        self.metrics_only = metrics_only
        self.dimensions_only = dimensions_only
        self.workers = workers
        self.continue_on_error = continue_on_error
        self.log_level = log_level
        self.log_format = log_format
        self.output_format = output_format
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.quiet = quiet
        self.skip_validation = skip_validation
        self.max_issues = max_issues
        self.profile = profile
        self.shared_cache_enabled = shared_cache
        self.api_tuning_config = api_tuning_config
        self.circuit_breaker_config = circuit_breaker_config
        self.include_derived_inventory = include_derived_inventory
        self.include_calculated_metrics = include_calculated_metrics
        self.include_segments_inventory = include_segments_inventory
        self.inventory_only = inventory_only
        self.inventory_order = inventory_order
        self.quality_report_only = quality_report_only
        self.batch_id = str(uuid.uuid4())[:8]  # Short correlation ID for log tracing
        self.logger = setup_logging(batch_mode=True, log_level=log_level, log_format=log_format)
        self.logger.info(f"Batch ID: {self.batch_id}")

        # Create shared validation cache if enabled
        self._shared_cache: SharedValidationCache | None = None
        if shared_cache and enable_cache and not skip_validation:
            self._shared_cache = SharedValidationCache(max_size=cache_size, ttl_seconds=cache_ttl)
            self.logger.info(f"[{self.batch_id}] Shared validation cache enabled (max_size={cache_size})")

        # Create output directory if it doesn't exist
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise OutputError(
                f"Permission denied creating output directory: {output_dir}. "
                "Check that you have write permissions for the parent directory."
            ) from e
        except OSError as e:
            raise OutputError(
                f"Cannot create output directory '{output_dir}': {e}. "
                "Verify the path is valid and the disk has available space."
            ) from e

    def process_batch(self, data_view_ids: list[str]) -> dict:
        """
        Process multiple data views in parallel

        Args:
            data_view_ids: List of data view IDs to process

        Returns:
            Dictionary with processing results
        """
        self.logger.info("=" * BANNER_WIDTH)
        self.logger.info(f"[{self.batch_id}] BATCH PROCESSING START")
        self.logger.info("=" * BANNER_WIDTH)
        self.logger.info(f"[{self.batch_id}] Data views to process: {len(data_view_ids)}")
        self.logger.info(f"[{self.batch_id}] Parallel workers: {self.workers}")
        self.logger.info(f"[{self.batch_id}] Continue on error: {self.continue_on_error}")
        self.logger.info(f"[{self.batch_id}] Output directory: {self.output_dir}")
        self.logger.info(f"[{self.batch_id}] Output format: {self.output_format}")
        self.logger.info("=" * BANNER_WIDTH)

        batch_start_time = time.time()

        results = {"successful": [], "failed": [], "total": len(data_view_ids), "total_duration": 0}

        # Prepare arguments for each worker
        worker_args = [
            WorkerArgs(
                data_view_id=dv_id,
                config_file=self.config_file,
                output_dir=self.output_dir,
                log_level=self.log_level,
                log_format=self.log_format,
                output_format=self.output_format,
                enable_cache=self.enable_cache,
                cache_size=self.cache_size,
                cache_ttl=self.cache_ttl,
                quiet=self.quiet,
                skip_validation=self.skip_validation,
                max_issues=self.max_issues,
                clear_cache=self.clear_cache,
                show_timings=self.show_timings,
                metrics_only=self.metrics_only,
                dimensions_only=self.dimensions_only,
                profile=self.profile,
                shared_cache=self._shared_cache,
                api_tuning_config=self.api_tuning_config,
                circuit_breaker_config=self.circuit_breaker_config,
                include_derived_inventory=self.include_derived_inventory,
                include_calculated_metrics=self.include_calculated_metrics,
                include_segments_inventory=self.include_segments_inventory,
                inventory_only=self.inventory_only,
                inventory_order=self.inventory_order,
                quality_report_only=self.quality_report_only,
            )
            for dv_id in data_view_ids
        ]

        # Process with ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_dv = {executor.submit(process_single_dataview_worker, wa): wa.data_view_id for wa in worker_args}

            # Collect results as they complete with progress bar
            with tqdm(
                total=len(data_view_ids),
                desc="Processing data views",
                unit="view",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                disable=self.quiet,
            ) as pbar:
                for future in as_completed(future_to_dv):
                    dv_id = future_to_dv[future]
                    try:
                        result = future.result()

                        if result.success:
                            results["successful"].append(result)
                            pbar.set_postfix_str(f"✓ {dv_id[:20]}", refresh=True)
                            self.logger.info(f"[{self.batch_id}] ✓ {dv_id}: SUCCESS ({result.duration:.1f}s)")
                        else:
                            results["failed"].append(result)
                            pbar.set_postfix_str(f"✗ {dv_id[:20]}", refresh=True)
                            self.logger.error(f"[{self.batch_id}] ✗ {dv_id}: FAILED - {result.error_message}")

                            if not self.continue_on_error:
                                self.logger.warning(
                                    f"[{self.batch_id}] Stopping batch processing due to error (use --continue-on-error to continue)"
                                )
                                # Cancel remaining tasks
                                for f in future_to_dv:
                                    f.cancel()
                                break

                    except KeyboardInterrupt, SystemExit:
                        # Allow graceful shutdown on Ctrl+C
                        self.logger.warning(f"[{self.batch_id}] Interrupted - cancelling remaining tasks...")
                        for f in future_to_dv:
                            f.cancel()
                        raise
                    except Exception as e:
                        self.logger.error(f"[{self.batch_id}] ✗ {dv_id}: EXCEPTION - {e!s}")
                        results["failed"].append(
                            ProcessingResult(
                                data_view_id=dv_id,
                                data_view_name="Unknown",
                                success=False,
                                duration=0,
                                error_message=str(e),
                            )
                        )

                        if not self.continue_on_error:
                            self.logger.warning(f"[{self.batch_id}] Stopping batch processing due to error")
                            break

                    pbar.update(1)

        results["total_duration"] = time.time() - batch_start_time

        # Log shared cache statistics if enabled
        if self._shared_cache is not None:
            cache_stats = self._shared_cache.get_statistics()
            self.logger.info(
                f"[{self.batch_id}] Shared cache stats: {cache_stats['hits']} hits, "
                f"{cache_stats['misses']} misses ({cache_stats['hit_rate']:.1f}% hit rate)"
            )
            # Cleanup shared cache resources
            self._shared_cache.shutdown()
            self._shared_cache = None

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: dict):
        """Print detailed batch processing summary with color-coded output"""
        total = results["total"]
        successful_count = len(results["successful"])
        failed_count = len(results["failed"])
        success_rate = (successful_count / total * 100) if total > 0 else 0
        total_duration = results["total_duration"]
        avg_duration = (total_duration / total) if total > 0 else 0

        # Calculate total output size
        total_file_size = sum(r.file_size_bytes for r in results["successful"])
        total_size_formatted = format_file_size(total_file_size)

        # Log to file
        self.logger.info("")
        self.logger.info("=" * BANNER_WIDTH)
        self.logger.info(f"[{self.batch_id}] BATCH PROCESSING SUMMARY")
        self.logger.info("=" * BANNER_WIDTH)
        self.logger.info(f"[{self.batch_id}] Total data views: {total}")
        self.logger.info(f"[{self.batch_id}] Successful: {successful_count}")
        self.logger.info(f"[{self.batch_id}] Failed: {failed_count}")
        self.logger.info(f"[{self.batch_id}] Success rate: {success_rate:.1f}%")
        self.logger.info(f"[{self.batch_id}] Total output size: {total_size_formatted}")
        self.logger.info(f"[{self.batch_id}] Total duration: {total_duration:.1f}s")
        self.logger.info(f"[{self.batch_id}] Average per data view: {avg_duration:.1f}s")
        if total_duration > 0:
            throughput = total / total_duration
            self.logger.info(f"[{self.batch_id}] Throughput: {throughput:.2f} views/second")
        self.logger.info("=" * BANNER_WIDTH)

        # Print color-coded console output
        print()
        print("=" * BANNER_WIDTH)
        print(ConsoleColors.bold("BATCH PROCESSING SUMMARY"))
        print("=" * BANNER_WIDTH)
        print(f"Total data views: {total}")
        print(f"Successful: {ConsoleColors.success(str(successful_count))}")
        if failed_count > 0:
            print(f"Failed: {ConsoleColors.error(str(failed_count))}")
        else:
            print(f"Failed: {failed_count}")
        print(f"Success rate: {ConsoleColors.status(success_rate == 100, f'{success_rate:.1f}%')}")
        print(f"Total output size: {total_size_formatted}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Average per data view: {avg_duration:.1f}s")
        if total_duration > 0:
            throughput = total / total_duration
            print(f"Throughput: {throughput:.2f} views/second")
        print()

        if results["successful"]:
            print(ConsoleColors.success("Successful Data Views:"))
            self.logger.info("")
            self.logger.info("Successful Data Views:")
            for result in results["successful"]:
                size_str = result.file_size_formatted
                line = f"  {result.data_view_id:20s}  {result.data_view_name:30s}  {size_str:>10s}  {result.duration:5.1f}s"
                print(ConsoleColors.success("  ✓") + line[3:])
                self.logger.info(
                    f"  ✓ {result.data_view_id:20s}  {result.data_view_name:30s}  {size_str:>10s}  {result.duration:5.1f}s"
                )
            print()
            self.logger.info("")

        if results["failed"]:
            print(ConsoleColors.error("Failed Data Views:"))
            self.logger.info("Failed Data Views:")
            for result in results["failed"]:
                line = f"  {result.data_view_id:20s}  {result.error_message}"
                print(ConsoleColors.error("  ✗") + line[3:])
                self.logger.info(f"  ✗ {result.data_view_id:20s}  {result.error_message}")
            print()
            self.logger.info("")

        print("=" * BANNER_WIDTH)
        self.logger.info("=" * BANNER_WIDTH)

        if total > 0 and total_duration > 0:
            throughput = (total / total_duration) * 60  # per minute
            print(f"Throughput: {throughput:.1f} data views per minute")
            print("=" * BANNER_WIDTH)
            self.logger.info(f"Throughput: {throughput:.1f} data views per minute")
            self.logger.info("=" * BANNER_WIDTH)


# ==================== DRY-RUN MODE ====================


def run_dry_run(data_views: list[str], config_file: str, logger: logging.Logger, profile: str | None = None) -> bool:
    """
    Validate configuration and connectivity without generating reports.

    Performs the following checks:
    1. Credential validation (profile, environment variables, or config file)
    2. CJA API connection test
    3. Data view accessibility verification

    Args:
        data_views: List of data view IDs to validate
        config_file: Path to configuration file
        logger: Logger instance
        profile: Optional profile name to use for credentials

    Returns:
        True if all validations pass, False otherwise
    """
    print()
    print("=" * BANNER_WIDTH)
    print("DRY-RUN MODE - Validating configuration and connectivity")
    print("=" * BANNER_WIDTH)
    print()

    all_passed = True

    # Step 1: Validate credentials
    print("[1/3] Validating credentials...")
    if profile:
        # Validate profile credentials
        try:
            profile_creds = load_profile_credentials(profile, logger)
            if profile_creds:
                print(f"  ✓ Profile '{profile}' found and valid")
            else:
                print(f"  ✗ Profile '{profile}' has no valid credentials")
                all_passed = False
        except ProfileNotFoundError:
            print(f"  ✗ Profile '{profile}' not found")
            all_passed = False
        except ProfileConfigError as e:
            print(f"  ✗ Profile '{profile}' configuration error: {e}")
            all_passed = False
    else:
        # Validate config file
        if validate_config_file(config_file, logger):
            print(f"  ✓ Configuration file '{config_file}' is valid")
        else:
            print("  ✗ Configuration file validation failed")
            all_passed = False

    if not all_passed:
        print()
        print("=" * BANNER_WIDTH)
        print("DRY-RUN FAILED - Fix configuration issues before proceeding")
        print("=" * BANNER_WIDTH)
        return False

    # Step 2: Test CJA connection
    print()
    print("[2/3] Testing CJA API connection...")
    try:
        success, source, _ = configure_cjapy(profile=profile, config_file=config_file, logger=logger)
        if not success:
            print(f"  ✗ Credential configuration failed: {source}")
            print()
            print("=" * BANNER_WIDTH)
            print("DRY-RUN FAILED - Cannot configure credentials")
            print("=" * BANNER_WIDTH)
            return False
        cja = cjapy.CJA()

        # Test API with getDataViews call (with retry for transient errors)
        available_dvs = make_api_call_with_retry(
            cja.getDataViews, logger=logger, operation_name="getDataViews (dry-run)"
        )
        if available_dvs is not None:
            dv_count = len(available_dvs) if hasattr(available_dvs, "__len__") else 0
            print("  ✓ API connection successful")
            print(f"  ✓ Found {dv_count} accessible data view(s)")
        else:
            print("  ⚠ API connection returned None - may be unstable")
            available_dvs = []
    except KeyboardInterrupt, SystemExit:
        print()
        print(ConsoleColors.warning("Dry-run cancelled."))
        raise
    except Exception as e:
        print(f"  ✗ API connection failed: {e!s}")
        all_passed = False
        print()
        print("=" * BANNER_WIDTH)
        print("DRY-RUN FAILED - Cannot connect to CJA API")
        print("=" * BANNER_WIDTH)
        return False

    # Step 3: Validate each data view
    print()
    print(f"[3/3] Validating {len(data_views)} data view(s)...")

    # Build set of available data view IDs for quick lookup
    available_ids = set()
    if available_dvs is not None and (
        (isinstance(available_dvs, pd.DataFrame) and not available_dvs.empty)
        or (not isinstance(available_dvs, pd.DataFrame) and available_dvs)
    ):
        for dv in available_dvs:
            if isinstance(dv, dict):
                available_ids.add(dv.get("id", ""))

    valid_count = 0
    invalid_count = 0
    total_metrics = 0
    total_dimensions = 0
    dv_details = []

    for dv_id in data_views:
        # Try to get data view info (with retry for transient errors)
        try:
            dv_info = make_api_call_with_retry(
                cja.getDataView, dv_id, logger=logger, operation_name=f"getDataView({dv_id})"
            )
            if dv_info:
                dv_name = dv_info.get("name", "Unknown")

                # Fetch component counts for predictions
                metrics_count = 0
                dimensions_count = 0
                try:
                    metrics = make_api_call_with_retry(
                        cja.getMetrics, dv_id, logger=logger, operation_name=f"getMetrics({dv_id})"
                    )
                    if metrics is not None:
                        metrics_count = len(metrics) if hasattr(metrics, "__len__") else 0
                except Exception:
                    pass  # Count will be 0 if fetch fails

                try:
                    dimensions = make_api_call_with_retry(
                        cja.getDimensions, dv_id, logger=logger, operation_name=f"getDimensions({dv_id})"
                    )
                    if dimensions is not None:
                        dimensions_count = len(dimensions) if hasattr(dimensions, "__len__") else 0
                except Exception:
                    pass  # Count will be 0 if fetch fails

                total_metrics += metrics_count
                total_dimensions += dimensions_count
                dv_details.append(
                    {"id": dv_id, "name": dv_name, "metrics": metrics_count, "dimensions": dimensions_count}
                )

                print(f"  ✓ {dv_id}: {dv_name}")
                print(f"      Components: {metrics_count} metrics, {dimensions_count} dimensions")
                valid_count += 1
            else:
                print(f"  ✗ {dv_id}: Not found or no access")
                invalid_count += 1
                all_passed = False
        except KeyboardInterrupt, SystemExit:
            print()
            print(ConsoleColors.warning("Validation cancelled."))
            raise
        except Exception as e:
            print(f"  ✗ {dv_id}: Error - {e!s}")
            invalid_count += 1
            all_passed = False

    # Calculate time estimates
    # Based on benchmarks: ~0.5s per component for validation, ~0.1s without
    total_components = total_metrics + total_dimensions
    est_time_with_validation = total_components * 0.01 + len(data_views) * 2  # API overhead per view
    est_time_skip_validation = total_components * 0.005 + len(data_views) * 1.5

    # Summary
    print()
    print("=" * BANNER_WIDTH)
    print("DRY-RUN SUMMARY")
    print("=" * BANNER_WIDTH)
    print("  Configuration: ✓ Valid")
    print("  API Connection: ✓ Connected")
    print(f"  Data Views: {valid_count} valid, {invalid_count} invalid")
    print()

    if valid_count > 0:
        print("  Predictions:")
        print(f"    Total components: {total_metrics} metrics + {total_dimensions} dimensions = {total_components}")
        print(f"    Est. time (with validation): ~{est_time_with_validation:.0f}s")
        print(f"    Est. time (--skip-validation): ~{est_time_skip_validation:.0f}s")
        print()

    if all_passed:
        print("✓ All validations passed - ready to generate reports")
        print()
        print("Run without --dry-run to generate SDR reports:")
        print(f"  cja_auto_sdr {' '.join(data_views)}")
    else:
        print("✗ Some validations failed - please fix issues before proceeding")

    print("=" * BANNER_WIDTH)

    return all_passed


# ==================== COMMAND-LINE INTERFACE ====================


def _bounded_float(min_val: float, max_val: float):
    """Argparse type factory for a float bounded to [min_val, max_val]."""

    def _type(value: str) -> float:
        f = float(value)
        if f < min_val or f > max_val:
            raise argparse.ArgumentTypeError(f"must be between {min_val} and {max_val}, got {f}")
        return f

    _type.__name__ = f"float[{min_val}-{max_val}]"
    return _type


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="CJA SDR Generator - Generate System Design Records for CJA Data Views",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single data view
  cja_auto_sdr dv_12345

  # Multiple data views (automatically triggers parallel processing)
  cja_auto_sdr dv_12345 dv_67890 dv_abcde

  # Batch processing with explicit flag (same as above)
  cja_auto_sdr --batch dv_12345 dv_67890 dv_abcde

  # Auto-detect optimal workers (default)
  cja_auto_sdr --batch dv_12345 dv_67890 --workers auto

  # Or specify explicit worker count
  cja_auto_sdr --batch dv_12345 dv_67890 --workers 2

  # Custom output directory
  cja_auto_sdr dv_12345 --output-dir ./reports

  # Continue on errors
  cja_auto_sdr --batch dv_* --continue-on-error

  # With custom log level
  cja_auto_sdr --batch dv_* --log-level WARNING

  # JSON structured logging (for Splunk, ELK, CloudWatch)
  cja_auto_sdr dv_12345 --log-format json

  # Export as CSV files
  cja_auto_sdr dv_12345 --format csv

  # Export as JSON
  cja_auto_sdr dv_12345 --format json

  # Export as HTML
  cja_auto_sdr dv_12345 --format html

  # Export as Markdown (GitHub/Confluence)
  cja_auto_sdr dv_12345 --format markdown

  # Export in all formats
  cja_auto_sdr dv_12345 --format all

  # Dry-run to validate config and connectivity
  cja_auto_sdr dv_12345 --dry-run

  # Quiet mode (errors only)
  cja_auto_sdr dv_12345 --quiet

  # List all accessible data views
  cja_auto_sdr --list-dataviews

  # Skip data quality validation (faster processing)
  cja_auto_sdr dv_12345 --skip-validation

  # Generate sample configuration file
  cja_auto_sdr --sample-config

  # Limit data quality issues to top 10 by severity
  cja_auto_sdr dv_12345 --max-issues 10

  # Validate only (alias for --dry-run)
  cja_auto_sdr dv_12345 --validate-only

  # --- Data View Comparison (Diff) ---

  # Compare two live data views
  cja_auto_sdr --diff dv_prod_12345 dv_staging_67890
  cja_auto_sdr --diff "Production Analytics" "Staging Analytics"

  # Save a snapshot for later comparison
  cja_auto_sdr dv_12345 --snapshot ./snapshots/baseline.json

  # Compare current state against a saved snapshot
  cja_auto_sdr dv_12345 --diff-snapshot ./snapshots/baseline.json

  # Diff output options
  cja_auto_sdr --diff dv_A dv_B --format html --output-dir ./reports
  cja_auto_sdr --diff dv_A dv_B --format all
  cja_auto_sdr --diff dv_A dv_B --changes-only
  cja_auto_sdr --diff dv_A dv_B --summary

  # Advanced diff options
  cja_auto_sdr --diff dv_A dv_B --ignore-fields description,title
  cja_auto_sdr --diff dv_A dv_B --diff-labels Production Staging

  # Auto-snapshot: automatically save snapshots during diff for audit trail
  cja_auto_sdr --diff dv_A dv_B --auto-snapshot
  cja_auto_sdr --diff dv_A dv_B --auto-snapshot --snapshot-dir ./history
  cja_auto_sdr --diff dv_A dv_B --auto-snapshot --keep-last 10

  # --- Quick UX Features ---

  # Quick stats without full report
  cja_auto_sdr dv_12345 --stats
  cja_auto_sdr dv_1 dv_2 dv_3 --stats

  # Stats in JSON format for scripting
  cja_auto_sdr dv_12345 --stats --format json
  cja_auto_sdr dv_12345 --stats --output -    # Output to stdout

  # Open file after generation
  cja_auto_sdr dv_12345 --open

  # List data views in JSON format (for scripting/piping)
  cja_auto_sdr --list-dataviews --format json
  cja_auto_sdr --list-dataviews --output -    # JSON to stdout

  # List connections with their datasets
  cja_auto_sdr --list-connections
  cja_auto_sdr --list-connections --format json
  cja_auto_sdr --list-connections --format csv --output connections.csv

  # List data views with their backing connections and datasets
  cja_auto_sdr --list-datasets
  cja_auto_sdr --list-datasets --format json
  cja_auto_sdr --list-datasets --format csv --output datasets.csv

  # --- Profile Management ---

  # List all available profiles
  cja_auto_sdr --profile-list

  # Use a named profile
  cja_auto_sdr --profile client-a --list-dataviews
  cja_auto_sdr -p client-a "My Data View" --format excel

  # Create a new profile interactively
  cja_auto_sdr --profile-add client-c

  # Test profile credentials
  cja_auto_sdr --profile-test client-a

  # Show profile configuration (secrets masked)
  cja_auto_sdr --profile-show client-a

  # Use profile via environment variable
  export CJA_PROFILE=client-a
  cja_auto_sdr --list-dataviews

Note:
  At least one data view ID must be provided (except for --list-dataviews, --list-connections, --list-datasets, --sample-config, --stats).
  Use 'cja_auto_sdr --help' to see all options.

Exit Codes:
  0 - Success (diff: no differences found)
  1 - Error occurred
  2 - Policy threshold exceeded (diff changes, quality gate, or governance threshold)

Requirements:
  Python 3.14 or higher required. Verify with: python3 --version
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}", help="Show program version and exit"
    )

    parser.add_argument("--exit-codes", action="store_true", help="Display exit code reference and exit")

    parser.add_argument(
        "data_views",
        nargs="*",
        metavar="DATA_VIEW_ID_OR_NAME",
        help="Data view IDs (e.g., dv_12345) or exact names (use quotes for names with spaces). "
        "If a name matches multiple data views, all will be processed. "
        "At least one required unless using --version, --list-dataviews, etc.",
    )

    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode (parallel execution)")

    parser.add_argument(
        "--workers",
        type=str,
        default="auto",
        help=f"Number of parallel workers for batch mode (1-{MAX_BATCH_WORKERS}). "
        f'Use "auto" (default) for intelligent detection based on CPU cores, '
        f"data view count, and component complexity. Auto-reduces workers for "
        f"large data views (>5K components) to prevent memory exhaustion",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "."),
        help="Output directory for generated files (default: current directory, or OUTPUT_DIR env var)",
    )

    parser.add_argument(
        "--config-file", type=str, default="config.json", help="Path to CJA configuration file (default: config.json)"
    )

    parser.add_argument(
        "--continue-on-error", action="store_true", help="Continue processing remaining data views if one fails"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO, or LOG_LEVEL environment variable)",
    )

    parser.add_argument(
        "--log-format",
        type=str,
        default="text",
        choices=["text", "json"],
        help='Log output format: "text" (default) for human-readable, '
        '"json" for structured logging (Splunk, ELK, CloudWatch compatible)',
    )

    parser.add_argument(
        "--production", action="store_true", help="Enable production mode (minimal logging for maximum performance)"
    )

    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable validation result caching (50-90%% faster on repeated validations)",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear validation cache before processing (use with --enable-cache for fresh validation)",
    )

    parser.add_argument(
        "--cache-size",
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help=f"Maximum number of cached validation results (default: {DEFAULT_CACHE_SIZE})",
    )

    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=DEFAULT_CACHE_TTL,
        help=f"Cache time-to-live in seconds (default: {DEFAULT_CACHE_TTL} = 1 hour)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", DEFAULT_RETRY_CONFIG["max_retries"])),
        help=f"Maximum API retry attempts (default: {DEFAULT_RETRY_CONFIG['max_retries']}, or MAX_RETRIES env var)",
    )

    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=float(os.environ.get("RETRY_BASE_DELAY", DEFAULT_RETRY_CONFIG["base_delay"])),
        help=f"Initial retry delay in seconds (default: {DEFAULT_RETRY_CONFIG['base_delay']}, or RETRY_BASE_DELAY env var)",
    )

    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=float(os.environ.get("RETRY_MAX_DELAY", DEFAULT_RETRY_CONFIG["max_delay"])),
        help=f"Maximum retry delay in seconds (default: {DEFAULT_RETRY_CONFIG['max_delay']}, or RETRY_MAX_DELAY env var)",
    )

    # ==================== RELIABILITY/PERFORMANCE ARGUMENTS ====================

    reliability_group = parser.add_argument_group(
        "Reliability & Performance", "Options for API resilience and performance tuning"
    )

    reliability_group.add_argument(
        "--api-auto-tune",
        action="store_true",
        help="Enable automatic API worker tuning based on response times. "
        "Scales workers up when responses are fast, down when slow",
    )

    reliability_group.add_argument(
        "--api-min-workers",
        type=int,
        default=1,
        metavar="N",
        help="Minimum workers for auto-tuning (default: 1, requires --api-auto-tune)",
    )

    reliability_group.add_argument(
        "--api-max-workers",
        type=int,
        default=10,
        metavar="N",
        help="Maximum workers for auto-tuning (default: 10, requires --api-auto-tune)",
    )

    reliability_group.add_argument(
        "--circuit-breaker",
        action="store_true",
        help="Enable circuit breaker pattern for API calls. "
        "Prevents cascading failures by stopping requests after repeated failures",
    )

    reliability_group.add_argument(
        "--circuit-failure-threshold",
        type=int,
        default=5,
        metavar="N",
        help="Consecutive failures before opening circuit (default: 5, requires --circuit-breaker)",
    )

    reliability_group.add_argument(
        "--circuit-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds before attempting recovery from open circuit (default: 30, requires --circuit-breaker)",
    )

    reliability_group.add_argument(
        "--shared-cache",
        action="store_true",
        help="Share validation cache across batch workers (requires --batch and --enable-cache). "
        "Enables cache reuse across data views for common validation patterns",
    )

    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["console", "excel", "csv", "json", "html", "markdown", "all", "reports", "data", "ci"],
        help="Output format: excel, csv, json, html, markdown, all, or aliases (reports=excel+markdown, data=csv+json, ci=json+markdown). Default: excel for SDR, console for diff",
    )

    parser.add_argument(
        "--dry-run",
        "--validate-only",
        action="store_true",
        dest="dry_run",
        help="Validate configuration and connectivity without generating reports",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode - suppress all output except errors and final summary"
    )

    discovery_group = parser.add_argument_group(
        "Discovery", "Commands to explore available CJA resources (mutually exclusive)"
    )
    discovery_mx = discovery_group.add_mutually_exclusive_group()

    discovery_mx.add_argument(
        "--list-dataviews",
        action="store_true",
        help="List all accessible data views and exit (no data view ID required)",
    )

    discovery_mx.add_argument(
        "--list-connections", action="store_true", help="List all accessible connections with their datasets and exit"
    )

    discovery_mx.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all data views with their backing connections and datasets, then exit",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data quality validation for faster processing (20-30%% faster)",
    )

    parser.add_argument("--sample-config", action="store_true", help="Generate a sample configuration file and exit")

    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and API connectivity without processing any data views",
    )

    parser.add_argument(
        "--config-status",
        action="store_true",
        help="Show configuration status (source, fields, masked credentials) without API call. "
        "Faster than --validate-config for quick troubleshooting",
    )

    parser.add_argument(
        "--config-json",
        action="store_true",
        help="Output --config-status as machine-readable JSON (for scripting and CI/CD)",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        dest="assume_yes",
        help="Skip confirmation prompts (e.g., for large batch operations)",
    )

    parser.add_argument(
        "--max-issues",
        type=int,
        default=0,
        metavar="N",
        help="Limit data quality issues to top N by severity (0 = show all, default: 0)",
    )

    parser.add_argument(
        "--fail-on-quality",
        type=str.upper,
        choices=list(QUALITY_SEVERITY_ORDER),
        metavar="SEVERITY",
        help="Exit with code 2 when quality issues at or above SEVERITY are found "
        "(CRITICAL, HIGH, MEDIUM, LOW, INFO)",
    )

    parser.add_argument(
        "--quality-report",
        type=str,
        choices=["json", "csv"],
        metavar="FORMAT",
        help="Generate standalone quality issues report only (json or csv) without SDR files",
    )

    # ==================== UX ENHANCEMENT ARGUMENTS ====================

    parser.add_argument(
        "--show-timings",
        action="store_true",
        help="Display performance timing breakdown after processing (API calls, validation, output generation)",
    )

    parser.add_argument(
        "--open", action="store_true", help="Open the generated file(s) in the default application after creation"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show quick statistics about data view(s) without generating full reports. "
        "Displays counts of metrics, dimensions, and basic info",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive mode for guided SDR generation. "
        "Walks through: (1) data view selection, (2) output format, "
        "(3) inventory options (segments, calculated metrics, derived fields). "
        "Ideal for new users or one-off generation tasks",
    )

    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help='Output file path. Use "-" or "stdout" to write to standard output (JSON/CSV only). '
        "For stdout, implies --quiet to suppress other output",
    )

    # ==================== DIFF COMPARISON ARGUMENTS ====================

    diff_group = parser.add_argument_group("Diff Comparison", "Options for comparing data views")

    diff_group.add_argument(
        "--diff",
        action="store_true",
        help="Compare two data views. Provide exactly 2 data view IDs/names as positional arguments",
    )

    diff_group.add_argument(
        "--snapshot",
        type=str,
        metavar="FILE",
        help="Save a snapshot of the data view to a JSON file (use with single data view)",
    )

    diff_group.add_argument(
        "--diff-snapshot", type=str, metavar="FILE", help="Compare data view against a saved snapshot file"
    )

    diff_group.add_argument(
        "--compare-with-prev",
        action="store_true",
        help="Compare data view against its most recent snapshot in --snapshot-dir (default: ./snapshots)",
    )

    diff_group.add_argument(
        "--compare-snapshots",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="Compare two snapshot files directly (no API calls required). "
        "Example: --compare-snapshots baseline.json current.json",
    )

    diff_group.add_argument(
        "--changes-only", action="store_true", help="Only show changed items in diff output (hide unchanged)"
    )

    diff_group.add_argument("--summary", action="store_true", help="Show summary statistics only (no detailed changes)")

    diff_group.add_argument(
        "--ignore-fields",
        type=str,
        metavar="FIELDS",
        help='Comma-separated list of fields to ignore during comparison (e.g., "description,title")',
    )

    diff_group.add_argument(
        "--diff-labels",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="Custom labels for the two sides of the comparison (default: Source, Target)",
    )

    diff_group.add_argument(
        "--show-only",
        type=str,
        metavar="TYPES",
        help="Filter diff output to show only specific change types. "
        'Comma-separated list: added,removed,modified,unchanged (e.g., "added,modified")',
    )

    diff_group.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only include metrics (exclude dimensions). Works for both SDR generation and diff comparison",
    )

    diff_group.add_argument(
        "--dimensions-only",
        action="store_true",
        help="Only include dimensions (exclude metrics). Works for both SDR generation and diff comparison",
    )

    diff_group.add_argument(
        "--extended-fields",
        action="store_true",
        help="Use extended field comparison including attribution, format, bucketing, "
        "persistence settings (default: basic fields only)",
    )

    diff_group.add_argument(
        "--side-by-side",
        action="store_true",
        help="Show side-by-side comparison view for modified items (console and markdown)",
    )

    diff_group.add_argument("--no-color", action="store_true", help="Disable ANSI color codes in console output")

    diff_group.add_argument(
        "--color-theme",
        type=str,
        choices=["default", "accessible"],
        default="default",
        help='Color theme for diff output: "default" (green/red) or "accessible" (blue/orange for accessibility)',
    )

    diff_group.add_argument(
        "--quiet-diff",
        action="store_true",
        help="Suppress diff output, only return exit code (0=no changes, 2=changes found)",
    )

    diff_group.add_argument(
        "--reverse-diff", action="store_true", help="Reverse the comparison direction (swap source and target)"
    )

    diff_group.add_argument(
        "--warn-threshold",
        type=float,
        metavar="PERCENT",
        help="Exit with code 3 if change percentage exceeds threshold (e.g., --warn-threshold 10)",
    )

    diff_group.add_argument(
        "--group-by-field", action="store_true", help="Group changes by field name instead of by component"
    )

    diff_group.add_argument(
        "--group-by-field-limit",
        type=int,
        default=10,
        metavar="N",
        help="Max items per section in --group-by-field output (default: 10, 0 = unlimited)",
    )

    diff_group.add_argument(
        "--diff-output", type=str, metavar="FILE", help="Write diff output directly to file instead of stdout"
    )

    diff_group.add_argument(
        "--format-pr-comment",
        action="store_true",
        help="Output in GitHub/GitLab PR comment format (markdown with collapsible details)",
    )

    diff_group.add_argument(
        "--auto-snapshot",
        action="store_true",
        help="Automatically save snapshots of data views during diff comparison. "
        "Creates timestamped snapshots in --snapshot-dir for audit trail",
    )

    diff_group.add_argument(
        "--auto-prune",
        action="store_true",
        help=(
            "When used with --auto-snapshot, automatically apply default retention "
            f"(--keep-last {DEFAULT_AUTO_PRUNE_KEEP_LAST} and --keep-since {DEFAULT_AUTO_PRUNE_KEEP_SINCE}) "
            "if explicit retention flags are not provided"
        ),
    )

    diff_group.add_argument(
        "--snapshot-dir",
        type=str,
        default="./snapshots",
        metavar="DIR",
        help="Directory for auto-saved snapshots (default: ./snapshots). Used with --auto-snapshot",
    )

    diff_group.add_argument(
        "--keep-last",
        type=int,
        default=0,
        metavar="N",
        help="Retention policy: keep only the last N snapshots per data view (0 = keep all). Used with --auto-snapshot",
    )

    diff_group.add_argument(
        "--keep-since",
        type=str,
        default=None,
        metavar="PERIOD",
        help="Date-based retention: delete snapshots older than PERIOD. "
        "Formats: 7d (7 days), 2w (2 weeks), 1m (1 month), 30 (30 days). "
        "Used with --auto-snapshot. Can be combined with --keep-last.",
    )

    # ==================== PROFILE MANAGEMENT ARGUMENTS ====================

    profile_group = parser.add_argument_group(
        "Profile Management", "Manage organization/credential profiles stored in ~/.cja/orgs/"
    )

    profile_group.add_argument(
        "--profile",
        "-p",
        type=str,
        metavar="NAME",
        default=os.environ.get("CJA_PROFILE"),
        help="Use named profile from ~/.cja/orgs/<NAME>/. Can also be set via CJA_PROFILE environment variable",
    )

    profile_group.add_argument("--profile-list", action="store_true", help="List all available profiles and exit")

    profile_group.add_argument("--profile-add", type=str, metavar="NAME", help="Create a new profile interactively")

    profile_group.add_argument(
        "--profile-test", type=str, metavar="NAME", help="Test profile credentials and API connectivity"
    )

    profile_group.add_argument(
        "--profile-show", type=str, metavar="NAME", help="Show profile configuration (with masked secrets)"
    )

    # ==================== GIT INTEGRATION ARGUMENTS ====================

    git_group = parser.add_argument_group("Git Integration", "Options for version-controlled snapshots")

    git_group.add_argument(
        "--git-commit",
        action="store_true",
        help="Save snapshot in Git-friendly format and commit to Git repository. "
        "Creates separate JSON files (metrics.json, dimensions.json, metadata.json) "
        "for easy diffing in Git",
    )

    git_group.add_argument(
        "--git-push", action="store_true", help="Push to remote after committing (requires --git-commit)"
    )

    git_group.add_argument(
        "--git-message", type=str, metavar="MESSAGE", help="Custom message for Git commit (used with --git-commit)"
    )

    git_group.add_argument(
        "--git-dir",
        type=str,
        default="./sdr-snapshots",
        metavar="DIR",
        help="Directory for Git-tracked snapshots (default: ./sdr-snapshots). "
        "Will be initialized as Git repo if not already",
    )

    git_group.add_argument(
        "--git-init", action="store_true", help="Initialize a new Git repository for snapshots at --git-dir location"
    )

    # ==================== DERIVED FIELD INVENTORY ARGUMENTS ====================

    derived_group = parser.add_argument_group(
        "Derived Field Inventory", "Include summary inventory of derived fields in SDR output"
    )

    derived_group.add_argument(
        "--include-derived",
        action="store_true",
        dest="include_derived_inventory",
        help='Include derived field inventory in SDR output. Adds a "Derived Fields" sheet/section '
        "with complexity scores, functions used, and logic summaries. "
        "Note: For SDR generation only; not used in snapshot diff comparisons since derived "
        "fields are already captured in standard metrics/dimensions output.",
    )

    # ==================== CALCULATED METRICS INVENTORY ARGUMENTS ====================

    calc_metrics_group = parser.add_argument_group(
        "Calculated Metrics Inventory", "Include summary inventory of calculated metrics in SDR output"
    )

    calc_metrics_group.add_argument(
        "--include-calculated",
        action="store_true",
        dest="include_calculated_metrics",
        help='Include calculated metrics inventory in SDR output. Adds a "Calculated Metrics" sheet/section '
        "with complexity scores, formula summaries, and metric references",
    )

    # ==================== SEGMENTS INVENTORY ARGUMENTS ====================

    segments_group = parser.add_argument_group(
        "Segments Inventory", "Include summary inventory of segments (filters) in SDR output"
    )

    segments_group.add_argument(
        "--include-segments",
        action="store_true",
        dest="include_segments_inventory",
        help='Include segments inventory in SDR output. Adds a "Segments" sheet/section '
        "with complexity scores, definition summaries, and dimension/metric references",
    )

    # ==================== INVENTORY-ONLY MODE ====================

    inventory_only_group = parser.add_argument_group(
        "Inventory-Only Mode", "Generate output with only inventory sheets (no standard SDR content)"
    )

    inventory_only_group.add_argument(
        "--inventory-only",
        action="store_true",
        dest="inventory_only",
        help="Output only inventory sheets (Calculated Metrics, Segments, Derived Fields). "
        "Skips standard SDR sheets (Metadata, Data Quality, DataView, Metrics, Dimensions). "
        "Requires at least one --include-* flag.",
    )

    inventory_only_group.add_argument(
        "--inventory-summary",
        action="store_true",
        dest="inventory_summary",
        help="Display quick inventory statistics without generating full output files. "
        "Shows counts, complexity distribution, and high-complexity warnings. "
        "Requires at least one --include-* flag. Cannot be used with --inventory-only.",
    )

    inventory_only_group.add_argument(
        "--include-all-inventory",
        action="store_true",
        dest="include_all_inventory",
        help="Enable all inventory options. In SDR mode, enables --include-segments, "
        "--include-calculated, and --include-derived. With --snapshot or --git-commit, "
        "enables only --include-segments and --include-calculated (derived fields are "
        "not supported in snapshots).",
    )

    # ==================== ORG-WIDE ANALYSIS ARGUMENTS ====================

    org_group = parser.add_argument_group(
        "Org-Wide Analysis", "Analyze component distribution across all data views in the organization"
    )

    org_group.add_argument(
        "--org-report",
        action="store_true",
        help="Generate org-wide component analysis report. Analyzes all accessible "
        "data views to show component distribution, similarity matrix, and "
        "governance recommendations. No data view arguments required.",
    )

    org_group.add_argument(
        "--filter",
        type=str,
        metavar="PATTERN",
        dest="org_filter",
        help="Include only data views whose name matches this regex pattern "
        '(e.g., "Prod.*" or "prod|production"). Avoid complex nested quantifiers',
    )

    org_group.add_argument(
        "--exclude",
        type=str,
        metavar="PATTERN",
        dest="org_exclude",
        help="Exclude data views whose name matches this regex pattern "
        '(e.g., "Test.*|Dev.*|sandbox"). Avoid complex nested quantifiers',
    )

    org_group.add_argument(
        "--limit",
        type=int,
        metavar="N",
        dest="org_limit",
        help="Limit the number of data views to analyze (useful for testing or large orgs)",
    )

    org_group.add_argument(
        "--core-threshold",
        type=float,
        default=0.5,
        metavar="PERCENT",
        help='Threshold for "core" components as fraction of data views '
        "(default: 0.5 = components in >= 50%% of DVs are core)",
    )

    org_group.add_argument(
        "--core-min-count",
        type=int,
        metavar="N",
        help='Minimum absolute count for "core" components (overrides --core-threshold). '
        "E.g., --core-min-count 5 means components in >= 5 DVs are core",
    )

    org_group.add_argument(
        "--overlap-threshold",
        type=_bounded_float(0.0, 1.0),
        default=0.8,
        metavar="PERCENT",
        help='Threshold for "high overlap" pairs in similarity analysis '
        "(default: 0.8). Note: For governance checks (--fail-on-threshold), "
        "values above 0.9 are capped at 90%% to ensure duplicate detection",
    )

    org_group.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the O(n^2) pairwise similarity matrix calculation. Useful for very large orgs with many data views",
    )

    org_group.add_argument(
        "--similarity-max-dvs",
        type=int,
        metavar="N",
        dest="org_similarity_max_dvs",
        default=250,
        help="Guardrail to skip similarity when data views exceed N (default: 250). "
        "Use --force-similarity to override.",
    )

    org_group.add_argument(
        "--force-similarity",
        action="store_true",
        dest="org_force_similarity",
        help="Force similarity matrix even if guardrails would skip it",
    )

    org_group.add_argument(
        "--include-names",
        action="store_true",
        dest="org_include_names",
        help="Include component names in the report (slower - requires fetching full component details)",
    )

    org_group.add_argument(
        "--org-summary", action="store_true", help="Show only summary statistics, suppress detailed component lists"
    )

    org_group.add_argument(
        "--org-verbose", action="store_true", help="Include full component lists and detailed breakdowns in output"
    )

    # Component type breakdown (enabled by default)
    org_group.add_argument(
        "--no-component-types",
        action="store_true",
        dest="no_component_types",
        help="Disable component type breakdown (standard vs derived metrics/dimensions)",
    )

    # Metadata
    org_group.add_argument(
        "--include-metadata",
        action="store_true",
        dest="org_include_metadata",
        help="Include data view metadata (owner, creation/modification dates, descriptions)",
    )

    # Drift detection
    org_group.add_argument(
        "--include-drift",
        action="store_true",
        dest="org_include_drift",
        help="Include component drift details showing exact differences between similar DV pairs",
    )

    org_group.add_argument(
        "--org-shared-client",
        action="store_true",
        dest="org_shared_client",
        help="Use a single shared cjapy client across threads. WARNING: This is experimental "
        "and may cause race conditions if cjapy is not thread-safe. Use only if you have "
        "tested with your cjapy version. Default creates one client per thread (safer)",
    )

    # Sampling options
    org_group.add_argument(
        "--sample",
        type=int,
        metavar="N",
        dest="org_sample_size",
        help="Randomly sample N data views (useful for very large orgs)",
    )

    org_group.add_argument(
        "--sample-seed", type=int, metavar="SEED", dest="org_sample_seed", help="Random seed for reproducible sampling"
    )

    org_group.add_argument(
        "--sample-stratified",
        action="store_true",
        dest="org_sample_stratified",
        help="Stratify sample by data view name prefix",
    )

    # Caching options
    org_group.add_argument(
        "--use-cache",
        action="store_true",
        dest="org_use_cache",
        help="Enable caching of data view components for faster repeat runs",
    )

    org_group.add_argument(
        "--cache-max-age",
        type=int,
        metavar="HOURS",
        default=24,
        dest="org_cache_max_age",
        help="Maximum cache age in hours (default: 24)",
    )

    org_group.add_argument(
        "--refresh-cache",
        action="store_true",
        dest="org_clear_cache",
        help="Clear the org-report cache and fetch fresh data",
    )

    org_group.add_argument(
        "--validate-cache",
        action="store_true",
        dest="org_validate_cache",
        help="Validate cached entries against data view modification timestamps before using",
    )

    org_group.add_argument(
        "--memory-warning",
        type=int,
        metavar="MB",
        default=100,
        dest="org_memory_warning",
        help="Warn if component index estimated memory exceeds this threshold in MB (default: 100, 0 to disable)",
    )

    org_group.add_argument(
        "--memory-limit",
        type=int,
        metavar="MB",
        default=None,
        dest="org_memory_limit",
        help="Abort if component index exceeds this size in MB. Protects against OOM for very large orgs (default: no limit)",
    )

    # Clustering options
    org_group.add_argument(
        "--cluster",
        action="store_true",
        dest="org_cluster",
        help="Enable hierarchical clustering to group related data views (requires 'clustering' extra: uv pip install 'cja-auto-sdr[clustering]')",
    )

    org_group.add_argument(
        "--cluster-method",
        type=str,
        choices=["average", "complete"],
        default="average",
        dest="org_cluster_method",
        help="Clustering linkage method: average (recommended) or complete. Both work correctly with Jaccard distances",
    )

    # Feature 1: Governance exit codes
    org_group.add_argument(
        "--duplicate-threshold",
        type=int,
        metavar="N",
        dest="org_duplicate_threshold",
        help="Maximum allowed high-similarity pairs (>=90%%). Exit with code 2 if exceeded when --fail-on-threshold is set",
    )

    org_group.add_argument(
        "--isolated-threshold",
        type=_bounded_float(0.0, 1.0),
        metavar="PERCENT",
        dest="org_isolated_threshold",
        help="Maximum isolated component percentage (0.0-1.0). Exit with code 2 if exceeded when --fail-on-threshold is set",
    )

    org_group.add_argument(
        "--fail-on-threshold",
        action="store_true",
        dest="org_fail_on_threshold",
        help="Enable exit code 2 when governance thresholds are exceeded (for CI/CD integration)",
    )

    # Feature 2: Org summary stats mode
    org_group.add_argument(
        "--org-stats",
        action="store_true",
        dest="org_stats_only",
        help="Quick summary stats only - skips similarity matrix and clustering for faster results",
    )

    # Feature 3: Naming convention audit
    org_group.add_argument(
        "--audit-naming",
        action="store_true",
        dest="org_audit_naming",
        help="Detect naming pattern inconsistencies (snake_case vs camelCase, stale prefixes, etc.)",
    )

    # Feature 4: Trending/drift report
    org_group.add_argument(
        "--compare-org-report",
        type=str,
        metavar="PREV.json",
        dest="org_compare_report",
        help="Compare current org-report to a previous JSON report for trending/drift analysis",
    )

    # Feature 5: Owner/team summary
    org_group.add_argument(
        "--owner-summary",
        action="store_true",
        dest="org_owner_summary",
        help="Group statistics by data view owner (requires --include-metadata)",
    )

    # Feature 6: Stale component heuristics
    org_group.add_argument(
        "--flag-stale",
        action="store_true",
        dest="org_flag_stale",
        help="Flag components with stale naming patterns (test, old, temp, deprecated, version suffixes, date patterns)",
    )

    # Enable shell tab-completion if argcomplete is installed
    if _ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    return parser.parse_args()


# ==================== DATA VIEW NAME RESOLUTION ====================


def is_data_view_id(identifier: str) -> bool:
    """
    Check if a string is a data view ID (starts with 'dv_')

    Args:
        identifier: String to check

    Returns:
        True if identifier is a data view ID, False if it's a name
    """
    return identifier.startswith("dv_")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    This is used to find similar data view names when exact match fails.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The minimum number of single-character edits needed to transform s1 into s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_similar_names(
    target: str, available_names: list[str], max_suggestions: int = 3, max_distance: int | None = None
) -> list[tuple[str, int]]:
    """
    Find names similar to the target using Levenshtein distance.

    Args:
        target: The name to find matches for
        available_names: List of available names to search
        max_suggestions: Maximum number of suggestions to return
        max_distance: Maximum edit distance to consider (default: half of target length + 2)

    Returns:
        List of (name, distance) tuples, sorted by distance (closest first)
    """
    if max_distance is None:
        max_distance = len(target) // 2 + 2

    # Calculate distances
    suggestions = []
    target_lower = target.lower()

    for name in available_names:
        # Check exact case-insensitive match first
        if name.lower() == target_lower:
            suggestions.append((name, 0))
            continue

        # Calculate distance
        distance = levenshtein_distance(target_lower, name.lower())
        if distance <= max_distance:
            suggestions.append((name, distance))

    # Sort by distance and return top matches
    suggestions.sort(key=lambda x: (x[1], x[0]))
    return suggestions[:max_suggestions]


# ==================== DATA VIEW CACHE ====================


class DataViewCache:
    """
    Thread-safe cache for data view listings to avoid repeated API calls.

    The cache has a configurable TTL and is automatically invalidated after
    the TTL expires. This is useful when performing multiple diff operations
    in the same session.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._cache: dict[str, tuple[list[dict], float]] = {}
        self._ttl_seconds = 300  # 5 minute default TTL
        self._initialized = True

    def get(self, config_file: str) -> list[dict] | None:
        """
        Get cached data views for a config file.

        Args:
            config_file: The config file key

        Returns:
            List of data view dicts if cached and not expired, None otherwise
        """
        with self._lock:
            if config_file in self._cache:
                data, timestamp = self._cache[config_file]
                if time.time() - timestamp < self._ttl_seconds:
                    return data
                # Expired - remove from cache
                del self._cache[config_file]
            return None

    def set(self, config_file: str, data: list[dict]) -> None:
        """
        Cache data views for a config file.

        Args:
            config_file: The config file key
            data: List of data view dicts to cache
        """
        with self._lock:
            self._cache[config_file] = (data, time.time())

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()

    def set_ttl(self, seconds: int) -> None:
        """Set the cache TTL in seconds."""
        self._ttl_seconds = seconds


# Global cache instance
_data_view_cache = DataViewCache()


def get_cached_data_views(cja, config_file: str, logger: logging.Logger) -> list[dict]:
    """
    Get data views with caching support.

    Args:
        cja: CJA API instance
        config_file: Config file path (used as cache key)
        logger: Logger instance

    Returns:
        List of data view dicts
    """
    # Check cache first
    cached = _data_view_cache.get(config_file)
    if cached is not None:
        logger.debug(f"Using cached data views ({len(cached)} entries)")
        return cached

    # Fetch from API
    logger.debug("Fetching data views from API (cache miss)")
    available_dvs = cja.getDataViews()

    if available_dvs is None:
        return []

    # Convert to list if DataFrame
    if isinstance(available_dvs, pd.DataFrame):
        available_dvs = available_dvs.to_dict("records")

    # Cache the result
    _data_view_cache.set(config_file, available_dvs)
    logger.debug(f"Cached {len(available_dvs)} data views")

    return available_dvs


def prompt_for_selection(options: list[tuple[str, str]], prompt_text: str) -> str | None:
    """
    Prompt user to select from a list of options interactively.

    Args:
        options: List of (id, display_text) tuples
        prompt_text: Text to display before options

    Returns:
        Selected ID or None if user cancels
    """
    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        return None

    print(f"\n{prompt_text}")
    print("-" * 40)

    for i, (opt_id, display) in enumerate(options, 1):
        print(f"  [{i}] {display}")
        print(f"      ID: {opt_id}")

    print("  [0] Cancel")
    print()

    while True:
        try:
            choice = input("Enter selection (number): ").strip()
            if choice == "0" or choice.lower() in ("q", "quit", "cancel"):
                return None

            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]

            print(f"Invalid selection. Enter 1-{len(options)} or 0 to cancel.")
        except ValueError:
            print("Please enter a number.")
        except EOFError, KeyboardInterrupt:
            print("\nCancelled.")
            return None


def resolve_data_view_names(
    identifiers: list[str],
    config_file: str = "config.json",
    logger: logging.Logger | None = None,
    suggest_similar: bool = True,
    profile: str | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Resolve data view names to IDs. If an identifier is already an ID, keep it as-is.
    If it's a name, look up all data views with that exact name.

    Features:
    - Caches API calls for performance when resolving multiple names
    - Suggests similar names using fuzzy matching when exact match fails

    Args:
        identifiers: List of data view IDs or names
        config_file: Path to CJA configuration file
        logger: Logger instance for logging
        suggest_similar: If True, suggest similar names when exact match fails
        profile: Optional profile name to use for credentials

    Returns:
        Tuple of (resolved_ids, name_to_ids_map)
        - resolved_ids: List of all resolved data view IDs
        - name_to_ids_map: Dict mapping names to their resolved IDs (for reporting)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    resolved_ids = []
    name_to_ids_map = {}

    try:
        # Initialize CJA connection
        logger.info(f"Resolving data view identifiers: {identifiers}")
        success, source, _ = configure_cjapy(profile, config_file, logger)
        if not success:
            logger.error(f"Failed to configure credentials: {source}")
            return [], {}
        cja = cjapy.CJA()

        # Get all available data views (with caching)
        logger.debug("Fetching all data views for name resolution")
        available_dvs = get_cached_data_views(cja, config_file, logger)

        if not available_dvs:
            logger.error("No data views found or no access to any data views")
            return [], {}

        # Build a lookup map: name -> list of IDs
        name_to_id_lookup = {}
        id_to_name_lookup = {}

        for dv in available_dvs:
            if isinstance(dv, dict):
                dv_id = dv.get("id")
                dv_name = dv.get("name")

                if dv_id and dv_name:
                    id_to_name_lookup[dv_id] = dv_name
                    if dv_name not in name_to_id_lookup:
                        name_to_id_lookup[dv_name] = []
                    name_to_id_lookup[dv_name].append(dv_id)

        logger.debug(f"Built lookup map with {len(name_to_id_lookup)} unique names and {len(id_to_name_lookup)} IDs")

        # Process each identifier
        for identifier in identifiers:
            if is_data_view_id(identifier):
                # It's an ID - validate it exists
                if identifier in id_to_name_lookup:
                    resolved_ids.append(identifier)
                    logger.debug(f"ID '{identifier}' validated: {id_to_name_lookup[identifier]}")
                else:
                    logger.warning(f"Data view ID '{identifier}' not found in accessible data views")
                    # Still add it - will fail during processing with proper error message
                    resolved_ids.append(identifier)
            else:
                # It's a name - look up all matching IDs
                if identifier in name_to_id_lookup:
                    matching_ids = name_to_id_lookup[identifier]
                    resolved_ids.extend(matching_ids)
                    name_to_ids_map[identifier] = matching_ids

                    if len(matching_ids) == 1:
                        logger.info(f"Name '{identifier}' resolved to ID: {matching_ids[0]}")
                    else:
                        logger.info(f"Name '{identifier}' matched {len(matching_ids)} data views: {matching_ids}")
                else:
                    # Name not found - try to find similar names for helpful error message
                    logger.error(f"Data view name '{identifier}' not found in accessible data views")

                    if suggest_similar:
                        similar = find_similar_names(identifier, list(name_to_id_lookup.keys()))
                        if similar:
                            # Check for case-insensitive match first
                            case_match = [s for s in similar if s[1] == 0]
                            if case_match:
                                logger.error(f"  → Did you mean '{case_match[0][0]}'? (case mismatch)")
                            else:
                                suggestions = [f"'{s[0]}'" for s in similar]
                                logger.error(f"  → Did you mean: {', '.join(suggestions)}?")

                    logger.error("  → Name matching is CASE-SENSITIVE and requires EXACT match")
                    logger.error("  → Run 'cja_auto_sdr --list-dataviews' to see all available names")
                    # Don't add to resolved_ids - this is an error

        logger.info(f"Resolved {len(identifiers)} identifier(s) to {len(resolved_ids)} data view ID(s)")
        return resolved_ids, name_to_ids_map

    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found")
        return [], {}
    except Exception as e:
        logger.error(f"Failed to resolve data view names: {e!s}")
        return [], {}


# ==================== DATASET INFO HELPER ====================


def _extract_dataset_info(dataset: Any) -> dict:
    """
    Resilient parser for dataset objects from connection API responses.

    The dataSets field in connection responses may use varying field names.
    This helper tries common field names and falls back gracefully.

    Args:
        dataset: A dataset object (dict or other) from the dataSets array

    Returns:
        Dict with 'id' and 'name' keys (values may be 'N/A' if not found)
    """
    if not isinstance(dataset, dict):
        return {"id": str(dataset) if dataset else "N/A", "name": "N/A"}

    # Try common ID field names (prefer canonical/camelCase, keep snake_case for compatibility)
    ds_id = (
        dataset.get("id") or dataset.get("datasetId") or dataset.get("dataSetId") or dataset.get("dataset_id") or "N/A"
    )

    # Try common name field names
    ds_name = (
        dataset.get("name")
        or dataset.get("datasetName")
        or dataset.get("dataSetName")
        or dataset.get("dataset_name")
        or dataset.get("title")
        or "N/A"
    )

    return {"id": str(ds_id), "name": str(ds_name)}


# ==================== LIST HELPERS ====================


def _emit_output(data: str, output_file: str | None, is_stdout: bool) -> None:
    """Emit output data to a file, stdout pipe, or the console.

    When output_file is None (no --output flag), falls through to print().
    When writing to a TTY and the output exceeds the terminal height,
    the text is piped through the system pager (``$PAGER``, defaulting
    to ``less -R``).
    """
    if output_file and not is_stdout:
        parent = os.path.dirname(output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(data)
    else:
        text = data.rstrip("\n")
        # Use a pager when output is longer than the terminal and stdout
        # is an interactive TTY (not a pipe / redirect).
        if not is_stdout and sys.stdout.isatty():
            line_count = text.count("\n") + 1
            try:
                term_height = os.get_terminal_size().lines
            except OSError:
                term_height = 0
            if term_height and line_count > term_height:
                pager = os.environ.get("PAGER", "less")
                if shutil.which(pager):
                    try:
                        proc = subprocess.Popen(
                            [pager, "-R"] if pager == "less" else [pager],
                            stdin=subprocess.PIPE,
                        )
                        proc.communicate(text.encode(), timeout=300)
                        return
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    except OSError:
                        pass
                    # pager unavailable or timed out — fall through to plain print
        print(text)


# ==================== SHARED DISCOVERY FORMATTERS ====================


def _format_as_json(payload: dict) -> str:
    """Format a discovery result dict as indented JSON."""
    return json.dumps(payload, indent=2)


def _format_as_csv(columns: list[str], rows: list[dict]) -> str:
    """Format discovery rows as CSV with the given column headers."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(col, "") for col in columns])
    return buf.getvalue()


def _format_as_table(
    header_line: str,
    items: list[dict],
    columns: list[str],
    col_labels: list[str] | None = None,
) -> str:
    """Format discovery items as an aligned text table.

    Args:
        header_line: Summary line (e.g. "Found 5 accessible data view(s):").
        items: List of dicts, one per row.
        columns: Dict keys to include, in order.
        col_labels: Display labels for each column (defaults to title-cased keys).

    Returns:
        Formatted table string with leading/trailing blank lines.
    """
    labels = col_labels or [c.replace("_", " ").title() for c in columns]
    widths = [
        max(len(lbl), max((len(str(item.get(col, ""))) for item in items), default=0)) + 2
        for col, lbl in zip(columns, labels, strict=True)
    ]
    lines: list[str] = ["", header_line, ""]
    lines.append("".join(f"{lbl:<{w}}" for lbl, w in zip(labels, widths, strict=True)))
    lines.append("-" * sum(widths))
    lines.extend(
        "".join(f"{item.get(col, '')!s:<{w}}" for col, w in zip(columns, widths, strict=True)) for item in items
    )
    lines.append("")
    return "\n".join(lines)


def _extract_owner_name(owner_data: Any) -> str:
    """Extract a displayable owner name from an API owner object.

    The owner field varies across CJA API endpoints:
    - Data views may return ``{"name": "Jane Doe"}``
    - Connections may return ``{"imsUserId": "ABC@AdobeID"}``
    - Some endpoints return ``None`` or a bare string.
    """
    if owner_data is None:
        return "N/A"
    if isinstance(owner_data, str):
        return owner_data or "N/A"
    if isinstance(owner_data, dict):
        for key in ("name", "login", "email", "imsUserId", "id"):
            val = owner_data.get(key)
            if val:
                return str(val)
        return "N/A"
    return str(owner_data) or "N/A"


def _extract_connections_list(raw_connections: Any) -> list:
    """Extract the connection list from a raw getConnections API response."""
    if isinstance(raw_connections, dict):
        connections = raw_connections.get("content", raw_connections.get("result", []))
        if not isinstance(connections, list):
            # content/result was an unexpected type (e.g. a string); wrap the
            # whole dict so downstream isinstance(conn, dict) checks can decide
            # whether to process or skip it.
            return [raw_connections]
        return connections
    elif isinstance(raw_connections, list):
        return raw_connections
    return []


def _run_list_command(
    banner_text: str,
    command_name: str,
    fetch_and_format: Callable,
    config_file: str = "config.json",
    output_format: str = "table",
    output_file: str | None = None,
    profile: str | None = None,
) -> bool:
    """Shared boilerplate for list-* discovery commands.

    Handles profile resolution, CJA configuration, banner display, and
    error handling.  The caller-specific logic lives in *fetch_and_format*,
    which receives ``(cja, is_machine_readable)`` and must return the
    formatted output string.  The returned string is routed through
    ``_emit_output`` (file, stdout pipe, or console).  Return ``None``
    to skip output entirely (e.g. when there are no results and a
    warning was already printed to the console).

    When ``output_file`` is set and the format is table (not
    machine-readable), the banner / progress text is printed to the
    console while the data payload is written to the file.

    Args:
        banner_text: Banner heading shown in table mode (e.g. "LISTING ACCESSIBLE DATA VIEWS").
        command_name: Logger name / short label for the command.
        fetch_and_format: ``(cja, is_machine_readable) -> Optional[str]``.
        config_file: Path to CJA configuration file.
        output_format: "table", "json", or "csv".
        output_file: File path, "-" for stdout pipe, or None.
        profile: Optional profile name.

    Returns:
        True if successful, False otherwise.
    """
    is_stdout = output_file in ("-", "stdout")
    is_machine_readable = output_format in ("json", "csv") or is_stdout

    active_profile = resolve_active_profile(profile)

    if not is_machine_readable:
        print()
        print("=" * BANNER_WIDTH)
        print(banner_text)
        print("=" * BANNER_WIDTH)
        print()
        if active_profile:
            print(f"Using profile: {active_profile}")
        else:
            print(f"Using configuration: {config_file}")
        print()

    try:
        logger = logging.getLogger(command_name)
        logger.setLevel(logging.WARNING)
        success, source, _ = configure_cjapy(profile=active_profile, config_file=config_file, logger=logger)
        if not success:
            if is_machine_readable:
                print(json.dumps({"error": f"Configuration error: {source}"}), file=sys.stderr)
            else:
                print(f"Configuration error: {source}")
            return False
        cja = cjapy.CJA()

        if not is_machine_readable:
            print("Connecting to CJA API...")

        output_data = fetch_and_format(cja, is_machine_readable)
        if output_data is not None:
            _emit_output(output_data, output_file, is_stdout)

        return True

    except FileNotFoundError:
        if is_machine_readable:
            print(json.dumps({"error": f"Configuration file '{config_file}' not found"}), file=sys.stderr)
        else:
            print(ConsoleColors.error(f"ERROR: Configuration file '{config_file}' not found"))
            print()
            print("Generate a sample configuration file with:")
            print("  cja_auto_sdr --sample-config")
        return False

    except KeyboardInterrupt, SystemExit:
        if not is_machine_readable:
            print()
            print(ConsoleColors.warning("Operation cancelled."))
        raise

    except Exception as e:
        if is_machine_readable:
            print(json.dumps({"error": f"Failed to connect to CJA API: {e!s}"}), file=sys.stderr)
        else:
            print(ConsoleColors.error(f"ERROR: Failed to connect to CJA API: {e!s}"))
        return False


# ==================== LIST DATA VIEWS ====================


def _fetch_dataviews(output_format: str) -> Callable:
    """Return a fetch_and_format callback for list_dataviews."""

    def _inner(cja: Any, is_machine_readable: bool) -> str | None:
        available_dvs = cja.getDataViews()

        if available_dvs is None or (hasattr(available_dvs, "__len__") and len(available_dvs) == 0):
            if is_machine_readable:
                if output_format == "json":
                    return json.dumps({"dataViews": [], "count": 0}, indent=2)
                return "id,name,owner\n"
            return "\nNo data views found or no access to any data views.\n"

        if isinstance(available_dvs, pd.DataFrame):
            available_dvs = available_dvs.to_dict("records")

        display_data = []
        for dv in available_dvs:
            if isinstance(dv, dict):
                dv_id = dv.get("id", "N/A")
                dv_name = dv.get("name", "N/A")
                owner_name = _extract_owner_name(dv.get("owner"))
                display_data.append({"id": dv_id, "name": dv_name, "owner": owner_name})

        if output_format == "json":
            return _format_as_json({"dataViews": display_data, "count": len(display_data)})
        elif output_format == "csv":
            return _format_as_csv(["id", "name", "owner"], display_data)
        else:
            table = _format_as_table(
                f"Found {len(display_data)} accessible data view(s):",
                display_data,
                columns=["id", "name", "owner"],
                col_labels=["ID", "Name", "Owner"],
            )
            # Compute total width to match table separator for usage footer
            labels = ["ID", "Name", "Owner"]
            cols = ["id", "name", "owner"]
            widths = [
                max(len(lbl), max((len(str(item.get(col, ""))) for item in display_data), default=0)) + 2
                for col, lbl in zip(cols, labels, strict=True)
            ]
            total_width = sum(widths)
            footer_lines = [
                "=" * total_width,
                "Usage:",
                "  cja_auto_sdr <DATA_VIEW_ID>       # Use ID directly",
                '  cja_auto_sdr "<DATA_VIEW_NAME>"   # Use exact name (quotes recommended)',
                "",
                "Note: If multiple data views share the same name, all will be processed.",
                "=" * total_width,
            ]
            return table + "\n".join(footer_lines)

    return _inner


def list_dataviews(
    config_file: str = "config.json",
    output_format: str = "table",
    output_file: str | None = None,
    profile: str | None = None,
) -> bool:
    """List all accessible data views and exit."""
    return _run_list_command(
        banner_text="LISTING ACCESSIBLE DATA VIEWS",
        command_name="list_dataviews",
        fetch_and_format=_fetch_dataviews(output_format),
        config_file=config_file,
        output_format=output_format,
        output_file=output_file,
        profile=profile,
    )


# ==================== LIST CONNECTIONS ====================


def _fetch_connections(output_format: str) -> Callable:
    """Return a fetch_and_format callback for list_connections."""

    def _inner(cja: Any, is_machine_readable: bool) -> str | None:
        raw_connections = cja.getConnections(output="raw", expansion="name,ownerFullName,dataSets")
        connections = _extract_connections_list(raw_connections)

        if not connections:
            # Check whether data views reference connections we can't see
            # (the GET /connections API requires product-admin privileges).
            available_dvs = cja.getDataViews()
            if isinstance(available_dvs, pd.DataFrame):
                available_dvs = available_dvs.to_dict("records")

            conn_ids_from_dvs: dict[str, int] = {}  # conn_id -> count of data views
            for dv in available_dvs or []:
                if not isinstance(dv, dict):
                    continue
                pid = dv.get("parentDataGroupId")
                if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                    conn_ids_from_dvs[pid] = conn_ids_from_dvs.get(pid, 0) + 1

            if conn_ids_from_dvs:
                # Permissions issue — derive connection IDs from data views
                _PERM_WARNING = (
                    "Note: The GET /connections API requires product-admin privileges.\n"
                    "Connection details are unavailable. Showing connection IDs derived\n"
                    "from data views instead."
                )
                derived = [
                    {"id": cid, "name": None, "owner": None, "datasets": [], "dataview_count": cnt}
                    for cid, cnt in sorted(conn_ids_from_dvs.items())
                ]

                if output_format == "json":
                    return _format_as_json(
                        {
                            "connections": derived,
                            "count": len(derived),
                            "warning": _PERM_WARNING.replace("\n", " "),
                        }
                    )
                elif output_format == "csv":
                    flat = [
                        {
                            "connection_id": d["id"],
                            "connection_name": "",
                            "owner": "",
                            "dataset_id": "",
                            "dataset_name": "",
                            "dataview_count": d["dataview_count"],
                        }
                        for d in derived
                    ]
                    return _format_as_csv(
                        ["connection_id", "connection_name", "owner", "dataset_id", "dataset_name", "dataview_count"],
                        flat,
                    )
                else:
                    lines: list[str] = []
                    lines.append("")
                    lines.append(_PERM_WARNING)
                    lines.append("")
                    lines.append(f"Found {len(derived)} connection(s) referenced by data views:")
                    lines.append("")
                    lines.extend(f"  {d['id']}  ({d['dataview_count']} data view(s))" for d in derived)
                    lines.append("")
                    return "\n".join(lines)

            # Genuinely no connections
            if is_machine_readable:
                if output_format == "json":
                    return _format_as_json({"connections": [], "count": 0})
                return "connection_id,connection_name,owner,dataset_id,dataset_name\n"
            return "\nNo connections found or no access to any connections.\n"

        display_data = []
        for conn in connections:
            if not isinstance(conn, dict):
                continue
            conn_id = conn.get("id", "N/A")
            conn_name = conn.get("name", "N/A")
            owner_name = _extract_owner_name(conn.get("ownerFullName") or conn.get("owner"))

            raw_datasets = conn.get("dataSets", conn.get("datasets", []))
            if not isinstance(raw_datasets, list):
                raw_datasets = []
            datasets = [_extract_dataset_info(ds) for ds in raw_datasets]

            display_data.append(
                {
                    "id": conn_id,
                    "name": conn_name,
                    "owner": owner_name,
                    "datasets": datasets,
                }
            )

        if output_format == "json":
            return _format_as_json({"connections": display_data, "count": len(display_data)})
        elif output_format == "csv":
            # Flatten nested datasets: one CSV row per dataset
            flat_rows: list[dict] = []
            for conn in display_data:
                if conn["datasets"]:
                    flat_rows.extend(
                        {
                            "connection_id": conn["id"],
                            "connection_name": conn["name"],
                            "owner": conn["owner"],
                            "dataset_id": ds["id"],
                            "dataset_name": ds["name"],
                        }
                        for ds in conn["datasets"]
                    )
                else:
                    flat_rows.append(
                        {
                            "connection_id": conn["id"],
                            "connection_name": conn["name"],
                            "owner": conn["owner"],
                            "dataset_id": "",
                            "dataset_name": "",
                        }
                    )
            return _format_as_csv(
                ["connection_id", "connection_name", "owner", "dataset_id", "dataset_name"], flat_rows
            )
        else:
            lines: list[str] = []
            lines.append("")
            lines.append(f"Found {len(display_data)} accessible connection(s):")
            lines.append("")
            for conn in display_data:
                lines.append(f"Connection: {conn['name']} ({conn['id']})")
                lines.append(f"Owner: {conn['owner']}")
                if conn["datasets"]:
                    lines.append(f"Datasets ({len(conn['datasets'])}):")
                    for ds in conn["datasets"]:
                        lines.append(f"  {ds['id']}  {ds['name']}")
                else:
                    lines.append("Datasets: (none)")
                lines.append("")
            return "\n".join(lines)

    return _inner


def list_connections(
    config_file: str = "config.json",
    output_format: str = "table",
    output_file: str | None = None,
    profile: str | None = None,
) -> bool:
    """List all accessible connections with their datasets and exit."""
    return _run_list_command(
        banner_text="LISTING ACCESSIBLE CONNECTIONS",
        command_name="list_connections",
        fetch_and_format=_fetch_connections(output_format),
        config_file=config_file,
        output_format=output_format,
        output_file=output_file,
        profile=profile,
    )


# ==================== LIST DATASETS ====================


def _fetch_datasets(output_format: str) -> Callable:
    """Return a fetch_and_format callback for list_datasets."""

    def _inner(cja: Any, is_machine_readable: bool) -> str | None:
        # Step 1: Fetch all connections and build lookup map
        raw_connections = cja.getConnections(output="raw", expansion="name,ownerFullName,dataSets")
        conn_map: dict = {}  # connection_id -> {name, datasets}
        for conn in _extract_connections_list(raw_connections):
            if not isinstance(conn, dict):
                continue
            conn_id = conn.get("id", "")
            conn_name = conn.get("name", "N/A")
            raw_datasets = conn.get("dataSets", conn.get("datasets", []))
            if not isinstance(raw_datasets, list):
                raw_datasets = []
            conn_map[conn_id] = {
                "name": conn_name,
                "datasets": [_extract_dataset_info(ds) for ds in raw_datasets],
            }

        # Step 2: Fetch all data views
        available_dvs = cja.getDataViews()
        if isinstance(available_dvs, pd.DataFrame):
            available_dvs = available_dvs.to_dict("records")

        if not available_dvs:
            if is_machine_readable:
                if output_format == "json":
                    return _format_as_json({"dataViews": [], "count": 0})
                return "dataview_id,dataview_name,connection_id,connection_name,dataset_id,dataset_name\n"
            return "\nNo data views found or no access to any data views.\n"

        # Detect permissions gap: conn_map is empty but data views have connections
        _no_conn_details = False
        if not conn_map:
            for dv in available_dvs or []:
                if not isinstance(dv, dict):
                    continue
                pid = dv.get("parentDataGroupId")
                if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                    _no_conn_details = True
                    break

        # Step 3: Build output records using parentDataGroupId
        if not is_machine_readable:
            print(f"Processing {len(available_dvs)} data view(s)...")
        display_data = []
        for i, dv in enumerate(available_dvs):
            if not isinstance(dv, dict):
                continue
            dv_id = dv.get("id", "N/A")
            dv_name = dv.get("name", "N/A")

            if not is_machine_readable:
                print(f"  [{i + 1}/{len(available_dvs)}] {dv_name}...", end="\r")

            parent_conn_id = dv.get("parentDataGroupId")
            # DataFrame-backed records can carry missing values as NaN/NA.
            # Normalize to None so machine-readable output emits "N/A", not NaN.
            if parent_conn_id is not None and pd.isna(parent_conn_id):
                parent_conn_id = None

            conn_info = conn_map.get(parent_conn_id) if parent_conn_id else None
            conn_name = conn_info.get("name", "N/A") if conn_info else None
            datasets = conn_info.get("datasets", []) if conn_info else []

            display_data.append(
                {
                    "id": dv_id,
                    "name": dv_name,
                    "connection": {"id": parent_conn_id or "N/A", "name": conn_name},
                    "datasets": datasets,
                }
            )

        if not is_machine_readable:
            # Clear progress line with ANSI erase-line escape
            print("\033[2K", end="\r")

        _CONN_PERM_WARNING = (
            "Note: Connection details are unavailable (the GET /connections API\n"
            "requires product-admin privileges). Showing connection IDs only."
        )

        result_payload: dict = {"dataViews": display_data, "count": len(display_data)}
        if _no_conn_details:
            result_payload["warning"] = _CONN_PERM_WARNING.replace("\n", " ")

        if output_format == "json":
            return _format_as_json(result_payload)
        elif output_format == "csv":
            # Flatten nested datasets: one CSV row per dataset per data view
            flat_rows: list[dict] = []
            for entry in display_data:
                conn_id = entry["connection"]["id"]
                conn_name_val = entry["connection"]["name"] or ""
                if entry["datasets"]:
                    flat_rows.extend(
                        {
                            "dataview_id": entry["id"],
                            "dataview_name": entry["name"],
                            "connection_id": conn_id,
                            "connection_name": conn_name_val,
                            "dataset_id": ds["id"],
                            "dataset_name": ds["name"],
                        }
                        for ds in entry["datasets"]
                    )
                else:
                    flat_rows.append(
                        {
                            "dataview_id": entry["id"],
                            "dataview_name": entry["name"],
                            "connection_id": conn_id,
                            "connection_name": conn_name_val,
                            "dataset_id": "",
                            "dataset_name": "",
                        }
                    )
            return _format_as_csv(
                ["dataview_id", "dataview_name", "connection_id", "connection_name", "dataset_id", "dataset_name"],
                flat_rows,
            )
        else:
            lines: list[str] = []
            lines.append("")
            if _no_conn_details:
                lines.append(_CONN_PERM_WARNING)
                lines.append("")
            lines.append(f"Found {len(display_data)} data view(s) with dataset information:")
            lines.append("")
            for entry in display_data:
                lines.append(f"Data View: {entry['name']} ({entry['id']})")
                c_name = entry["connection"]["name"]
                c_id = entry["connection"]["id"]
                if c_name:
                    lines.append(f"Connection: {c_name} ({c_id})")
                else:
                    lines.append(f"Connection: {c_id}")
                if entry["datasets"]:
                    lines.append(f"Datasets ({len(entry['datasets'])}):")
                    lines.extend(f"  {ds['id']}  {ds['name']}" for ds in entry["datasets"])
                elif not _no_conn_details:
                    lines.append("Datasets: (none)")
                lines.append("")
            return "\n".join(lines)

    return _inner


def list_datasets(
    config_file: str = "config.json",
    output_format: str = "table",
    output_file: str | None = None,
    profile: str | None = None,
) -> bool:
    """List all data views with their backing connections and underlying datasets."""
    return _run_list_command(
        banner_text="LISTING DATA VIEWS WITH DATASETS",
        command_name="list_datasets",
        fetch_and_format=_fetch_datasets(output_format),
        config_file=config_file,
        output_format=output_format,
        output_file=output_file,
        profile=profile,
    )


# ==================== INTERACTIVE DATA VIEW SELECTION ====================


def interactive_select_dataviews(config_file: str = "config.json", profile: str | None = None) -> list[str]:
    """
    Interactively select data views from a numbered list.

    Supports selection formats:
    - Single number: "3"
    - Multiple numbers: "1,3,5"
    - Range: "1-5"
    - Combined: "1,3-5,7"
    - All: "all" or "a"

    Args:
        config_file: Path to CJA configuration file
        profile: Optional profile name to use for credentials

    Returns:
        List of selected data view IDs, or empty list on error/cancel
    """
    print()
    print("=" * BANNER_WIDTH)
    print("INTERACTIVE DATA VIEW SELECTION")
    print("=" * BANNER_WIDTH)
    print()
    if profile:
        print(f"Using profile: {profile}")
    else:
        print(f"Using configuration: {config_file}")
    print()

    try:
        success, source, _ = configure_cjapy(profile, config_file)
        if not success:
            print(ConsoleColors.error(f"ERROR: {source}"))
            return []
        cja = cjapy.CJA()

        print("Fetching available data views...")
        available_dvs = cja.getDataViews()

        if available_dvs is None or (hasattr(available_dvs, "__len__") and len(available_dvs) == 0):
            print()
            print(ConsoleColors.warning("No data views found or no access to any data views."))
            return []

        # Convert to list if DataFrame
        if isinstance(available_dvs, pd.DataFrame):
            available_dvs = available_dvs.to_dict("records")

        # Build display data
        display_data = []
        for dv in available_dvs:
            if isinstance(dv, dict):
                dv_id = dv.get("id", "N/A")
                dv_name = dv.get("name", "N/A")
                owner_name = _extract_owner_name(dv.get("owner"))
                display_data.append({"id": dv_id, "name": dv_name, "owner": owner_name})

        if not display_data:
            print(ConsoleColors.warning("No data views available."))
            return []

        # Calculate column widths
        num_width = len(str(len(display_data))) + 2
        max_id_width = max(len("ID"), max(len(item["id"]) for item in display_data)) + 2
        max_name_width = max(len("Name"), max(len(item["name"]) for item in display_data)) + 2
        max_owner_width = max(len("Owner"), max(len(item["owner"]) for item in display_data)) + 2

        total_width = num_width + max_id_width + max_name_width + max_owner_width

        print()
        print(f"Found {len(display_data)} accessible data view(s):")
        print()
        print(f"{'#':<{num_width}} {'ID':<{max_id_width}} {'Name':<{max_name_width}} {'Owner':<{max_owner_width}}")
        print("-" * total_width)

        for idx, item in enumerate(display_data, 1):
            print(
                f"{idx:<{num_width}} {item['id']:<{max_id_width}} {item['name']:<{max_name_width}} {item['owner']:<{max_owner_width}}"
            )

        print()
        print("-" * total_width)
        print("Selection options:")
        print("  Single:   3         (selects #3)")
        print("  Multiple: 1,3,5     (selects #1, #3, #5)")
        print("  Range:    1-5       (selects #1 through #5)")
        print("  Combined: 1,3-5,7   (selects #1, #3, #4, #5, #7)")
        print("  All:      all or a  (selects all data views)")
        print("  Cancel:   q or quit (exit without selection)")
        print()

        while True:
            try:
                selection = input("Enter selection: ").strip().lower()
            except EOFError:
                print()
                print(ConsoleColors.warning("No input available (non-interactive terminal)."))
                return []

            if not selection:
                print("Please enter a selection.")
                continue

            if selection in ("q", "quit", "exit", "cancel"):
                print(ConsoleColors.warning("Selection cancelled."))
                return []

            # Parse selection
            selected_indices = set()

            if selection in ("all", "a", "*"):
                selected_indices = set(range(1, len(display_data) + 1))
            else:
                # Parse comma-separated parts
                parts = selection.replace(" ", "").split(",")
                valid = True
                for part in parts:
                    if not part:
                        continue
                    if "-" in part:
                        # Range like "1-5"
                        try:
                            range_parts = part.split("-")
                            if len(range_parts) != 2:
                                raise ValueError("Invalid range format")
                            start = int(range_parts[0])
                            end = int(range_parts[1])
                            if start > end:
                                start, end = end, start
                            for i in range(start, end + 1):
                                selected_indices.add(i)
                        except ValueError:
                            print(ConsoleColors.error(f"Invalid range: '{part}'. Use format like '1-5'."))
                            valid = False
                            break
                    else:
                        # Single number
                        try:
                            num = int(part)
                            selected_indices.add(num)
                        except ValueError:
                            print(ConsoleColors.error(f"Invalid number: '{part}'."))
                            valid = False
                            break

                if not valid:
                    continue

            # Validate indices
            invalid_indices = [i for i in selected_indices if i < 1 or i > len(display_data)]
            if invalid_indices:
                print(
                    ConsoleColors.error(f"Invalid selection(s): {invalid_indices}. Valid range: 1-{len(display_data)}")
                )
                continue

            if not selected_indices:
                print("No valid selections. Please try again.")
                continue

            # Convert indices to IDs
            selected_ids = [display_data[i - 1]["id"] for i in sorted(selected_indices)]

            print()
            print(f"Selected {len(selected_ids)} data view(s):")
            for idx in sorted(selected_indices):
                item = display_data[idx - 1]
                print(f"  {idx}. {item['name']} ({item['id']})")

            return selected_ids

    except FileNotFoundError:
        print(ConsoleColors.error(f"ERROR: Configuration file '{config_file}' not found"))
        print()
        print("Generate a sample configuration file with:")
        print("  cja_auto_sdr --sample-config")
        return []

    except KeyboardInterrupt, SystemExit:
        print()
        print(ConsoleColors.warning("Operation cancelled."))
        return []

    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to connect to CJA API: {e!s}"))
        return []


# ==================== INTERACTIVE MODE ====================


@dataclass
class WizardConfig:
    """Configuration collected from interactive mode"""

    data_view_ids: list[str]
    output_format: str = "excel"
    output_dir: str | None = None
    include_segments: bool = False
    include_calculated: bool = False
    include_derived: bool = False
    inventory_only: bool = False


def interactive_wizard(config_file: str = "config.json", profile: str | None = None) -> WizardConfig | None:
    """
    Interactive wizard for guided SDR generation.

    Walks users through:
    1. Data view selection
    2. Output format selection
    3. Inventory options
    4. Summary and confirmation

    Args:
        config_file: Path to CJA configuration file
        profile: Optional profile name to use for credentials

    Returns:
        WizardConfig with user selections, or None if cancelled
    """

    def prompt_choice(prompt: str, options: list[tuple[str, str]], default: str | None = None) -> str | None:
        """Prompt user to select from numbered options."""
        print()
        print(prompt)
        print()
        for i, (key, label) in enumerate(options, 1):
            default_marker = " (default)" if key == default else ""
            print(f"  {i}. {label}{default_marker}")
        print()
        print("  q. Cancel and exit")
        print()

        while True:
            try:
                default_hint = f" [{options[[k for k, _ in options].index(default)][1]}]" if default else ""
                choice = input(f"Enter choice (1-{len(options)}){default_hint}: ").strip().lower()
            except EOFError, KeyboardInterrupt:
                print()
                return None

            if choice in ("q", "quit", "exit", "cancel"):
                return None

            if not choice and default:
                return default

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
                print(f"Please enter a number between 1 and {len(options)}.")
            except ValueError:
                print(f"Please enter a number between 1 and {len(options)}, or 'q' to cancel.")

    def prompt_yes_no(prompt: str, default: bool = False) -> bool | None:
        """Prompt user for yes/no answer."""
        if default:
            prompt_hint = "[Y/n] (Enter=yes)"
        else:
            prompt_hint = "[y/N] (Enter=no)"
        valid_yes = ("y", "yes", "1", "true")
        valid_no = ("n", "no", "0", "false")
        valid_quit = ("q", "quit", "exit", "cancel")

        while True:
            print()
            try:
                answer = input(f"{prompt} {prompt_hint}: ").strip().lower()
            except EOFError, KeyboardInterrupt:
                print()
                return None

            if answer in valid_quit:
                return None
            if not answer:
                return default
            if answer in valid_yes:
                return True
            if answer in valid_no:
                return False

            # Invalid input - show error and retry
            print(ConsoleColors.warning(f"Invalid input '{answer}'. Please enter 'y' or 'n' (or 'q' to quit)."))

    print()
    print("=" * BANNER_WIDTH)
    print("  CJA SDR GENERATOR - INTERACTIVE MODE")
    print("=" * BANNER_WIDTH)
    print()
    print("This interactive mode will guide you through generating an SDR.")
    print("Press 'q' at any prompt to cancel.")

    # Step 1: Connect and show data views
    print()
    print("-" * 60)
    print("STEP 1: Select Data View(s)")
    print("-" * 60)

    if profile:
        print(f"Using profile: {profile}")
    else:
        print(f"Using configuration: {config_file}")
    print()

    try:
        success, source, _ = configure_cjapy(profile, config_file)
        if not success:
            print(ConsoleColors.error(f"ERROR: {source}"))
            return None
        cja = cjapy.CJA()

        print("Fetching available data views...")
        available_dvs = cja.getDataViews()

        if available_dvs is None or (hasattr(available_dvs, "__len__") and len(available_dvs) == 0):
            print()
            print(ConsoleColors.warning("No data views found or no access to any data views."))
            return None

        # Convert to list if DataFrame
        if isinstance(available_dvs, pd.DataFrame):
            available_dvs = available_dvs.to_dict("records")

        # Build display data
        display_data = []
        for dv in available_dvs:
            if isinstance(dv, dict):
                dv_id = dv.get("id", "N/A")
                dv_name = dv.get("name", "N/A")
                display_data.append({"id": dv_id, "name": dv_name})

        if not display_data:
            print(ConsoleColors.warning("No data views available."))
            return None

        # Show data views
        print()
        print(f"Found {len(display_data)} accessible data view(s):")
        print()
        for idx, item in enumerate(display_data, 1):
            print(f"  {idx}. {item['name']}")
            print(f"      {ConsoleColors.dim(item['id'])}")

        print()
        print("Selection options: single (3), multiple (1,3,5), range (1-3), all")
        print()

        while True:
            try:
                selection = input("Select data view(s): ").strip().lower()
            except EOFError, KeyboardInterrupt:
                print()
                return None

            if selection in ("q", "quit", "exit", "cancel"):
                print(ConsoleColors.warning("Cancelled."))
                return None

            if not selection:
                print("Please enter a selection.")
                continue

            # Parse selection (reuse logic from interactive_select_dataviews)
            selected_indices = set()
            valid = True

            if selection in ("all", "a", "*"):
                selected_indices = set(range(1, len(display_data) + 1))
            else:
                parts = selection.replace(" ", "").split(",")
                for part in parts:
                    if not part:
                        continue
                    if "-" in part:
                        try:
                            range_parts = part.split("-")
                            if len(range_parts) != 2:
                                raise ValueError()
                            start, end = int(range_parts[0]), int(range_parts[1])
                            if start > end:
                                start, end = end, start
                            for i in range(start, end + 1):
                                selected_indices.add(i)
                        except ValueError:
                            print(ConsoleColors.error(f"Invalid range: '{part}'"))
                            valid = False
                            break
                    else:
                        try:
                            selected_indices.add(int(part))
                        except ValueError:
                            print(ConsoleColors.error(f"Invalid number: '{part}'"))
                            valid = False
                            break

            if not valid:
                continue

            # Validate
            invalid = [i for i in selected_indices if i < 1 or i > len(display_data)]
            if invalid:
                print(ConsoleColors.error(f"Invalid: {invalid}. Valid range: 1-{len(display_data)}"))
                continue

            if not selected_indices:
                continue

            selected_ids = [display_data[i - 1]["id"] for i in sorted(selected_indices)]
            selected_names = [display_data[i - 1]["name"] for i in sorted(selected_indices)]

            print()
            print(f"Selected: {', '.join(selected_names)}")
            break

    except FileNotFoundError:
        print(ConsoleColors.error(f"ERROR: Configuration file '{config_file}' not found"))
        print("Run: cja_auto_sdr --sample-config")
        return None
    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to connect to CJA API: {e!s}"))
        return None

    # Step 2: Output format
    print()
    print("-" * 60)
    print("STEP 2: Choose Output Format")
    print("-" * 60)

    format_options = [
        ("excel", "Excel (.xlsx) - Best for review and sharing"),
        ("json", "JSON - Best for automation and APIs"),
        ("csv", "CSV - Best for data processing"),
        ("html", "HTML - Best for web viewing"),
        ("markdown", "Markdown - Best for documentation/GitHub"),
        ("all", "All formats - Generate everything"),
    ]

    output_format = prompt_choice("Which output format would you like?", format_options, default="excel")
    if output_format is None:
        print(ConsoleColors.warning("Cancelled."))
        return None

    # Step 3: Inventory options
    print()
    print("-" * 60)
    print("STEP 3: Include Inventory Data?")
    print("-" * 60)
    print()
    print("Inventory data provides additional documentation beyond the standard SDR:")
    print("  • Segments: Filter definitions, complexity scores, references")
    print("  • Calculated Metrics: Formulas, complexity, metric dependencies")
    print("  • Derived Fields: Logic analysis, functions used, schema references")

    include_segments = prompt_yes_no("Include Segments inventory?", default=False)
    if include_segments is None:
        print(ConsoleColors.warning("Cancelled."))
        return None

    include_calculated = prompt_yes_no("Include Calculated Metrics inventory?", default=False)
    if include_calculated is None:
        print(ConsoleColors.warning("Cancelled."))
        return None

    include_derived = prompt_yes_no("Include Derived Fields inventory?", default=False)
    if include_derived is None:
        print(ConsoleColors.warning("Cancelled."))
        return None

    # Step 4: Summary and confirmation
    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print()
    print(f"  Data View(s):        {', '.join(selected_names)}")
    print(f"  Output Format:       {output_format.upper()}")
    print(f"  Include Segments:    {'Yes' if include_segments else 'No'}")
    print(f"  Include Calc Metrics: {'Yes' if include_calculated else 'No'}")
    print(f"  Include Derived:     {'Yes' if include_derived else 'No'}")
    print()

    confirm = prompt_yes_no("Generate SDR with these settings?", default=True)
    if not confirm:
        print(ConsoleColors.warning("Cancelled."))
        return None

    return WizardConfig(
        data_view_ids=selected_ids,
        output_format=output_format,
        include_segments=include_segments,
        include_calculated=include_calculated,
        include_derived=include_derived,
    )


# ==================== SAMPLE CONFIG GENERATOR ====================


def generate_sample_config(output_path: str = "config.sample.json") -> bool:
    """
    Generate a sample configuration file

    Args:
        output_path: Path to write the sample config file

    Returns:
        True if successful, False otherwise
    """
    sample_config = {
        "org_id": "YOUR_ORG_ID@AdobeOrg",
        "client_id": "your_client_id_here",
        "secret": "your_client_secret_here",
        "scopes": "your_scopes_from_developer_console",
    }

    print()
    print("=" * BANNER_WIDTH)
    print("GENERATING SAMPLE CONFIGURATION FILE")
    print("=" * BANNER_WIDTH)
    print()

    try:
        with open(output_path, "w") as f:
            json.dump(sample_config, f, indent=2)

        print(f"✓ Sample configuration file created: {output_path}")
        print()
        print("Next steps:")
        print("  1. Copy the sample file to 'config.json':")
        print(f"     cp {output_path} config.json")
        print()
        print("  2. Edit config.json with your Adobe Developer Console credentials")
        print()
        print("  3. Test your configuration:")
        print("     cja_auto_sdr --list-dataviews")
        print()
        print("=" * BANNER_WIDTH)

        return True

    except (PermissionError, OSError) as e:
        print(ConsoleColors.error(f"ERROR: Failed to create sample config: {e!s}"))
        return False


# ==================== CONFIG STATUS ====================


def show_config_status(config_file: str = "config.json", profile: str | None = None, output_json: bool = False) -> bool:
    """
    Show configuration status without connecting to API.

    Displays:
        - Active configuration source (profile, env vars, or config file)
        - Fields that are set (with masked sensitive values)
        - Quick troubleshooting information

    Args:
        config_file: Path to CJA configuration file
        profile: Optional profile name to use for credentials
        output_json: If True, output machine-readable JSON instead of human-readable text

    Returns:
        True if valid configuration found, False otherwise
    """
    # For JSON output, we'll collect data and print at the end
    if not output_json:
        print()
        print("=" * BANNER_WIDTH)
        print("CONFIGURATION STATUS")
        print("=" * BANNER_WIDTH)
        print()

    config_source = None
    config_source_type = None  # 'profile', 'environment', 'file'
    config_data = {}
    logger = logging.getLogger(__name__)
    error_message = None

    # Priority 1: Profile credentials
    if profile:
        try:
            profile_creds = load_profile_credentials(profile, logger)
            if profile_creds:
                config_source = f"Profile: {profile}"
                config_source_type = "profile"
                config_data = profile_creds
        except (ProfileNotFoundError, ProfileConfigError) as e:
            error_message = f"Profile '{profile}' - {e}"
            if output_json:
                print(json.dumps({"error": error_message, "valid": False}, indent=2))
            else:
                print(ConsoleColors.error(f"ERROR: {error_message}"))
            return False

    # Priority 2: Environment variables
    if not config_source:
        env_credentials = load_credentials_from_env()
        if env_credentials and validate_env_credentials(env_credentials, logger):
            config_source = "Environment variables"
            config_source_type = "environment"
            config_data = env_credentials

    # Priority 3: Config file
    if not config_source:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                config_source = f"Config file: {config_path.resolve()}"
                config_source_type = "file"
            except json.JSONDecodeError:
                error_message = f"{config_file} is not valid JSON"
                if output_json:
                    print(json.dumps({"error": error_message, "valid": False}, indent=2))
                else:
                    print(ConsoleColors.error(f"ERROR: {error_message}"))
                return False
            except Exception as e:
                error_message = f"Cannot read {config_file}: {e}"
                if output_json:
                    print(json.dumps({"error": error_message, "valid": False}, indent=2))
                else:
                    print(ConsoleColors.error(f"ERROR: {error_message}"))
                return False
        else:
            error_message = "No configuration found"
            if output_json:
                print(json.dumps({"error": error_message, "valid": False}, indent=2))
            else:
                print(ConsoleColors.error(f"ERROR: {error_message}"))
                print()
                print("Options:")
                print(f"  1. Create config file: {config_file}")
                print("  2. Set environment variables: ORG_ID, CLIENT_ID, SECRET, SCOPES")
                print("  3. Create a profile: cja_auto_sdr --profile-add <name>")
                print()
                print("Generate a sample config with:")
                print("  cja_auto_sdr --sample-config")
            return False

    # Define field display order and metadata
    fields = [
        ("org_id", "ORG_ID", True, False),  # (key, display_name, required, sensitive)
        ("client_id", "CLIENT_ID", True, True),
        ("secret", "SECRET", True, True),
        ("scopes", "SCOPES", False, False),
        ("sandbox", "SANDBOX", False, False),
    ]

    # Build credentials info with masked values
    all_required_set = True
    credentials_info = {}

    for key, _display_name, required, sensitive in fields:
        value = config_data.get(key, "")
        if value:
            if sensitive:
                # Mask sensitive values
                if isinstance(value, str) and len(value) > 8:
                    masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
                else:
                    masked = "****"
                display_value = masked
            else:
                display_value = value
            credentials_info[key] = {"value": display_value, "set": True, "required": required}
        else:
            credentials_info[key] = {"value": None, "set": False, "required": required}
            if required:
                all_required_set = False

    # Output based on format
    if output_json:
        result = {
            "source": config_source,
            "source_type": config_source_type,
            "profile": profile,
            "config_file": str(Path(config_file).resolve()) if config_source_type == "file" else None,
            "credentials": credentials_info,
            "valid": all_required_set,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Source: {config_source}")
        print()
        print("Credentials:")

        for key, display_name, required, _sensitive in fields:
            info = credentials_info[key]
            if info["set"]:
                status = ConsoleColors.success("✓")
                print(f"  {status} {display_name}: {info['value']}")
            else:
                if required:
                    status = ConsoleColors.error("✗")
                    print(f"  {status} {display_name}: not set (required)")
                else:
                    print(f"  - {display_name}: not set (optional)")

        print()
        if all_required_set:
            print(ConsoleColors.success("Configuration is complete."))
            print()
            print("To verify API connectivity, run:")
            print("  cja_auto_sdr --validate-config")
        else:
            print(ConsoleColors.error("Configuration is incomplete."))
            print()
            print("See documentation:")
            print("  https://github.com/brian-a-au/cja_auto_sdr/blob/main/docs/CONFIGURATION.md")

        print()

    return all_required_set


# ==================== VALIDATE CONFIG ====================


def validate_config_only(config_file: str = "config.json", profile: str | None = None) -> bool:
    """
    Validate configuration and API connectivity without processing data views.

    Tests:
        1. Profile, environment variables, or config file exists
        2. Required credentials are present
        3. CJA API connection works

    Args:
        config_file: Path to CJA configuration file
        profile: Optional profile name to use for credentials

    Returns:
        True if configuration is valid and API is reachable
    """
    print()
    print("=" * BANNER_WIDTH)
    print("CONFIGURATION VALIDATION")
    print("=" * BANNER_WIDTH)
    print()

    all_passed = True
    active_credentials = None
    credential_source = None
    logger = logging.getLogger(__name__)

    # Helper to display credentials
    def display_credentials(creds: dict[str, str], source_name: str):
        required_fields = ["org_id", "client_id", "secret"]
        optional_fields = ["scopes", "sandbox"]
        missing = []

        print()
        print("  Credential status:")
        for field in required_fields:
            if creds.get(field):
                value = creds[field]
                if field in ["secret", "client_id"]:
                    masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
                else:
                    masked = value
                print(f"    ✓ {field}: {masked}")
            else:
                print(f"    ✗ {field}: not set (required)")
                missing.append(field)

        for field in optional_fields:
            if creds.get(field):
                print(f"    ✓ {field}: {creds[field]}")
            else:
                print(f"    - {field}: not set (optional)")

        print()
        if missing:
            print(f"  ✗ Missing required fields: {', '.join(missing)}")
            return False
        else:
            print("  ✓ All required fields present")
            print(f"  → Using: {source_name}")
            return True

    # Step 1: Check credentials (priority: profile > env > config file)
    print("[1/3] Checking credentials...")

    # Priority 1: Profile
    if profile:
        print(f"  Checking profile '{profile}'...")
        try:
            profile_creds = load_profile_credentials(profile, logger)
            if profile_creds:
                print(f"  ✓ Profile '{profile}' found")
                if display_credentials(profile_creds, f"Profile: {profile}"):
                    active_credentials = profile_creds
                    credential_source = "profile"
                else:
                    all_passed = False
        except ProfileNotFoundError:
            print(f"  ✗ Profile '{profile}' not found")
            print()
            print("  Create the profile with:")
            print(f"    cja_auto_sdr --profile-add {profile}")
            all_passed = False
        except ProfileConfigError as e:
            print(f"  ✗ Profile '{profile}' has invalid configuration: {e}")
            all_passed = False

    # Priority 2: Environment variables (if no profile or profile failed)
    if not active_credentials and all_passed:
        env_credentials = load_credentials_from_env()
        if env_credentials:
            print("  ✓ Environment variables detected")
            if validate_env_credentials(env_credentials, logger):
                if display_credentials(env_credentials, "Environment variables"):
                    active_credentials = env_credentials
                    credential_source = "env"
            else:
                print("  ⚠ Environment credentials incomplete, checking config file...")
        else:
            if not profile:
                print("  - No environment variables set")

    # Priority 3: Config file (if no profile and no env)
    if not active_credentials and all_passed:
        print()
        print("[2/3] Checking configuration file...")
        config_path = Path(config_file)
        if config_path.exists():
            abs_path = config_path.resolve()
            print(f"  ✓ Config file found: {abs_path}")
            try:
                with open(config_file) as f:
                    config = json.load(f)
                print("  ✓ Config file is valid JSON")
                if display_credentials(config, f"Config file ({config_file})"):
                    active_credentials = config
                    credential_source = "file"
                else:
                    all_passed = False
            except json.JSONDecodeError as e:
                print(f"  ✗ Invalid JSON: {e!s}")
                all_passed = False
        else:
            print(f"  ✗ Config file not found: {config_file}")
            print()
            print("  To create a sample config file:")
            print("    cja_auto_sdr --sample-config")
            print()
            print("  Or set environment variables:")
            print("    export ORG_ID=your_org_id@AdobeOrg")
            print("    export CLIENT_ID=your_client_id")
            print("    export SECRET=your_client_secret")
            print()
            print("  Or create a profile:")
            print("    cja_auto_sdr --profile-add <name>")
            all_passed = False
    elif active_credentials and credential_source in ("profile", "env"):
        print()
        print(f"[2/3] Skipping config file check (using {credential_source} credentials)")

    if not all_passed:
        print()
        print("=" * BANNER_WIDTH)
        print(ConsoleColors.error("VALIDATION FAILED - Fix issues above"))
        print("=" * BANNER_WIDTH)
        return False

    # Step 3: Test API connection
    print()
    print("[3/3] Testing API connection...")
    try:
        # Configure cjapy with the active credentials
        if credential_source in ("profile", "env"):
            _config_from_env(active_credentials, logger)
        else:
            cjapy.importConfigFile(config_file)

        cja = cjapy.CJA()
        print("  ✓ CJA client initialized")

        # Test connection with API call
        available_dvs = cja.getDataViews()
        if available_dvs is not None:
            dv_count = len(available_dvs) if hasattr(available_dvs, "__len__") else 0
            print("  ✓ API connection successful")
            print(f"  ✓ Found {dv_count} accessible data view(s)")
        else:
            print("  ⚠ API returned empty response - connection may be unstable")

    except KeyboardInterrupt, SystemExit:
        print()
        print(ConsoleColors.warning("Validation cancelled."))
        raise
    except Exception as e:
        print(f"  ✗ API connection failed: {e!s}")
        all_passed = False

    # Summary
    print()
    print("=" * BANNER_WIDTH)
    if all_passed:
        print(ConsoleColors.success("VALIDATION PASSED - Configuration is valid!"))
    else:
        print(ConsoleColors.error("VALIDATION FAILED - Check errors above"))
    print("=" * BANNER_WIDTH)

    return all_passed


# ==================== STATS COMMAND ====================


def show_stats(
    data_views: list[str],
    config_file: str = "config.json",
    output_format: str = "table",
    output_file: str | None = None,
    quiet: bool = False,
    profile: str | None = None,
) -> bool:
    """
    Show quick statistics about data view(s) without generating full reports.

    Args:
        data_views: List of data view IDs to get stats for
        config_file: Path to CJA configuration file
        output_format: Output format - "table" (default), "json", or "csv"
        output_file: Optional file path to write output (or "-" for stdout)
        quiet: Suppress decorative output
        profile: Optional profile name to use for credentials

    Returns:
        True if successful, False otherwise
    """
    is_stdout = output_file in ("-", "stdout")
    is_machine_readable = output_format in ("json", "csv") or is_stdout

    if not is_machine_readable and not quiet:
        print()
        print("=" * BANNER_WIDTH)
        print("DATA VIEW STATISTICS")
        print("=" * BANNER_WIDTH)
        if profile:
            print(f"\nUsing profile: {profile}")
        print()

    try:
        success, source, _ = configure_cjapy(profile, config_file)
        if not success:
            print(ConsoleColors.error(f"ERROR: {source}"))
            return False
        cja = cjapy.CJA()

        stats_data = []

        for dv_id in data_views:
            try:
                # Get data view info
                dv_info = cja.getDataView(dv_id)
                dv_name = dv_info.get("name", "Unknown") if isinstance(dv_info, dict) else "Unknown"

                # Get metrics and dimensions
                metrics_df = cja.getMetrics(dv_id)
                dimensions_df = cja.getDimensions(dv_id)

                metrics_count = len(metrics_df) if metrics_df is not None and not metrics_df.empty else 0
                dimensions_count = len(dimensions_df) if dimensions_df is not None and not dimensions_df.empty else 0

                # Get owner info
                owner_info = dv_info.get("owner", {}) if isinstance(dv_info, dict) else {}
                owner_name = owner_info.get("name", "N/A") if isinstance(owner_info, dict) else "N/A"

                # Get description
                description = dv_info.get("description", "") if isinstance(dv_info, dict) else ""

                stats_data.append(
                    {
                        "id": dv_id,
                        "name": dv_name,
                        "owner": owner_name,
                        "metrics": metrics_count,
                        "dimensions": dimensions_count,
                        "total_components": metrics_count + dimensions_count,
                        "description": description[:100] + "..." if len(description) > 100 else description,
                    }
                )

            except Exception as e:
                stats_data.append(
                    {
                        "id": dv_id,
                        "name": "ERROR",
                        "owner": "N/A",
                        "metrics": 0,
                        "dimensions": 0,
                        "total_components": 0,
                        "description": f"Error: {e!s}",
                    }
                )

        # Output based on format
        if output_format == "json" or (is_stdout and output_format != "csv"):
            output_data = json.dumps(
                {
                    "stats": stats_data,
                    "count": len(stats_data),
                    "totals": {
                        "metrics": sum(s["metrics"] for s in stats_data),
                        "dimensions": sum(s["dimensions"] for s in stats_data),
                        "components": sum(s["total_components"] for s in stats_data),
                    },
                },
                indent=2,
            )
            if is_stdout:
                print(output_data)
            elif output_file:
                with open(output_file, "w") as f:
                    f.write(output_data)
            else:
                print(output_data)
        elif output_format == "csv":
            lines = ["id,name,owner,metrics,dimensions,total_components"]
            for item in stats_data:
                name = item["name"].replace('"', '""')
                owner = item["owner"].replace('"', '""')
                lines.append(
                    f'{item["id"]},"{name}","{owner}",{item["metrics"]},{item["dimensions"]},{item["total_components"]}'
                )
            output_data = "\n".join(lines)
            if is_stdout:
                print(output_data)
            elif output_file:
                with open(output_file, "w") as f:
                    f.write(output_data)
            else:
                print(output_data)
        else:
            # Table format
            if stats_data:
                # Calculate column widths
                max_id_width = max(len("ID"), max(len(s["id"]) for s in stats_data)) + 2
                max_name_width = min(40, max(len("Name"), max(len(s["name"]) for s in stats_data)) + 2)

                # Print header
                header = (
                    f"{'ID':<{max_id_width}} {'Name':<{max_name_width}} {'Metrics':>8} {'Dimensions':>10} {'Total':>8}"
                )
                print(header)
                print("-" * len(header))

                # Print data
                for item in stats_data:
                    name = (
                        item["name"][: max_name_width - 2] + ".."
                        if len(item["name"]) > max_name_width - 2
                        else item["name"]
                    )
                    print(
                        f"{item['id']:<{max_id_width}} {name:<{max_name_width}} {item['metrics']:>8} {item['dimensions']:>10} {item['total_components']:>8}"
                    )

                # Print totals
                print("-" * len(header))
                total_metrics = sum(s["metrics"] for s in stats_data)
                total_dims = sum(s["dimensions"] for s in stats_data)
                total_all = sum(s["total_components"] for s in stats_data)
                print(
                    f"{'TOTAL':<{max_id_width}} {'':<{max_name_width}} {total_metrics:>8} {total_dims:>10} {total_all:>8}"
                )

            print()
            print("=" * BANNER_WIDTH)

        return True

    except FileNotFoundError:
        if is_machine_readable:
            error_json = json.dumps({"error": f"Configuration file '{config_file}' not found"})
            print(error_json, file=sys.stderr if is_stdout else sys.stdout)
        else:
            print(ConsoleColors.error(f"ERROR: Configuration file '{config_file}' not found"))
        return False

    except KeyboardInterrupt, SystemExit:
        if not is_machine_readable:
            print()
            print(ConsoleColors.warning("Operation cancelled."))
        raise

    except Exception as e:
        if is_machine_readable:
            error_json = json.dumps({"error": f"Failed to get stats: {e!s}"})
            print(error_json, file=sys.stderr if is_stdout else sys.stdout)
        else:
            print(ConsoleColors.error(f"ERROR: Failed to get stats: {e!s}"))
        return False


def compare_org_reports(current: OrgReportResult, previous_path: str) -> OrgReportComparison:
    """Compare current org-report to a previous report for trending analysis (Feature 4).

    Args:
        current: Current OrgReportResult
        previous_path: Path to previous org-report JSON file

    Returns:
        OrgReportComparison with delta information
    """
    # Load previous report
    with open(previous_path, encoding="utf-8") as f:
        prev_data = json.load(f)

    # Extract data view IDs from both
    current_dv_ids = {s.data_view_id for s in current.data_view_summaries if not s.error}
    current_dv_names = {s.data_view_id: s.data_view_name for s in current.data_view_summaries}

    prev_dv_ids = set()
    prev_dv_names = {}
    for dv in prev_data.get("data_views", []):
        dv_id = dv.get("data_view_id", dv.get("id", ""))
        if dv_id:
            prev_dv_ids.add(dv_id)
            prev_dv_names[dv_id] = dv.get("data_view_name", dv.get("name", "Unknown"))

    # Compute deltas
    added_ids = list(current_dv_ids - prev_dv_ids)
    removed_ids = list(prev_dv_ids - current_dv_ids)

    # Component counts
    current_components = len(current.component_index)
    prev_components = prev_data.get("summary", {}).get("total_unique_components", 0)

    # Distribution deltas
    current_core = current.distribution.total_core
    current_isolated = current.distribution.total_isolated
    prev_dist = prev_data.get("distribution", {})
    prev_core = prev_dist.get("core", {}).get("total")
    prev_isolated = prev_dist.get("isolated", {}).get("total")

    # Backward/forward compatible totals from metrics/dimensions counts
    if prev_core is None:
        prev_core = prev_dist.get("core", {}).get("metrics_count", 0) + prev_dist.get("core", {}).get(
            "dimensions_count", 0
        )
    if prev_isolated is None:
        prev_isolated = prev_dist.get("isolated", {}).get("metrics_count", 0) + prev_dist.get("isolated", {}).get(
            "dimensions_count", 0
        )

    # High-similarity pairs comparison (normalize order for stability)
    def _pair_key(dv1: str, dv2: str) -> tuple[str, str]:
        return tuple(sorted([dv1, dv2]))

    current_high_sim = set()
    if current.similarity_pairs:
        for p in current.similarity_pairs:
            if p.jaccard_similarity >= 0.9:
                current_high_sim.add(_pair_key(p.dv1_id, p.dv2_id))

    prev_high_sim = set()
    for p in prev_data.get("similarity_pairs", []):
        if p.get("jaccard_similarity", 0) >= 0.9:
            if "dv1_id" in p or "dv2_id" in p:
                dv1 = p.get("dv1_id", "")
                dv2 = p.get("dv2_id", "")
            else:
                dv1 = p.get("data_view_1", {}).get("id", "")
                dv2 = p.get("data_view_2", {}).get("id", "")
            if dv1 and dv2:
                prev_high_sim.add(_pair_key(dv1, dv2))

    new_pairs = current_high_sim - prev_high_sim
    resolved_pairs = prev_high_sim - current_high_sim

    return OrgReportComparison(
        current_timestamp=current.timestamp,
        previous_timestamp=prev_data.get("generated_at", prev_data.get("timestamp", "unknown")),
        data_views_added=added_ids,
        data_views_removed=removed_ids,
        data_views_added_names=[current_dv_names.get(dv_id, "Unknown") for dv_id in added_ids],
        data_views_removed_names=[prev_dv_names.get(dv_id, "Unknown") for dv_id in removed_ids],
        components_added=max(0, current_components - prev_components),
        components_removed=max(0, prev_components - current_components),
        core_delta=current_core - prev_core,
        isolated_delta=current_isolated - prev_isolated,
        new_high_similarity_pairs=[{"dv1_id": p[0], "dv2_id": p[1]} for p in new_pairs],
        resolved_pairs=[{"dv1_id": p[0], "dv2_id": p[1]} for p in resolved_pairs],
        summary={
            "data_views_delta": len(current_dv_ids) - len(prev_dv_ids),
            "components_delta": current_components - prev_components,
            "core_delta": current_core - prev_core,
            "isolated_delta": current_isolated - prev_isolated,
            "new_duplicates": len(new_pairs),
            "resolved_duplicates": len(resolved_pairs),
        },
    )


def _render_distribution_bar(count: int, total: int, width: int = 30) -> str:
    """Render ASCII progress bar for distribution visualization.

    Args:
        count: Number of items in this bucket
        total: Total items across all buckets
        width: Width of the bar in characters

    Returns:
        ASCII bar string like "████████░░░░░░░░ 45%"
    """
    if total == 0:
        return "░" * width + "  0%"

    pct = count / total
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct * 100:>3.0f}%"


def write_org_report_console(result: OrgReportResult, config: OrgReportConfig, quiet: bool = False) -> None:
    """Write org report to console with ASCII distribution bars.

    Args:
        result: OrgReportResult from analysis
        config: OrgReportConfig used for analysis
        quiet: Suppress decorative output
    """
    if quiet:
        return

    total_dvs = result.successful_data_views
    print()
    print("=" * 110)
    title = f"ORG-WIDE COMPONENT ANALYSIS REPORT: {result.org_id}"
    if result.is_sampled:
        title += " [SAMPLED]"
    print(title)
    print("=" * 110)
    print(f"Generated: {result.timestamp}")
    if result.is_sampled:
        print(
            f"Data Views Analyzed: {result.successful_data_views} / {result.total_data_views} (sampled from {result.total_available_data_views})"
        )
    else:
        print(f"Data Views Analyzed: {result.successful_data_views} / {result.total_data_views}")
    print(f"Analysis Duration: {result.duration:.2f}s")
    print()

    # Data Views Summary Table
    print("-" * 110)
    print("DATA VIEWS")
    print("-" * 110)
    print(f"{'Name':<50} {'ID':<30} {'Metrics':>8} {'Dimensions':>10} {'Status':<8}")
    print("-" * 110)

    for dv in sorted(result.data_view_summaries, key=lambda x: x.data_view_name):
        name = dv.data_view_name[:48] + ".." if len(dv.data_view_name) > 50 else dv.data_view_name
        if dv.error:
            print(f"{name:<50} {dv.data_view_id:<30} {'ERROR':>8} {'':>10} {dv.status:<8}")
        else:
            print(f"{name:<50} {dv.data_view_id:<30} {dv.metric_count:>8} {dv.dimension_count:>10} {dv.status:<8}")

    print()

    # Component Summary
    print("-" * 110)
    print("COMPONENT SUMMARY")
    print("-" * 110)

    total_metrics = result.total_unique_metrics
    total_dims = result.total_unique_dimensions
    total_all = result.total_unique_components

    # Calculate total aggregates (non-unique counts across all data views)
    total_metrics_aggregate = sum(dv.metric_count for dv in result.data_view_summaries if not dv.error)
    total_dimensions_aggregate = sum(dv.dimension_count for dv in result.data_view_summaries if not dv.error)
    total_components_aggregate = sum(dv.total_components for dv in result.data_view_summaries if not dv.error)
    total_derived_metrics = sum(dv.derived_metric_count for dv in result.data_view_summaries if not dv.error)
    total_derived_dimensions = sum(dv.derived_dimension_count for dv in result.data_view_summaries if not dv.error)
    total_derived_fields = total_derived_metrics + total_derived_dimensions

    dist = result.distribution
    # Build correct label for core threshold: "50% DVs" for percentage, ">=5 DVs" for absolute count
    if config.core_min_count is None:
        core_threshold_label = f"{int(config.core_threshold * 100)}%"
    else:
        core_threshold_label = f">={config.core_min_count}"

    print(f"{'':30} {'Metrics':>12} {'Dimensions':>12} {'Total':>10}")
    print(f"{'Total unique components':<30} {total_metrics:>12} {total_dims:>12} {total_all:>10}")
    print(
        f"{'Total (non-unique)':<30} {total_metrics_aggregate:>12} {total_dimensions_aggregate:>12} {total_components_aggregate:>10}"
    )
    print(
        f"{'Derived fields (non-unique)':<30} {total_derived_metrics:>12} {total_derived_dimensions:>12} {total_derived_fields:>10}"
    )
    # Add "+" suffix only for percentage labels; absolute labels already have ">="
    core_label_suffix = " DVs" if config.core_min_count is not None else "+ DVs"
    print(
        f"{'Core (in ' + core_threshold_label + core_label_suffix + ')':<30} {len(dist.core_metrics):>12} {len(dist.core_dimensions):>12} {dist.total_core:>10}"
    )
    print(
        f"{'Common (in 25-49% DVs)':<30} {len(dist.common_metrics):>12} {len(dist.common_dimensions):>12} {dist.total_common:>10}"
    )
    print(
        f"{'Limited (in 2+ DVs)':<30} {len(dist.limited_metrics):>12} {len(dist.limited_dimensions):>12} {dist.total_limited:>10}"
    )
    print(
        f"{'Isolated (in 1 DV only)':<30} {len(dist.isolated_metrics):>12} {len(dist.isolated_dimensions):>12} {dist.total_isolated:>10}"
    )
    print()

    # Distribution Visualization
    print("-" * 110)
    print("DISTRIBUTION")
    print("-" * 110)

    print("Metrics by data view coverage:")
    print(f"  Core:     {_render_distribution_bar(len(dist.core_metrics), total_metrics)} ({len(dist.core_metrics)})")
    print(
        f"  Common:   {_render_distribution_bar(len(dist.common_metrics), total_metrics)} ({len(dist.common_metrics)})"
    )
    print(
        f"  Limited:  {_render_distribution_bar(len(dist.limited_metrics), total_metrics)} ({len(dist.limited_metrics)})"
    )
    print(
        f"  Isolated: {_render_distribution_bar(len(dist.isolated_metrics), total_metrics)} ({len(dist.isolated_metrics)})"
    )
    print()

    print("Dimensions by data view coverage:")
    print(
        f"  Core:     {_render_distribution_bar(len(dist.core_dimensions), total_dims)} ({len(dist.core_dimensions)})"
    )
    print(
        f"  Common:   {_render_distribution_bar(len(dist.common_dimensions), total_dims)} ({len(dist.common_dimensions)})"
    )
    print(
        f"  Limited:  {_render_distribution_bar(len(dist.limited_dimensions), total_dims)} ({len(dist.limited_dimensions)})"
    )
    print(
        f"  Isolated: {_render_distribution_bar(len(dist.isolated_dimensions), total_dims)} ({len(dist.isolated_dimensions)})"
    )
    print()

    # Core Components (if not summary only)
    if not config.summary_only and dist.total_core > 0:
        print("-" * 110)
        print(f"CORE COMPONENTS (in {core_threshold_label}{core_label_suffix})")
        print("-" * 110)

        if dist.core_metrics:
            print("\nCore Metrics:")
            for comp_id in dist.core_metrics[:15]:  # Limit to 15
                info = result.component_index.get(comp_id)
                if info:
                    if info.name:
                        display = f"{info.name} ({comp_id})"
                        display = display[:55] + ".." if len(display) > 57 else display
                        print(f"  {display:<57} {info.presence_count}/{total_dvs} DVs")
                    else:
                        print(f"  {comp_id:<57} {info.presence_count}/{total_dvs} DVs")
            if len(dist.core_metrics) > 15:
                print(f"  ... and {len(dist.core_metrics) - 15} more")

        if dist.core_dimensions:
            print("\nCore Dimensions:")
            for comp_id in dist.core_dimensions[:15]:
                info = result.component_index.get(comp_id)
                if info:
                    if info.name:
                        display = f"{info.name} ({comp_id})"
                        display = display[:55] + ".." if len(display) > 57 else display
                        print(f"  {display:<57} {info.presence_count}/{total_dvs} DVs")
                    else:
                        print(f"  {comp_id:<57} {info.presence_count}/{total_dvs} DVs")
            if len(dist.core_dimensions) > 15:
                print(f"  ... and {len(dist.core_dimensions) - 15} more")
        print()

    # Similarity Matrix (if computed)
    if result.similarity_pairs:
        print("-" * 110)
        print("HIGH OVERLAP PAIRS")
        print("-" * 110)
        effective_threshold = min(config.overlap_threshold, 0.9)
        threshold_note = ""
        if config.overlap_threshold > 0.9:
            threshold_note = f" (configured {config.overlap_threshold * 100:.0f}%, capped at 90% for governance checks)"
        print(f"Pairs with >= {effective_threshold * 100:.0f}% Jaccard similarity{threshold_note}:")
        print()

        for pair in result.similarity_pairs[:10]:  # Limit to top 10
            name1 = pair.dv1_name[:25] + ".." if len(pair.dv1_name) > 27 else pair.dv1_name
            name2 = pair.dv2_name[:25] + ".." if len(pair.dv2_name) > 27 else pair.dv2_name
            print(f"  {name1:<27} <-> {name2:<27} {pair.jaccard_similarity * 100:>5.1f}%")
            print(f"  {'':27}     {'':27} ({pair.shared_count} shared)")

        if len(result.similarity_pairs) > 10:
            print(f"\n  ... and {len(result.similarity_pairs) - 10} more pairs")

        # Show drift details if enabled
        if config.include_drift:
            has_drift = any(pair.only_in_dv1 or pair.only_in_dv2 for pair in result.similarity_pairs[:5])
            if has_drift:
                print()
                print("Drift Details (top pairs):")
                for pair in result.similarity_pairs[:5]:
                    if pair.only_in_dv1 or pair.only_in_dv2:
                        print(f"\n  {pair.dv1_name} <-> {pair.dv2_name}:")
                        if pair.only_in_dv1:
                            print(f"    Only in {pair.dv1_name[:20]}: {len(pair.only_in_dv1)} components")
                            for comp_id in pair.only_in_dv1[:3]:
                                name = pair.only_in_dv1_names.get(comp_id, "") if pair.only_in_dv1_names else ""
                                display = f"{name} ({comp_id})" if name else comp_id
                                print(f"      - {display[:50]}")
                            if len(pair.only_in_dv1) > 3:
                                print(f"      ... and {len(pair.only_in_dv1) - 3} more")
                        if pair.only_in_dv2:
                            print(f"    Only in {pair.dv2_name[:20]}: {len(pair.only_in_dv2)} components")
                            for comp_id in pair.only_in_dv2[:3]:
                                name = pair.only_in_dv2_names.get(comp_id, "") if pair.only_in_dv2_names else ""
                                display = f"{name} ({comp_id})" if name else comp_id
                                print(f"      - {display[:50]}")
                            if len(pair.only_in_dv2) > 3:
                                print(f"      ... and {len(pair.only_in_dv2) - 3} more")
        print()

    # Component Type Breakdown (if enabled)
    if config.include_component_types and not config.summary_only:
        # Aggregate type counts
        total_standard_metrics = sum(s.standard_metric_count for s in result.data_view_summaries if not s.error)
        total_derived_metrics = sum(s.derived_metric_count for s in result.data_view_summaries if not s.error)
        total_standard_dims = sum(s.standard_dimension_count for s in result.data_view_summaries if not s.error)
        total_derived_dims = sum(s.derived_dimension_count for s in result.data_view_summaries if not s.error)

        if total_standard_metrics + total_derived_metrics > 0:
            print("-" * 110)
            print("COMPONENT TYPES (aggregate counts across DVs, standard vs derived field breakdown)")
            print("-" * 110)
            total_metrics = total_standard_metrics + total_derived_metrics
            total_dims = total_standard_dims + total_derived_dims
            # Calculate percentages
            std_metric_pct = (total_standard_metrics / total_metrics * 100) if total_metrics > 0 else 0
            der_metric_pct = (total_derived_metrics / total_metrics * 100) if total_metrics > 0 else 0
            std_dim_pct = (total_standard_dims / total_dims * 100) if total_dims > 0 else 0
            der_dim_pct = (total_derived_dims / total_dims * 100) if total_dims > 0 else 0
            print(f"{'':18} {'Total':>10} {'Standard':>12} {'% Total':>8} {'Derived':>10} {'% Total':>8}")
            print(
                f"{'Metrics':<18} {total_metrics:>10} {total_standard_metrics:>12} {std_metric_pct:>7.1f}% {total_derived_metrics:>10} {der_metric_pct:>7.1f}%"
            )
            print(
                f"{'Dimensions':<18} {total_dims:>10} {total_standard_dims:>12} {std_dim_pct:>7.1f}% {total_derived_dims:>10} {der_dim_pct:>7.1f}%"
            )
            print()

    # Clusters (if enabled)
    if result.clusters:
        print("-" * 110)
        print("DATA VIEW CLUSTERS")
        print("-" * 110)
        print(f"Found {len(result.clusters)} clusters:")
        print()
        for cluster in result.clusters[:10]:
            name = cluster.cluster_name or f"Cluster {cluster.cluster_id}"
            print(f"  [{cluster.cluster_id}] {name} ({cluster.size} DVs, cohesion: {cluster.cohesion_score:.0%})")
            for dv_name in cluster.data_view_names[:3]:
                print(f"      - {dv_name[:50]}")
            if cluster.size > 3:
                print(f"      ... and {cluster.size - 3} more")
        if len(result.clusters) > 10:
            print(f"\n  ... and {len(result.clusters) - 10} more clusters")
        print()

    # Governance Violations (Feature 1)
    if result.governance_violations:
        print("-" * 110)
        print("GOVERNANCE VIOLATIONS")
        print("-" * 110)
        for violation in result.governance_violations:
            print(f"\n[!] {violation.get('message', '')}")
            print(f"    Threshold: {violation.get('threshold')}, Actual: {violation.get('actual')}")
        print()

    # Owner Summary (Feature 5)
    if result.owner_summary:
        print("-" * 110)
        print("OWNER SUMMARY")
        print("-" * 110)
        owner_data = result.owner_summary.get("by_owner", {})
        sorted_owners = result.owner_summary.get("owners_sorted_by_dv_count", [])
        print(f"{'Owner':<30} {'DVs':>6} {'Metrics':>10} {'Dimensions':>12} {'Avg/DV':>8}")
        print("-" * 110)
        for owner in sorted_owners[:15]:
            stats = owner_data.get(owner, {})
            print(
                f"{owner[:30]:<30} {stats.get('data_view_count', 0):>6} "
                f"{stats.get('total_metrics', 0):>10} {stats.get('total_dimensions', 0):>12} "
                f"{stats.get('avg_components_per_dv', 0):>8.1f}"
            )
        if len(sorted_owners) > 15:
            print(f"  ... and {len(sorted_owners) - 15} more owners")
        print()

    # Naming Audit (Feature 3)
    if result.naming_audit:
        print("-" * 110)
        print("NAMING AUDIT")
        print("-" * 110)
        audit = result.naming_audit
        styles = audit.get("case_styles", {})
        print("Case styles distribution:")
        for style, count in sorted(styles.items(), key=lambda x: -x[1]):
            pct = count / audit.get("total_components", 1) * 100
            print(f"  {style:<15} {count:>6} ({pct:>5.1f}%)")
        if audit.get("recommendations"):
            print("\nNaming Recommendations:")
            for rec in audit["recommendations"]:
                print(f"  [{rec.get('severity', 'info')}] {rec.get('message', '')}")
        print()

    # Stale Components (Feature 6)
    if result.stale_components:
        print("-" * 110)
        print("STALE COMPONENTS")
        print("-" * 110)
        print(f"Found {len(result.stale_components)} components with stale naming patterns:")
        print()
        # Group by pattern type
        by_pattern: dict[str, list] = {}
        for comp in result.stale_components:
            pattern = comp.get("pattern", "unknown")
            if pattern not in by_pattern:
                by_pattern[pattern] = []
            by_pattern[pattern].append(comp)
        for pattern, comps in by_pattern.items():
            print(f"  {pattern} ({len(comps)} components):")
            for comp in comps[:5]:
                name = comp.get("name", comp.get("component_id", ""))[:50]
                print(f"    - {name}")
            if len(comps) > 5:
                print(f"    ... and {len(comps) - 5} more")
        print()

    # Recommendations
    if result.recommendations:
        print("-" * 110)
        print("RECOMMENDATIONS")
        print("-" * 110)

        for _i, rec in enumerate(result.recommendations, 1):
            severity_icon = {"high": "!", "medium": "?", "low": "i"}.get(rec.get("severity", "low"), "·")
            print(f"\n[{severity_icon}] {rec.get('reason', 'No details')}")

            if rec.get("data_view"):
                print(f"    Data View: {rec.get('data_view_name', '')} ({rec.get('data_view')})")
            if rec.get("data_view_1"):
                print(f"    Pair: {rec.get('data_view_1_name', '')} <-> {rec.get('data_view_2_name', '')}")

        print()


def write_org_report_stats_only(result: OrgReportResult, quiet: bool = False) -> None:
    """Write minimal org-report stats to console (Feature 2: --org-stats mode).

    Args:
        result: OrgReportResult from analysis
        quiet: Suppress output
    """
    if quiet:
        return

    print()
    print("=" * BANNER_WIDTH)
    print(f"ORG STATS: {result.org_id}")
    print("=" * BANNER_WIDTH)
    print(f"Data Views: {result.successful_data_views} analyzed")
    print(f"Components: {result.total_unique_components} unique")
    print(f"  Metrics:    {result.total_unique_metrics}")
    print(f"  Dimensions: {result.total_unique_dimensions}")
    print("Distribution:")
    dist = result.distribution
    total = result.total_unique_components or 1
    print(f"  Core:     {dist.total_core:>6} ({dist.total_core / total * 100:>5.1f}%)")
    print(f"  Common:   {dist.total_common:>6} ({dist.total_common / total * 100:>5.1f}%)")
    print(f"  Limited:  {dist.total_limited:>6} ({dist.total_limited / total * 100:>5.1f}%)")
    print(f"  Isolated: {dist.total_isolated:>6} ({dist.total_isolated / total * 100:>5.1f}%)")
    print(f"Duration: {result.duration:.2f}s")
    print("=" * BANNER_WIDTH)
    print()


def write_org_report_comparison_console(comparison: OrgReportComparison, quiet: bool = False) -> None:
    """Write org-report comparison to console with trending arrows (Feature 4).

    Args:
        comparison: OrgReportComparison from compare_org_reports()
        quiet: Suppress output
    """
    if quiet:
        return

    def trend_arrow(delta: int) -> str:
        if delta > 0:
            return f"↑{delta}"
        elif delta < 0:
            return f"↓{abs(delta)}"
        return "→0"

    print()
    print("=" * 70)
    print("ORG REPORT COMPARISON (TRENDING)")
    print("=" * 70)
    print(f"Previous: {comparison.previous_timestamp}")
    print(f"Current:  {comparison.current_timestamp}")
    print()

    summary = comparison.summary
    print("-" * 70)
    print("CHANGES")
    print("-" * 70)
    print(f"Data Views:  {trend_arrow(summary.get('data_views_delta', 0))}")
    print(f"Components:  {trend_arrow(summary.get('components_delta', 0))}")
    print(f"Core:        {trend_arrow(summary.get('core_delta', 0))}")
    print(f"Isolated:    {trend_arrow(summary.get('isolated_delta', 0))}")
    print()

    if comparison.data_views_added:
        print(f"Data Views Added ({len(comparison.data_views_added)}):")
        for _i, dv_name in enumerate(comparison.data_views_added_names[:5]):
            print(f"  + {dv_name}")
        if len(comparison.data_views_added) > 5:
            print(f"  ... and {len(comparison.data_views_added) - 5} more")

    if comparison.data_views_removed:
        print(f"Data Views Removed ({len(comparison.data_views_removed)}):")
        for dv_name in comparison.data_views_removed_names[:5]:
            print(f"  - {dv_name}")
        if len(comparison.data_views_removed) > 5:
            print(f"  ... and {len(comparison.data_views_removed) - 5} more")

    if comparison.new_high_similarity_pairs:
        print(f"\nNew High-Similarity Pairs ({len(comparison.new_high_similarity_pairs)}):")
        for pair in comparison.new_high_similarity_pairs[:3]:
            print(f"  ! {pair.get('dv1_id', '')} <-> {pair.get('dv2_id', '')}")

    if comparison.resolved_pairs:
        print(f"\nResolved High-Similarity Pairs ({len(comparison.resolved_pairs)}):")
        for pair in comparison.resolved_pairs[:3]:
            print(f"  ✓ {pair.get('dv1_id', '')} <-> {pair.get('dv2_id', '')}")

    print()
    print("=" * 70)
    print()


def build_org_report_json_data(result: OrgReportResult) -> dict[str, Any]:
    """Build org report JSON payload."""
    effective_overlap_threshold = min(result.parameters.overlap_threshold, 0.9)
    return {
        "report_type": "org_analysis",
        "version": "1.0",
        "generated_at": result.timestamp,
        "org_id": result.org_id,
        "parameters": {
            "filter_pattern": result.parameters.filter_pattern,
            "exclude_pattern": result.parameters.exclude_pattern,
            "limit": result.parameters.limit,
            "core_threshold": result.parameters.core_threshold,
            "core_min_count": result.parameters.core_min_count,
            "overlap_threshold": result.parameters.overlap_threshold,
            "overlap_threshold_effective": effective_overlap_threshold,
            "include_component_types": result.parameters.include_component_types,
            "include_metadata": result.parameters.include_metadata,
            "include_drift": result.parameters.include_drift,
            "sample_size": result.parameters.sample_size,
            "sample_seed": result.parameters.sample_seed,
            "enable_clustering": result.parameters.enable_clustering,
        },
        "summary": {
            "data_views_total": result.total_data_views,
            "data_views_analyzed": result.successful_data_views,
            "total_available_data_views": result.total_available_data_views,
            "is_sampled": result.is_sampled,
            "total_unique_metrics": result.total_unique_metrics,
            "total_unique_dimensions": result.total_unique_dimensions,
            "total_unique_components": result.total_unique_components,
            "total_metrics_non_unique": sum(dv.metric_count for dv in result.data_view_summaries if not dv.error),
            "total_dimensions_non_unique": sum(dv.dimension_count for dv in result.data_view_summaries if not dv.error),
            "total_components_non_unique": sum(
                dv.total_components for dv in result.data_view_summaries if not dv.error
            ),
            "derived_metrics_non_unique": sum(
                dv.derived_metric_count for dv in result.data_view_summaries if not dv.error
            ),
            "derived_dimensions_non_unique": sum(
                dv.derived_dimension_count for dv in result.data_view_summaries if not dv.error
            ),
            "total_derived_fields_non_unique": sum(
                dv.derived_metric_count + dv.derived_dimension_count
                for dv in result.data_view_summaries
                if not dv.error
            ),
            "analysis_duration_seconds": round(result.duration, 2),
        },
        "distribution": {
            "core": {
                "metrics_count": len(result.distribution.core_metrics),
                "dimensions_count": len(result.distribution.core_dimensions),
                "metrics": result.distribution.core_metrics,
                "dimensions": result.distribution.core_dimensions,
            },
            "common": {
                "metrics_count": len(result.distribution.common_metrics),
                "dimensions_count": len(result.distribution.common_dimensions),
                "metrics": result.distribution.common_metrics,
                "dimensions": result.distribution.common_dimensions,
            },
            "limited": {
                "metrics_count": len(result.distribution.limited_metrics),
                "dimensions_count": len(result.distribution.limited_dimensions),
                "metrics": result.distribution.limited_metrics,
                "dimensions": result.distribution.limited_dimensions,
            },
            "isolated": {
                "metrics_count": len(result.distribution.isolated_metrics),
                "dimensions_count": len(result.distribution.isolated_dimensions),
                "metrics": result.distribution.isolated_metrics,
                "dimensions": result.distribution.isolated_dimensions,
            },
        },
        "data_views": [
            {
                "id": dv.data_view_id,
                "name": dv.data_view_name,
                "metrics_count": dv.metric_count,
                "dimensions_count": dv.dimension_count,
                "total_components": dv.total_components,
                "status": dv.status,
                "error": dv.error,
                # Component type breakdown
                "standard_metrics": dv.standard_metric_count,
                "derived_metrics": dv.derived_metric_count,
                "standard_dimensions": dv.standard_dimension_count,
                "derived_dimensions": dv.derived_dimension_count,
                # Metadata
                "owner": dv.owner,
                "owner_id": dv.owner_id,
                "created": dv.created,
                "modified": dv.modified,
                "description": dv.description,
                "has_description": dv.has_description,
            }
            for dv in result.data_view_summaries
        ],
        "component_index": {
            comp_id: {
                "type": info.component_type,
                "name": info.name,
                "data_view_count": info.presence_count,
                "data_views": list(info.data_views),
            }
            for comp_id, info in result.component_index.items()
        },
        "similarity_pairs": [
            {
                "data_view_1": {"id": pair.dv1_id, "name": pair.dv1_name},
                "data_view_2": {"id": pair.dv2_id, "name": pair.dv2_name},
                "jaccard_similarity": pair.jaccard_similarity,
                "shared_components": pair.shared_count,
                "union_size": pair.union_count,
                # Drift detection
                "only_in_dv1": pair.only_in_dv1,
                "only_in_dv2": pair.only_in_dv2,
                "only_in_dv1_names": pair.only_in_dv1_names,
                "only_in_dv2_names": pair.only_in_dv2_names,
            }
            for pair in (result.similarity_pairs or [])
        ],
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "data_view_ids": cluster.data_view_ids,
                "data_view_names": cluster.data_view_names,
                "size": cluster.size,
                "cohesion_score": cluster.cohesion_score,
            }
            for cluster in (result.clusters or [])
        ]
        if result.clusters
        else None,
        "recommendations": result.recommendations,
        # New features
        "governance_violations": result.governance_violations,
        "thresholds_exceeded": result.thresholds_exceeded,
        "naming_audit": result.naming_audit,
        "owner_summary": result.owner_summary,
        "stale_components": result.stale_components,
    }


def write_org_report_json(
    result: OrgReportResult, output_path: Path | None, output_dir: str, logger: logging.Logger
) -> str:
    """Write org report as structured JSON.

    Args:
        result: OrgReportResult from analysis
        output_path: Optional specific output path
        output_dir: Output directory if no path specified
        logger: Logger instance

    Returns:
        Path to created JSON file
    """
    if output_path:
        file_path = output_path if str(output_path).endswith(".json") else Path(f"{output_path}.json")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_dir) / f"org_report_{result.org_id}_{timestamp}.json"

    json_data = build_org_report_json_data(result)

    # Write JSON
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON report written to {file_path}")
    return str(file_path)


def write_org_report_excel(
    result: OrgReportResult, output_path: Path | None, output_dir: str, logger: logging.Logger
) -> str:
    """Write org report as multi-sheet Excel workbook.

    Sheets:
    - Summary: Overview statistics
    - Data Views: List of all analyzed data views
    - Core Components: Components in threshold% of DVs
    - Distribution: Component distribution breakdown
    - Similarity: High-overlap pairs
    - Recommendations: Actionable items

    Args:
        result: OrgReportResult from analysis
        output_path: Optional specific output path
        output_dir: Output directory if no path specified
        logger: Logger instance

    Returns:
        Path to created Excel file
    """
    if output_path:
        file_path = output_path if str(output_path).endswith(".xlsx") else Path(f"{output_path}.xlsx")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_dir) / f"org_report_{result.org_id}_{timestamp}.xlsx"

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        # Sheet 1: Summary
        # Calculate total aggregates (non-unique counts across all data views)
        total_metrics_aggregate = sum(dv.metric_count for dv in result.data_view_summaries if not dv.error)
        total_dimensions_aggregate = sum(dv.dimension_count for dv in result.data_view_summaries if not dv.error)
        total_components_aggregate = sum(dv.total_components for dv in result.data_view_summaries if not dv.error)
        total_derived_metrics = sum(dv.derived_metric_count for dv in result.data_view_summaries if not dv.error)
        total_derived_dimensions = sum(dv.derived_dimension_count for dv in result.data_view_summaries if not dv.error)
        total_derived_fields = total_derived_metrics + total_derived_dimensions
        effective_overlap_threshold = min(result.parameters.overlap_threshold, 0.9)

        metrics = [
            "Organization ID",
            "Report Generated",
            "Data Views Total",
            "Data Views Analyzed",
            "Total Unique Metrics",
            "Total Unique Dimensions",
            "Total Unique Components",
            "Total Metrics (Non-Unique)",
            "Total Dimensions (Non-Unique)",
            "Total Components (Non-Unique)",
            "Derived Metrics (Non-Unique)",
            "Derived Dimensions (Non-Unique)",
            "Total Derived Fields (Non-Unique)",
            "Core Components",
            "Common Components",
            "Limited Components",
            "Isolated Components",
            "Overlap Threshold (Configured)",
            "Overlap Threshold (Effective)",
            "Analysis Duration (seconds)",
        ]
        values = [
            result.org_id,
            result.timestamp,
            result.total_data_views,
            result.successful_data_views,
            result.total_unique_metrics,
            result.total_unique_dimensions,
            result.total_unique_components,
            total_metrics_aggregate,
            total_dimensions_aggregate,
            total_components_aggregate,
            total_derived_metrics,
            total_derived_dimensions,
            total_derived_fields,
            result.distribution.total_core,
            result.distribution.total_common,
            result.distribution.total_limited,
            result.distribution.total_isolated,
            result.parameters.overlap_threshold,
            effective_overlap_threshold,
            round(result.duration, 2),
        ]
        # Add sampling info
        if result.is_sampled:
            metrics.extend(["Is Sampled", "Total Available DVs", "Sample Seed"])
            values.extend(["Yes", result.total_available_data_views, result.parameters.sample_seed])
        # Add clustering info
        if result.clusters:
            metrics.append("Cluster Count")
            values.append(len(result.clusters))
        summary_data = {"Metric": metrics, "Value": values}
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        worksheet = writer.sheets["Summary"]
        worksheet.set_column("A:A", 30)
        worksheet.set_column("B:B", 25)

        # Sheet 2: Data Views
        dv_data = []
        for dv in result.data_view_summaries:
            row = {
                "ID": dv.data_view_id,
                "Name": dv.data_view_name,
                "Metrics": dv.metric_count,
                "Dimensions": dv.dimension_count,
                "Total": dv.total_components,
                "Status": dv.status,
                "Error": dv.error or "",
            }
            # Add component type columns if enabled
            if result.parameters.include_component_types:
                row["Standard Metrics"] = dv.standard_metric_count
                row["Derived Metrics"] = dv.derived_metric_count
                row["Standard Dimensions"] = dv.standard_dimension_count
                row["Derived Dimensions"] = dv.derived_dimension_count
            # Add metadata columns if enabled
            if result.parameters.include_metadata:
                row["Owner"] = dv.owner or ""
                row["Created"] = dv.created or ""
                row["Modified"] = dv.modified or ""
                row["Has Description"] = "Yes" if dv.has_description else "No"
            dv_data.append(row)
        dv_df = pd.DataFrame(dv_data)
        dv_df.to_excel(writer, sheet_name="Data Views", index=False)

        worksheet = writer.sheets["Data Views"]
        worksheet.set_column("A:A", 20)
        worksheet.set_column("B:B", 40)
        worksheet.set_column("C:G", 12)
        if result.parameters.include_component_types:
            worksheet.set_column("H:K", 18)  # 4 columns: Standard/Derived Metrics/Dimensions
        if result.parameters.include_metadata:
            worksheet.set_column("L:O", 18)

        # Sheet 3: Core Components
        core_data = []
        for comp_id in result.distribution.core_metrics:
            info = result.component_index.get(comp_id)
            if info:
                core_data.append(
                    {
                        "Component ID": comp_id,
                        "Type": "Metric",
                        "Name": info.name or "",
                        "Data View Count": info.presence_count,
                        "Coverage %": info.presence_count / result.successful_data_views
                        if result.successful_data_views > 0
                        else 0,
                    }
                )
        for comp_id in result.distribution.core_dimensions:
            info = result.component_index.get(comp_id)
            if info:
                core_data.append(
                    {
                        "Component ID": comp_id,
                        "Type": "Dimension",
                        "Name": info.name or "",
                        "Data View Count": info.presence_count,
                        "Coverage %": info.presence_count / result.successful_data_views
                        if result.successful_data_views > 0
                        else 0,
                    }
                )

        if core_data:
            core_df = pd.DataFrame(core_data)
            core_df.to_excel(writer, sheet_name="Core Components", index=False)
            worksheet = writer.sheets["Core Components"]
            worksheet.set_column("A:A", 40)
            worksheet.set_column("B:B", 12)
            worksheet.set_column("C:C", 30)
            worksheet.set_column("D:E", 15)

        # Sheet 4: Isolated by Data View
        isolated_data = []
        for dv in result.data_view_summaries:
            if dv.error:
                continue
            isolated_metrics = [
                c
                for c in result.distribution.isolated_metrics
                if dv.data_view_id in result.component_index.get(c, ComponentInfo("", "")).data_views
            ]
            isolated_dims = [
                c
                for c in result.distribution.isolated_dimensions
                if dv.data_view_id in result.component_index.get(c, ComponentInfo("", "")).data_views
            ]
            if isolated_metrics or isolated_dims:
                isolated_data.append(
                    {
                        "Data View ID": dv.data_view_id,
                        "Data View Name": dv.data_view_name,
                        "Isolated Metrics": len(isolated_metrics),
                        "Isolated Dimensions": len(isolated_dims),
                        "Total Isolated": len(isolated_metrics) + len(isolated_dims),
                    }
                )

        if isolated_data:
            isolated_df = pd.DataFrame(isolated_data)
            isolated_df = isolated_df.sort_values("Total Isolated", ascending=False)
            isolated_df.to_excel(writer, sheet_name="Isolated by DV", index=False)
            worksheet = writer.sheets["Isolated by DV"]
            worksheet.set_column("A:A", 20)
            worksheet.set_column("B:B", 40)
            worksheet.set_column("C:E", 18)

        # Sheet 5: Similarity Matrix
        if result.similarity_pairs:
            sim_data = []
            for pair in result.similarity_pairs:
                row = {
                    "Data View 1 ID": pair.dv1_id,
                    "Data View 1 Name": pair.dv1_name,
                    "Data View 2 ID": pair.dv2_id,
                    "Data View 2 Name": pair.dv2_name,
                    "Similarity %": pair.jaccard_similarity,
                    "Shared Components": pair.shared_count,
                    "Union Size": pair.union_count,
                }
                if result.parameters.include_drift:
                    row["Only in DV1"] = len(pair.only_in_dv1)
                    row["Only in DV2"] = len(pair.only_in_dv2)
                    row["Drift Total"] = len(pair.only_in_dv1) + len(pair.only_in_dv2)
                sim_data.append(row)
            sim_df = pd.DataFrame(sim_data)
            sim_df.to_excel(writer, sheet_name="Similarity", index=False)
            worksheet = writer.sheets["Similarity"]
            worksheet.set_column("A:A", 20)
            worksheet.set_column("B:B", 35)
            worksheet.set_column("C:C", 20)
            worksheet.set_column("D:D", 35)
            worksheet.set_column("E:J", 15)

        # Sheet 5b: Drift Details (if enabled)
        if result.parameters.include_drift and result.similarity_pairs:
            drift_data = []
            for pair in result.similarity_pairs:
                if pair.only_in_dv1 or pair.only_in_dv2:
                    for comp_id in pair.only_in_dv1:
                        name = pair.only_in_dv1_names.get(comp_id, "") if pair.only_in_dv1_names else ""
                        drift_data.append(
                            {
                                "DV1 ID": pair.dv1_id,
                                "DV1 Name": pair.dv1_name,
                                "DV2 ID": pair.dv2_id,
                                "DV2 Name": pair.dv2_name,
                                "Component ID": comp_id,
                                "Component Name": name,
                                "Location": f"Only in {pair.dv1_name}",
                            }
                        )
                    for comp_id in pair.only_in_dv2:
                        name = pair.only_in_dv2_names.get(comp_id, "") if pair.only_in_dv2_names else ""
                        drift_data.append(
                            {
                                "DV1 ID": pair.dv1_id,
                                "DV1 Name": pair.dv1_name,
                                "DV2 ID": pair.dv2_id,
                                "DV2 Name": pair.dv2_name,
                                "Component ID": comp_id,
                                "Component Name": name,
                                "Location": f"Only in {pair.dv2_name}",
                            }
                        )
            if drift_data:
                drift_df = pd.DataFrame(drift_data)
                drift_df.to_excel(writer, sheet_name="Drift Details", index=False)
                worksheet = writer.sheets["Drift Details"]
                worksheet.set_column("A:A", 20)
                worksheet.set_column("B:B", 30)
                worksheet.set_column("C:C", 20)
                worksheet.set_column("D:D", 30)
                worksheet.set_column("E:E", 40)
                worksheet.set_column("F:F", 30)
                worksheet.set_column("G:G", 25)

        # Sheet 5c: Clusters (if enabled)
        if result.clusters:
            cluster_data = []
            for cluster in result.clusters:
                cluster_data.extend(
                    {
                        "Cluster ID": cluster.cluster_id,
                        "Cluster Name": cluster.cluster_name or f"Cluster {cluster.cluster_id}",
                        "Cluster Size": cluster.size,
                        "Cohesion": cluster.cohesion_score,
                        "Data View ID": dv_id,
                        "Data View Name": dv_name,
                    }
                    for dv_id, dv_name in zip(cluster.data_view_ids, cluster.data_view_names, strict=True)
                )
            if cluster_data:
                cluster_df = pd.DataFrame(cluster_data)
                cluster_df.to_excel(writer, sheet_name="Clusters", index=False)
                worksheet = writer.sheets["Clusters"]
                worksheet.set_column("A:A", 12)
                worksheet.set_column("B:B", 25)
                worksheet.set_column("C:C", 12)
                worksheet.set_column("D:D", 12)
                worksheet.set_column("E:E", 20)
                worksheet.set_column("F:F", 40)

        # Sheet 6: Recommendations
        if result.recommendations:
            rec_data = [
                {
                    "Type": rec.get("type", ""),
                    "Severity": rec.get("severity", ""),
                    "Description": rec.get("reason", ""),
                    "Data View": rec.get("data_view", rec.get("data_view_1", "")),
                    "Details": json.dumps(
                        {
                            k: v
                            for k, v in rec.items()
                            if k not in ["type", "severity", "reason", "data_view", "data_view_1"]
                        }
                    ),
                }
                for rec in result.recommendations
            ]
            rec_df = pd.DataFrame(rec_data)
            rec_df.to_excel(writer, sheet_name="Recommendations", index=False)
            worksheet = writer.sheets["Recommendations"]
            worksheet.set_column("A:B", 20)
            worksheet.set_column("C:C", 60)
            worksheet.set_column("D:E", 25)

    logger.info(f"Excel report written to {file_path}")
    return str(file_path)


def write_org_report_markdown(
    result: OrgReportResult, output_path: Path | None, output_dir: str, logger: logging.Logger
) -> str:
    """Write org report as GitHub-flavored markdown.

    Args:
        result: OrgReportResult from analysis
        output_path: Optional specific output path
        output_dir: Output directory if no path specified
        logger: Logger instance

    Returns:
        Path to created Markdown file
    """
    if output_path:
        file_path = output_path if str(output_path).endswith(".md") else Path(f"{output_path}.md")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_dir) / f"org_report_{result.org_id}_{timestamp}.md"

    file_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append(f"# Org-Wide Component Analysis Report: {result.org_id}")
    lines.append("")
    lines.append(f"**Organization:** {result.org_id}")
    lines.append(f"**Generated:** {result.timestamp}")
    lines.append("")

    # Summary Table
    # Calculate total aggregates (non-unique counts across all data views)
    total_metrics_aggregate = sum(dv.metric_count for dv in result.data_view_summaries if not dv.error)
    total_dimensions_aggregate = sum(dv.dimension_count for dv in result.data_view_summaries if not dv.error)
    total_components_aggregate = sum(dv.total_components for dv in result.data_view_summaries if not dv.error)
    total_derived_metrics = sum(dv.derived_metric_count for dv in result.data_view_summaries if not dv.error)
    total_derived_dimensions = sum(dv.derived_dimension_count for dv in result.data_view_summaries if not dv.error)
    total_derived_fields = total_derived_metrics + total_derived_dimensions

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Data Views Analyzed | {result.successful_data_views} / {result.total_data_views} |")
    lines.append(f"| Total Unique Metrics | {result.total_unique_metrics:,} |")
    lines.append(f"| Total Unique Dimensions | {result.total_unique_dimensions:,} |")
    lines.append(f"| Total Unique Components | {result.total_unique_components:,} |")
    lines.append(f"| Total Metrics (Non-Unique) | {total_metrics_aggregate:,} |")
    lines.append(f"| Total Dimensions (Non-Unique) | {total_dimensions_aggregate:,} |")
    lines.append(f"| Total Components (Non-Unique) | {total_components_aggregate:,} |")
    lines.append(f"| Derived Metrics (Non-Unique) | {total_derived_metrics:,} |")
    lines.append(f"| Derived Dimensions (Non-Unique) | {total_derived_dimensions:,} |")
    lines.append(f"| Total Derived Fields (Non-Unique) | {total_derived_fields:,} |")
    lines.append(f"| Analysis Duration | {result.duration:.2f}s |")
    lines.append("")

    # Distribution
    lines.append("## Component Distribution")
    lines.append("")
    lines.append("| Category | Metrics | Dimensions | Total |")
    lines.append("|----------|--------:|----------:|------:|")

    dist = result.distribution
    # Build correct label for core threshold
    if result.parameters.core_min_count is None:
        core_label = f"{int(result.parameters.core_threshold * 100)}%+"
        core_desc = f"{int(result.parameters.core_threshold * 100)}% or more of"
    else:
        core_label = f">={result.parameters.core_min_count}"
        core_desc = f"{result.parameters.core_min_count} or more"

    lines.append(
        f"| Core ({core_label} DVs) | {len(dist.core_metrics)} | {len(dist.core_dimensions)} | {dist.total_core} |"
    )
    lines.append(
        f"| Common (25-49% DVs) | {len(dist.common_metrics)} | {len(dist.common_dimensions)} | {dist.total_common} |"
    )
    lines.append(
        f"| Limited (2+ DVs) | {len(dist.limited_metrics)} | {len(dist.limited_dimensions)} | {dist.total_limited} |"
    )
    lines.append(
        f"| Isolated (1 DV only) | {len(dist.isolated_metrics)} | {len(dist.isolated_dimensions)} | {dist.total_isolated} |"
    )
    lines.append("")

    # Data Views Table
    lines.append("## Data Views")
    lines.append("")
    lines.append("| Name | ID | Metrics | Dimensions | Status |")
    lines.append("|------|----|---------:|----------:|--------|")

    for dv in sorted(result.data_view_summaries, key=lambda x: x.data_view_name):
        name = dv.data_view_name.replace("|", "\\|")
        if dv.error:
            lines.append(f"| {name} | `{dv.data_view_id}` | ERROR | - | {dv.status} |")
        else:
            lines.append(f"| {name} | `{dv.data_view_id}` | {dv.metric_count} | {dv.dimension_count} | {dv.status} |")
    lines.append("")

    # Core Components
    if dist.total_core > 0:
        lines.append("## Core Components")
        lines.append("")
        lines.append(f"Components present in {core_desc} data views.")
        lines.append("")

        # Check if any components have names
        has_names = any(info.name for info in result.component_index.values())

        if dist.core_metrics:
            lines.append("### Core Metrics")
            lines.append("")
            if has_names:
                lines.append("| Component ID | Name | Data View Count |")
                lines.append("|--------------|------|----------------:|")
            else:
                lines.append("| Component ID | Data View Count |")
                lines.append("|--------------|----------------:|")
            for comp_id in dist.core_metrics[:20]:
                info = result.component_index.get(comp_id)
                if info:
                    if has_names:
                        name = (info.name or "-").replace("|", "\\|")
                        lines.append(f"| `{comp_id}` | {name} | {info.presence_count} |")
                    else:
                        lines.append(f"| `{comp_id}` | {info.presence_count} |")
            if len(dist.core_metrics) > 20:
                if has_names:
                    lines.append(f"| *... {len(dist.core_metrics) - 20} more* | | |")
                else:
                    lines.append(f"| *... {len(dist.core_metrics) - 20} more* | |")
            lines.append("")

        if dist.core_dimensions:
            lines.append("### Core Dimensions")
            lines.append("")
            if has_names:
                lines.append("| Component ID | Name | Data View Count |")
                lines.append("|--------------|------|----------------:|")
            else:
                lines.append("| Component ID | Data View Count |")
                lines.append("|--------------|----------------:|")
            for comp_id in dist.core_dimensions[:20]:
                info = result.component_index.get(comp_id)
                if info:
                    if has_names:
                        name = (info.name or "-").replace("|", "\\|")
                        lines.append(f"| `{comp_id}` | {name} | {info.presence_count} |")
                    else:
                        lines.append(f"| `{comp_id}` | {info.presence_count} |")
            if len(dist.core_dimensions) > 20:
                if has_names:
                    lines.append(f"| *... {len(dist.core_dimensions) - 20} more* | | |")
                else:
                    lines.append(f"| *... {len(dist.core_dimensions) - 20} more* | |")
            lines.append("")

    # Similarity Matrix
    if result.similarity_pairs:
        lines.append("## High Overlap Pairs")
        lines.append("")
        effective_threshold = min(result.parameters.overlap_threshold, 0.9)
        threshold_note = ""
        if result.parameters.overlap_threshold > 0.9:
            threshold_note = (
                f" (configured {int(result.parameters.overlap_threshold * 100)}%, capped at 90% for governance checks)"
            )
        lines.append(f"Data view pairs with >= {int(effective_threshold * 100)}% Jaccard similarity{threshold_note}.")
        lines.append("")
        lines.append("| Data View 1 | Data View 2 | Similarity | Shared |")
        lines.append("|-------------|-------------|------------|-------:|")

        for pair in result.similarity_pairs[:15]:
            name1 = pair.dv1_name.replace("|", "\\|")
            name2 = pair.dv2_name.replace("|", "\\|")
            lines.append(f"| {name1} | {name2} | {pair.jaccard_similarity * 100:.1f}% | {pair.shared_count} |")

        if len(result.similarity_pairs) > 15:
            lines.append(f"| *... {len(result.similarity_pairs) - 15} more pairs* | | | |")
        lines.append("")

    # Recommendations
    if result.recommendations:
        lines.append("## Recommendations")
        lines.append("")

        for i, rec in enumerate(result.recommendations, 1):
            severity = rec.get("severity", "low")
            severity_badge = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")
            lines.append(f"### {i}. {severity_badge} {rec.get('type', 'Unknown').replace('_', ' ').title()}")
            lines.append("")
            lines.append(rec.get("reason", "No details provided."))
            lines.append("")

            if rec.get("data_view"):
                lines.append(f"- **Data View:** {rec.get('data_view_name', '')} (`{rec.get('data_view')}`)")
            if rec.get("data_view_1"):
                lines.append(f"- **Pair:** {rec.get('data_view_1_name', '')} ↔ {rec.get('data_view_2_name', '')}")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by CJA SDR Generator v{__version__}*")

    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown report written to {file_path}")
    return str(file_path)


def write_org_report_html(
    result: OrgReportResult, output_path: Path | None, output_dir: str, logger: logging.Logger
) -> str:
    """Write org report as styled HTML.

    Args:
        result: OrgReportResult from analysis
        output_path: Optional specific output path
        output_dir: Output directory if no path specified
        logger: Logger instance

    Returns:
        Path to created HTML file
    """
    if output_path:
        file_path = output_path if str(output_path).endswith(".html") else Path(f"{output_path}.html")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_dir) / f"org_report_{result.org_id}_{timestamp}.html"

    file_path.parent.mkdir(parents=True, exist_ok=True)

    dist = result.distribution
    has_names = any(info.name for info in result.component_index.values())

    # Escape org_id for HTML output
    org_id_escaped = html.escape(result.org_id)

    # Calculate total aggregates (non-unique counts across all data views)
    total_components_aggregate = sum(dv.total_components for dv in result.data_view_summaries if not dv.error)
    total_derived_metrics = sum(dv.derived_metric_count for dv in result.data_view_summaries if not dv.error)
    total_derived_dimensions = sum(dv.derived_dimension_count for dv in result.data_view_summaries if not dv.error)
    total_derived_fields = total_derived_metrics + total_derived_dimensions

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Org-Wide Component Analysis Report</title>
    <style>
        :root {{
            --primary: #1a73e8;
            --success: #34a853;
            --warning: #fbbc04;
            --danger: #ea4335;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #202124;
            --text-secondary: #5f6368;
            --border: #dadce0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        h2 {{ color: var(--text); margin: 2rem 0 1rem; border-bottom: 2px solid var(--primary); padding-bottom: 0.5rem; }}
        h3 {{ color: var(--text-secondary); margin: 1.5rem 0 0.75rem; }}
        .meta {{ color: var(--text-secondary); margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .stat-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid var(--border);
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: var(--primary); }}
        .stat-label {{ color: var(--text-secondary); font-size: 0.875rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--bg); font-weight: 600; }}
        tr:hover {{ background: #f1f3f4; }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        .badge-core {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-common {{ background: #e3f2fd; color: #1565c0; }}
        .badge-limited {{ background: #fff3e0; color: #ef6c00; }}
        .badge-isolated {{ background: #fce4ec; color: #c2185b; }}
        .badge-high {{ background: var(--danger); color: white; }}
        .badge-medium {{ background: var(--warning); color: #333; }}
        .badge-low {{ background: var(--primary); color: white; }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: var(--primary);
            transition: width 0.3s;
        }}
        code {{ background: #f1f3f4; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.85rem; }}
        .recommendation {{ padding: 1rem; border-left: 4px solid var(--primary); margin: 1rem 0; background: #f8f9fa; }}
        .recommendation.high {{ border-color: var(--danger); }}
        .recommendation.medium {{ border-color: var(--warning); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Org-Wide Component Analysis Report</h1>
        <p class="meta">Organization: {org_id_escaped} | Generated: {result.timestamp} | Duration: {result.duration:.2f}s</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{result.successful_data_views}/{result.total_data_views}</div>
                <div class="stat-label">Data Views Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{result.total_unique_metrics:,}</div>
                <div class="stat-label">Unique Metrics</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{result.total_unique_dimensions:,}</div>
                <div class="stat-label">Unique Dimensions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{result.total_unique_components:,}</div>
                <div class="stat-label">Total Unique Components</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_components_aggregate:,}</div>
                <div class="stat-label">Total Components (Non-Unique)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_derived_fields:,}</div>
                <div class="stat-label">Total Derived Fields (Non-Unique)</div>
            </div>
        </div>

        <h2>Component Distribution</h2>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Category</th><th>Metrics</th><th>Dimensions</th><th>Total</th><th>Distribution</th></tr>
                </thead>
                <tbody>
"""

    total = result.total_unique_components or 1

    # Build correct label for core threshold
    if result.parameters.core_min_count is None:
        core_label = f"{int(result.parameters.core_threshold * 100)}%+"
        core_desc = f"{int(result.parameters.core_threshold * 100)}% or more of"
    else:
        core_label = f"&gt;={result.parameters.core_min_count}"
        core_desc = f"{result.parameters.core_min_count} or more"

    for bucket, m_list, d_list, badge_class in [
        (f"Core ({core_label} DVs)", dist.core_metrics, dist.core_dimensions, "core"),
        ("Common (25-49% DVs)", dist.common_metrics, dist.common_dimensions, "common"),
        ("Limited (2+ DVs)", dist.limited_metrics, dist.limited_dimensions, "limited"),
        ("Isolated (1 DV)", dist.isolated_metrics, dist.isolated_dimensions, "isolated"),
    ]:
        bucket_total = len(m_list) + len(d_list)
        pct = bucket_total / total * 100
        html_out += f"""                    <tr>
                        <td><span class="badge badge-{badge_class}">{bucket}</span></td>
                        <td>{len(m_list)}</td>
                        <td>{len(d_list)}</td>
                        <td>{bucket_total}</td>
                        <td><div class="progress-bar"><div class="progress-fill" style="width: {pct:.1f}%"></div></div></td>
                    </tr>
"""

    html_out += """                </tbody>
            </table>
        </div>

        <h2>Data Views</h2>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Name</th><th>ID</th><th>Metrics</th><th>Dimensions</th><th>Status</th></tr>
                </thead>
                <tbody>
"""

    for dv in sorted(result.data_view_summaries, key=lambda x: x.data_view_name):
        # Escape user-sourced strings to prevent HTML injection
        dv_name_escaped = html.escape(dv.data_view_name)
        dv_id_escaped = html.escape(dv.data_view_id)
        if dv.error:
            error_escaped = html.escape(dv.error)
            html_out += f'                    <tr><td>{dv_name_escaped}</td><td><code>{dv_id_escaped}</code></td><td colspan="2">ERROR: {error_escaped}</td><td>{dv.status}</td></tr>\n'
        else:
            html_out += f"                    <tr><td>{dv_name_escaped}</td><td><code>{dv_id_escaped}</code></td><td>{dv.metric_count}</td><td>{dv.dimension_count}</td><td>{dv.status}</td></tr>\n"

    html_out += """                </tbody>
            </table>
        </div>
"""

    # Core Components
    if dist.total_core > 0:
        html_out += f"""
        <h2>Core Components</h2>
        <p>Components present in {core_desc} data views.</p>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Component ID</th>{"<th>Name</th>" if has_names else ""}<th>Type</th><th>Data View Count</th></tr>
                </thead>
                <tbody>
"""
        for comp_id in (dist.core_metrics + dist.core_dimensions)[:30]:
            info = result.component_index.get(comp_id)
            if info:
                comp_id_escaped = html.escape(comp_id)
                name_escaped = html.escape(info.name) if info.name else "-"
                name_col = f"<td>{name_escaped}</td>" if has_names else ""
                html_out += f"                    <tr><td><code>{comp_id_escaped}</code></td>{name_col}<td>{info.component_type.title()}</td><td>{info.presence_count}</td></tr>\n"

        if dist.total_core > 30:
            html_out += f'                    <tr><td colspan="{"4" if has_names else "3"}"><em>... and {dist.total_core - 30} more</em></td></tr>\n'

        html_out += """                </tbody>
            </table>
        </div>
"""

    # Similarity Pairs
    if result.similarity_pairs:
        effective_threshold = min(result.parameters.overlap_threshold, 0.9)
        threshold_note = ""
        if result.parameters.overlap_threshold > 0.9:
            threshold_note = (
                f" (configured {int(result.parameters.overlap_threshold * 100)}%, capped at 90% for governance checks)"
            )
        html_out += f"""
        <h2>High Overlap Pairs</h2>
        <p>Data view pairs with &gt;= {int(effective_threshold * 100)}% Jaccard similarity{threshold_note}.</p>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Data View 1</th><th>Data View 2</th><th>Similarity</th><th>Shared</th></tr>
                </thead>
                <tbody>
"""
        for pair in result.similarity_pairs[:20]:
            dv1_escaped = html.escape(pair.dv1_name)
            dv2_escaped = html.escape(pair.dv2_name)
            html_out += f"                    <tr><td>{dv1_escaped}</td><td>{dv2_escaped}</td><td>{pair.jaccard_similarity * 100:.1f}%</td><td>{pair.shared_count}</td></tr>\n"

        if len(result.similarity_pairs) > 20:
            html_out += f'                    <tr><td colspan="4"><em>... and {len(result.similarity_pairs) - 20} more pairs</em></td></tr>\n'

        html_out += """                </tbody>
            </table>
        </div>
"""

    # Recommendations
    if result.recommendations:
        html_out += """
        <h2>Recommendations</h2>
"""
        for rec in result.recommendations:
            severity = rec.get("severity", "low")
            rec_type = html.escape(rec.get("type", "Unknown").replace("_", " ").title())
            rec_reason = html.escape(rec.get("reason", "No details provided."))
            html_out += f"""        <div class="recommendation {severity}">
            <strong><span class="badge badge-{severity}">{severity.upper()}</span> {rec_type}</strong>
            <p>{rec_reason}</p>
        </div>
"""

    html_out += """
        <hr style="margin: 2rem 0; border: none; border-top: 1px solid var(--border);">
        <p style="color: var(--text-secondary); font-size: 0.875rem;">Generated by CJA SDR Generator</p>
    </div>
</body>
</html>"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    logger.info(f"HTML report written to {file_path}")
    return str(file_path)


def write_org_report_csv(
    result: OrgReportResult, output_path: Path | None, output_dir: str, logger: logging.Logger
) -> str:
    """Write org report as multiple CSV files.

    Creates the following CSV files:
    - org_report_summary.csv: High-level statistics
    - org_report_data_views.csv: Per-data-view breakdown
    - org_report_components.csv: Component index with names and coverage
    - org_report_similarity.csv: Similarity pairs (if computed)
    - org_report_distribution.csv: Distribution bucket counts

    Args:
        result: OrgReportResult from analysis
        output_path: Optional base path (suffix will be added)
        output_dir: Output directory if no path specified
        logger: Logger instance

    Returns:
        Path to the created directory containing CSV files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output directory
    if output_path:
        csv_dir = Path(output_path)
        if csv_dir.suffix == ".csv":
            csv_dir = csv_dir.parent / csv_dir.stem
    else:
        csv_dir = Path(output_dir) / f"org_report_{result.org_id}_{timestamp}"

    csv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary CSV
    # Calculate total aggregates (non-unique counts across all data views)
    total_metrics_aggregate = sum(dv.metric_count for dv in result.data_view_summaries if not dv.error)
    total_dimensions_aggregate = sum(dv.dimension_count for dv in result.data_view_summaries if not dv.error)
    total_components_aggregate = sum(dv.total_components for dv in result.data_view_summaries if not dv.error)
    total_derived_metrics = sum(dv.derived_metric_count for dv in result.data_view_summaries if not dv.error)
    total_derived_dimensions = sum(dv.derived_dimension_count for dv in result.data_view_summaries if not dv.error)
    total_derived_fields = total_derived_metrics + total_derived_dimensions
    effective_overlap_threshold = min(result.parameters.overlap_threshold, 0.9)

    summary_data = [
        {
            "Report Type": "Org-Wide Component Analysis",
            "Generated At": result.timestamp,
            "Org ID": result.org_id,
            "Total Data Views": result.total_data_views,
            "Successful Data Views": result.successful_data_views,
            "Total Unique Metrics": result.total_unique_metrics,
            "Total Unique Dimensions": result.total_unique_dimensions,
            "Total Unique Components": result.total_unique_components,
            "Total Metrics (Non-Unique)": total_metrics_aggregate,
            "Total Dimensions (Non-Unique)": total_dimensions_aggregate,
            "Total Components (Non-Unique)": total_components_aggregate,
            "Derived Metrics (Non-Unique)": total_derived_metrics,
            "Derived Dimensions (Non-Unique)": total_derived_dimensions,
            "Total Derived Fields (Non-Unique)": total_derived_fields,
            "Core Threshold": result.parameters.core_threshold,
            "Overlap Threshold (Configured)": result.parameters.overlap_threshold,
            "Overlap Threshold (Effective)": effective_overlap_threshold,
            "Analysis Duration (s)": round(result.duration, 2),
        }
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_path = csv_dir / "org_report_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # 2. Data Views CSV
    dv_data = [
        {
            "Data View ID": dv.data_view_id,
            "Data View Name": dv.data_view_name,
            "Metrics Count": dv.metric_count,
            "Dimensions Count": dv.dimension_count,
            "Total Components": dv.total_components,
            "Status": dv.status,
            "Error": dv.error or "",
            "Fetch Duration (s)": round(dv.fetch_duration, 3),
        }
        for dv in result.data_view_summaries
    ]
    dv_df = pd.DataFrame(dv_data)
    dv_path = csv_dir / "org_report_data_views.csv"
    dv_df.to_csv(dv_path, index=False)

    # 3. Components CSV
    comp_data = []
    for comp_id, info in result.component_index.items():
        # Determine distribution bucket
        if comp_id in result.distribution.core_metrics or comp_id in result.distribution.core_dimensions:
            bucket = "Core"
        elif comp_id in result.distribution.common_metrics or comp_id in result.distribution.common_dimensions:
            bucket = "Common"
        elif comp_id in result.distribution.limited_metrics or comp_id in result.distribution.limited_dimensions:
            bucket = "Limited"
        else:
            bucket = "Isolated"

        coverage_pct = (
            (info.presence_count / result.successful_data_views * 100) if result.successful_data_views > 0 else 0
        )

        comp_data.append(
            {
                "Component ID": comp_id,
                "Component Type": info.component_type.title(),
                "Name": info.name or "",
                "Data View Count": info.presence_count,
                "Coverage (%)": round(coverage_pct, 1),
                "Distribution Bucket": bucket,
                "Data Views": ";".join(sorted(info.data_views)),
            }
        )
    comp_df = pd.DataFrame(comp_data)
    comp_df = comp_df.sort_values(["Distribution Bucket", "Data View Count"], ascending=[True, False])
    comp_path = csv_dir / "org_report_components.csv"
    comp_df.to_csv(comp_path, index=False)

    # 4. Distribution CSV
    dist = result.distribution
    dist_data = [
        {
            "Bucket": "Core",
            "Metrics": len(dist.core_metrics),
            "Dimensions": len(dist.core_dimensions),
            "Total": dist.total_core,
        },
        {
            "Bucket": "Common",
            "Metrics": len(dist.common_metrics),
            "Dimensions": len(dist.common_dimensions),
            "Total": dist.total_common,
        },
        {
            "Bucket": "Limited",
            "Metrics": len(dist.limited_metrics),
            "Dimensions": len(dist.limited_dimensions),
            "Total": dist.total_limited,
        },
        {
            "Bucket": "Isolated",
            "Metrics": len(dist.isolated_metrics),
            "Dimensions": len(dist.isolated_dimensions),
            "Total": dist.total_isolated,
        },
    ]
    dist_df = pd.DataFrame(dist_data)
    dist_path = csv_dir / "org_report_distribution.csv"
    dist_df.to_csv(dist_path, index=False)

    # 5. Similarity CSV (if computed)
    if result.similarity_pairs:
        effective_overlap_threshold = min(result.parameters.overlap_threshold, 0.9)
        sim_data = [
            {
                "Data View 1 ID": pair.dv1_id,
                "Data View 1 Name": pair.dv1_name,
                "Data View 2 ID": pair.dv2_id,
                "Data View 2 Name": pair.dv2_name,
                "Jaccard Similarity": pair.jaccard_similarity,
                "Shared Components": pair.shared_count,
                "Union Size": pair.union_count,
                "Overlap Threshold (Configured)": result.parameters.overlap_threshold,
                "Overlap Threshold (Effective)": effective_overlap_threshold,
            }
            for pair in result.similarity_pairs
        ]
        sim_df = pd.DataFrame(sim_data)
        sim_path = csv_dir / "org_report_similarity.csv"
        sim_df.to_csv(sim_path, index=False)

    # 6. Recommendations CSV (if any)
    if result.recommendations:
        rec_data = [
            {
                "Type": rec.get("type", ""),
                "Severity": rec.get("severity", ""),
                "Description": rec.get("reason", ""),
                "Data View": rec.get("data_view", rec.get("data_view_1", "")),
                "Data View Name": rec.get("data_view_name", rec.get("data_view_1_name", "")),
            }
            for rec in result.recommendations
        ]
        rec_df = pd.DataFrame(rec_data)
        rec_path = csv_dir / "org_report_recommendations.csv"
        rec_df.to_csv(rec_path, index=False)

    logger.info(f"CSV reports written to {csv_dir}")
    return str(csv_dir)


def run_org_report(
    config_file: str,
    output_format: str,
    output_path: str | None,
    output_dir: str,
    org_config: OrgReportConfig,
    profile: str | None = None,
    quiet: bool = False,
) -> tuple[bool, bool]:
    """Run org-wide component analysis and generate report.

    Args:
        config_file: Path to CJA configuration file
        output_format: Output format (console, json, excel, markdown, html, csv, all)
        output_path: Optional specific output file path
        output_dir: Output directory for generated files
        org_config: OrgReportConfig with analysis parameters
        profile: Optional profile name for credentials
        quiet: Suppress progress output

    Returns:
        Tuple of (success, thresholds_exceeded) - thresholds_exceeded triggers exit code 2
    """
    # Setup logging
    logger = logging.getLogger("org_report")
    logger.setLevel(logging.INFO if not quiet else logging.WARNING)

    output_to_stdout = output_path in ("-", "stdout")
    status_to_stderr = output_to_stdout and output_format == "json"
    status_stream = sys.stderr if status_to_stderr else sys.stdout

    def _status_print(*args, **kwargs) -> None:
        kwargs.setdefault("file", status_stream)
        print(*args, **kwargs)

    if not quiet:
        _status_print()
        _status_print("=" * 110)
        _status_print("ORG-WIDE COMPONENT ANALYSIS")
        _status_print("=" * 110)
        if profile:
            _status_print(f"\nUsing profile: {profile}")
        _status_print()

    try:
        # Configure CJA connection
        success, source, credentials = configure_cjapy(profile, config_file)
        if not success:
            _status_print(ConsoleColors.error(f"ERROR: {source}"))
            return False, False

        # Extract org_id from credentials
        org_id = credentials.get("org_id", "unknown") if credentials else "unknown"

        cja = cjapy.CJA()

        # Create cache if caching enabled
        cache = None
        if org_config.use_cache:
            cache = OrgReportCache()
            if not quiet:
                stats = cache.get_stats()
                if stats["entries"] > 0:
                    _status_print(f"Cache: {stats['entries']} entries available")

        # Run analysis
        analyzer = OrgComponentAnalyzer(cja, org_config, logger, org_id=org_id, cache=cache)
        result = analyzer.run_analysis()

        if result.total_data_views == 0:
            _status_print(ConsoleColors.warning("No data views found matching criteria"))
            return False, False

        # Handle org-report comparison (Feature 4)
        comparison = None
        if org_config.compare_org_report:
            try:
                if not quiet:
                    _status_print(f"\nComparing to previous report: {org_config.compare_org_report}")
                comparison = compare_org_reports(result, org_config.compare_org_report)
                with contextlib.redirect_stdout(status_stream):
                    write_org_report_comparison_console(comparison, quiet)
            except FileNotFoundError:
                _status_print(ConsoleColors.error(f"ERROR: Previous report not found: {org_config.compare_org_report}"))
            except json.JSONDecodeError:
                _status_print(
                    ConsoleColors.error(f"ERROR: Invalid JSON in previous report: {org_config.compare_org_report}")
                )
            except Exception as e:
                _status_print(ConsoleColors.warning(f"Warning: Could not compare reports: {e}"))

        # Generate output based on format
        output_path_obj = Path(output_path) if output_path and not output_to_stdout else None

        # Handle org-stats mode (Feature 2) - minimal output
        if org_config.org_stats_only:
            with contextlib.redirect_stdout(status_stream):
                write_org_report_stats_only(result, quiet)
            # Still output JSON if requested for CI integration
            if output_format == "json":
                if output_to_stdout:
                    json.dump(build_org_report_json_data(result), sys.stdout, indent=2, ensure_ascii=False)
                    print()
                else:
                    file_path = write_org_report_json(result, output_path_obj, output_dir, logger)
                    if not quiet:
                        _status_print(f"JSON saved to: {file_path}")
            append_github_step_summary(build_org_step_summary(result), logger)
            return True, result.thresholds_exceeded

        # Handle format aliases (reports, data, ci)
        if output_format in FORMAT_ALIASES:
            formats_to_generate = FORMAT_ALIASES[output_format]
            generated_files = []
            alias_base = output_path_obj.with_suffix("") if output_path_obj else None
            for fmt in formats_to_generate:
                if fmt == "json":
                    path = write_org_report_json(result, alias_base, output_dir, logger)
                    generated_files.append(f"JSON: {path}")
                elif fmt == "excel":
                    path = write_org_report_excel(result, alias_base, output_dir, logger)
                    generated_files.append(f"Excel: {path}")
                elif fmt == "markdown":
                    path = write_org_report_markdown(result, alias_base, output_dir, logger)
                    generated_files.append(f"Markdown: {path}")
                elif fmt == "csv":
                    path = write_org_report_csv(result, alias_base, output_dir, logger)
                    generated_files.append(f"CSV: {path}")
                elif fmt == "html":
                    path = write_org_report_html(result, alias_base, output_dir, logger)
                    generated_files.append(f"HTML: {path}")
            if not quiet:
                _status_print(f"\n{ConsoleColors.success('✓')} Reports saved ({output_format} alias):")
                for f in generated_files:
                    _status_print(f"  - {f}")
        elif output_format == "console" or (output_format is None and output_path is None):
            write_org_report_console(result, org_config, quiet)
        elif output_format == "json":
            if output_to_stdout:
                json.dump(build_org_report_json_data(result), sys.stdout, indent=2, ensure_ascii=False)
                print()
            else:
                file_path = write_org_report_json(result, output_path_obj, output_dir, logger)
                if not quiet:
                    _status_print(f"\n{ConsoleColors.success('✓')} JSON report saved to: {file_path}")
        elif output_format == "excel":
            file_path = write_org_report_excel(result, output_path_obj, output_dir, logger)
            if not quiet:
                _status_print(f"\n{ConsoleColors.success('✓')} Excel report saved to: {file_path}")
        elif output_format == "markdown":
            file_path = write_org_report_markdown(result, output_path_obj, output_dir, logger)
            if not quiet:
                _status_print(f"\n{ConsoleColors.success('✓')} Markdown report saved to: {file_path}")
        elif output_format == "html":
            file_path = write_org_report_html(result, output_path_obj, output_dir, logger)
            if not quiet:
                _status_print(f"\n{ConsoleColors.success('✓')} HTML report saved to: {file_path}")
        elif output_format == "csv":
            if output_to_stdout:
                _status_print(
                    ConsoleColors.error(
                        "ERROR: CSV output for org-report writes multiple files and cannot be sent to stdout. Use --output-dir or a file path."
                    )
                )
                return False, False
            csv_dir = write_org_report_csv(result, output_path_obj, output_dir, logger)
            if not quiet:
                _status_print(f"\n{ConsoleColors.success('✓')} CSV reports saved to: {csv_dir}")
        elif output_format == "all":
            # Generate all formats
            write_org_report_console(result, org_config, quiet)
            all_base = output_path_obj.with_suffix("") if output_path_obj else None
            json_path = write_org_report_json(result, all_base, output_dir, logger)
            excel_path = write_org_report_excel(result, all_base, output_dir, logger)
            md_path = write_org_report_markdown(result, all_base, output_dir, logger)
            html_path = write_org_report_html(result, all_base, output_dir, logger)
            csv_dir = write_org_report_csv(result, all_base, output_dir, logger)
            if not quiet:
                _status_print(f"\n{ConsoleColors.success('✓')} Reports saved:")
                _status_print(f"  - JSON: {json_path}")
                _status_print(f"  - Excel: {excel_path}")
                _status_print(f"  - Markdown: {md_path}")
                _status_print(f"  - HTML: {html_path}")
                _status_print(f"  - CSV: {csv_dir}")
        else:
            # Unknown format - raise clear error instead of silent fallback
            _status_print(
                ConsoleColors.error(
                    f"ERROR: Unknown format '{output_format}'. Valid formats: console, json, excel, markdown, html, csv, all, reports, data, ci"
                )
            )
            return False, False

        if not quiet:
            _status_print()
            _status_print("=" * 110)
            _status_print(f"Analysis completed in {result.duration:.2f}s")
            # Show governance violation summary if thresholds exceeded
            if result.thresholds_exceeded and org_config.fail_on_threshold:
                _status_print(
                    ConsoleColors.warning(
                        f"GOVERNANCE THRESHOLDS EXCEEDED - {len(result.governance_violations or [])} violation(s)"
                    )
                )
            _status_print("=" * 110)

        append_github_step_summary(build_org_step_summary(result), logger)
        return True, result.thresholds_exceeded

    except FileNotFoundError:
        _status_print(ConsoleColors.error(f"ERROR: Configuration file '{config_file}' not found"))
        return False, False

    except KeyboardInterrupt, SystemExit:
        if not quiet:
            _status_print()
            _status_print(ConsoleColors.warning("Operation cancelled."))
        raise

    except Exception as e:
        _status_print(ConsoleColors.error(f"ERROR: Org report failed: {e!s}"))
        logger.exception("Org report error")
        return False, False


# ==================== DIFF AND SNAPSHOT COMMAND HANDLERS ====================


def handle_snapshot_command(
    data_view_id: str,
    snapshot_file: str,
    config_file: str = "config.json",
    quiet: bool = False,
    profile: str | None = None,
    include_calculated_metrics: bool = False,
    include_segments: bool = False,
) -> bool:
    """
    Handle the --snapshot command to save a data view snapshot.

    Args:
        data_view_id: The data view ID to snapshot
        snapshot_file: Path to save the snapshot
        config_file: Path to CJA configuration file
        quiet: Suppress progress output
        profile: Optional profile name for credentials
        include_calculated_metrics: Include calculated metrics inventory in snapshot
        include_segments: Include segments inventory in snapshot

    Returns:
        True if successful, False otherwise
    """
    try:
        # Build inventory info string for header
        inventory_info = []
        if include_calculated_metrics:
            inventory_info.append("calculated metrics")
        if include_segments:
            inventory_info.append("segments")

        if not quiet:
            print()
            print("=" * BANNER_WIDTH)
            print("CREATING DATA VIEW SNAPSHOT")
            print("=" * BANNER_WIDTH)
            print(f"Data View: {data_view_id}")
            print(f"Output: {snapshot_file}")
            if inventory_info:
                print(f"Including: {', '.join(inventory_info)} inventory")
            print()

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if not quiet else logging.WARNING)

        # Initialize CJA with profile support
        success, source, _ = configure_cjapy(profile=profile, config_file=config_file, logger=logger)
        if not success:
            print(ConsoleColors.error(f"ERROR: Configuration failed: {source}"), file=sys.stderr)
            return False
        cja = cjapy.CJA()

        # Create and save snapshot
        snapshot_manager = SnapshotManager(logger)
        snapshot = snapshot_manager.create_snapshot(
            cja,
            data_view_id,
            quiet,
            include_calculated_metrics=include_calculated_metrics,
            include_segments=include_segments,
        )
        saved_path = snapshot_manager.save_snapshot(snapshot, snapshot_file)

        if not quiet:
            print()
            print("=" * BANNER_WIDTH)
            print(ConsoleColors.success("SNAPSHOT CREATED SUCCESSFULLY"))
            print("=" * BANNER_WIDTH)
            print(f"Data View: {snapshot.data_view_name} ({snapshot.data_view_id})")
            print(f"Metrics: {len(snapshot.metrics)}")
            print(f"Dimensions: {len(snapshot.dimensions)}")
            # Show inventory counts if included
            if snapshot.calculated_metrics_inventory is not None:
                print(f"Calculated Metrics: {len(snapshot.calculated_metrics_inventory)}")
            if snapshot.segments_inventory is not None:
                print(f"Segments: {len(snapshot.segments_inventory)}")
            print(f"Snapshot Version: {snapshot.snapshot_version}")
            print(f"Saved to: {saved_path}")
            print("=" * BANNER_WIDTH)

        return True

    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to create snapshot: {e!s}"), file=sys.stderr)
        return False


def handle_diff_command(
    source_id: str,
    target_id: str,
    config_file: str = "config.json",
    output_format: str = "console",
    output_dir: str = ".",
    changes_only: bool = False,
    summary_only: bool = False,
    ignore_fields: list[str] | None = None,
    labels: tuple[str, str] | None = None,
    quiet: bool = False,
    show_only: list[str] | None = None,
    metrics_only: bool = False,
    dimensions_only: bool = False,
    extended_fields: bool = False,
    side_by_side: bool = False,
    no_color: bool = False,
    quiet_diff: bool = False,
    reverse_diff: bool = False,
    warn_threshold: float | None = None,
    group_by_field: bool = False,
    group_by_field_limit: int = 10,
    diff_output: str | None = None,
    format_pr_comment: bool = False,
    auto_snapshot: bool = False,
    auto_prune: bool = False,
    snapshot_dir: str = "./snapshots",
    keep_last: int = 0,
    keep_since: str | None = None,
    keep_last_specified: bool = False,
    keep_since_specified: bool = False,
    profile: str | None = None,
) -> tuple[bool, bool, int | None]:
    """
    Handle the --diff command to compare two data views.

    Args:
        source_id: Source data view ID
        target_id: Target data view ID
        config_file: Path to CJA configuration file
        output_format: Output format
        output_dir: Output directory
        changes_only: Only show changed items
        summary_only: Only show summary
        ignore_fields: Fields to ignore
        labels: Custom labels (source_label, target_label)
        quiet: Suppress progress output
        show_only: Filter to show only specific change types
        metrics_only: Only compare metrics
        dimensions_only: Only compare dimensions
        extended_fields: Use extended field comparison
        side_by_side: Show side-by-side comparison view
        no_color: Disable ANSI color codes
        quiet_diff: Suppress output, only return exit code
        reverse_diff: Swap source and target
        warn_threshold: Exit with code 3 if change % exceeds threshold
        group_by_field: Group changes by field name
        group_by_field_limit: Max items per section in group-by-field output (0 = unlimited)
        diff_output: Write output to file instead of stdout
        format_pr_comment: Output in PR comment format
        auto_snapshot: Automatically save snapshots during diff
        auto_prune: Apply default retention when --auto-snapshot is enabled
        snapshot_dir: Directory for auto-saved snapshots
        keep_last: Retention policy - keep only last N snapshots per data view (0 = keep all)
        keep_since: Date-based retention - delete snapshots older than this period (e.g., '7d', '2w', '1m')
        keep_last_specified: Whether --keep-last was explicitly provided
        keep_since_specified: Whether --keep-since was explicitly provided
        profile: Optional profile name for credentials

    Returns:
        Tuple of (success, has_changes, exit_code_override)
        exit_code_override is 3 if warn_threshold exceeded, None otherwise
    """
    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if not quiet else logging.WARNING)

        # Handle reverse diff - swap source and target
        if reverse_diff:
            source_id, target_id = target_id, source_id

        if not quiet and not quiet_diff:
            print()
            print("=" * BANNER_WIDTH)
            print("COMPARING DATA VIEWS")
            print("=" * BANNER_WIDTH)
            print(f"Source: {source_id}")
            print(f"Target: {target_id}")
            if reverse_diff:
                print("(Reversed comparison)")
            print()

        # Initialize CJA with profile support
        success, source, _ = configure_cjapy(profile=profile, config_file=config_file, logger=logger)
        if not success:
            if not quiet and not quiet_diff:
                print(ConsoleColors.error(f"ERROR: Configuration failed: {source}"), file=sys.stderr)
            return False, False, None
        cja = cjapy.CJA()

        # Create snapshots from live data views
        snapshot_manager = SnapshotManager(logger)

        if not quiet and not quiet_diff:
            print("Fetching source data view...")
        source_snapshot = snapshot_manager.create_snapshot(cja, source_id, quiet or quiet_diff)

        if not quiet and not quiet_diff:
            print("Fetching target data view...")
        target_snapshot = snapshot_manager.create_snapshot(cja, target_id, quiet or quiet_diff)

        # Auto-save snapshots if enabled
        if auto_snapshot:
            os.makedirs(snapshot_dir, exist_ok=True)

            effective_keep_last, effective_keep_since = resolve_auto_prune_retention(
                keep_last=keep_last,
                keep_since=keep_since,
                auto_prune=auto_prune,
                keep_last_specified=keep_last_specified,
                keep_since_specified=keep_since_specified,
            )

            # Save source snapshot
            source_filename = snapshot_manager.generate_snapshot_filename(source_id, source_snapshot.data_view_name)
            source_path = os.path.join(snapshot_dir, source_filename)
            snapshot_manager.save_snapshot(source_snapshot, source_path)

            # Save target snapshot
            target_filename = snapshot_manager.generate_snapshot_filename(target_id, target_snapshot.data_view_name)
            target_path = os.path.join(snapshot_dir, target_filename)
            snapshot_manager.save_snapshot(target_snapshot, target_path)

            if not quiet and not quiet_diff:
                print(f"Auto-saved snapshots to: {snapshot_dir}/")
                print(f"  - {source_filename}")
                print(f"  - {target_filename}")

            # Apply retention policies if configured
            total_deleted = 0

            # Count-based retention
            if effective_keep_last > 0:
                deleted_source = snapshot_manager.apply_retention_policy(snapshot_dir, source_id, effective_keep_last)
                deleted_target = snapshot_manager.apply_retention_policy(snapshot_dir, target_id, effective_keep_last)
                total_deleted += len(deleted_source) + len(deleted_target)

            # Date-based retention
            if effective_keep_since:
                days = parse_retention_period(effective_keep_since)
                if days:
                    deleted_source = snapshot_manager.apply_date_retention_policy(
                        snapshot_dir, source_id, keep_since_days=days
                    )
                    deleted_target = snapshot_manager.apply_date_retention_policy(
                        snapshot_dir, target_id, keep_since_days=days
                    )
                    total_deleted += len(deleted_source) + len(deleted_target)

            if total_deleted > 0 and not quiet and not quiet_diff:
                print(f"  Retention policy: Deleted {total_deleted} old snapshot(s)")

            if not quiet and not quiet_diff:
                print()

        # Compare
        source_label = labels[0] if labels else "Source"
        target_label = labels[1] if labels else "Target"

        comparator = DataViewComparator(
            logger,
            ignore_fields=ignore_fields,
            use_extended_fields=extended_fields,
            show_only=show_only,
            metrics_only=metrics_only,
            dimensions_only=dimensions_only,
        )
        diff_result = comparator.compare(source_snapshot, target_snapshot, source_label, target_label)

        # Check warn threshold
        exit_code_override = None
        if warn_threshold is not None:
            max_change_pct = max(
                diff_result.summary.metrics_change_percent, diff_result.summary.dimensions_change_percent
            )
            if max_change_pct > warn_threshold:
                exit_code_override = 3
                if not quiet_diff:
                    print(
                        ConsoleColors.warning(
                            f"WARNING: Change threshold exceeded! {max_change_pct:.1f}% > {warn_threshold}%"
                        ),
                        file=sys.stderr,
                    )

        # Generate output (unless quiet_diff is set)
        if not quiet_diff:
            # Determine effective format
            effective_format = "pr-comment" if format_pr_comment else output_format

            base_filename = f"diff_{source_id}_{target_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_content = write_diff_output(
                diff_result,
                effective_format,
                base_filename,
                output_dir,
                logger,
                changes_only,
                summary_only,
                side_by_side,
                use_color=ConsoleColors.is_enabled() and not no_color,
                group_by_field=group_by_field,
                group_by_field_limit=group_by_field_limit,
            )

            # Handle --diff-output flag
            if diff_output and output_content:
                with open(diff_output, "w", encoding="utf-8") as f:
                    f.write(output_content)
                if not quiet:
                    print(f"Diff output written to: {diff_output}")

            if not quiet and output_format != "console":
                print()
                print(ConsoleColors.success("Diff report generated successfully"))

        append_github_step_summary(build_diff_step_summary(diff_result), logger)
        return True, diff_result.summary.has_changes, exit_code_override

    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to compare data views: {e!s}"), file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False, False, None


def handle_diff_snapshot_command(
    data_view_id: str,
    snapshot_file: str,
    config_file: str = "config.json",
    output_format: str = "console",
    output_dir: str = ".",
    changes_only: bool = False,
    summary_only: bool = False,
    ignore_fields: list[str] | None = None,
    labels: tuple[str, str] | None = None,
    quiet: bool = False,
    show_only: list[str] | None = None,
    metrics_only: bool = False,
    dimensions_only: bool = False,
    extended_fields: bool = False,
    side_by_side: bool = False,
    no_color: bool = False,
    quiet_diff: bool = False,
    reverse_diff: bool = False,
    warn_threshold: float | None = None,
    group_by_field: bool = False,
    group_by_field_limit: int = 10,
    diff_output: str | None = None,
    format_pr_comment: bool = False,
    auto_snapshot: bool = False,
    auto_prune: bool = False,
    snapshot_dir: str = "./snapshots",
    keep_last: int = 0,
    keep_since: str | None = None,
    keep_last_specified: bool = False,
    keep_since_specified: bool = False,
    profile: str | None = None,
    include_calc_metrics: bool = False,
    include_segments: bool = False,
) -> tuple[bool, bool, int | None]:
    """
    Handle the --diff-snapshot command to compare a data view against a saved snapshot.

    Args:
        data_view_id: The current data view ID to compare
        snapshot_file: Path to the saved snapshot file
        config_file: Path to CJA configuration file
        output_format: Output format
        output_dir: Output directory
        changes_only: Only show changed items
        summary_only: Only show summary
        ignore_fields: Fields to ignore
        labels: Custom labels (source_label, target_label)
        quiet: Suppress progress output
        show_only: Filter to show only specific change types
        metrics_only: Only compare metrics
        dimensions_only: Only compare dimensions
        extended_fields: Use extended field comparison
        side_by_side: Show side-by-side comparison view
        no_color: Disable ANSI color codes
        quiet_diff: Suppress output, only return exit code
        reverse_diff: Swap source and target
        warn_threshold: Exit with code 3 if change % exceeds threshold
        group_by_field: Group changes by field name
        group_by_field_limit: Max items per section in group-by-field output (0 = unlimited)
        diff_output: Write output to file instead of stdout
        format_pr_comment: Output in PR comment format
        auto_snapshot: Automatically save snapshot of current data view state
        auto_prune: Apply default retention when --auto-snapshot is enabled
        snapshot_dir: Directory for auto-saved snapshots
        keep_last: Retention policy - keep only last N snapshots per data view (0 = keep all)
        keep_since: Date-based retention - delete snapshots older than this period (e.g., '7d', '2w', '1m')
        keep_last_specified: Whether --keep-last was explicitly provided
        keep_since_specified: Whether --keep-since was explicitly provided
        profile: Optional profile name for credentials
        include_calc_metrics: Include calculated metrics inventory in comparison
        include_segments: Include segments inventory in comparison

    Returns:
        Tuple of (success, has_changes, exit_code_override)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if not quiet else logging.WARNING)

        if not quiet and not quiet_diff:
            print()
            print("=" * BANNER_WIDTH)
            print("COMPARING DATA VIEW AGAINST SNAPSHOT")
            print("=" * BANNER_WIDTH)
            print(f"Data View: {data_view_id}")
            print(f"Snapshot: {snapshot_file}")
            if reverse_diff:
                print("(Reversed comparison)")
            if include_calc_metrics or include_segments:
                inv_types = []
                if include_calc_metrics:
                    inv_types.append("calculated metrics")
                if include_segments:
                    inv_types.append("segments")
                print(f"Including inventory: {', '.join(inv_types)}")
            print()

        # Load the saved snapshot (source/baseline)
        snapshot_manager = SnapshotManager(logger)
        source_snapshot = snapshot_manager.load_snapshot(snapshot_file)

        # Validate snapshot has required inventory data
        missing_inventory = []
        if include_calc_metrics and not source_snapshot.has_calculated_metrics_inventory:
            missing_inventory.append("calculated metrics")
        if include_segments and not source_snapshot.has_segments_inventory:
            missing_inventory.append("segments")

        if missing_inventory:
            inv_summary = source_snapshot.get_inventory_summary()
            print(
                ConsoleColors.error("ERROR: Cannot perform inventory diff - snapshot missing requested data."),
                file=sys.stderr,
            )
            print(file=sys.stderr)
            print(f"Snapshot '{snapshot_file}' contains:", file=sys.stderr)
            print(f"  {'✓' if True else '✗'} Metrics ({len(source_snapshot.metrics)} items)", file=sys.stderr)
            print(f"  {'✓' if True else '✗'} Dimensions ({len(source_snapshot.dimensions)} items)", file=sys.stderr)
            print(
                f"  {'✓' if inv_summary['calculated_metrics']['present'] else '✗'} Calculated Metrics Inventory ({inv_summary['calculated_metrics']['count']} items)",
                file=sys.stderr,
            )
            print(
                f"  {'✓' if inv_summary['segments']['present'] else '✗'} Segments Inventory ({inv_summary['segments']['count']} items)",
                file=sys.stderr,
            )
            print(file=sys.stderr)
            print(f"You requested: {', '.join(missing_inventory)}", file=sys.stderr)
            print(file=sys.stderr)
            print("To create a compatible snapshot:", file=sys.stderr)
            flags = []
            if include_calc_metrics:
                flags.append("--include-calculated")
            if include_segments:
                flags.append("--include-segments")
            print(f"  cja-auto-sdr --sdr {data_view_id} {' '.join(flags)} --auto-snapshot", file=sys.stderr)
            return False, False, None

        # Initialize CJA with profile support
        success, source, _ = configure_cjapy(profile=profile, config_file=config_file, logger=logger)
        if not success:
            if not quiet and not quiet_diff:
                print(ConsoleColors.error(f"ERROR: Configuration failed: {source}"), file=sys.stderr)
            return False, False, None
        cja = cjapy.CJA()

        if not quiet and not quiet_diff:
            print("Fetching current data view state...")
        target_snapshot = snapshot_manager.create_snapshot(cja, data_view_id, quiet or quiet_diff)

        # Build inventory for target snapshot if requested
        if include_calc_metrics or include_segments:
            if not quiet and not quiet_diff:
                print("Building inventory for current state...")

            if include_calc_metrics:
                try:
                    from cja_auto_sdr.inventory.calculated_metrics import CalculatedMetricsInventoryBuilder

                    builder = CalculatedMetricsInventoryBuilder(logger=logger)
                    inventory = builder.build(cja, data_view_id, target_snapshot.data_view_name)
                    target_snapshot.calculated_metrics_inventory = [m.to_full_dict() for m in inventory.metrics]
                    if not quiet and not quiet_diff:
                        print(f"  Calculated metrics: {len(target_snapshot.calculated_metrics_inventory)} items")
                except Exception as e:
                    logger.warning(f"Failed to build calculated metrics inventory: {e}")

            if include_segments:
                try:
                    from cja_auto_sdr.inventory.segments import SegmentsInventoryBuilder

                    builder = SegmentsInventoryBuilder(logger=logger)
                    inventory = builder.build(cja, data_view_id, target_snapshot.data_view_name)
                    target_snapshot.segments_inventory = [s.to_full_dict() for s in inventory.segments]
                    if not quiet and not quiet_diff:
                        print(f"  Segments: {len(target_snapshot.segments_inventory)} items")
                except Exception as e:
                    logger.warning(f"Failed to build segments inventory: {e}")

            if not quiet and not quiet_diff:
                print()

        # Auto-save current state snapshot if enabled
        if auto_snapshot:
            os.makedirs(snapshot_dir, exist_ok=True)

            effective_keep_last, effective_keep_since = resolve_auto_prune_retention(
                keep_last=keep_last,
                keep_since=keep_since,
                auto_prune=auto_prune,
                keep_last_specified=keep_last_specified,
                keep_since_specified=keep_since_specified,
            )

            # Save current state snapshot
            current_filename = snapshot_manager.generate_snapshot_filename(data_view_id, target_snapshot.data_view_name)
            current_path = os.path.join(snapshot_dir, current_filename)
            snapshot_manager.save_snapshot(target_snapshot, current_path)

            if not quiet and not quiet_diff:
                print(f"Auto-saved current state to: {snapshot_dir}/{current_filename}")

            # Apply retention policies if configured
            total_deleted = 0

            # Count-based retention
            if effective_keep_last > 0:
                deleted = snapshot_manager.apply_retention_policy(snapshot_dir, data_view_id, effective_keep_last)
                total_deleted += len(deleted)

            # Date-based retention
            if effective_keep_since:
                days = parse_retention_period(effective_keep_since)
                if days:
                    deleted = snapshot_manager.apply_date_retention_policy(
                        snapshot_dir, data_view_id, keep_since_days=days
                    )
                    total_deleted += len(deleted)

            if total_deleted > 0 and not quiet and not quiet_diff:
                print(f"  Retention policy: Deleted {total_deleted} old snapshot(s)")

            if not quiet and not quiet_diff:
                print()

        # Handle reverse_diff - swap source and target
        if reverse_diff:
            source_snapshot, target_snapshot = target_snapshot, source_snapshot

        # Compare (snapshot is baseline/source, current state is target)
        source_label = labels[0] if labels else f"Snapshot ({source_snapshot.created_at[:10]})"
        target_label = labels[1] if labels else "Current"

        comparator = DataViewComparator(
            logger,
            ignore_fields=ignore_fields,
            use_extended_fields=extended_fields,
            show_only=show_only,
            metrics_only=metrics_only,
            dimensions_only=dimensions_only,
            include_calc_metrics=include_calc_metrics,
            include_segments=include_segments,
        )
        diff_result = comparator.compare(source_snapshot, target_snapshot, source_label, target_label)

        # Check warn threshold
        exit_code_override = None
        if warn_threshold is not None:
            max_change_pct = max(
                diff_result.summary.metrics_change_percent, diff_result.summary.dimensions_change_percent
            )
            if max_change_pct > warn_threshold:
                exit_code_override = 3
                if not quiet_diff:
                    print(
                        ConsoleColors.warning(
                            f"WARNING: Change threshold exceeded! {max_change_pct:.1f}% > {warn_threshold}%"
                        ),
                        file=sys.stderr,
                    )

        # Generate output (unless quiet_diff is set)
        if not quiet_diff:
            # Determine effective format
            effective_format = "pr-comment" if format_pr_comment else output_format

            base_filename = f"diff_{data_view_id}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_content = write_diff_output(
                diff_result,
                effective_format,
                base_filename,
                output_dir,
                logger,
                changes_only,
                summary_only,
                side_by_side,
                use_color=ConsoleColors.is_enabled() and not no_color,
                group_by_field=group_by_field,
                group_by_field_limit=group_by_field_limit,
            )

            # Handle --diff-output flag
            if diff_output and output_content:
                with open(diff_output, "w", encoding="utf-8") as f:
                    f.write(output_content)
                if not quiet:
                    print(f"Diff output written to: {diff_output}")

            if not quiet and output_format != "console":
                print()
                print(ConsoleColors.success("Diff report generated successfully"))

        append_github_step_summary(build_diff_step_summary(diff_result), logger)
        return True, diff_result.summary.has_changes, exit_code_override

    except FileNotFoundError:
        print(ConsoleColors.error(f"ERROR: Snapshot file not found: {snapshot_file}"), file=sys.stderr)
        return False, False, None
    except ValueError as e:
        print(ConsoleColors.error(f"ERROR: Invalid snapshot file: {e!s}"), file=sys.stderr)
        return False, False, None
    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to compare against snapshot: {e!s}"), file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False, False, None


def handle_compare_snapshots_command(
    source_file: str,
    target_file: str,
    output_format: str = "console",
    output_dir: str = ".",
    changes_only: bool = False,
    summary_only: bool = False,
    ignore_fields: list[str] | None = None,
    labels: tuple[str, str] | None = None,
    quiet: bool = False,
    show_only: list[str] | None = None,
    metrics_only: bool = False,
    dimensions_only: bool = False,
    extended_fields: bool = False,
    side_by_side: bool = False,
    no_color: bool = False,
    quiet_diff: bool = False,
    reverse_diff: bool = False,
    warn_threshold: float | None = None,
    group_by_field: bool = False,
    group_by_field_limit: int = 10,
    diff_output: str | None = None,
    format_pr_comment: bool = False,
    include_calc_metrics: bool = False,
    include_segments: bool = False,
) -> tuple[bool, bool, int | None]:
    """
    Handle the --compare-snapshots command to compare two snapshot files directly.

    This is useful for:
    - Comparing snapshots from different points in time
    - Offline comparison without API access
    - CI/CD pipelines where you want to compare pre/post snapshots

    Args:
        source_file: Path to the source (baseline) snapshot file
        target_file: Path to the target snapshot file
        output_format: Output format
        output_dir: Output directory
        changes_only: Only show changed items
        summary_only: Only show summary
        ignore_fields: Fields to ignore
        labels: Custom labels (source_label, target_label)
        quiet: Suppress progress output
        show_only: Filter to show only specific change types
        metrics_only: Only compare metrics
        dimensions_only: Only compare dimensions
        extended_fields: Use extended field comparison
        side_by_side: Show side-by-side comparison view
        no_color: Disable ANSI color codes
        quiet_diff: Suppress output, only return exit code
        reverse_diff: Swap source and target
        warn_threshold: Exit with code 3 if change % exceeds threshold
        group_by_field: Group changes by field name
        group_by_field_limit: Max items per section in group-by-field output (0 = unlimited)
        diff_output: Write output to file instead of stdout
        format_pr_comment: Output in PR comment format
        include_calc_metrics: Include calculated metrics inventory in comparison
        include_segments: Include segments inventory in comparison

    Returns:
        Tuple of (success, has_changes, exit_code_override)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if not quiet else logging.WARNING)

        if not quiet and not quiet_diff:
            print()
            print("=" * BANNER_WIDTH)
            print("COMPARING TWO SNAPSHOTS")
            print("=" * BANNER_WIDTH)
            print(f"Source: {source_file}")
            print(f"Target: {target_file}")
            if reverse_diff:
                print("(Reversed comparison)")
            print()

        # Load both snapshots
        snapshot_manager = SnapshotManager(logger)

        if not quiet and not quiet_diff:
            print("Loading source snapshot...")
        source_snapshot = snapshot_manager.load_snapshot(source_file)

        if not quiet and not quiet_diff:
            print("Loading target snapshot...")
        target_snapshot = snapshot_manager.load_snapshot(target_file)

        # Validate same data view for inventory comparison
        if (include_calc_metrics or include_segments) and source_snapshot.data_view_id != target_snapshot.data_view_id:
            print(
                ConsoleColors.error("ERROR: Inventory comparison requires snapshots from the same data view."),
                file=sys.stderr,
            )
            print(f"  Source: {source_snapshot.data_view_name} ({source_snapshot.data_view_id})", file=sys.stderr)
            print(f"  Target: {target_snapshot.data_view_name} ({target_snapshot.data_view_id})", file=sys.stderr)
            print(file=sys.stderr)
            print(
                "Inventory IDs are data-view-scoped and cannot be matched across different data views.",
                file=sys.stderr,
            )
            print(
                "Remove --include-segments, --include-calculated, --include-derived for cross-data-view comparison.",
                file=sys.stderr,
            )
            return False, False, None

        # Handle reverse_diff - swap source and target
        if reverse_diff:
            source_snapshot, target_snapshot = target_snapshot, source_snapshot
            source_file, target_file = target_file, source_file

        # Determine labels
        if labels:
            source_label, target_label = labels
        else:
            # Use snapshot metadata for labels
            source_label = f"{source_snapshot.data_view_name} ({source_snapshot.created_at[:10]})"
            target_label = f"{target_snapshot.data_view_name} ({target_snapshot.created_at[:10]})"

        if not quiet and not quiet_diff:
            print(f"Comparing: {source_label} vs {target_label}")
            print()

            # Show snapshot metadata
            print("Snapshot Details:")
            print("-" * 40)

            # Source snapshot info
            source_size = os.path.getsize(source_file)
            source_size_str = f"{source_size:,} bytes" if source_size < 1024 else f"{source_size / 1024:.1f} KB"
            print("  Source:")
            print(f"    File: {Path(source_file).name} ({source_size_str})")
            print(f"    Created: {source_snapshot.created_at}")
            print(f"    Data View: {source_snapshot.data_view_name} ({source_snapshot.data_view_id})")
            print(f"    Metrics: {len(source_snapshot.metrics):,} | Dimensions: {len(source_snapshot.dimensions):,}")

            # Target snapshot info
            target_size = os.path.getsize(target_file)
            target_size_str = f"{target_size:,} bytes" if target_size < 1024 else f"{target_size / 1024:.1f} KB"
            print("  Target:")
            print(f"    File: {Path(target_file).name} ({target_size_str})")
            print(f"    Created: {target_snapshot.created_at}")
            print(f"    Data View: {target_snapshot.data_view_name} ({target_snapshot.data_view_id})")
            print(f"    Metrics: {len(target_snapshot.metrics):,} | Dimensions: {len(target_snapshot.dimensions):,}")

            print("-" * 40)
            print()

        # Compare snapshots
        comparator = DataViewComparator(
            logger,
            ignore_fields=ignore_fields,
            use_extended_fields=extended_fields,
            show_only=show_only,
            metrics_only=metrics_only,
            dimensions_only=dimensions_only,
            include_calc_metrics=include_calc_metrics,
            include_segments=include_segments,
        )
        diff_result = comparator.compare(source_snapshot, target_snapshot, source_label, target_label)

        # Check warn threshold
        exit_code_override = None
        if warn_threshold is not None:
            max_change_pct = max(
                diff_result.summary.metrics_change_percent, diff_result.summary.dimensions_change_percent
            )
            if max_change_pct > warn_threshold:
                exit_code_override = 3
                if not quiet_diff:
                    print(
                        ConsoleColors.warning(
                            f"WARNING: Change threshold exceeded! {max_change_pct:.1f}% > {warn_threshold}%"
                        ),
                        file=sys.stderr,
                    )

        # Generate output (unless quiet_diff is set)
        if not quiet_diff:
            # Determine effective format
            effective_format = "pr-comment" if format_pr_comment else output_format

            # Generate base filename from snapshot names
            source_base = Path(source_file).stem
            target_base = Path(target_file).stem
            base_filename = f"diff_{source_base}_vs_{target_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            output_content = write_diff_output(
                diff_result,
                effective_format,
                base_filename,
                output_dir,
                logger,
                changes_only,
                summary_only,
                side_by_side,
                use_color=ConsoleColors.is_enabled() and not no_color,
                group_by_field=group_by_field,
                group_by_field_limit=group_by_field_limit,
            )

            # Handle --diff-output flag
            if diff_output and output_content:
                with open(diff_output, "w", encoding="utf-8") as f:
                    f.write(output_content)
                if not quiet:
                    print(f"Diff output written to: {diff_output}")

            if not quiet and output_format != "console":
                print()
                print(ConsoleColors.success("Diff report generated successfully"))

        append_github_step_summary(build_diff_step_summary(diff_result), logger)
        return True, diff_result.summary.has_changes, exit_code_override

    except FileNotFoundError as e:
        print(ConsoleColors.error(f"ERROR: Snapshot file not found: {e!s}"), file=sys.stderr)
        return False, False, None
    except ValueError as e:
        print(ConsoleColors.error(f"ERROR: Invalid snapshot file: {e!s}"), file=sys.stderr)
        return False, False, None
    except Exception as e:
        print(ConsoleColors.error(f"ERROR: Failed to compare snapshots: {e!s}"), file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False, False, None


# ==================== MAIN FUNCTION ====================


def main():
    """Main entry point for the script"""

    # Parse arguments (will show error and help if no data views provided)
    args = parse_arguments()

    # Configure global color policy for all ConsoleColors call sites.
    ConsoleColors.configure(no_color=getattr(args, "no_color", False))

    # Parse and validate --workers argument
    workers_auto = False
    if args.workers.lower() == "auto":
        workers_auto = True
        # Will be set later based on data view count
        args.workers = DEFAULT_BATCH_WORKERS  # Temporary default
    else:
        try:
            args.workers = int(args.workers)
        except ValueError:
            _exit_error(f"--workers must be 'auto' or an integer, got '{args.workers}'")

    # Validate numeric parameter bounds
    if not workers_auto and args.workers < 1:
        _exit_error("--workers must be at least 1")
    if not workers_auto and args.workers > MAX_BATCH_WORKERS:
        _exit_error(f"--workers cannot exceed {MAX_BATCH_WORKERS}")
    if args.cache_size < 1:
        _exit_error("--cache-size must be at least 1")
    if args.cache_ttl < 1:
        _exit_error("--cache-ttl must be at least 1 second")
    if args.max_issues < 0:
        _exit_error("--max-issues cannot be negative")
    if args.max_retries < 0:
        _exit_error("--max-retries cannot be negative")
    if args.retry_base_delay < 0:
        _exit_error("--retry-base-delay cannot be negative")
    if args.retry_max_delay < args.retry_base_delay:
        _exit_error("--retry-max-delay must be >= --retry-base-delay")

    # Track whether retention flags were explicitly provided so auto-prune
    # defaults don't override intentional values like --keep-last 0.
    keep_last_specified = _cli_option_specified("--keep-last")
    keep_since_specified = _cli_option_specified("--keep-since")

    non_sdr_modes_for_quality_options = (
        getattr(args, "diff", False)
        or getattr(args, "snapshot", None)
        or getattr(args, "diff_snapshot", None)
        or getattr(args, "compare_with_prev", False)
        or getattr(args, "compare_snapshots", None)
        or getattr(args, "org_report", False)
        or getattr(args, "inventory_summary", False)
        or getattr(args, "list_dataviews", False)
        or getattr(args, "list_connections", False)
        or getattr(args, "list_datasets", False)
        or getattr(args, "validate_config", False)
        or getattr(args, "config_status", False)
        or getattr(args, "config_json", False)
        or getattr(args, "sample_config", False)
        or getattr(args, "exit_codes", False)
        or getattr(args, "profile_list", False)
        or getattr(args, "profile_add", None)
        or getattr(args, "profile_test", None)
        or getattr(args, "profile_show", None)
        or getattr(args, "git_init", False)
        or getattr(args, "org_compare_report", None)
        or getattr(args, "stats", False)
        or getattr(args, "dry_run", False)
    )
    if getattr(args, "fail_on_quality", None) and non_sdr_modes_for_quality_options:
        _exit_error("--fail-on-quality is only supported in SDR generation mode")

    if getattr(args, "auto_prune", False) and not getattr(args, "auto_snapshot", False):
        _exit_error("--auto-prune requires --auto-snapshot")
    if getattr(args, "fail_on_quality", None) and args.skip_validation:
        _exit_error("--fail-on-quality cannot be used with --skip-validation")
    if getattr(args, "quality_report", None) and args.skip_validation:
        _exit_error("--quality-report cannot be used with --skip-validation")
    if getattr(args, "quality_report", None) and non_sdr_modes_for_quality_options:
        _exit_error("--quality-report is only supported in SDR generation mode")

    # Propagate retry config via env vars so both the current process
    # (read by _effective_retry_config in resilience.py) and child
    # processes (ProcessPoolExecutor workers) pick up CLI overrides.
    os.environ["MAX_RETRIES"] = str(args.max_retries)
    os.environ["RETRY_BASE_DELAY"] = str(args.retry_base_delay)
    os.environ["RETRY_MAX_DELAY"] = str(args.retry_max_delay)

    # Handle --output for stdout - implies quiet mode
    output_to_stdout = getattr(args, "output", None) in ("-", "stdout")
    if output_to_stdout:
        args.quiet = True

    # Auto-detect format from output file extension if --format not explicitly set
    output_path = getattr(args, "output", None)
    if output_path and not args.format:
        inferred_format = infer_format_from_path(output_path)
        if inferred_format:
            args.format = inferred_format
            if not args.quiet:
                print(f"Auto-detected format '{inferred_format}' from output file extension")

    # Set color theme for diff output (accessible accessibility)
    color_theme = getattr(args, "color_theme", "default")
    if color_theme and color_theme != "default":
        ConsoleColors.set_theme(color_theme)

    # Handle --exit-codes mode (no data view required)
    if getattr(args, "exit_codes", False):
        print("=" * BANNER_WIDTH)
        print("EXIT CODE REFERENCE")
        print("=" * BANNER_WIDTH)
        print()
        print("  Code  Meaning")
        print("  ----  " + "-" * 50)
        print("    0   Success")
        print("        - SDR generated successfully")
        print("        - Diff comparison: no changes found")
        print("        - Validation passed")
        print()
        print("    1   Error occurred")
        print("        - Configuration error (invalid credentials, missing file)")
        print("        - API error (network, authentication, rate limit)")
        print("        - Validation failed")
        print("        - File I/O error")
        print()
        print("    2   Policy threshold exceeded (not a runtime error)")
        print("        - Diff mode: changes found")
        print("        - SDR mode: quality gate failed (--fail-on-quality)")
        print("        - Org mode: governance threshold failed (--fail-on-threshold)")
        print()
        print("    3   Diff: Warning threshold exceeded")
        print("        - Triggered by --warn-threshold PERCENT")
        print("        - Example: cja_auto_sdr --diff dv_A dv_B --warn-threshold 10")
        print("        - Exits 3 if change percentage > threshold")
        print()
        print("=" * BANNER_WIDTH)
        print("CI/CD Examples:")
        print("=" * BANNER_WIDTH)
        print()
        print("  # Fail CI if any changes detected")
        print("  cja_auto_sdr --diff dv_prod dv_staging --quiet")
        print("  if [ $? -eq 2 ]; then echo 'Changes detected!'; exit 1; fi")
        print()
        print("  # Fail CI only if >10% changes")
        print("  cja_auto_sdr --diff dv_A dv_B --warn-threshold 10 --quiet")
        print()
        sys.exit(0)

    # Handle --sample-config mode (no data view required)
    if args.sample_config:
        success = generate_sample_config()
        sys.exit(0 if success else 1)

    # ==================== PROFILE MANAGEMENT COMMANDS ====================

    # Handle --profile-list mode (no data view required)
    if getattr(args, "profile_list", False):
        list_format = "json" if args.format == "json" else "table"
        success = list_profiles(output_format=list_format)
        sys.exit(0 if success else 1)

    # Handle --profile-add mode (no data view required)
    if getattr(args, "profile_add", None):
        success = add_profile_interactive(args.profile_add)
        sys.exit(0 if success else 1)

    # Handle --profile-test mode (no data view required)
    if getattr(args, "profile_test", None):
        success = test_profile(args.profile_test)
        sys.exit(0 if success else 1)

    # Handle --profile-show mode (no data view required)
    if getattr(args, "profile_show", None):
        success = show_profile(args.profile_show)
        sys.exit(0 if success else 1)

    # Handle --git-init mode (no data view required)
    if getattr(args, "git_init", False):
        git_dir = Path(getattr(args, "git_dir", "./sdr-snapshots"))
        print(f"Initializing Git repository at: {git_dir}")
        success, message = git_init_snapshot_repo(git_dir)
        if success:
            print(ConsoleColors.success(f"SUCCESS: {message}"))
            print(f"  Directory: {git_dir.absolute()}")
            print()
            print("Next steps:")
            print("  1. Run SDR generation with --git-commit to save and commit snapshots")
            print(f"  2. Add a remote: cd {git_dir} && git remote add origin <url>")
            print("  3. Use --git-push to push commits to remote")
        else:
            print(ConsoleColors.error(f"FAILED: {message}"))
        sys.exit(0 if success else 1)

    # Validate Git argument combinations
    if getattr(args, "git_push", False) and not getattr(args, "git_commit", False):
        print(ConsoleColors.error("ERROR: --git-push requires --git-commit"), file=sys.stderr)
        sys.exit(1)

    # Validate: --include-derived is not supported with --git-commit
    # Derived fields are for SDR generation only - they're computed from metrics/dimensions
    if getattr(args, "git_commit", False) and getattr(args, "include_derived_inventory", False):
        print(ConsoleColors.error("ERROR: --include-derived cannot be used with --git-commit"), file=sys.stderr)
        print("Derived fields inventory is only available in SDR generation mode.", file=sys.stderr)
        print("Derived field changes are captured in the standard Metrics/Dimensions diff.", file=sys.stderr)
        sys.exit(1)

    # Handle discovery commands (no data view required)
    _discovery_commands = {
        "list_dataviews": list_dataviews,
        "list_connections": list_connections,
        "list_datasets": list_datasets,
    }
    for attr, func in _discovery_commands.items():
        if getattr(args, attr, False):
            list_format = "table"
            if args.format in ("json", "csv"):
                list_format = args.format
            elif output_to_stdout:
                list_format = "json"
            success = func(
                args.config_file,
                output_format=list_format,
                output_file=getattr(args, "output", None),
                profile=getattr(args, "profile", None),
            )
            sys.exit(0 if success else 1)

    # Handle --config-status mode (no data view required, no API call)
    # --config-json implies --config-status
    if getattr(args, "config_status", False) or getattr(args, "config_json", False):
        output_json = getattr(args, "config_json", False)
        success = show_config_status(args.config_file, profile=getattr(args, "profile", None), output_json=output_json)
        sys.exit(0 if success else 1)

    # Handle --validate-config mode (no data view required)
    if args.validate_config:
        success = validate_config_only(args.config_file, profile=getattr(args, "profile", None))
        sys.exit(0 if success else 1)

    # Get data views from arguments
    data_view_inputs = args.data_views

    # Handle --interactive mode (full wizard for guided SDR generation)
    if getattr(args, "interactive", False):
        if data_view_inputs:
            print(ConsoleColors.warning("Note: --interactive mode ignores command line arguments"))

        wizard_config = interactive_wizard(args.config_file, profile=getattr(args, "profile", None))
        if wizard_config is None:
            print("Cancelled. Exiting.")
            sys.exit(0)

        # Apply wizard selections to args
        data_view_inputs = wizard_config.data_view_ids
        args.format = wizard_config.output_format
        args.include_segments_inventory = wizard_config.include_segments
        args.include_calculated_metrics = wizard_config.include_calculated
        args.include_derived_inventory = wizard_config.include_derived
        args.inventory_only = wizard_config.inventory_only

        print()
        print("=" * BANNER_WIDTH)
        print("GENERATING SDR...")
        print("=" * BANNER_WIDTH)
        print()

    # Handle --stats mode (requires data views)
    if getattr(args, "stats", False):
        if not data_view_inputs:
            print(ConsoleColors.error("ERROR: --stats requires at least one data view ID or name"), file=sys.stderr)
            sys.exit(1)

        # Resolve data view names first
        temp_logger = logging.getLogger("name_resolution")
        temp_logger.setLevel(logging.WARNING)
        resolved_ids, _ = resolve_data_view_names(
            data_view_inputs, args.config_file, temp_logger, profile=getattr(args, "profile", None)
        )

        if not resolved_ids:
            print(ConsoleColors.error("ERROR: No valid data views found"), file=sys.stderr)
            sys.exit(1)

        # Determine format for stats output
        stats_format = "table"
        if args.format in ("json", "csv"):
            stats_format = args.format
        elif output_to_stdout:
            stats_format = "json"

        success = show_stats(
            resolved_ids,
            config_file=args.config_file,
            output_format=stats_format,
            output_file=getattr(args, "output", None),
            quiet=args.quiet,
            profile=getattr(args, "profile", None),
        )
        sys.exit(0 if success else 1)

    # Handle --org-report mode (no data views required)
    if getattr(args, "org_report", False):
        # Build config from args
        org_config = OrgReportConfig(
            filter_pattern=getattr(args, "org_filter", None),
            exclude_pattern=getattr(args, "org_exclude", None),
            limit=getattr(args, "org_limit", None),
            core_threshold=getattr(args, "core_threshold", 0.5),
            core_min_count=getattr(args, "core_min_count", None),
            overlap_threshold=getattr(args, "overlap_threshold", 0.8),
            summary_only=getattr(args, "org_summary", False) or getattr(args, "summary", False),
            verbose=getattr(args, "org_verbose", False),
            include_names=getattr(args, "org_include_names", False),
            skip_similarity=getattr(args, "skip_similarity", False),
            similarity_max_dvs=getattr(args, "org_similarity_max_dvs", 250),
            force_similarity=getattr(args, "org_force_similarity", False),
            # Existing options
            include_component_types=not getattr(args, "no_component_types", False),
            include_metadata=getattr(args, "org_include_metadata", False),
            include_drift=getattr(args, "org_include_drift", False),
            sample_size=getattr(args, "org_sample_size", None),
            sample_seed=getattr(args, "org_sample_seed", None),
            sample_stratified=getattr(args, "org_sample_stratified", False),
            use_cache=getattr(args, "org_use_cache", False),
            cache_max_age_hours=getattr(args, "org_cache_max_age", 24),
            clear_cache=getattr(args, "org_clear_cache", False),
            validate_cache=getattr(args, "org_validate_cache", False),
            memory_warning_threshold_mb=getattr(args, "org_memory_warning", 100),
            memory_limit_mb=getattr(args, "org_memory_limit", None),
            enable_clustering=getattr(args, "org_cluster", False),
            cluster_method=getattr(args, "org_cluster_method", "average"),
            quiet=args.quiet,
            cja_per_thread=not getattr(args, "org_shared_client", False),
            # Feature 1: Governance thresholds
            duplicate_threshold=getattr(args, "org_duplicate_threshold", None),
            isolated_threshold=getattr(args, "org_isolated_threshold", None),
            fail_on_threshold=getattr(args, "org_fail_on_threshold", False),
            # Feature 2: Org-stats mode
            org_stats_only=getattr(args, "org_stats_only", False),
            # Feature 3: Naming audit
            audit_naming=getattr(args, "org_audit_naming", False),
            # Feature 4: Trending/drift report
            compare_org_report=getattr(args, "org_compare_report", None),
            # Feature 5: Owner summary
            include_owner_summary=getattr(args, "org_owner_summary", False),
            # Feature 6: Stale component heuristics
            flag_stale=getattr(args, "org_flag_stale", False),
        )

        # Determine output format (default to console for org reports)
        output_format = args.format if args.format else "console"

        success, thresholds_exceeded = run_org_report(
            config_file=args.config_file,
            output_format=output_format,
            output_path=getattr(args, "output", None),
            output_dir=args.output_dir,
            org_config=org_config,
            profile=getattr(args, "profile", None),
            quiet=args.quiet,
        )

        # Exit code: 0 = success, 1 = error, 2 = thresholds exceeded (with --fail-on-threshold)
        if success:
            if thresholds_exceeded and org_config.fail_on_threshold:
                sys.exit(2)
            sys.exit(0)
        else:
            sys.exit(1)

    # Parse ignore_fields if provided
    ignore_fields = None
    if hasattr(args, "ignore_fields") and args.ignore_fields:
        ignore_fields = [f.strip() for f in args.ignore_fields.split(",")]

    # Parse show_only filter if provided
    show_only = None
    if hasattr(args, "show_only") and args.show_only:
        show_only = [t.strip().lower() for t in args.show_only.split(",")]
        valid_types = {"added", "removed", "modified", "unchanged"}
        invalid = set(show_only) - valid_types
        if invalid:
            print(ConsoleColors.error(f"ERROR: Invalid --show-only types: {invalid}"), file=sys.stderr)
            print(f"Valid types: {', '.join(valid_types)}", file=sys.stderr)
            sys.exit(1)

    # Parse labels if provided
    labels = None
    if hasattr(args, "diff_labels") and args.diff_labels:
        labels = tuple(args.diff_labels)

    # Handle --compare-snapshots mode (compare two snapshot files directly)
    if hasattr(args, "compare_snapshots") and args.compare_snapshots:
        source_file, target_file = args.compare_snapshots

        # Check for conflicting options
        if getattr(args, "metrics_only", False) and getattr(args, "dimensions_only", False):
            print(ConsoleColors.error("ERROR: Cannot use both --metrics-only and --dimensions-only"), file=sys.stderr)
            sys.exit(1)

        # Check for inventory-only (not supported in diff mode - inventory comparison is part of diff output)
        if getattr(args, "inventory_only", False):
            print(
                ConsoleColors.error(
                    "ERROR: --inventory-only is only available in SDR mode, not with --compare-snapshots"
                ),
                file=sys.stderr,
            )
            sys.exit(1)

        # Default to console for diff commands
        diff_format = args.format if args.format else "console"
        success, has_changes, exit_code_override = handle_compare_snapshots_command(
            source_file=source_file,
            target_file=target_file,
            output_format=diff_format,
            output_dir=args.output_dir,
            changes_only=getattr(args, "changes_only", False),
            summary_only=getattr(args, "summary", False),
            ignore_fields=ignore_fields,
            labels=labels,
            quiet=args.quiet,
            show_only=show_only,
            metrics_only=getattr(args, "metrics_only", False),
            dimensions_only=getattr(args, "dimensions_only", False),
            extended_fields=getattr(args, "extended_fields", False),
            side_by_side=getattr(args, "side_by_side", False),
            no_color=getattr(args, "no_color", False),
            quiet_diff=getattr(args, "quiet_diff", False),
            reverse_diff=getattr(args, "reverse_diff", False),
            warn_threshold=getattr(args, "warn_threshold", None),
            group_by_field=getattr(args, "group_by_field", False),
            group_by_field_limit=getattr(args, "group_by_field_limit", 10),
            diff_output=getattr(args, "diff_output", None),
            format_pr_comment=getattr(args, "format_pr_comment", False),
            include_calc_metrics=getattr(args, "include_calculated_metrics", False),
            include_segments=getattr(args, "include_segments_inventory", False),
        )

        # Exit with code 3 if threshold exceeded, 2 if differences found, 0 if no changes
        if success:
            if exit_code_override is not None:
                sys.exit(exit_code_override)
            sys.exit(2 if has_changes else 0)
        else:
            sys.exit(1)

    # Handle --diff mode (compare two data views)
    if hasattr(args, "diff") and args.diff:
        if len(data_view_inputs) != 2:
            print(ConsoleColors.error("ERROR: --diff requires exactly 2 data view IDs or names"), file=sys.stderr)
            print("Usage: cja_auto_sdr --diff DATA_VIEW_A DATA_VIEW_B", file=sys.stderr)
            sys.exit(1)

        # Check for conflicting options
        if getattr(args, "metrics_only", False) and getattr(args, "dimensions_only", False):
            print(ConsoleColors.error("ERROR: Cannot use both --metrics-only and --dimensions-only"), file=sys.stderr)
            sys.exit(1)

        # Check for inventory options (not supported in cross-DV diff - IDs are data-view-scoped)
        # Inventory diff is only supported for same-data-view snapshot comparisons
        if getattr(args, "include_derived_inventory", False):
            print(
                ConsoleColors.error("ERROR: --include-derived cannot be used with --diff (cross-data-view comparison)"),
                file=sys.stderr,
            )
            print(
                "Inventory IDs are data-view-scoped and cannot be matched across different data views.", file=sys.stderr
            )
            print(
                "For same-data-view comparisons, use: --diff-snapshot, --compare-snapshots, or --compare-with-prev",
                file=sys.stderr,
            )
            sys.exit(1)
        if getattr(args, "include_calculated_metrics", False):
            print(
                ConsoleColors.error(
                    "ERROR: --include-calculated cannot be used with --diff (cross-data-view comparison)"
                ),
                file=sys.stderr,
            )
            print(
                "Inventory IDs are data-view-scoped and cannot be matched across different data views.", file=sys.stderr
            )
            print(
                "For same-data-view comparisons, use: --diff-snapshot, --compare-snapshots, or --compare-with-prev",
                file=sys.stderr,
            )
            sys.exit(1)
        if getattr(args, "include_segments_inventory", False):
            print(
                ConsoleColors.error(
                    "ERROR: --include-segments cannot be used with --diff (cross-data-view comparison)"
                ),
                file=sys.stderr,
            )
            print(
                "Inventory IDs are data-view-scoped and cannot be matched across different data views.", file=sys.stderr
            )
            print(
                "For same-data-view comparisons, use: --diff-snapshot, --compare-snapshots, or --compare-with-prev",
                file=sys.stderr,
            )
            sys.exit(1)
        if getattr(args, "inventory_only", False):
            print(
                ConsoleColors.error("ERROR: --inventory-only is only available in SDR mode, not with --diff"),
                file=sys.stderr,
            )
            sys.exit(1)

        # Resolve names to IDs if needed - resolve EACH identifier separately
        # to ensure 1:1 mapping for diff comparison
        temp_logger = logging.getLogger("name_resolution")
        temp_logger.setLevel(logging.WARNING)

        source_input = data_view_inputs[0]
        target_input = data_view_inputs[1]

        # Resolve source identifier
        source_resolved, _source_map = resolve_data_view_names(
            [source_input], args.config_file, temp_logger, profile=getattr(args, "profile", None)
        )
        if not source_resolved:
            print(ConsoleColors.error(f"ERROR: Could not resolve source data view: '{source_input}'"), file=sys.stderr)
            sys.exit(1)
        if len(source_resolved) > 1:
            # Ambiguous - try interactive selection if in terminal
            options = [(dv_id, f"{source_input} ({dv_id})") for dv_id in source_resolved]
            selected = prompt_for_selection(
                options, f"Source name '{source_input}' matches {len(source_resolved)} data views. Please select one:"
            )
            if selected:
                source_resolved = [selected]
            else:
                # Not interactive or user cancelled
                print(
                    ConsoleColors.error(
                        f"ERROR: Source name '{source_input}' is ambiguous - matches {len(source_resolved)} data views:"
                    ),
                    file=sys.stderr,
                )
                for dv_id in source_resolved:
                    print(f"  • {dv_id}", file=sys.stderr)
                print("\nPlease specify the exact data view ID instead of the name.", file=sys.stderr)
                sys.exit(1)

        # Resolve target identifier
        target_resolved, _target_map = resolve_data_view_names(
            [target_input], args.config_file, temp_logger, profile=getattr(args, "profile", None)
        )
        if not target_resolved:
            print(ConsoleColors.error(f"ERROR: Could not resolve target data view: '{target_input}'"), file=sys.stderr)
            sys.exit(1)
        if len(target_resolved) > 1:
            # Ambiguous - try interactive selection if in terminal
            options = [(dv_id, f"{target_input} ({dv_id})") for dv_id in target_resolved]
            selected = prompt_for_selection(
                options, f"Target name '{target_input}' matches {len(target_resolved)} data views. Please select one:"
            )
            if selected:
                target_resolved = [selected]
            else:
                # Not interactive or user cancelled
                print(
                    ConsoleColors.error(
                        f"ERROR: Target name '{target_input}' is ambiguous - matches {len(target_resolved)} data views:"
                    ),
                    file=sys.stderr,
                )
                for dv_id in target_resolved:
                    print(f"  • {dv_id}", file=sys.stderr)
                print("\nPlease specify the exact data view ID instead of the name.", file=sys.stderr)
                sys.exit(1)

        resolved_ids = [source_resolved[0], target_resolved[0]]

        # Default to console for diff commands
        diff_format = args.format if args.format else "console"
        success, has_changes, exit_code_override = handle_diff_command(
            source_id=resolved_ids[0],
            target_id=resolved_ids[1],
            config_file=args.config_file,
            output_format=diff_format,
            output_dir=args.output_dir,
            changes_only=getattr(args, "changes_only", False),
            summary_only=getattr(args, "summary", False),
            ignore_fields=ignore_fields,
            labels=labels,
            quiet=args.quiet,
            show_only=show_only,
            metrics_only=getattr(args, "metrics_only", False),
            dimensions_only=getattr(args, "dimensions_only", False),
            extended_fields=getattr(args, "extended_fields", False),
            side_by_side=getattr(args, "side_by_side", False),
            no_color=getattr(args, "no_color", False),
            quiet_diff=getattr(args, "quiet_diff", False),
            reverse_diff=getattr(args, "reverse_diff", False),
            warn_threshold=getattr(args, "warn_threshold", None),
            group_by_field=getattr(args, "group_by_field", False),
            group_by_field_limit=getattr(args, "group_by_field_limit", 10),
            diff_output=getattr(args, "diff_output", None),
            format_pr_comment=getattr(args, "format_pr_comment", False),
            auto_snapshot=getattr(args, "auto_snapshot", False),
            auto_prune=getattr(args, "auto_prune", False),
            snapshot_dir=getattr(args, "snapshot_dir", "./snapshots"),
            keep_last=getattr(args, "keep_last", 0),
            keep_since=getattr(args, "keep_since", None),
            keep_last_specified=keep_last_specified,
            keep_since_specified=keep_since_specified,
            profile=getattr(args, "profile", None),
        )

        # Exit with code 3 if threshold exceeded, 2 if differences found, 0 if no changes
        if success:
            if exit_code_override is not None:
                sys.exit(exit_code_override)
            sys.exit(2 if has_changes else 0)
        else:
            sys.exit(1)

    # Handle --snapshot mode (save a data view snapshot)
    if hasattr(args, "snapshot") and args.snapshot:
        if len(data_view_inputs) != 1:
            print(ConsoleColors.error("ERROR: --snapshot requires exactly 1 data view ID or name"), file=sys.stderr)
            print("Usage: cja_auto_sdr DATA_VIEW --snapshot ./snapshots/baseline.json", file=sys.stderr)
            sys.exit(1)

        # Validate: --include-derived is not supported with --snapshot
        # Derived fields are for SDR generation only - they're computed from metrics/dimensions
        if getattr(args, "include_derived_inventory", False):
            print(ConsoleColors.error("ERROR: --include-derived cannot be used with --snapshot"), file=sys.stderr)
            print("Derived fields inventory is only available in SDR generation mode.", file=sys.stderr)
            print("Derived field changes are captured in the standard Metrics/Dimensions diff.", file=sys.stderr)
            sys.exit(1)

        # Resolve name to ID if needed - ensure 1:1 mapping
        temp_logger = logging.getLogger("name_resolution")
        temp_logger.setLevel(logging.WARNING)
        resolved_ids, _ = resolve_data_view_names(
            data_view_inputs, args.config_file, temp_logger, profile=getattr(args, "profile", None)
        )

        if not resolved_ids:
            print(ConsoleColors.error(f"ERROR: Could not resolve data view: '{data_view_inputs[0]}'"), file=sys.stderr)
            sys.exit(1)
        if len(resolved_ids) > 1:
            # Ambiguous - try interactive selection if in terminal
            dv_name = data_view_inputs[0]
            options = [(dv_id, f"{dv_name} ({dv_id})") for dv_id in resolved_ids]
            selected = prompt_for_selection(
                options, f"Name '{dv_name}' matches {len(resolved_ids)} data views. Please select one:"
            )
            if selected:
                resolved_ids = [selected]
            else:
                print(
                    ConsoleColors.error(
                        f"ERROR: Name '{dv_name}' is ambiguous - matches {len(resolved_ids)} data views:"
                    ),
                    file=sys.stderr,
                )
                for dv_id in resolved_ids:
                    print(f"  • {dv_id}", file=sys.stderr)
                print("\nPlease specify the exact data view ID instead of the name.", file=sys.stderr)
                sys.exit(1)

        success = handle_snapshot_command(
            data_view_id=resolved_ids[0],
            snapshot_file=args.snapshot,
            config_file=args.config_file,
            quiet=args.quiet,
            profile=getattr(args, "profile", None),
            include_calculated_metrics=getattr(args, "include_calculated_metrics", False),
            include_segments=getattr(args, "include_segments_inventory", False),
        )
        sys.exit(0 if success else 1)

    # Handle --compare-with-prev mode (find most recent snapshot and compare)
    if getattr(args, "compare_with_prev", False):
        if len(data_view_inputs) != 1:
            print(
                ConsoleColors.error("ERROR: --compare-with-prev requires exactly 1 data view ID or name"),
                file=sys.stderr,
            )
            print("Usage: cja_auto_sdr DATA_VIEW --compare-with-prev", file=sys.stderr)
            sys.exit(1)

        # Check for inventory-only (not supported in diff mode - inventory comparison is part of diff output)
        if getattr(args, "inventory_only", False):
            print(
                ConsoleColors.error(
                    "ERROR: --inventory-only is only available in SDR mode, not with --compare-with-prev"
                ),
                file=sys.stderr,
            )
            sys.exit(1)

        # Resolve name to ID if needed
        temp_logger = logging.getLogger("name_resolution")
        temp_logger.setLevel(logging.WARNING)
        resolved_ids, _ = resolve_data_view_names(
            data_view_inputs, args.config_file, temp_logger, profile=getattr(args, "profile", None)
        )

        if not resolved_ids:
            print(ConsoleColors.error(f"ERROR: Could not resolve data view: '{data_view_inputs[0]}'"), file=sys.stderr)
            sys.exit(1)
        if len(resolved_ids) > 1:
            # Ambiguous - try interactive selection if in terminal
            dv_name = data_view_inputs[0]
            options = [(dv_id, f"{dv_name} ({dv_id})") for dv_id in resolved_ids]
            selected = prompt_for_selection(
                options, f"Name '{dv_name}' matches {len(resolved_ids)} data views. Please select one:"
            )
            if selected:
                resolved_ids = [selected]
            else:
                print(
                    ConsoleColors.error(
                        f"ERROR: Name '{dv_name}' is ambiguous - matches {len(resolved_ids)} data views:"
                    ),
                    file=sys.stderr,
                )
                for dv_id in resolved_ids:
                    print(f"  • {dv_id}", file=sys.stderr)
                print("\nPlease specify the exact data view ID instead of the name.", file=sys.stderr)
                sys.exit(1)

        # Find most recent snapshot
        snapshot_dir = getattr(args, "snapshot_dir", "./snapshots")
        snapshot_mgr = SnapshotManager()
        prev_snapshot = snapshot_mgr.get_most_recent_snapshot(snapshot_dir, resolved_ids[0])

        if not prev_snapshot:
            print(
                ConsoleColors.error(
                    f"ERROR: No previous snapshots found for data view '{resolved_ids[0]}' in {snapshot_dir}"
                ),
                file=sys.stderr,
            )
            print(
                f"Create a snapshot first with: cja_auto_sdr {resolved_ids[0]} --snapshot {snapshot_dir}/baseline.json",
                file=sys.stderr,
            )
            print("Or use --auto-snapshot with --diff to automatically save snapshots.", file=sys.stderr)
            sys.exit(1)

        if not args.quiet:
            print(f"Comparing against previous snapshot: {prev_snapshot}")

        # Set diff_snapshot and let the existing handler process it
        args.diff_snapshot = prev_snapshot

    # Handle --diff-snapshot mode (compare against a saved snapshot)
    if hasattr(args, "diff_snapshot") and args.diff_snapshot:
        if len(data_view_inputs) != 1:
            print(
                ConsoleColors.error("ERROR: --diff-snapshot requires exactly 1 data view ID or name"), file=sys.stderr
            )
            print("Usage: cja_auto_sdr DATA_VIEW --diff-snapshot ./snapshots/baseline.json", file=sys.stderr)
            sys.exit(1)

        # Check for conflicting options
        if getattr(args, "metrics_only", False) and getattr(args, "dimensions_only", False):
            print(ConsoleColors.error("ERROR: Cannot use both --metrics-only and --dimensions-only"), file=sys.stderr)
            sys.exit(1)

        # Check for inventory options
        # Note: --include-calculated, --include-segments ARE supported with --diff-snapshot
        # for inventory diff over time. --include-derived is NOT supported for diff since
        # derived fields are already captured in metrics/dimensions output.
        if getattr(args, "inventory_only", False):
            print(
                ConsoleColors.error("ERROR: --inventory-only is only available in SDR mode, not with --diff-snapshot"),
                file=sys.stderr,
            )
            sys.exit(1)

        # Get inventory flags (derived fields not supported in diff mode)
        include_calc_metrics = getattr(args, "include_calculated_metrics", False)
        include_segments = getattr(args, "include_segments_inventory", False)

        # Resolve name to ID if needed - ensure 1:1 mapping
        temp_logger = logging.getLogger("name_resolution")
        temp_logger.setLevel(logging.WARNING)
        resolved_ids, _ = resolve_data_view_names(
            data_view_inputs, args.config_file, temp_logger, profile=getattr(args, "profile", None)
        )

        if not resolved_ids:
            print(ConsoleColors.error(f"ERROR: Could not resolve data view: '{data_view_inputs[0]}'"), file=sys.stderr)
            sys.exit(1)
        if len(resolved_ids) > 1:
            # Ambiguous - try interactive selection if in terminal
            dv_name = data_view_inputs[0]
            options = [(dv_id, f"{dv_name} ({dv_id})") for dv_id in resolved_ids]
            selected = prompt_for_selection(
                options, f"Name '{dv_name}' matches {len(resolved_ids)} data views. Please select one:"
            )
            if selected:
                resolved_ids = [selected]
            else:
                print(
                    ConsoleColors.error(
                        f"ERROR: Name '{dv_name}' is ambiguous - matches {len(resolved_ids)} data views:"
                    ),
                    file=sys.stderr,
                )
                for dv_id in resolved_ids:
                    print(f"  • {dv_id}", file=sys.stderr)
                print("\nPlease specify the exact data view ID instead of the name.", file=sys.stderr)
                sys.exit(1)

        # Default to console for diff commands
        diff_format = args.format if args.format else "console"
        success, has_changes, exit_code_override = handle_diff_snapshot_command(
            data_view_id=resolved_ids[0],
            snapshot_file=args.diff_snapshot,
            config_file=args.config_file,
            output_format=diff_format,
            output_dir=args.output_dir,
            changes_only=getattr(args, "changes_only", False),
            summary_only=getattr(args, "summary", False),
            ignore_fields=ignore_fields,
            labels=labels,
            quiet=args.quiet,
            show_only=show_only,
            metrics_only=getattr(args, "metrics_only", False),
            dimensions_only=getattr(args, "dimensions_only", False),
            extended_fields=getattr(args, "extended_fields", False),
            side_by_side=getattr(args, "side_by_side", False),
            no_color=getattr(args, "no_color", False),
            quiet_diff=getattr(args, "quiet_diff", False),
            reverse_diff=getattr(args, "reverse_diff", False),
            warn_threshold=getattr(args, "warn_threshold", None),
            group_by_field=getattr(args, "group_by_field", False),
            group_by_field_limit=getattr(args, "group_by_field_limit", 10),
            diff_output=getattr(args, "diff_output", None),
            format_pr_comment=getattr(args, "format_pr_comment", False),
            auto_snapshot=getattr(args, "auto_snapshot", False),
            auto_prune=getattr(args, "auto_prune", False),
            snapshot_dir=getattr(args, "snapshot_dir", "./snapshots"),
            keep_last=getattr(args, "keep_last", 0),
            keep_since=getattr(args, "keep_since", None),
            keep_last_specified=keep_last_specified,
            keep_since_specified=keep_since_specified,
            profile=getattr(args, "profile", None),
            include_calc_metrics=include_calc_metrics,
            include_segments=include_segments,
        )

        # Exit with code 3 if threshold exceeded, 2 if differences found, 0 if no changes
        if success:
            if exit_code_override is not None:
                sys.exit(exit_code_override)
            sys.exit(2 if has_changes else 0)
        else:
            sys.exit(1)

    # Validate that at least one data view is provided
    if not data_view_inputs:
        print(ConsoleColors.error("ERROR: At least one data view ID or name is required"), file=sys.stderr)
        print("Usage: cja_auto_sdr DATA_VIEW_ID_OR_NAME [DATA_VIEW_ID_OR_NAME ...]", file=sys.stderr)
        print("       Use --help for more information", file=sys.stderr)
        sys.exit(1)

    # Resolve data view names to IDs
    # Create a temporary logger for name resolution
    temp_logger = logging.getLogger("name_resolution")
    temp_logger.setLevel(logging.INFO if not args.quiet else logging.ERROR)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    temp_logger.addHandler(handler)
    temp_logger.propagate = False

    # Show what we're resolving
    names_provided = [dv for dv in data_view_inputs if not is_data_view_id(dv)]

    if names_provided and not args.quiet:
        print()
        print(ConsoleColors.info(f"Resolving {len(names_provided)} data view name(s)..."))

    data_views, name_to_ids_map = resolve_data_view_names(
        data_view_inputs, args.config_file, temp_logger, profile=getattr(args, "profile", None)
    )

    # Remove the temporary handler
    temp_logger.removeHandler(handler)

    # Check if resolution failed
    if not data_views:
        print()
        print(ConsoleColors.error("ERROR: No valid data views found"), file=sys.stderr)
        print()
        print("Possible issues:", file=sys.stderr)
        print("  - Data view ID(s) or name(s) not found or you don't have access", file=sys.stderr)
        print("  - Data view name is not an EXACT match (names are case-sensitive)", file=sys.stderr)
        print("  - Configuration issue preventing data view lookup", file=sys.stderr)
        print()
        print("Tips for using Data View Names:", file=sys.stderr)
        print("  • Names must match EXACTLY: 'Production Analytics' ≠ 'production analytics'", file=sys.stderr)
        print('  • Use quotes around names: cja_auto_sdr "Production Analytics"', file=sys.stderr)
        print("  • IDs start with 'dv_': cja_auto_sdr dv_12345", file=sys.stderr)
        print()
        print("Try running: cja_auto_sdr --list-dataviews", file=sys.stderr)
        print("  to see all accessible data view IDs and names", file=sys.stderr)
        sys.exit(1)

    # Show resolution summary if names were used
    if name_to_ids_map and not args.quiet:
        print()
        print(ConsoleColors.success("Data view name resolution:"))
        for name, ids in name_to_ids_map.items():
            if len(ids) == 1:
                print(f"  ✓ '{name}' → {ids[0]}")
            else:
                print(f"  ✓ '{name}' → {len(ids)} matching data views:")
                for dv_id in ids:
                    print(f"      - {dv_id}")
        print()

    # Check for duplicate data view IDs and deduplicate
    original_count = len(data_views)
    seen = set()
    duplicates = []
    unique_data_views = []
    for dv in data_views:
        if dv in seen:
            duplicates.append(dv)
        else:
            seen.add(dv)
            unique_data_views.append(dv)

    if duplicates and not args.quiet:
        print(ConsoleColors.warning(f"Duplicate data view IDs removed: {set(duplicates)}"))
        print(f"  Processing {len(unique_data_views)} unique data view(s) instead of {original_count}")
        print()

    data_views = unique_data_views

    # Large batch confirmation (unless --yes or --quiet)
    LARGE_BATCH_THRESHOLD = 20
    if (
        len(data_views) >= LARGE_BATCH_THRESHOLD
        and not getattr(args, "assume_yes", False)
        and not args.quiet
        and not getattr(args, "dry_run", False)
        and sys.stdin.isatty()
    ):
        print(ConsoleColors.warning(f"Large batch detected: {len(data_views)} data views"))
        print()
        print("Estimated processing:")
        print(f"  • API calls: ~{len(data_views) * 3} requests (metrics, dimensions, info per DV)")
        print(f"  • Duration: ~{len(data_views) * 2}-{len(data_views) * 5} seconds")
        print()
        print("Tips:")
        print("  • Use --filter to narrow scope: --filter 'prod*'")
        print("  • Use --limit N to process only first N data views")
        print("  • Use --yes to skip this prompt in CI/CD")
        print()

        try:
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("Cancelled.")
                sys.exit(0)
            print()
        except EOFError, KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Validate the resolved data view IDs
    if not args.quiet and names_provided:
        print(ConsoleColors.info(f"Processing {len(data_views)} data view(s) total..."))
        print()

    # Priority logic for log level: --quiet > --production > --log-level
    if args.quiet:
        effective_log_level = "ERROR"
    elif args.production:
        effective_log_level = "WARNING"
    else:
        effective_log_level = args.log_level

    # Handle dry-run mode
    if args.dry_run:
        logger = setup_logging(batch_mode=True, log_level="WARNING", log_format=args.log_format)
        success = run_dry_run(data_views, args.config_file, logger, profile=getattr(args, "profile", None))
        sys.exit(0 if success else 1)

    quality_report_format = getattr(args, "quality_report", None)
    quality_report_only = quality_report_format is not None

    # Default to excel for SDR generation
    sdr_format = args.format if args.format else "excel"
    if quality_report_only:
        sdr_format = "json"

    # Validate format - console is only supported for diff comparison
    if sdr_format == "console" and not quality_report_only:
        print(ConsoleColors.error("Error: Console format is only supported for diff comparison."))
        print()
        print("For SDR generation, use one of these formats:")
        print("  --format excel     Excel workbook with multiple sheets (default)")
        print("  --format csv       CSV files (one per data type)")
        print("  --format json      JSON file with all data")
        print("  --format html      HTML report")
        print("  --format markdown  Markdown document")
        print("  --format all       Generate all formats")
        print()
        print("For diff comparison, console is the default:")
        print("  cja_auto_sdr --diff dv_A dv_B              # Console output")
        print("  cja_auto_sdr --diff dv_A dv_B --format json  # JSON output")
        sys.exit(1)

    # Check for conflicting component filter options
    if getattr(args, "metrics_only", False) and getattr(args, "dimensions_only", False):
        print(ConsoleColors.error("ERROR: Cannot use both --metrics-only and --dimensions-only"), file=sys.stderr)
        sys.exit(1)

    # Proactive output directory write check - fail fast before API calls
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        # Directory exists - check write permissions
        if not os.access(output_dir, os.W_OK):
            print(ConsoleColors.error(f"ERROR: Cannot write to output directory: {output_dir}"), file=sys.stderr)
            print("Check permissions and try again.", file=sys.stderr)
            sys.exit(1)
    else:
        # Directory doesn't exist - check we can create it by checking parent
        parent_dir = output_dir.parent
        if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
            print(ConsoleColors.error(f"ERROR: Cannot create output directory: {output_dir}"), file=sys.stderr)
            print(f"Parent directory {parent_dir} is not writable.", file=sys.stderr)
            sys.exit(1)

    # Process data views - start timing here for accurate processing-only runtime
    processing_start_time = time.time()

    # Create API tuning config if enabled
    api_tuning_config = None
    if getattr(args, "api_auto_tune", False):
        api_tuning_config = APITuningConfig(
            min_workers=getattr(args, "api_min_workers", 1), max_workers=getattr(args, "api_max_workers", 10)
        )
        if not args.quiet:
            print(
                ConsoleColors.info(
                    f"API auto-tuning enabled (workers: {api_tuning_config.min_workers}-{api_tuning_config.max_workers})"
                )
            )

    # Create circuit breaker config if enabled
    circuit_breaker_config = None
    if getattr(args, "circuit_breaker", False):
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=getattr(args, "circuit_failure_threshold", 5),
            timeout_seconds=getattr(args, "circuit_timeout", 30.0),
        )
        if not args.quiet:
            print(
                ConsoleColors.info(
                    f"Circuit breaker enabled (threshold: {circuit_breaker_config.failure_threshold}, timeout: {circuit_breaker_config.timeout_seconds}s)"
                )
            )

    # Expand --include-all-inventory into individual flags
    if getattr(args, "include_all_inventory", False):
        # Always enable segments and calculated metrics
        args.include_segments_inventory = True
        args.include_calculated_metrics = True

        # Only enable derived fields if NOT using snapshots (derived not supported with snapshots)
        is_snapshot_mode = (
            getattr(args, "snapshot", None)
            or getattr(args, "git_commit", False)
            or getattr(args, "diff_snapshot", None)
            or getattr(args, "compare_snapshots", None)
            or getattr(args, "compare_with_prev", False)
        )
        if not is_snapshot_mode:
            args.include_derived_inventory = True

        if not args.quiet:
            enabled = ["--include-segments", "--include-calculated"]
            if not is_snapshot_mode:
                enabled.append("--include-derived")
            print(ConsoleColors.info(f"--include-all-inventory enabled: {', '.join(enabled)}"))

    # Validate --inventory-only requires at least one --include-* flag
    if getattr(args, "inventory_only", False):
        has_inventory = (
            getattr(args, "include_derived_inventory", False)
            or getattr(args, "include_calculated_metrics", False)
            or getattr(args, "include_segments_inventory", False)
        )
        if not has_inventory:
            print(ConsoleColors.error("ERROR: --inventory-only requires at least one inventory flag"), file=sys.stderr)
            print("Use: --include-derived, --include-calculated, and/or --include-segments", file=sys.stderr)
            print("\nExample: cja_auto_sdr dv_12345 --include-segments --inventory-only", file=sys.stderr)
            sys.exit(1)

    # Validate --inventory-summary requires at least one --include-* flag and is mutually exclusive with --inventory-only
    # Determine inventory order based on CLI argument order (used for both sheets and summaries)
    inventory_order = []
    if (
        getattr(args, "include_derived_inventory", False)
        or getattr(args, "include_calculated_metrics", False)
        or getattr(args, "include_segments_inventory", False)
    ):
        # Check which flag appears first in sys.argv
        derived_pos = None
        calculated_pos = None
        segments_pos = None
        for i, arg in enumerate(sys.argv):
            if arg == "--include-derived" and derived_pos is None:
                derived_pos = i
            elif arg == "--include-calculated" and calculated_pos is None:
                calculated_pos = i
            elif arg == "--include-segments" and segments_pos is None:
                segments_pos = i

        # Build order based on position
        positions = []
        if derived_pos is not None:
            positions.append(("derived", derived_pos))
        if calculated_pos is not None:
            positions.append(("calculated", calculated_pos))
        if segments_pos is not None:
            positions.append(("segments", segments_pos))

        # Sort by position and extract names
        positions.sort(key=lambda x: x[1])
        inventory_order = [name for name, _ in positions]

    if getattr(args, "inventory_summary", False):
        has_inventory = (
            getattr(args, "include_derived_inventory", False)
            or getattr(args, "include_calculated_metrics", False)
            or getattr(args, "include_segments_inventory", False)
        )
        if not has_inventory:
            print(
                ConsoleColors.error("ERROR: --inventory-summary requires at least one inventory flag"), file=sys.stderr
            )
            print("Use: --include-derived, --include-calculated, and/or --include-segments", file=sys.stderr)
            print("\nExample: cja_auto_sdr dv_12345 --include-segments --inventory-summary", file=sys.stderr)
            sys.exit(1)
        if getattr(args, "inventory_only", False):
            print(
                ConsoleColors.error("ERROR: --inventory-summary cannot be used with --inventory-only"), file=sys.stderr
            )
            print(
                "Use --inventory-summary alone for quick stats, or --inventory-only for inventory sheets without full SDR.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Handle --inventory-summary mode (quick stats without full output)
    if getattr(args, "inventory_summary", False):
        # Determine output format for summary
        summary_format = args.format if args.format in ("json", "all") else "console"

        if len(data_views) > 1:
            # Process multiple data views in summary mode
            for dv_id in data_views:
                process_inventory_summary(
                    data_view_id=dv_id,
                    config_file=args.config_file,
                    output_dir=args.output_dir,
                    log_level=effective_log_level,
                    output_format=summary_format,
                    quiet=args.quiet,
                    profile=getattr(args, "profile", None),
                    include_derived=getattr(args, "include_derived_inventory", False),
                    include_calculated=getattr(args, "include_calculated_metrics", False),
                    include_segments=getattr(args, "include_segments_inventory", False),
                    inventory_order=inventory_order,
                )
                print()  # Blank line between data views
        else:
            # Single data view
            process_inventory_summary(
                data_view_id=data_views[0],
                config_file=args.config_file,
                output_dir=args.output_dir,
                log_level=effective_log_level,
                output_format=summary_format,
                quiet=args.quiet,
                profile=getattr(args, "profile", None),
                include_derived=getattr(args, "include_derived_inventory", False),
                include_calculated=getattr(args, "include_calculated_metrics", False),
                include_segments=getattr(args, "include_segments_inventory", False),
                inventory_order=inventory_order,
            )
        sys.exit(0)

    successful_results: list[ProcessingResult] = []
    quality_report_results: list[ProcessingResult] = []
    processed_results: list[ProcessingResult] = []
    overall_failure = False
    processing_failures_detected = False

    if quality_report_only:
        if not args.quiet:
            print(ConsoleColors.info(f"Validating data quality for {len(data_views)} data view(s)..."))
            print()

        for dv_id in data_views:
            result = process_single_dataview(
                dv_id,
                config_file=args.config_file,
                output_dir=args.output_dir,
                log_level=effective_log_level,
                log_format=args.log_format,
                output_format=sdr_format,
                enable_cache=args.enable_cache,
                cache_size=args.cache_size,
                cache_ttl=args.cache_ttl,
                quiet=args.quiet,
                skip_validation=args.skip_validation,
                max_issues=args.max_issues,
                clear_cache=args.clear_cache,
                show_timings=args.show_timings,
                metrics_only=getattr(args, "metrics_only", False),
                dimensions_only=getattr(args, "dimensions_only", False),
                profile=getattr(args, "profile", None),
                api_tuning_config=api_tuning_config,
                circuit_breaker_config=circuit_breaker_config,
                include_derived_inventory=False,
                include_calculated_metrics=False,
                include_segments_inventory=False,
                inventory_only=False,
                inventory_order=None,
                quality_report_only=True,
            )
            quality_report_results.append(result)
            processed_results.append(result)

            if result.success:
                successful_results.append(result)
                if not args.quiet:
                    print(
                        ConsoleColors.success(
                            f"✓ {result.data_view_name} ({result.data_view_id}): {result.dq_issues_count} issues"
                        )
                    )
            else:
                # Quality report mode should still fail overall if any DV fails,
                # even when --continue-on-error is used to keep processing.
                overall_failure = True
                processing_failures_detected = True
                print(ConsoleColors.error(f"FAILED: {result.data_view_id} - {result.error_message}"), file=sys.stderr)
                if not args.continue_on_error:
                    break

        if not args.quiet:
            total_runtime = time.time() - processing_start_time
            print()
            print(ConsoleColors.bold(f"Total runtime: {total_runtime:.1f}s"))

    elif args.batch or len(data_views) > 1:
        # Batch mode - parallel processing

        # Apply auto-detection for workers if requested
        if workers_auto:
            args.workers = auto_detect_workers(num_data_views=len(data_views))
            if not args.quiet:
                cpu_count = os.cpu_count() or 4
                print(
                    ConsoleColors.info(
                        f"Auto-detected workers: {args.workers} (based on {cpu_count} CPU cores, {len(data_views)} data views)"
                    )
                )

        if not args.quiet:
            print(
                ConsoleColors.info(
                    f"Processing {len(data_views)} data view(s) in batch mode with {args.workers} workers..."
                )
            )
            print()

        processor = BatchProcessor(
            config_file=args.config_file,
            output_dir=args.output_dir,
            workers=args.workers,
            continue_on_error=args.continue_on_error,
            log_level=effective_log_level,
            log_format=args.log_format,
            output_format=sdr_format,
            enable_cache=args.enable_cache,
            cache_size=args.cache_size,
            cache_ttl=args.cache_ttl,
            quiet=args.quiet,
            skip_validation=args.skip_validation,
            max_issues=args.max_issues,
            clear_cache=args.clear_cache,
            show_timings=args.show_timings,
            metrics_only=getattr(args, "metrics_only", False),
            dimensions_only=getattr(args, "dimensions_only", False),
            profile=getattr(args, "profile", None),
            shared_cache=getattr(args, "shared_cache", False),
            api_tuning_config=api_tuning_config,
            circuit_breaker_config=circuit_breaker_config,
            include_derived_inventory=getattr(args, "include_derived_inventory", False),
            include_calculated_metrics=getattr(args, "include_calculated_metrics", False),
            include_segments_inventory=getattr(args, "include_segments_inventory", False),
            inventory_only=getattr(args, "inventory_only", False),
            inventory_order=inventory_order if inventory_order else None,
            quality_report_only=quality_report_only,
        )

        results = processor.process_batch(data_views)
        successful_results = list(results.get("successful", []))
        processed_results = successful_results + list(results.get("failed", []))

        # Print total runtime
        total_runtime = time.time() - processing_start_time
        print()
        print(ConsoleColors.bold(f"Total runtime: {total_runtime:.1f}s"))

        # Handle --open flag for batch mode (open all successful files)
        if getattr(args, "open", False) and results.get("successful") and not quality_report_only:
            files_to_open = []
            for success_info in results["successful"]:
                if isinstance(success_info, dict) and success_info.get("output_file"):
                    files_to_open.append(success_info["output_file"])
                elif hasattr(success_info, "output_file") and success_info.output_file:
                    files_to_open.append(success_info.output_file)

            if files_to_open:
                print()
                print(f"Opening {len(files_to_open)} file(s)...")
                for file_path in files_to_open:
                    if not open_file_in_default_app(file_path):
                        print(ConsoleColors.warning(f"  Could not open: {file_path}"))

        # Track processing failures separately for exit-code precedence logic.
        processing_failures_detected = bool(results.get("failed"))

        # Exit with error code if any failed (unless continue-on-error)
        overall_failure = bool(results["failed"] and not args.continue_on_error)

    else:
        # Single mode - process one data view
        if not args.quiet:
            print(ConsoleColors.info(f"Processing data view: {data_views[0]}"))
            print()

        result = process_single_dataview(
            data_views[0],
            config_file=args.config_file,
            output_dir=args.output_dir,
            log_level=effective_log_level,
            log_format=args.log_format,
            output_format=sdr_format,
            enable_cache=args.enable_cache,
            cache_size=args.cache_size,
            cache_ttl=args.cache_ttl,
            quiet=args.quiet,
            skip_validation=args.skip_validation,
            max_issues=args.max_issues,
            clear_cache=args.clear_cache,
            show_timings=args.show_timings,
            metrics_only=getattr(args, "metrics_only", False),
            dimensions_only=getattr(args, "dimensions_only", False),
            profile=getattr(args, "profile", None),
            api_tuning_config=api_tuning_config,
            circuit_breaker_config=circuit_breaker_config,
            include_derived_inventory=getattr(args, "include_derived_inventory", False),
            include_calculated_metrics=getattr(args, "include_calculated_metrics", False),
            include_segments_inventory=getattr(args, "include_segments_inventory", False),
            inventory_only=getattr(args, "inventory_only", False),
            inventory_order=inventory_order if inventory_order else None,
            quality_report_only=quality_report_only,
        )
        processed_results = [result]

        # Print final status with color and total runtime
        total_runtime = time.time() - processing_start_time
        print()
        if result.success:
            successful_results = [result]
            if quality_report_only:
                print(ConsoleColors.success(f"SUCCESS: Quality validation completed for {result.data_view_name}"))
                print(f"  Metrics: {result.metrics_count}, Dimensions: {result.dimensions_count}")
                print(f"  Data Quality Issues: {result.dq_issues_count}")
            else:
                print(ConsoleColors.success(f"SUCCESS: SDR generated for {result.data_view_name}"))
                print(f"  Output: {result.output_file}")
                print(f"  Size: {result.file_size_formatted}")
                print(f"  Metrics: {result.metrics_count}, Dimensions: {result.dimensions_count}")
                if result.dq_issues_count > 0:
                    print(ConsoleColors.warning(f"  Data Quality Issues: {result.dq_issues_count}"))

                # Display inventory summary if any inventory was requested
                include_segs = getattr(args, "include_segments_inventory", False)
                include_calc = getattr(args, "include_calculated_metrics", False)
                include_derived = getattr(args, "include_derived_inventory", False)

                if include_segs or include_calc or include_derived:
                    inv_parts = []
                    # Use inventory_order to maintain consistent ordering with sheets
                    inv_order = inventory_order if inventory_order else ["segments", "calculated", "derived"]
                    for inv_type in inv_order:
                        if inv_type == "segments" and include_segs:
                            seg_str = f"Segments: {result.segments_count}"
                            if result.segments_high_complexity > 0:
                                seg_str += f" ({result.segments_high_complexity} high-complexity)"
                            inv_parts.append(seg_str)
                        elif inv_type == "calculated" and include_calc:
                            calc_str = f"Calculated Metrics: {result.calculated_metrics_count}"
                            if result.calculated_metrics_high_complexity > 0:
                                calc_str += f" ({result.calculated_metrics_high_complexity} high-complexity)"
                            inv_parts.append(calc_str)
                        elif inv_type == "derived" and include_derived:
                            derived_str = f"Derived Fields: {result.derived_fields_count}"
                            if result.derived_fields_high_complexity > 0:
                                derived_str += f" ({result.derived_fields_high_complexity} high-complexity)"
                            inv_parts.append(derived_str)

                    if inv_parts:
                        print(f"  Inventory: {', '.join(inv_parts)}")

                    # Warn about high-complexity items
                    if result.total_high_complexity > 0:
                        print(
                            ConsoleColors.warning(
                                f"  ⚠ {result.total_high_complexity} high-complexity items (≥75) - review recommended"
                            )
                        )

                # Handle --git-commit for single mode
                if getattr(args, "git_commit", False):
                    print()
                    git_dir = Path(getattr(args, "git_dir", "./sdr-snapshots"))

                    # Initialize repo if needed
                    if not is_git_repository(git_dir):
                        print(f"Initializing Git repository at: {git_dir}")
                        init_success, init_msg = git_init_snapshot_repo(git_dir)
                        if not init_success:
                            print(ConsoleColors.error(f"Git init failed: {init_msg}"))
                        else:
                            print(ConsoleColors.success("  Repository initialized"))

                    # Create snapshot for Git
                    # Check if inventory flags are set
                    include_calc = getattr(args, "include_calculated_metrics", False)
                    include_segs = getattr(args, "include_segments_inventory", False)

                    snapshot = DataViewSnapshot(
                        data_view_id=result.data_view_id,
                        data_view_name=result.data_view_name,
                        metrics=result.metrics_data if hasattr(result, "metrics_data") else [],
                        dimensions=result.dimensions_data if hasattr(result, "dimensions_data") else [],
                    )

                    # If we don't have the raw data in result, or if inventory is requested,
                    # we need to fetch it via create_snapshot
                    needs_fetch = not snapshot.metrics and not snapshot.dimensions
                    needs_inventory = include_calc or include_segs

                    if needs_fetch or needs_inventory:
                        # Re-fetch data for Git snapshot (with optional inventory)
                        fetch_reason = "inventory" if needs_inventory and not needs_fetch else "data"
                        print(f"Fetching {fetch_reason} for Git snapshot...")
                        try:
                            temp_logger = logging.getLogger("git_snapshot")
                            temp_logger.setLevel(logging.WARNING)
                            cja = initialize_cja(args.config_file, temp_logger, profile=getattr(args, "profile", None))
                            if cja:
                                snapshot_mgr = SnapshotManager(temp_logger)
                                snapshot = snapshot_mgr.create_snapshot(
                                    cja,
                                    result.data_view_id,
                                    quiet=True,
                                    include_calculated_metrics=include_calc,
                                    include_segments=include_segs,
                                )
                        except Exception as e:
                            print(ConsoleColors.warning(f"  Could not fetch snapshot data: {e}"))

                    # Save Git-friendly snapshot
                    print(f"Saving snapshot to: {git_dir}")
                    save_git_friendly_snapshot(
                        snapshot=snapshot,
                        output_dir=git_dir,
                        quality_issues=result.dq_issues if hasattr(result, "dq_issues") else None,
                    )

                    # Commit to Git
                    git_push = getattr(args, "git_push", False)
                    git_message = getattr(args, "git_message", None)

                    commit_success, commit_result = git_commit_snapshot(
                        snapshot_dir=git_dir,
                        data_view_id=result.data_view_id,
                        data_view_name=result.data_view_name,
                        metrics_count=result.metrics_count,
                        dimensions_count=result.dimensions_count,
                        quality_issues=result.dq_issues if hasattr(result, "dq_issues") else None,
                        custom_message=git_message,
                        push=git_push,
                    )

                    if commit_success:
                        if commit_result == "no_changes":
                            print(ConsoleColors.info("  No changes to commit (snapshot unchanged)"))
                        else:
                            print(ConsoleColors.success(f"  Committed: {commit_result}"))
                            if git_push:
                                print(ConsoleColors.success("  Pushed to remote"))
                    else:
                        print(ConsoleColors.error(f"  Git commit failed: {commit_result}"))

                # Handle --open flag for single mode
                if getattr(args, "open", False) and result.output_file:
                    print()
                    print("Opening file...")
                    if not open_file_in_default_app(result.output_file):
                        print(ConsoleColors.warning(f"  Could not open: {result.output_file}"))
        else:
            print(ConsoleColors.error(f"FAILED: {result.error_message}"))
            overall_failure = True
            processing_failures_detected = True

        print(ConsoleColors.bold(f"Total runtime: {total_runtime:.1f}s"))

    all_quality_issues = aggregate_quality_issues(successful_results)

    if quality_report_only:
        summary_results = quality_report_results
    else:
        summary_results = processed_results
    if summary_results:
        append_github_step_summary(build_quality_step_summary(summary_results))

    if quality_report_only:
        try:
            report_target = write_quality_report_output(
                all_quality_issues,
                report_format=quality_report_format,
                output=getattr(args, "output", None),
                output_dir=args.output_dir,
            )
            if not args.quiet:
                if report_target == "stdout":
                    print(ConsoleColors.success("Quality report written to stdout"))
                else:
                    print(ConsoleColors.success(f"Quality report written to: {report_target}"))
        except Exception as e:
            print(ConsoleColors.error(f"ERROR: Failed to write quality report: {e!s}"), file=sys.stderr)
            overall_failure = True

    fail_on_quality = getattr(args, "fail_on_quality", None)
    quality_gate_failed = False
    if fail_on_quality and has_quality_issues_at_or_above(all_quality_issues, fail_on_quality):
        quality_gate_failed = True
        threshold_rank = QUALITY_SEVERITY_RANK[fail_on_quality]
        failing_counts = count_quality_issues_by_severity(
            [
                issue
                for issue in all_quality_issues
                if QUALITY_SEVERITY_RANK.get(str(issue.get("Severity", "")).upper(), 99) <= threshold_rank
            ]
        )
        if not args.quiet:
            print(
                ConsoleColors.error(
                    f"QUALITY GATE FAILED: Found issues at or above {fail_on_quality} severity."
                ),
                file=sys.stderr,
            )
            for severity in QUALITY_SEVERITY_ORDER:
                count = failing_counts.get(severity, 0)
                if count > 0:
                    print(f"  {severity}: {count}", file=sys.stderr)

    if quality_gate_failed:
        # Exit code 1 has precedence when processing failed.
        if processing_failures_detected or overall_failure:
            sys.exit(1)
        sys.exit(2)

    if overall_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
