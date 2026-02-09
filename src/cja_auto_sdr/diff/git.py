from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from cja_auto_sdr.core.version import __version__
from cja_auto_sdr.diff.models import DataViewSnapshot, DiffResult


def is_git_repository(path: Path) -> bool:
    """Check if the given path is inside a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=str(path), capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def git_get_user_info() -> tuple[str, str]:
    """Get Git user name and email from config."""
    name = "CJA SDR Generator"
    email = ""

    try:
        result = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        result = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            email = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return name, email


def save_git_friendly_snapshot(
    snapshot: DataViewSnapshot,
    output_dir: Path,
    quality_issues: list[dict] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Save snapshot in Git-friendly format (separate JSON files)."""
    logger = logger or logging.getLogger(__name__)

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in snapshot.data_view_name)
    safe_name = safe_name[:50] if safe_name else snapshot.data_view_id
    dv_dir = output_dir / f"{safe_name}_{snapshot.data_view_id}"
    dv_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    metrics_sorted = sorted(snapshot.metrics, key=lambda x: x.get("id", ""))
    metrics_file = dv_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_sorted, f, indent=2, ensure_ascii=False, default=str)
    saved_files["metrics"] = metrics_file
    logger.debug(f"Saved {len(metrics_sorted)} metrics to {metrics_file}")

    dimensions_sorted = sorted(snapshot.dimensions, key=lambda x: x.get("id", ""))
    dimensions_file = dv_dir / "dimensions.json"
    with open(dimensions_file, "w", encoding="utf-8") as f:
        json.dump(dimensions_sorted, f, indent=2, ensure_ascii=False, default=str)
    saved_files["dimensions"] = dimensions_file
    logger.debug(f"Saved {len(dimensions_sorted)} dimensions to {dimensions_file}")

    if snapshot.calculated_metrics_inventory:
        calc_metrics_sorted = sorted(
            snapshot.calculated_metrics_inventory, key=lambda x: x.get("id", x.get("metric_id", ""))
        )
        calc_metrics_file = dv_dir / "calculated-metrics.json"
        with open(calc_metrics_file, "w", encoding="utf-8") as f:
            json.dump(calc_metrics_sorted, f, indent=2, ensure_ascii=False, default=str)
        saved_files["calculated_metrics"] = calc_metrics_file
        logger.debug(f"Saved {len(calc_metrics_sorted)} calculated metrics to {calc_metrics_file}")

    if snapshot.segments_inventory:
        segments_sorted = sorted(snapshot.segments_inventory, key=lambda x: x.get("id", x.get("segment_id", "")))
        segments_file = dv_dir / "segments.json"
        with open(segments_file, "w", encoding="utf-8") as f:
            json.dump(segments_sorted, f, indent=2, ensure_ascii=False, default=str)
        saved_files["segments"] = segments_file
        logger.debug(f"Saved {len(segments_sorted)} segments to {segments_file}")

    metadata = {
        "snapshot_version": snapshot.snapshot_version,
        "created_at": snapshot.created_at,
        "data_view_id": snapshot.data_view_id,
        "data_view_name": snapshot.data_view_name,
        "owner": snapshot.owner,
        "description": snapshot.description,
        "tool_version": __version__,
        "summary": {
            "metrics_count": len(snapshot.metrics),
            "dimensions_count": len(snapshot.dimensions),
            "total_components": len(snapshot.metrics) + len(snapshot.dimensions),
        },
    }

    if snapshot.calculated_metrics_inventory or snapshot.segments_inventory:
        metadata["inventory"] = {}
        if snapshot.calculated_metrics_inventory:
            metadata["inventory"]["calculated_metrics_count"] = len(snapshot.calculated_metrics_inventory)
        if snapshot.segments_inventory:
            metadata["inventory"]["segments_count"] = len(snapshot.segments_inventory)

    if quality_issues:
        severity_counts = {}
        for issue in quality_issues:
            sev = issue.get("Severity", "UNKNOWN")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        metadata["quality"] = {"total_issues": len(quality_issues), "by_severity": severity_counts}

    metadata_file = dv_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    saved_files["metadata"] = metadata_file
    logger.debug(f"Saved metadata to {metadata_file}")

    return saved_files


def generate_git_commit_message(
    data_view_id: str,
    data_view_name: str,
    metrics_count: int,
    dimensions_count: int,
    quality_issues: list[dict] | None = None,
    diff_result: DiffResult = None,
    custom_message: str | None = None,
) -> str:
    """Generate a descriptive Git commit message for SDR snapshot."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if custom_message:
        subject = f"[{data_view_id}] {custom_message}"
    else:
        subject = f"[{data_view_id}] SDR snapshot {timestamp}"

    lines = [subject, ""]

    lines.append(f"Data View: {data_view_name}")
    lines.append(f"ID: {data_view_id}")
    lines.append(f"Components: {metrics_count} metrics, {dimensions_count} dimensions")
    lines.append("")

    if diff_result and diff_result.summary.has_changes:
        summary = diff_result.summary
        lines.append("Changes:")
        if summary.metrics_added > 0:
            lines.append(f"  + {summary.metrics_added} metrics added")
        if summary.metrics_removed > 0:
            lines.append(f"  - {summary.metrics_removed} metrics removed")
        if summary.metrics_modified > 0:
            lines.append(f"  ~ {summary.metrics_modified} metrics modified")
        if summary.dimensions_added > 0:
            lines.append(f"  + {summary.dimensions_added} dimensions added")
        if summary.dimensions_removed > 0:
            lines.append(f"  - {summary.dimensions_removed} dimensions removed")
        if summary.dimensions_modified > 0:
            lines.append(f"  ~ {summary.dimensions_modified} dimensions modified")
        lines.append("")

    if quality_issues:
        severity_counts = {}
        for issue in quality_issues:
            sev = issue.get("Severity", "UNKNOWN")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        lines.append("Quality:")
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                lines.append(f"  {sev}: {count}")
        lines.append("")

    lines.append(f"Generated by CJA SDR Generator v{__version__}")

    return "\n".join(lines)


def git_commit_snapshot(
    snapshot_dir: Path,
    data_view_id: str,
    data_view_name: str,
    metrics_count: int,
    dimensions_count: int,
    quality_issues: list[dict] | None = None,
    diff_result: DiffResult = None,
    custom_message: str | None = None,
    push: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[bool, str]:
    """Commit snapshot files to Git with auto-generated message."""
    logger = logger or logging.getLogger(__name__)

    if not is_git_repository(snapshot_dir):
        return False, f"Not a Git repository: {snapshot_dir}"

    try:
        logger.info(f"Staging snapshot files in {snapshot_dir}")
        result = subprocess.run(["git", "add", "."], cwd=str(snapshot_dir), capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"git add failed: {result.stderr}"

        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(snapshot_dir), capture_output=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("No changes to commit (snapshot unchanged)")
            return True, "no_changes"

        commit_message = generate_git_commit_message(
            data_view_id=data_view_id,
            data_view_name=data_view_name,
            metrics_count=metrics_count,
            dimensions_count=dimensions_count,
            quality_issues=quality_issues,
            diff_result=diff_result,
            custom_message=custom_message,
        )

        logger.info("Committing snapshot to Git")
        result = subprocess.run(
            ["git", "commit", "-m", commit_message], cwd=str(snapshot_dir), capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False, f"git commit failed: {result.stderr}"

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(snapshot_dir), capture_output=True, text=True, timeout=10
        )
        commit_sha = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"

        logger.info(f"Committed snapshot: {commit_sha}")

        if push:
            logger.info("Pushing to remote")
            result = subprocess.run(["git", "push"], cwd=str(snapshot_dir), capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.warning(f"git push failed: {result.stderr}")
                return True, f"{commit_sha} (push failed: {result.stderr.strip()})"
            logger.info("Pushed to remote successfully")

        return True, commit_sha

    except subprocess.TimeoutExpired:
        return False, "Git operation timed out"
    except FileNotFoundError:
        return False, "Git not found - ensure Git is installed and in PATH"
    except Exception as e:
        return False, f"Git error: {e!s}"


def git_init_snapshot_repo(directory: Path, logger: logging.Logger | None = None) -> tuple[bool, str]:
    """Initialize a new Git repository for snapshots."""
    logger = logger or logging.getLogger(__name__)

    try:
        directory.mkdir(parents=True, exist_ok=True)

        if is_git_repository(directory):
            return True, "Already a Git repository"

        logger.info(f"Initializing Git repository in {directory}")
        result = subprocess.run(["git", "init"], cwd=str(directory), capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"git init failed: {result.stderr}"

        gitignore = directory / ".gitignore"
        gitignore.write_text("# CJA SDR Snapshots\n*.log\n*.tmp\n.DS_Store\n")

        readme = directory / "README.md"
        readme.write_text(f"""# CJA SDR Snapshots

This repository contains Solution Design Reference (SDR) snapshots from Adobe Customer Journey Analytics.

## Structure

```
<data_view_name>_<data_view_id>/
├── metrics.json      # All metrics (sorted by ID)
├── dimensions.json   # All dimensions (sorted by ID)
└── metadata.json     # Data view info and quality summary
```

## Usage

View history:
```bash
git log --oneline
```

Compare versions:
```bash
git diff HEAD~1 HEAD -- <data_view_dir>/metrics.json
```

---
Generated by CJA SDR Generator v{__version__}
""")

        subprocess.run(["git", "add", "."], cwd=str(directory), capture_output=True, timeout=30)
        subprocess.run(
            ["git", "commit", "-m", "Initialize SDR snapshot repository"],
            cwd=str(directory),
            capture_output=True,
            timeout=30,
        )

        logger.info(f"Initialized Git repository: {directory}")
        return True, "Repository initialized"

    except Exception as e:
        return False, f"Initialization failed: {e!s}"
