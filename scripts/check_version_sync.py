"""
Verify that version strings are consistent across the project.

Usage:
  uv run python scripts/check_version_sync.py          # check and report
  uv run python scripts/check_version_sync.py --require-tag
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical version source
VERSION_FILE = ROOT / "src" / "cja_auto_sdr" / "core" / "version.py"


def get_canonical_version() -> str:
    """Read the canonical version from version.py."""
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise SystemExit(f"FAIL: Could not parse version from {VERSION_FILE.relative_to(ROOT)}")
    return match.group(1)


# Each entry: (relative_path, regex_pattern_with_one_capture_group, description)
VERSION_LOCATIONS: list[tuple[str, str, str]] = [
    (
        "CHANGELOG.md",
        r"^## \[(\d+\.\d+\.\d+)\]",
        "CHANGELOG.md latest release entry",
    ),
    (
        "docs/QUICK_REFERENCE.md",
        r"v(\d+\.\d+\.\d+)",
        "Quick Reference version",
    ),
    (
        "docs/QUICKSTART_GUIDE.md",
        r"cja_auto_sdr\s+(\d+\.\d+\.\d+)",
        "Quickstart Guide version output",
    ),
    (
        "docs/CONFIGURATION.md",
        r"CJA SDR Generator v(\d+\.\d+\.\d+)",
        "Configuration startup diagnostics version example",
    ),
    (
        "CLAUDE.md",
        r"Current version:\s*v(\d+\.\d+\.\d+)",
        "Claude project notes current version",
    ),
    (
        "tests/test_ux_features.py",
        r'assert __version__ == "(\d+\.\d+\.\d+)"',
        "UX features version assertion",
    ),
    (
        "tests/test_output_content_validation.py",
        r'"Tool Version":\s*"(\d+\.\d+\.\d+)"',
        "Output content validation version fixture",
    ),
]


def check_all(canonical: str) -> list[str]:
    """Check all version locations against canonical. Return list of errors."""
    errors = []

    for rel_path, pattern, description in VERSION_LOCATIONS:
        filepath = ROOT / rel_path
        if not filepath.exists():
            errors.append(f"  {rel_path}: file not found")
            continue

        content = filepath.read_text(encoding="utf-8")
        matches = re.findall(pattern, content, re.MULTILINE)

        if not matches:
            errors.append(f"  {rel_path}: no version match found ({description})")
            continue

        # Check first match (most relevant — e.g. latest changelog entry)
        found = matches[0]
        if found != canonical:
            errors.append(f"  {rel_path}: expected {canonical}, found {found} ({description})")

    return errors


def check_release_tag(canonical: str) -> str | None:
    """Return an error message when v<canonical> tag is missing, else None."""
    tag_name = f"v{canonical}"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag_name}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f"Unable to verify release tag {tag_name}: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "not a git repository" in stderr:
            return f"Unable to verify release tag {tag_name}: {ROOT} is not a git repository"
        return (
            f"Missing required release tag: {tag_name}. "
            f"Create it with: git tag {tag_name}. "
            "If it already exists remotely, run: git fetch --tags"
        )
    return None


def check_ci_tag_ref_match(canonical: str, *, github_ref: str | None = None) -> str | None:
    """Return an error if CI tag ref doesn't exactly match v<canonical>; otherwise None.

    Applies only when running under a tag ref (`refs/tags/...`). For branch refs
    or missing refs, this check is a no-op.
    """
    ref = github_ref if github_ref is not None else os.environ.get("GITHUB_REF")
    if not ref or not ref.startswith("refs/tags/"):
        return None

    expected_ref = f"refs/tags/v{canonical}"
    if ref != expected_ref:
        return f"GITHUB_REF tag mismatch: expected {expected_ref}, found {ref}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify version/docs/changelog sync.")
    parser.add_argument(
        "--require-tag",
        action="store_true",
        help="Require git tag v<canonical version> to exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = get_canonical_version()
    errors = check_all(canonical)

    if args.require_tag:
        tag_error = check_release_tag(canonical)
        if tag_error is not None:
            errors.append(f"  {tag_error}")
        ref_error = check_ci_tag_ref_match(canonical)
        if ref_error is not None:
            errors.append(f"  {ref_error}")

    if errors:
        lines = [f"Version sync check FAILED (canonical: {canonical})", ""]
        lines.extend(errors)
        lines.append("")
        lines.append(f"Canonical source: {VERSION_FILE.relative_to(ROOT)}")
        sys.stdout.write("\n".join(lines) + "\n")
        raise SystemExit(1)

    sys.stdout.write(f"Version sync OK: all references match {canonical}\n")


if __name__ == "__main__":
    main()
