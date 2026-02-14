#!/usr/bin/env python3
"""
Verify that version strings are consistent across the project.

Usage:
  uv run python scripts/check_version_sync.py          # check and report
  uv run python scripts/check_version_sync.py --fix    # not yet supported
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical version source
VERSION_FILE = ROOT / "src" / "cja_auto_sdr" / "core" / "version.py"


def get_canonical_version() -> str:
    """Read the canonical version from version.py."""
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        print(f"FAIL: Could not parse version from {VERSION_FILE.relative_to(ROOT)}")
        sys.exit(2)
    return match.group(1)


# Each entry: (relative_path, regex_pattern_with_one_capture_group, description)
VERSION_LOCATIONS: list[tuple[str, str, str]] = [
    (
        "CLAUDE.md",
        r"Current version:\s*v(\d+\.\d+\.\d+)",
        "CLAUDE.md version reference",
    ),
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

        content = filepath.read_text()
        matches = re.findall(pattern, content, re.MULTILINE)

        if not matches:
            errors.append(f"  {rel_path}: no version match found ({description})")
            continue

        # Check first match (most relevant — e.g. latest changelog entry)
        found = matches[0]
        if found != canonical:
            errors.append(
                f"  {rel_path}: expected {canonical}, found {found} ({description})"
            )

    return errors


def main() -> None:
    canonical = get_canonical_version()
    errors = check_all(canonical)

    if errors:
        print(f"Version sync check FAILED (canonical: {canonical})")
        print()
        for error in errors:
            print(error)
        print()
        print(f"Canonical source: {VERSION_FILE.relative_to(ROOT)}")
        sys.exit(1)
    else:
        print(f"Version sync OK: all references match {canonical}")


if __name__ == "__main__":
    main()
