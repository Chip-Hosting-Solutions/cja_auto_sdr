"""Shared exit-code reference output.

Lightweight — safe to import from ``__main__.py`` without triggering
heavyweight dependencies (pandas, cjapy, tqdm).
"""

from __future__ import annotations


def print_exit_codes(banner_width: int = 60) -> None:
    """Print the exit-code reference table to stdout."""
    print("=" * banner_width)
    print("EXIT CODE REFERENCE")
    print("=" * banner_width)
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
    print("=" * banner_width)
    print("CI/CD Examples:")
    print("=" * banner_width)
    print()
    print("  # Fail CI if any changes detected")
    print("  cja_auto_sdr --diff dv_prod dv_staging --quiet")
    print("  if [ $? -eq 2 ]; then echo 'Changes detected!'; exit 1; fi")
    print()
    print("  # Fail CI only if >10% changes")
    print("  cja_auto_sdr --diff dv_A dv_B --warn-threshold 10 --quiet")
    print()
