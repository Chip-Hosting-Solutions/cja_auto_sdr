"""Fast-path entry point for CJA Auto SDR.

Lightweight flags (``--version``, ``--help``, ``--exit-codes``) are handled
here *before* any heavyweight imports (pandas, cjapy, tqdm) so that simple
informational queries return in under 100 ms.

All other invocations fall through to the full ``generator.main()`` path.

Used by both ``python -m cja_auto_sdr`` and the console-script entry points.
"""

from __future__ import annotations

import sys


def _is_fast_path_flag(argv: list[str]) -> str | None:
    """Return the fast-path flag present in *argv*, or ``None``."""
    # Only consider the very first real argument (ignore argv[0] which is
    # the script/module path).  This matches argparse behaviour:
    # ``--version`` and ``--help`` short-circuit regardless of other
    # arguments, and ``--exit-codes`` is a standalone informational flag.
    args = argv[1:]
    if not args:
        return None

    # --version / -V
    if args[0] in ("--version", "-V"):
        return "--version"

    # --help / -h — still needs the full parser for complete output,
    # so we don't intercept it here.

    # --exit-codes (standalone flag, no other args needed)
    if args == ["--exit-codes"]:
        return "--exit-codes"

    return None


def _print_version() -> None:
    from cja_auto_sdr.core.version import __version__

    print(f"cja_auto_sdr {__version__}")


def _print_exit_codes() -> None:
    banner_width = 78  # matches core.constants.BANNER_WIDTH

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


def main() -> None:
    """Entry point with fast-path for lightweight flags."""
    flag = _is_fast_path_flag(sys.argv)

    if flag == "--version":
        _print_version()
        raise SystemExit(0)

    if flag == "--exit-codes":
        _print_exit_codes()
        raise SystemExit(0)

    # All other invocations need the full generator
    from cja_auto_sdr.generator import main as _generator_main

    _generator_main()


if __name__ == "__main__":
    main()
