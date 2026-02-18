"""Fast-path entry point for CJA Auto SDR.

Lightweight flags (``--version``, ``--help``, ``--exit-codes``) are handled
here *before* any heavyweight imports (pandas, cjapy, tqdm) so that simple
informational queries return in under 100 ms.

All other invocations fall through to the full ``generator.main()`` path.

Used by both ``python -m cja_auto_sdr`` and the console-script entry points.
"""

from __future__ import annotations

import os
import sys


def _has_run_summary_flag(args: list[str]) -> bool:
    """Return True when argv contains --run-summary-json (or abbreviation)."""
    for token in args:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option in ("--run-summary-json", "--run-summary-j"):
            return True
        if "--run-summary-json".startswith(option) or "--run-summary-j".startswith(option):
            return True
    return False


def _is_fast_path_flag(argv: list[str]) -> str | None:
    """Return the fast-path flag present in *argv*, or ``None``."""
    # Only consider the very first real argument (ignore argv[0] which is
    # the script/module path).  This matches argparse behaviour:
    # ``--version`` and ``--help`` short-circuit regardless of other
    # arguments, and ``--exit-codes`` is a standalone informational flag.
    args = argv[1:]
    if not args:
        return None

    # Preserve run-summary contract: when requested, always route through
    # generator.main() so summary emission is consistent and order-independent.
    if _has_run_summary_flag(args):
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


def _resolve_program_name(
    argv0: str | None,
    module_name: str | None = None,
    interpreter_name: str | None = None,
) -> str:
    """Return the display program name argparse would use for version output.

    For ``python -m cja_auto_sdr`` invocation, argparse reports
    ``python -m cja_auto_sdr`` rather than ``__main__.py``. Mirror that to keep
    fast-path and full-parser behavior consistent.
    """
    if not argv0:
        return "cja_auto_sdr"
    program_name = os.path.basename(argv0)
    if program_name == "__main__.py":
        resolved_module = module_name
        if resolved_module is None:
            spec = globals().get("__spec__")
            resolved_module = getattr(spec, "name", None) if spec else None
        if resolved_module:
            module_target = resolved_module.removesuffix(".__main__")
            resolved_interpreter = interpreter_name
            if resolved_interpreter is None:
                resolved_interpreter = os.path.basename(sys.executable)
            return f"{resolved_interpreter or 'python'} -m {module_target}"
    return program_name or "cja_auto_sdr"


def _print_version(program_name: str = "cja_auto_sdr") -> None:
    from cja_auto_sdr.core.version import __version__

    print(f"{program_name} {__version__}")


def _print_exit_codes() -> None:
    banner_width = 78  # wider than BANNER_WIDTH for exit-code reference output

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
        _print_version(_resolve_program_name(sys.argv[0] if sys.argv else None))
        raise SystemExit(0)

    if flag == "--exit-codes":
        _print_exit_codes()
        raise SystemExit(0)

    # All other invocations need the full generator
    from cja_auto_sdr.generator import main as _generator_main

    _generator_main()


if __name__ == "__main__":
    main()
