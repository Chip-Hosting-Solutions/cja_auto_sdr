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
from functools import lru_cache

_VERSION_OPTION = "--version"
_VERSION_SHORT_OPTION = "-V"

_RUN_SUMMARY_OPTION = "--run-summary-json"

def _minimum_option_arity(nargs: object) -> int:
    """Return the minimum positional values an option must consume."""
    if nargs is None:
        return 1
    if isinstance(nargs, int):
        return max(nargs, 0)
    if nargs == "+":
        return 1
    # Includes 0, "?", "*", argparse.REMAINDER, argparse.PARSER.
    return 0


@lru_cache(maxsize=1)
def _fast_path_option_spec() -> tuple[frozenset[str], dict[str, int]]:
    """Return (known_long_options, option_min_arity) from the configured CLI parser."""
    from cja_auto_sdr.cli.parser import parse_arguments

    parser = parse_arguments(return_parser=True, enable_autocomplete=False)
    option_min_arity: dict[str, int] = {}
    known_long_options: set[str] = set()

    for action in parser._actions:
        if not action.option_strings:
            continue
        min_arity = _minimum_option_arity(getattr(action, "nargs", None))
        for option in action.option_strings:
            option_min_arity[option] = min_arity
            if option.startswith("--"):
                known_long_options.add(option)

    return frozenset(known_long_options), option_min_arity


def _resolve_long_option_token(token_name: str, known_long_options: frozenset[str]) -> str | None:
    """Resolve a token to a canonical long option if argparse would accept it."""
    if not token_name.startswith("--") or token_name == "--":
        return None
    if token_name in known_long_options:
        return token_name

    matches = [option for option in known_long_options if option.startswith(token_name)]
    if len(matches) == 1:
        return matches[0]
    return None


def _iter_option_tokens(args: list[str]):
    """Yield canonical option tokens, skipping tokens consumed as option values."""
    known_long_options, option_min_arity = _fast_path_option_spec()
    pending_option_values = 0

    for token in args:
        if token == "--":
            break

        if pending_option_values > 0:
            pending_option_values -= 1
            continue

        if token.startswith("--"):
            option_name, has_equals, _inline_value = token.partition("=")
            canonical_option = _resolve_long_option_token(option_name, known_long_options)
            if canonical_option is None:
                continue

            yield canonical_option

            min_arity = option_min_arity.get(canonical_option, 0)
            inline_values = 1 if has_equals else 0
            if min_arity > inline_values:
                pending_option_values = min_arity - inline_values
            continue

        if token.startswith("-") and token != "-":
            if token in option_min_arity:
                yield token
                min_arity = option_min_arity[token]
                if min_arity > 0:
                    pending_option_values = min_arity
                continue

            # Support short-option clusters like -qV and attached values like -pVALUE.
            if len(token) > 2:
                cluster = token[1:]
                for index, short_name in enumerate(cluster):
                    short_option = f"-{short_name}"
                    min_arity = option_min_arity.get(short_option)
                    if min_arity is None:
                        break
                    yield short_option
                    if min_arity > 0:
                        attached_value = cluster[index + 1 :]
                        pending_option_values = max(min_arity - 1, 0) if attached_value else min_arity
                        break
            continue


def _has_run_summary_flag(args: list[str]) -> bool:
    """Return True when argv contains --run-summary-json (or unambiguous prefix)."""
    return any(option == _RUN_SUMMARY_OPTION for option in _iter_option_tokens(args))


def _has_version_flag(args: list[str]) -> bool:
    """Return True when argv contains --version/-V in option context."""
    return any(option in (_VERSION_OPTION, _VERSION_SHORT_OPTION) for option in _iter_option_tokens(args))


def _is_fast_path_flag(argv: list[str]) -> str | None:
    """Return the fast-path flag present in *argv*, or ``None``."""
    # Only consider real arguments (ignore argv[0] script/module path).
    args = argv[1:]
    if not args:
        return None

    # Preserve run-summary contract: when requested, always route through
    # generator.main() so summary emission is consistent and order-independent.
    if _has_run_summary_flag(args):
        return None

    # argparse's version action can short-circuit even when version appears
    # after other options; mirror that without importing heavy runtime modules.
    if _has_version_flag(args):
        return _VERSION_OPTION

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
    from cja_auto_sdr.core.constants import BANNER_WIDTH
    from cja_auto_sdr.core.exit_codes import print_exit_codes

    print_exit_codes(banner_width=BANNER_WIDTH)


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
