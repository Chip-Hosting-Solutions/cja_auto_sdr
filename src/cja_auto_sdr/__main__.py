"""Fast-path entry point for CJA Auto SDR.

Lightweight flags (``--version``, ``--help``, ``--exit-codes``,
``--completion``) are handled here *before* any heavyweight imports
(pandas, cjapy, tqdm) so that simple informational queries return in
under 100 ms.

All other invocations fall through to the full ``generator.main()`` path.

Used by both ``python -m cja_auto_sdr`` and the console-script entry points.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Mapping
from functools import lru_cache
from typing import NamedTuple

from cja_auto_sdr.cli.option_resolution import resolve_long_option_token as _resolve_long_option_token

_VERSION_OPTION = "--version"
_VERSION_SHORT_OPTION = "-V"

_COMPLETION_OPTION = "--completion"
_RUN_SUMMARY_OPTION = "--run-summary-json"
_ARGCOMPLETE_ENV_VAR = "_ARGCOMPLETE"
_FALSEY_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})

_SUPPORTED_SHELLS = frozenset({"bash", "zsh", "fish"})

_COMPLETION_SCRIPTS: dict[str, str] = {
    "bash": 'eval "$(register-python-argcomplete cja_auto_sdr)"',
    "zsh": ('autoload -U bashcompinit && bashcompinit\neval "$(register-python-argcomplete cja_auto_sdr)"'),
    "fish": "register-python-argcomplete --shell fish cja_auto_sdr | source",
}


class _OptionSpec(NamedTuple):
    """Minimal option metadata needed for fast-path token scanning."""

    min_arity: int
    accepts_inline_value: bool


class _OptionScanResult(NamedTuple):
    """Fast-path parse outcome: recognized options plus parse validity."""

    options: tuple[str, ...]
    has_parse_error: bool


class _ArgparseProbeExit(Exception):
    """Internal sentinel used to capture argparse exits without emitting output."""

    def __init__(self, status: int = 0, message: str | None = None) -> None:
        super().__init__(status, message)
        self.status = status
        self.message = message


class _ArgparseProbeResult(NamedTuple):
    """Result of a lightweight argparse probe parse."""

    status: int
    output: str | None


def _is_argcomplete_completion_active(environ: Mapping[str, str] | None = None) -> bool:
    """Return True when argcomplete shell-completion invocation is active."""
    env = os.environ if environ is None else environ
    raw_value = env.get(_ARGCOMPLETE_ENV_VAR)
    if raw_value is None:
        return False
    return raw_value.strip().lower() not in _FALSEY_ENV_VALUES


def _accepts_inline_option_value(nargs: object) -> bool:
    """Return True when an option can legally consume explicit inline values."""
    return nargs != 0


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
def _fast_path_option_spec() -> tuple[frozenset[str], dict[str, _OptionSpec]]:
    """Return (known_long_options, option_specs) from the configured CLI parser."""
    from cja_auto_sdr.cli.parser import parse_arguments

    parser = parse_arguments(return_parser=True, enable_autocomplete=False)
    option_specs: dict[str, _OptionSpec] = {}
    known_long_options: set[str] = set()

    # CPython argparse internals: `_actions` is intentionally used as the
    # canonical source of configured option metadata for fast-path scanning.
    for action in parser._actions:
        if not action.option_strings:
            continue
        action_nargs = getattr(action, "nargs", None)
        option_spec = _OptionSpec(
            min_arity=_minimum_option_arity(action_nargs),
            accepts_inline_value=_accepts_inline_option_value(action_nargs),
        )
        for option in action.option_strings:
            option_specs[option] = option_spec
            if option.startswith("--"):
                known_long_options.add(option)

    return frozenset(known_long_options), option_specs


def _scan_option_tokens(args: list[str]) -> _OptionScanResult:
    """Scan CLI tokens argparse-style for fast-path decisions.

    Unknown options are tolerated (argparse's version action can still exit
    before unknown-argument failures), but ambiguous long-option prefixes and
    explicit values for zero-arity options are treated as parse errors.
    """
    known_long_options, option_specs = _fast_path_option_spec()
    pending_option_values = 0
    resolved_options: list[str] = []

    for arg in args:
        if arg == "--":
            break

        if pending_option_values > 0:
            pending_option_values -= 1
            continue

        if arg.startswith("--"):
            option_name, has_equals, _inline_value = arg.partition("=")
            long_resolution = _resolve_long_option_token(option_name, known_long_options)
            if long_resolution.is_ambiguous:
                return _OptionScanResult(options=tuple(resolved_options), has_parse_error=True)
            canonical_option = long_resolution.canonical_option
            if canonical_option is None:
                continue

            option_spec = option_specs.get(canonical_option)
            if option_spec is None:
                continue
            if has_equals and not option_spec.accepts_inline_value:
                return _OptionScanResult(options=tuple(resolved_options), has_parse_error=True)

            resolved_options.append(canonical_option)

            inline_values = 1 if has_equals and option_spec.accepts_inline_value else 0
            if option_spec.min_arity > inline_values:
                pending_option_values = option_spec.min_arity - inline_values
            continue

        if arg.startswith("-") and arg != "-":
            option_spec = option_specs.get(arg)
            if option_spec is not None:
                resolved_options.append(arg)
                if option_spec.min_arity > 0:
                    pending_option_values = option_spec.min_arity
                continue

            # Support short-option clusters like -qV and attached values like -pVALUE.
            if len(arg) > 2:
                cluster = arg[1:]
                for index, short_name in enumerate(cluster):
                    short_option = f"-{short_name}"
                    option_spec = option_specs.get(short_option)
                    if option_spec is None:
                        break
                    resolved_options.append(short_option)
                    attached_value = cluster[index + 1 :]
                    if option_spec.min_arity > 0:
                        pending_option_values = (
                            max(option_spec.min_arity - 1, 0) if attached_value else option_spec.min_arity
                        )
                        break
                    if attached_value.startswith(("=", "-")):
                        return _OptionScanResult(options=tuple(resolved_options), has_parse_error=True)
            continue

    return _OptionScanResult(options=tuple(resolved_options), has_parse_error=False)


def _has_run_summary_contract_flag(args: list[str]) -> bool:
    """Return True when argv explicitly requests run-summary output.

    This token-level detector intentionally ignores option-value consumption so
    run-summary handling remains order-independent (e.g., `--version` followed
    by flags that would otherwise consume later tokens).
    """
    known_long_options, _ = _fast_path_option_spec()

    for arg in args:
        if arg == "--":
            break
        if not arg.startswith("--"):
            continue

        option_name, _, _inline_value = arg.partition("=")
        long_resolution = _resolve_long_option_token(option_name, known_long_options)
        if long_resolution.canonical_option == _RUN_SUMMARY_OPTION:
            return True

    return False


def _has_run_summary_flag(args: list[str]) -> bool:
    """Return True when argparse-style scan resolves --run-summary-json."""
    scan = _scan_option_tokens(args)
    if scan.has_parse_error:
        return False
    return any(option == _RUN_SUMMARY_OPTION for option in scan.options)


def _probe_argparse_termination(args: list[str], argv0: str | None = None) -> _ArgparseProbeResult | None:
    """Return argparse termination info for *args* without printing output.

    This uses the real parser as the source of truth for precedence and
    validation (help/version actions, missing values, and mutex conflicts).
    """
    from cja_auto_sdr.cli.parser import parse_arguments

    parser = parse_arguments(return_parser=True, enable_autocomplete=False)
    parser.prog = _resolve_program_name(argv0)
    captured_output: list[str] = []

    def _probe_exit(_self, status: int = 0, message: str | None = None) -> None:
        raise _ArgparseProbeExit(status, message)

    def _capture_output(_self, message: str | None, _file=None) -> None:
        if message:
            captured_output.append(message)

    parser.exit = types.MethodType(_probe_exit, parser)
    # CPython argparse internals: `_print_message` is overridden so probe parses
    # can capture output without emitting to real stdio.
    parser._print_message = types.MethodType(_capture_output, parser)

    try:
        parser.parse_args(args)
    except _ArgparseProbeExit as probe_exit:
        rendered_output = probe_exit.message or "".join(captured_output) or None
        return _ArgparseProbeResult(status=probe_exit.status, output=rendered_output)
    return None


def _is_fast_path_flag(argv: list[str]) -> str | None:
    """Return the fast-path flag present in *argv*, or ``None``."""
    # Only consider real arguments (ignore argv[0] script/module path).
    args = argv[1:]
    if not args:
        return None

    # Preserve run-summary contract: when requested, always route through
    # generator.main() so summary emission is consistent and order-independent.
    if _has_run_summary_contract_flag(args):
        return None

    scan = _scan_option_tokens(args)

    has_version_candidate = any(option in (_VERSION_OPTION, _VERSION_SHORT_OPTION) for option in scan.options)
    if has_version_candidate:
        probe = _probe_argparse_termination(args, argv[0] if argv else None)
        if probe is not None:
            # argparse exits with status 0 for both help and version actions.
            # Treat non-help output as version-action termination.
            probe_text = (probe.output or "").lower()
            if probe.status == 0 and probe_text and "usage:" not in probe_text:
                return _VERSION_OPTION
        return None

    # For non-version requests, parse errors still disable fast-path.
    if scan.has_parse_error:
        return None

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


def _extract_completion_shell(argv: list[str]) -> str | None:
    """Extract the shell name following ``--completion`` in *argv*.

    Returns the shell name (e.g. ``"bash"``) or ``None`` if ``--completion``
    is not present.  Raises ``SystemExit(1)`` when ``--completion`` appears
    with no following argument or with an unsupported shell name.
    """
    args = argv[1:]  # skip argv[0]
    try:
        idx = args.index(_COMPLETION_OPTION)
    except ValueError:
        return None

    if idx + 1 >= len(args):
        print(
            f"error: --completion requires a shell argument ({', '.join(sorted(_SUPPORTED_SHELLS))})",
            file=sys.stderr,
        )
        raise SystemExit(1)

    shell = args[idx + 1].lower()
    if shell not in _SUPPORTED_SHELLS:
        print(
            f"error: unsupported shell '{shell}'. Supported shells: {', '.join(sorted(_SUPPORTED_SHELLS))}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return shell


def _handle_completion(shell: str) -> None:
    """Print the shell completion activation script and exit.

    Exits 0 on success, 1 if argcomplete is not installed.
    """
    try:
        import argcomplete as _argcomplete  # noqa: F401
    except ImportError:
        print(
            "error: argcomplete is not installed. Install it with:\n  pip install argcomplete",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    print(_COMPLETION_SCRIPTS[shell])
    raise SystemExit(0)


def _completion_fast_path_allowed(argv: list[str]) -> bool:
    """Return True when completion can be handled directly in __main__.

    When run-summary is requested, completion must flow through
    ``generator.main()`` so summary emission remains contract-consistent.
    """
    args = argv[1:] if len(argv) > 1 else []
    return not _has_run_summary_contract_flag(args)


def main() -> None:
    """Entry point with fast-path for lightweight flags."""
    # Argcomplete shell completion relies on parser-side hooks in
    # parse_arguments(); do not short-circuit fast-path in that mode.
    if _is_argcomplete_completion_active():
        from cja_auto_sdr.generator import main as _generator_main

        _generator_main()
        return

    # --completion fast-path: detect before heavyweight imports, except when
    # run-summary is requested (must route through generator.main()).
    if _completion_fast_path_allowed(sys.argv):
        completion_shell = _extract_completion_shell(sys.argv)
        if completion_shell is not None:
            _handle_completion(completion_shell)
            # _handle_completion always raises SystemExit; this is a safety net.
            return  # pragma: no cover

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
