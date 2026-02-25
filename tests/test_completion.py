"""Tests for --completion bash/zsh/fish flag.

Covers:
- Fast-path detection in __main__.py for each supported shell
- Correct activation script output on stdout for bash, zsh, fish
- Missing argcomplete triggers stderr message and exit 1
- Invalid shell name triggers stderr error and exit 1
- --completion with no shell argument triggers stderr error and exit 1
- Safety-net dispatch from generator._main_impl()
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_entrypoint_main(argv: list[str]):
    """Call __main__.main() with the given argv and capture SystemExit."""
    from cja_auto_sdr import __main__ as entrypoint

    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as exc_info:
            entrypoint.main()
    return exc_info


# ---------------------------------------------------------------------------
# _extract_completion_shell unit tests
# ---------------------------------------------------------------------------


class TestExtractCompletionShell:
    """Unit tests for _extract_completion_shell()."""

    def test_returns_none_when_no_completion_flag(self):
        from cja_auto_sdr.__main__ import _extract_completion_shell

        assert _extract_completion_shell(["prog", "--version"]) is None

    def test_returns_none_for_empty_args(self):
        from cja_auto_sdr.__main__ import _extract_completion_shell

        assert _extract_completion_shell(["prog"]) is None

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_extracts_valid_shell(self, shell):
        from cja_auto_sdr.__main__ import _extract_completion_shell

        result = _extract_completion_shell(["prog", "--completion", shell])
        assert result == shell

    def test_shell_name_is_case_insensitive(self):
        from cja_auto_sdr.__main__ import _extract_completion_shell

        assert _extract_completion_shell(["prog", "--completion", "BASH"]) == "bash"
        assert _extract_completion_shell(["prog", "--completion", "Zsh"]) == "zsh"
        assert _extract_completion_shell(["prog", "--completion", "FISH"]) == "fish"

    def test_missing_shell_argument_exits_1(self, capsys):
        from cja_auto_sdr.__main__ import _extract_completion_shell

        with pytest.raises(SystemExit) as exc_info:
            _extract_completion_shell(["prog", "--completion"])

        assert int(exc_info.value.code) == 1
        stderr = capsys.readouterr().err
        assert "--completion requires a shell argument" in stderr

    def test_invalid_shell_exits_1(self, capsys):
        from cja_auto_sdr.__main__ import _extract_completion_shell

        with pytest.raises(SystemExit) as exc_info:
            _extract_completion_shell(["prog", "--completion", "powershell"])

        assert int(exc_info.value.code) == 1
        stderr = capsys.readouterr().err
        assert "unsupported shell 'powershell'" in stderr
        assert "bash" in stderr
        assert "zsh" in stderr
        assert "fish" in stderr


# ---------------------------------------------------------------------------
# _handle_completion unit tests
# ---------------------------------------------------------------------------


class TestHandleCompletion:
    """Unit tests for _handle_completion()."""

    @pytest.mark.parametrize(
        ("shell", "expected_fragment"),
        [
            ("bash", 'eval "$(register-python-argcomplete cja_auto_sdr)"'),
            ("zsh", "autoload -U bashcompinit && bashcompinit"),
            ("zsh", 'eval "$(register-python-argcomplete cja_auto_sdr)"'),
            ("fish", "register-python-argcomplete --shell fish cja_auto_sdr | source"),
        ],
    )
    def test_prints_activation_script_to_stdout(self, shell, expected_fragment, capsys):
        """Each shell produces the correct activation snippet on stdout."""
        from cja_auto_sdr.__main__ import _handle_completion

        with patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}):
            with pytest.raises(SystemExit) as exc_info:
                _handle_completion(shell)

        assert int(exc_info.value.code) == 0
        stdout = capsys.readouterr().out
        assert expected_fragment in stdout

    def test_missing_argcomplete_prints_install_instructions(self, capsys):
        """When argcomplete is missing, print install instructions to stderr and exit 1."""
        from cja_auto_sdr.__main__ import _handle_completion

        with patch.dict("sys.modules", {"argcomplete": None}):
            with pytest.raises(SystemExit) as exc_info:
                _handle_completion("bash")

        assert int(exc_info.value.code) == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # nothing on stdout
        assert "argcomplete is not installed" in captured.err
        assert "pip install argcomplete" in captured.err

    def test_stdout_contains_no_stderr_content(self, capsys):
        """Verify script goes to stdout and nothing leaks to stderr on success."""
        from cja_auto_sdr.__main__ import _handle_completion

        with patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}):
            with pytest.raises(SystemExit):
                _handle_completion("bash")

        captured = capsys.readouterr()
        assert captured.err == ""
        assert "register-python-argcomplete" in captured.out


# ---------------------------------------------------------------------------
# Fast-path integration tests (via __main__.main)
# ---------------------------------------------------------------------------


class TestCompletionFastPath:
    """Integration tests verifying --completion is handled in the fast path."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_fast_path_does_not_import_generator(self, shell, capsys):
        """--completion should not fall through to generator.main()."""
        from cja_auto_sdr import __main__ as entrypoint

        with (
            patch.object(sys, "argv", ["cja_auto_sdr", "--completion", shell]),
            patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}),
            patch("cja_auto_sdr.generator.main") as mock_generator_main,
        ):
            with pytest.raises(SystemExit) as exc_info:
                entrypoint.main()

        assert int(exc_info.value.code) == 0
        mock_generator_main.assert_not_called()

    def test_fast_path_bash_output(self, capsys):
        """--completion bash prints the bash activation script."""
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS

        with (
            patch.object(sys, "argv", ["cja_auto_sdr", "--completion", "bash"]),
            patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from cja_auto_sdr import __main__ as entrypoint

                entrypoint.main()

        assert int(exc_info.value.code) == 0
        stdout = capsys.readouterr().out.strip()
        assert stdout == _COMPLETION_SCRIPTS["bash"]

    def test_fast_path_zsh_output(self, capsys):
        """--completion zsh prints the zsh activation script."""
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS

        with (
            patch.object(sys, "argv", ["cja_auto_sdr", "--completion", "zsh"]),
            patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from cja_auto_sdr import __main__ as entrypoint

                entrypoint.main()

        assert int(exc_info.value.code) == 0
        stdout = capsys.readouterr().out.strip()
        assert stdout == _COMPLETION_SCRIPTS["zsh"]

    def test_fast_path_fish_output(self, capsys):
        """--completion fish prints the fish activation script."""
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS

        with (
            patch.object(sys, "argv", ["cja_auto_sdr", "--completion", "fish"]),
            patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from cja_auto_sdr import __main__ as entrypoint

                entrypoint.main()

        assert int(exc_info.value.code) == 0
        stdout = capsys.readouterr().out.strip()
        assert stdout == _COMPLETION_SCRIPTS["fish"]

    def test_fast_path_missing_shell_exits_1(self, capsys):
        """--completion with no shell arg should exit 1 with error on stderr."""
        from cja_auto_sdr import __main__ as entrypoint

        with patch.object(sys, "argv", ["cja_auto_sdr", "--completion"]):
            with pytest.raises(SystemExit) as exc_info:
                entrypoint.main()

        assert int(exc_info.value.code) == 1
        stderr = capsys.readouterr().err
        assert "--completion requires a shell argument" in stderr

    def test_fast_path_invalid_shell_exits_1(self, capsys):
        """--completion with invalid shell name should exit 1 with error on stderr."""
        from cja_auto_sdr import __main__ as entrypoint

        with patch.object(sys, "argv", ["cja_auto_sdr", "--completion", "ksh"]):
            with pytest.raises(SystemExit) as exc_info:
                entrypoint.main()

        assert int(exc_info.value.code) == 1
        stderr = capsys.readouterr().err
        assert "unsupported shell 'ksh'" in stderr

    @pytest.mark.parametrize(
        "argv_tail",
        [
            ["--completion", "bash", "--run-summary-json", "stdout"],
            ["--run-summary-json", "stdout", "--completion", "bash"],
        ],
    )
    def test_completion_with_run_summary_routes_through_generator_contract(self, argv_tail, capsys):
        """Run-summary contract must bypass completion fast-path and emit JSON summary."""
        from cja_auto_sdr import __main__ as entrypoint

        argv = ["cja_auto_sdr", *argv_tail]
        fake_argcomplete = type(sys)("argcomplete")
        fake_argcomplete.autocomplete = lambda *_args, **_kwargs: None
        with (
            patch.object(sys, "argv", argv),
            patch.dict("sys.modules", {"argcomplete": fake_argcomplete}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                entrypoint.main()

        assert int(exc_info.value.code) == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["summary_version"] == "1.0"
        assert payload["exit_code"] == 0
        assert payload["command"]["argv"] == argv
        assert "register-python-argcomplete" in captured.err
        assert "register-python-argcomplete" not in captured.out


# ---------------------------------------------------------------------------
# Argparse integration (parser recognizes --completion)
# ---------------------------------------------------------------------------


class TestCompletionArgparse:
    """Tests that the argparse parser recognizes --completion."""

    def test_parser_accepts_completion_bash(self):
        from cja_auto_sdr.cli.parser import parse_arguments

        args = parse_arguments(["--completion", "bash", "dv_test"])
        assert args.completion == "bash"

    def test_parser_accepts_completion_zsh(self):
        from cja_auto_sdr.cli.parser import parse_arguments

        args = parse_arguments(["--completion", "zsh", "dv_test"])
        assert args.completion == "zsh"

    def test_parser_accepts_completion_fish(self):
        from cja_auto_sdr.cli.parser import parse_arguments

        args = parse_arguments(["--completion", "fish", "dv_test"])
        assert args.completion == "fish"

    def test_parser_rejects_invalid_shell(self):
        from cja_auto_sdr.cli.parser import parse_arguments

        with pytest.raises(SystemExit):
            parse_arguments(["--completion", "powershell", "dv_test"])

    def test_parser_completion_default_is_none(self):
        from cja_auto_sdr.cli.parser import parse_arguments

        args = parse_arguments(["dv_test"])
        assert args.completion is None


# ---------------------------------------------------------------------------
# Completion script content validation
# ---------------------------------------------------------------------------


class TestCompletionScriptContent:
    """Validate the static content of each shell's activation script."""

    def test_bash_script_content(self):
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS

        script = _COMPLETION_SCRIPTS["bash"]
        assert "register-python-argcomplete" in script
        assert "cja_auto_sdr" in script
        assert "eval" in script

    def test_zsh_script_content(self):
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS

        script = _COMPLETION_SCRIPTS["zsh"]
        assert "bashcompinit" in script
        assert "autoload" in script
        assert "register-python-argcomplete" in script
        assert "cja_auto_sdr" in script

    def test_fish_script_content(self):
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS

        script = _COMPLETION_SCRIPTS["fish"]
        assert "--shell fish" in script
        assert "register-python-argcomplete" in script
        assert "cja_auto_sdr" in script
        assert "| source" in script

    def test_all_supported_shells_have_scripts(self):
        from cja_auto_sdr.__main__ import _COMPLETION_SCRIPTS, _SUPPORTED_SHELLS

        assert set(_COMPLETION_SCRIPTS.keys()) == _SUPPORTED_SHELLS


# ---------------------------------------------------------------------------
# Safety-net: generator._main_impl handles --completion
# ---------------------------------------------------------------------------


class TestCompletionSafetyNet:
    """Verify the generator._main_impl safety net for --completion."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_generator_safety_net_handles_completion(self, shell, capsys):
        """If --completion reaches _main_impl, it should still be handled."""
        from cja_auto_sdr.generator import _main_impl

        with (
            patch.object(sys, "argv", ["cja_auto_sdr", "--completion", shell, "dv_test"]),
            patch.dict("sys.modules", {"argcomplete": type(sys)("argcomplete")}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _main_impl()

        assert int(exc_info.value.code) == 0
        stdout = capsys.readouterr().out
        assert "register-python-argcomplete" in stdout
