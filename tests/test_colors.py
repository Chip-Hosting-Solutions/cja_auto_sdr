"""Tests for ConsoleColors, format_file_size, open_file_in_default_app, and _format_error_msg.

Validates:
1. ConsoleColors formatting when enabled vs disabled
2. configure() priority: --no-color > FORCE_COLOR > NO_COLOR > TTY
3. Theme management (default and accessible)
4. All formatting methods (success/green, error/red, warning/yellow, etc.)
5. Theme-aware diff_* methods
6. visible_len, rjust, ljust with ANSI codes
7. format_file_size across all units
8. open_file_in_default_app cross-platform behavior
9. _format_error_msg parameter combinations
10. ANSIColors backward-compatibility alias
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cja_auto_sdr.core.colors import (
    ANSIColors,
    ConsoleColors,
    _format_error_msg,
    format_file_size,
    open_file_in_default_app,
)


@pytest.fixture(autouse=True)
def _reset_console_colors():
    """Reset ConsoleColors state before and after each test."""
    original_enabled = ConsoleColors._enabled
    original_theme = ConsoleColors._theme
    yield
    ConsoleColors._enabled = original_enabled
    ConsoleColors._theme = original_theme


# ---------------------------------------------------------------------------
# ConsoleColors enabled
# ---------------------------------------------------------------------------
class TestConsoleColorsEnabled:
    """Tests for formatting methods when colors are enabled."""

    def setup_method(self):
        ConsoleColors.set_enabled(True)

    def test_success_wraps_green(self):
        result = ConsoleColors.success("ok")
        assert result == "\033[92mok\033[0m"

    def test_green_alias_matches_success(self):
        assert ConsoleColors.green("x") == ConsoleColors.success("x")

    def test_error_wraps_red(self):
        result = ConsoleColors.error("fail")
        assert result == "\033[91mfail\033[0m"

    def test_red_alias_matches_error(self):
        assert ConsoleColors.red("x") == ConsoleColors.error("x")

    def test_warning_wraps_yellow(self):
        result = ConsoleColors.warning("warn")
        assert result == "\033[93mwarn\033[0m"

    def test_yellow_alias_matches_warning(self):
        assert ConsoleColors.yellow("x") == ConsoleColors.warning("x")

    def test_info_wraps_cyan(self):
        result = ConsoleColors.info("note")
        assert result == "\033[96mnote\033[0m"

    def test_cyan_alias_matches_info(self):
        assert ConsoleColors.cyan("x") == ConsoleColors.info("x")

    def test_bold_wraps_bold(self):
        result = ConsoleColors.bold("strong")
        assert result == "\033[1mstrong\033[0m"

    def test_dim_wraps_dim(self):
        result = ConsoleColors.dim("quiet")
        assert result == "\033[90mquiet\033[0m"

    def test_empty_string_enabled(self):
        """Formatting methods still wrap empty strings when enabled."""
        assert ConsoleColors.success("") == "\033[92m\033[0m"
        assert ConsoleColors.error("") == "\033[91m\033[0m"

    def test_multiline_text(self):
        text = "line1\nline2"
        result = ConsoleColors.success(text)
        assert result == "\033[92mline1\nline2\033[0m"


# ---------------------------------------------------------------------------
# ConsoleColors disabled
# ---------------------------------------------------------------------------
class TestConsoleColorsDisabled:
    """Tests for formatting methods when colors are disabled."""

    def setup_method(self):
        ConsoleColors.set_enabled(False)

    def test_success_returns_plain(self):
        assert ConsoleColors.success("ok") == "ok"

    def test_error_returns_plain(self):
        assert ConsoleColors.error("fail") == "fail"

    def test_warning_returns_plain(self):
        assert ConsoleColors.warning("warn") == "warn"

    def test_info_returns_plain(self):
        assert ConsoleColors.info("note") == "note"

    def test_bold_returns_plain(self):
        assert ConsoleColors.bold("strong") == "strong"

    def test_dim_returns_plain(self):
        assert ConsoleColors.dim("quiet") == "quiet"

    def test_diff_added_returns_plain(self):
        assert ConsoleColors.diff_added("new") == "new"

    def test_diff_removed_returns_plain(self):
        assert ConsoleColors.diff_removed("old") == "old"

    def test_diff_modified_returns_plain(self):
        assert ConsoleColors.diff_modified("changed") == "changed"


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------
class TestConfigure:
    """Tests for ConsoleColors.configure() priority logic."""

    def test_no_color_flag_always_disables(self):
        """--no-color flag overrides everything."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            result = ConsoleColors.configure(no_color=True)
            assert result is False
            assert ConsoleColors.is_enabled() is False

    def test_force_color_1_enables(self):
        """FORCE_COLOR=1 enables colors."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is True
            assert ConsoleColors.is_enabled() is True

    def test_force_color_nonempty_enables(self):
        """FORCE_COLOR with any non-empty, non-'0' value enables."""
        with patch.dict(os.environ, {"FORCE_COLOR": "yes"}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is True

    def test_force_color_0_disables(self):
        """FORCE_COLOR=0 disables colors."""
        with patch.dict(os.environ, {"FORCE_COLOR": "0"}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is False

    def test_force_color_empty_disables(self):
        """FORCE_COLOR='' (empty) disables colors."""
        with patch.dict(os.environ, {"FORCE_COLOR": ""}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is False

    def test_force_color_whitespace_only_disables(self):
        """FORCE_COLOR='  ' (whitespace only) disables colors after strip."""
        with patch.dict(os.environ, {"FORCE_COLOR": "  "}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is False

    def test_no_color_env_nonempty_disables(self):
        """NO_COLOR env var (non-empty) disables colors."""
        env = {"NO_COLOR": "1"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("FORCE_COLOR", None)
            result = ConsoleColors.configure(no_color=False)
            assert result is False

    def test_no_color_env_empty_falls_through(self):
        """NO_COLOR='' (empty) does not disable -- falls to TTY detection."""
        env = {"NO_COLOR": ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("FORCE_COLOR", None)
            with patch.object(ConsoleColors, "_detect_default_enabled", return_value=True):
                result = ConsoleColors.configure(no_color=False)
                assert result is True

    def test_no_color_env_whitespace_only_falls_through(self):
        """NO_COLOR='  ' (whitespace only) does not disable -- falls to TTY."""
        env = {"NO_COLOR": "  "}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("FORCE_COLOR", None)
            with patch.object(ConsoleColors, "_detect_default_enabled", return_value=True):
                result = ConsoleColors.configure(no_color=False)
                assert result is True

    def test_tty_fallback_enabled(self):
        """Falls back to TTY detection when no flags/env vars set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORCE_COLOR", None)
            os.environ.pop("NO_COLOR", None)
            with patch.object(ConsoleColors, "_detect_default_enabled", return_value=True):
                result = ConsoleColors.configure(no_color=False)
                assert result is True

    def test_tty_fallback_disabled(self):
        """Falls back to TTY detection returning False."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORCE_COLOR", None)
            os.environ.pop("NO_COLOR", None)
            with patch.object(ConsoleColors, "_detect_default_enabled", return_value=False):
                result = ConsoleColors.configure(no_color=False)
                assert result is False

    def test_force_color_takes_priority_over_no_color_env(self):
        """FORCE_COLOR has higher priority than NO_COLOR env."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1", "NO_COLOR": "1"}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is True

    def test_no_color_flag_beats_force_color(self):
        """--no-color flag beats FORCE_COLOR."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            result = ConsoleColors.configure(no_color=True)
            assert result is False

    def test_configure_returns_and_sets_state(self):
        """configure() both returns the value and sets _enabled."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            result = ConsoleColors.configure(no_color=False)
            assert result is ConsoleColors.is_enabled()


class TestDetectDefaultEnabled:
    """Tests for _detect_default_enabled TTY detection."""

    def test_tty_non_windows(self):
        with patch("sys.stdout") as mock_stdout, patch("os.name", "posix"):
            mock_stdout.isatty.return_value = True
            assert ConsoleColors._detect_default_enabled() is True

    def test_not_tty(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert ConsoleColors._detect_default_enabled() is False

    def test_windows_without_term(self):
        with patch("sys.stdout") as mock_stdout, patch("os.name", "nt"), patch.dict(os.environ, {}, clear=True):
            mock_stdout.isatty.return_value = True
            assert not ConsoleColors._detect_default_enabled()

    def test_windows_with_term(self):
        with (
            patch("sys.stdout") as mock_stdout,
            patch("os.name", "nt"),
            patch.dict(os.environ, {"TERM": "xterm"}, clear=False),
        ):
            mock_stdout.isatty.return_value = True
            assert ConsoleColors._detect_default_enabled()


# ---------------------------------------------------------------------------
# set_enabled / is_enabled
# ---------------------------------------------------------------------------
class TestSetEnabled:
    """Tests for set_enabled and is_enabled."""

    def test_set_enabled_true(self):
        ConsoleColors.set_enabled(True)
        assert ConsoleColors.is_enabled() is True

    def test_set_enabled_false(self):
        ConsoleColors.set_enabled(False)
        assert ConsoleColors.is_enabled() is False

    def test_set_enabled_coerces_to_bool(self):
        ConsoleColors.set_enabled(1)
        assert ConsoleColors._enabled is True
        ConsoleColors.set_enabled(0)
        assert ConsoleColors._enabled is False


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
class TestThemes:
    """Tests for theme management and theme-aware diff methods."""

    def setup_method(self):
        ConsoleColors.set_enabled(True)
        ConsoleColors.set_theme("default")

    def test_set_valid_theme_default(self):
        ConsoleColors.set_theme("default")
        assert ConsoleColors._theme == "default"

    def test_set_valid_theme_accessible(self):
        ConsoleColors.set_theme("accessible")
        assert ConsoleColors._theme == "accessible"

    def test_set_invalid_theme_raises(self):
        with pytest.raises(ValueError, match="Unknown theme: nope"):
            ConsoleColors.set_theme("nope")

    def test_invalid_theme_message_lists_valid(self):
        with pytest.raises(ValueError, match="default"):
            ConsoleColors.set_theme("invalid")

    def test_diff_added_default_theme(self):
        ConsoleColors.set_theme("default")
        result = ConsoleColors.diff_added("new")
        assert result == "\033[92mnew\033[0m"

    def test_diff_removed_default_theme(self):
        ConsoleColors.set_theme("default")
        result = ConsoleColors.diff_removed("old")
        assert result == "\033[91mold\033[0m"

    def test_diff_modified_default_theme(self):
        ConsoleColors.set_theme("default")
        result = ConsoleColors.diff_modified("changed")
        assert result == "\033[93mchanged\033[0m"

    def test_diff_added_accessible_theme(self):
        ConsoleColors.set_theme("accessible")
        result = ConsoleColors.diff_added("new")
        assert result == "\033[94mnew\033[0m"

    def test_diff_removed_accessible_theme(self):
        ConsoleColors.set_theme("accessible")
        result = ConsoleColors.diff_removed("old")
        assert result == "\033[38;5;208mold\033[0m"

    def test_diff_modified_accessible_theme(self):
        ConsoleColors.set_theme("accessible")
        result = ConsoleColors.diff_modified("changed")
        assert result == "\033[96mchanged\033[0m"


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------
class TestStatus:
    """Tests for status() conditional formatting."""

    def test_status_true_returns_green(self):
        ConsoleColors.set_enabled(True)
        result = ConsoleColors.status(True, "pass")
        assert result == ConsoleColors.success("pass")
        assert "\033[92m" in result

    def test_status_false_returns_red(self):
        ConsoleColors.set_enabled(True)
        result = ConsoleColors.status(False, "fail")
        assert result == ConsoleColors.error("fail")
        assert "\033[91m" in result

    def test_status_disabled_returns_plain(self):
        ConsoleColors.set_enabled(False)
        assert ConsoleColors.status(True, "ok") == "ok"
        assert ConsoleColors.status(False, "bad") == "bad"


# ---------------------------------------------------------------------------
# visible_len
# ---------------------------------------------------------------------------
class TestVisibleLen:
    """Tests for visible_len ANSI-stripping length calculation."""

    def test_plain_text(self):
        assert ConsoleColors.visible_len("hello") == 5

    def test_empty_string(self):
        assert ConsoleColors.visible_len("") == 0

    def test_text_with_single_ansi_code(self):
        colored = "\033[92mhello\033[0m"
        assert ConsoleColors.visible_len(colored) == 5

    def test_text_with_multiple_ansi_codes(self):
        colored = "\033[1m\033[92mbold green\033[0m"
        assert ConsoleColors.visible_len(colored) == 10

    def test_text_with_extended_256_color(self):
        colored = "\033[38;5;208morange\033[0m"
        assert ConsoleColors.visible_len(colored) == 6

    def test_only_ansi_codes(self):
        assert ConsoleColors.visible_len("\033[92m\033[0m") == 0

    def test_mixed_content(self):
        text = "prefix " + "\033[91mred\033[0m" + " suffix"
        assert ConsoleColors.visible_len(text) == len("prefix red suffix")


# ---------------------------------------------------------------------------
# rjust / ljust
# ---------------------------------------------------------------------------
class TestJustify:
    """Tests for rjust and ljust with ANSI-aware padding."""

    def test_rjust_plain_text(self):
        result = ConsoleColors.rjust("hi", 10)
        assert result == "        hi"
        assert len(result) == 10

    def test_ljust_plain_text(self):
        result = ConsoleColors.ljust("hi", 10)
        assert result == "hi        "
        assert len(result) == 10

    def test_rjust_ansi_text(self):
        colored = "\033[92mhi\033[0m"
        result = ConsoleColors.rjust(colored, 10)
        assert result.endswith(colored)
        visible = ConsoleColors.visible_len(result)
        assert visible == 10

    def test_ljust_ansi_text(self):
        colored = "\033[92mhi\033[0m"
        result = ConsoleColors.ljust(colored, 10)
        assert result.startswith(colored)
        visible = ConsoleColors.visible_len(result)
        assert visible == 10

    def test_rjust_text_longer_than_width(self):
        result = ConsoleColors.rjust("hello world", 5)
        assert result == "hello world"

    def test_ljust_text_longer_than_width(self):
        result = ConsoleColors.ljust("hello world", 5)
        assert result == "hello world"

    def test_rjust_exact_width(self):
        result = ConsoleColors.rjust("abc", 3)
        assert result == "abc"

    def test_ljust_exact_width(self):
        result = ConsoleColors.ljust("abc", 3)
        assert result == "abc"

    def test_rjust_zero_width(self):
        result = ConsoleColors.rjust("abc", 0)
        assert result == "abc"

    def test_ljust_zero_width(self):
        result = ConsoleColors.ljust("abc", 0)
        assert result == "abc"


# ---------------------------------------------------------------------------
# format_file_size
# ---------------------------------------------------------------------------
class TestFormatFileSize:
    """Tests for format_file_size human-readable formatting."""

    def test_zero_bytes(self):
        assert format_file_size(0) == "0 B"

    def test_small_bytes(self):
        assert format_file_size(512) == "512 B"

    def test_one_byte(self):
        assert format_file_size(1) == "1 B"

    def test_1023_bytes(self):
        assert format_file_size(1023) == "1023 B"

    def test_exactly_1024_kb(self):
        assert format_file_size(1024) == "1.0 KB"

    def test_kilobytes(self):
        assert format_file_size(1536) == "1.5 KB"

    def test_megabytes(self):
        assert format_file_size(1024 * 1024) == "1.0 MB"

    def test_megabytes_fractional(self):
        assert format_file_size(int(2.5 * 1024 * 1024)) == "2.5 MB"

    def test_gigabytes(self):
        assert format_file_size(1024**3) == "1.0 GB"

    def test_terabytes(self):
        assert format_file_size(1024**4) == "1.0 TB"

    def test_large_terabytes(self):
        assert format_file_size(5 * 1024**4) == "5.0 TB"

    def test_bytes_no_decimal(self):
        result = format_file_size(42)
        assert result == "42 B"
        assert "." not in result


# ---------------------------------------------------------------------------
# open_file_in_default_app
# ---------------------------------------------------------------------------
class TestOpenFile:
    """Tests for open_file_in_default_app cross-platform behavior."""

    @patch("subprocess.run")
    @patch("platform.system", return_value="Darwin")
    def test_darwin_uses_open(self, _mock_platform, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = open_file_in_default_app("/tmp/report.xlsx")
        assert result is True
        mock_run.assert_called_once_with(["open", "/tmp/report.xlsx"], check=True)

    @patch("subprocess.run")
    @patch("platform.system", return_value="Linux")
    def test_linux_uses_xdg_open(self, _mock_platform, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = open_file_in_default_app("/tmp/report.xlsx")
        assert result is True
        mock_run.assert_called_once_with(["xdg-open", "/tmp/report.xlsx"], check=True)

    @patch("os.startfile", create=True)
    @patch("platform.system", return_value="Windows")
    def test_windows_uses_startfile(self, _mock_platform, mock_startfile):
        result = open_file_in_default_app("C:\\report.xlsx")
        assert result is True
        mock_startfile.assert_called_once_with("C:\\report.xlsx")

    @patch("webbrowser.open")
    @patch("subprocess.run", side_effect=OSError("no such command"))
    @patch("platform.system", return_value="Darwin")
    def test_fallback_to_webbrowser_for_html(self, _mock_platform, _mock_run, mock_wb):
        result = open_file_in_default_app("/tmp/report.html")
        assert result is True
        mock_wb.assert_called_once()
        call_arg = mock_wb.call_args[0][0]
        assert call_arg.startswith("file://")
        assert "report.html" in call_arg

    @patch("subprocess.run", side_effect=OSError("no such command"))
    @patch("platform.system", return_value="Darwin")
    def test_non_html_failure_returns_false(self, _mock_platform, _mock_run):
        result = open_file_in_default_app("/tmp/report.xlsx")
        assert result is False

    @patch("webbrowser.open", side_effect=Exception("browser error"))
    @patch("subprocess.run", side_effect=OSError("no such command"))
    @patch("platform.system", return_value="Darwin")
    def test_html_fallback_also_fails(self, _mock_platform, _mock_run, _mock_wb):
        result = open_file_in_default_app("/tmp/report.html")
        assert result is False

    @patch("subprocess.run")
    @patch("platform.system", return_value="Darwin")
    def test_accepts_path_object(self, _mock_platform, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = open_file_in_default_app(Path("/tmp/report.xlsx"))
        assert result is True
        mock_run.assert_called_once_with(["open", "/tmp/report.xlsx"], check=True)


# ---------------------------------------------------------------------------
# _format_error_msg
# ---------------------------------------------------------------------------
class TestFormatErrorMsg:
    """Tests for _format_error_msg helper."""

    def test_operation_only(self):
        result = _format_error_msg("checking duplicates")
        assert result == "Error checking duplicates"

    def test_operation_with_item_type(self):
        result = _format_error_msg("checking duplicates", item_type="Metrics")
        assert result == "Error checking duplicates for Metrics"

    def test_operation_with_error(self):
        err = ValueError("bad value")
        result = _format_error_msg("checking duplicates", error=err)
        assert result == "Error checking duplicates: bad value"

    def test_operation_with_item_type_and_error(self):
        err = RuntimeError("timeout")
        result = _format_error_msg("fetching data", item_type="Dimensions", error=err)
        assert result == "Error fetching data for Dimensions: timeout"

    def test_none_item_type_excluded(self):
        result = _format_error_msg("processing", item_type=None)
        assert "for" not in result

    def test_none_error_excluded(self):
        result = _format_error_msg("processing", error=None)
        assert ":" not in result

    def test_empty_string_item_type_excluded(self):
        result = _format_error_msg("processing", item_type="")
        assert result == "Error processing"


# ---------------------------------------------------------------------------
# ANSIColors alias
# ---------------------------------------------------------------------------
class TestANSIColorsAlias:
    """Tests for the ANSIColors backward-compatibility alias."""

    def test_ansi_colors_is_console_colors(self):
        assert ANSIColors is ConsoleColors

    def test_ansi_colors_methods_work(self):
        ANSIColors.set_enabled(True)
        result = ANSIColors.success("ok")
        assert "\033[92m" in result

    def test_shared_state(self):
        ANSIColors.set_enabled(True)
        assert ConsoleColors.is_enabled() is True
        ConsoleColors.set_enabled(False)
        assert ANSIColors.is_enabled() is False


# ---------------------------------------------------------------------------
# ANSI_ESCAPE regex
# ---------------------------------------------------------------------------
class TestAnsiEscapeRegex:
    """Tests for the ANSI_ESCAPE compiled regex."""

    def test_strips_simple_codes(self):
        assert ConsoleColors.ANSI_ESCAPE.sub("", "\033[91mhello\033[0m") == "hello"

    def test_strips_extended_codes(self):
        assert ConsoleColors.ANSI_ESCAPE.sub("", "\033[38;5;208mtext\033[0m") == "text"

    def test_strips_bold_code(self):
        assert ConsoleColors.ANSI_ESCAPE.sub("", "\033[1mbold\033[0m") == "bold"

    def test_no_codes_unchanged(self):
        assert ConsoleColors.ANSI_ESCAPE.sub("", "plain") == "plain"

    def test_empty_string(self):
        assert ConsoleColors.ANSI_ESCAPE.sub("", "") == ""


# ---------------------------------------------------------------------------
# THEMES class variable
# ---------------------------------------------------------------------------
class TestThemeStructure:
    """Tests for the THEMES dictionary structure."""

    def test_default_theme_exists(self):
        assert "default" in ConsoleColors.THEMES

    def test_accessible_theme_exists(self):
        assert "accessible" in ConsoleColors.THEMES

    def test_default_theme_has_required_keys(self):
        theme = ConsoleColors.THEMES["default"]
        assert "added" in theme
        assert "removed" in theme
        assert "modified" in theme

    def test_accessible_theme_has_required_keys(self):
        theme = ConsoleColors.THEMES["accessible"]
        assert "added" in theme
        assert "removed" in theme
        assert "modified" in theme

    def test_default_theme_uses_standard_colors(self):
        theme = ConsoleColors.THEMES["default"]
        assert theme["added"] == ConsoleColors.GREEN
        assert theme["removed"] == ConsoleColors.RED
        assert theme["modified"] == ConsoleColors.YELLOW

    def test_accessible_theme_uses_accessible_colors(self):
        theme = ConsoleColors.THEMES["accessible"]
        assert theme["added"] == ConsoleColors.BLUE
        assert theme["removed"] == ConsoleColors.ORANGE
        assert theme["modified"] == ConsoleColors.CYAN


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
class TestColorConstants:
    """Tests for ANSI color code constants."""

    def test_green_code(self):
        assert ConsoleColors.GREEN == "\033[92m"

    def test_red_code(self):
        assert ConsoleColors.RED == "\033[91m"

    def test_yellow_code(self):
        assert ConsoleColors.YELLOW == "\033[93m"

    def test_blue_code(self):
        assert ConsoleColors.BLUE == "\033[94m"

    def test_cyan_code(self):
        assert ConsoleColors.CYAN == "\033[96m"

    def test_orange_code(self):
        assert ConsoleColors.ORANGE == "\033[38;5;208m"

    def test_bold_code(self):
        assert ConsoleColors.BOLD == "\033[1m"

    def test_dim_code(self):
        assert ConsoleColors.DIM == "\033[90m"

    def test_reset_code(self):
        assert ConsoleColors.RESET == "\033[0m"
