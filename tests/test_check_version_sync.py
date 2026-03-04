from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from scripts import check_version_sync


def test_get_canonical_version_reads_version_file(tmp_path, monkeypatch) -> None:
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "9.8.7"\n', encoding="utf-8")

    monkeypatch.setattr(check_version_sync, "VERSION_FILE", version_file)
    monkeypatch.setattr(check_version_sync, "ROOT", tmp_path)

    assert check_version_sync.get_canonical_version() == "9.8.7"


def test_check_all_reports_mismatch_for_first_match(tmp_path, monkeypatch) -> None:
    tracked = tmp_path / "tracked.md"
    tracked.write_text("Release v1.2.2\nRelease v1.2.3\n", encoding="utf-8")

    monkeypatch.setattr(
        check_version_sync,
        "VERSION_LOCATIONS",
        [("tracked.md", r"v(\d+\.\d+\.\d+)", "Tracked version")],
    )
    monkeypatch.setattr(check_version_sync, "ROOT", tmp_path)

    errors = check_version_sync.check_all("1.2.3")

    assert len(errors) == 1
    assert "expected 1.2.3, found 1.2.2" in errors[0]


def test_check_release_tag_returns_none_when_tag_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        check_version_sync.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    assert check_version_sync.check_release_tag("3.3.6") is None


def test_check_release_tag_reports_missing_tag(monkeypatch) -> None:
    monkeypatch.setattr(
        check_version_sync.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=""),
    )

    error = check_version_sync.check_release_tag("3.3.6")

    assert error is not None
    assert "v3.3.6" in error
    assert "git fetch --tags" in error


def test_check_release_tag_reports_non_git_repository(monkeypatch) -> None:
    monkeypatch.setattr(
        check_version_sync.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128,
            stderr="fatal: not a git repository (or any of the parent directories): .git",
        ),
    )

    error = check_version_sync.check_release_tag("3.3.6")

    assert error is not None
    assert "not a git repository" in error


def test_check_release_tag_reports_oserror(monkeypatch) -> None:
    def _raise_oserror(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(check_version_sync.subprocess, "run", _raise_oserror)

    error = check_version_sync.check_release_tag("3.3.6")

    assert error is not None
    assert "Unable to verify release tag v3.3.6" in error
    assert "git unavailable" in error


def test_check_ci_tag_ref_match_returns_none_for_matching_tag_ref() -> None:
    error = check_version_sync.check_ci_tag_ref_match("3.3.6", github_ref="refs/tags/v3.3.6")

    assert error is None


def test_check_ci_tag_ref_match_returns_none_for_non_tag_ref() -> None:
    error = check_version_sync.check_ci_tag_ref_match("3.3.6", github_ref="refs/heads/main")

    assert error is None


def test_check_ci_tag_ref_match_reports_mismatch() -> None:
    error = check_version_sync.check_ci_tag_ref_match("3.3.6", github_ref="refs/tags/v3.3.5")

    assert error is not None
    assert "expected refs/tags/v3.3.6" in error
    assert "found refs/tags/v3.3.5" in error


def test_parse_args_defaults_require_tag_false(monkeypatch) -> None:
    monkeypatch.setattr(check_version_sync.sys, "argv", ["check_version_sync.py"])

    args = check_version_sync.parse_args()

    assert args.require_tag is False


def test_parse_args_sets_require_tag_true(monkeypatch) -> None:
    monkeypatch.setattr(check_version_sync.sys, "argv", ["check_version_sync.py", "--require-tag"])

    args = check_version_sync.parse_args()

    assert args.require_tag is True


def test_main_prints_success_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_version_sync,
        "parse_args",
        lambda: argparse.Namespace(require_tag=False),
    )
    monkeypatch.setattr(check_version_sync, "get_canonical_version", lambda: "3.3.6")
    monkeypatch.setattr(check_version_sync, "check_all", lambda _canonical: [])
    monkeypatch.setattr(check_version_sync, "check_release_tag", lambda _canonical: None)
    monkeypatch.setattr(check_version_sync, "check_ci_tag_ref_match", lambda _canonical: None)

    check_version_sync.main()
    output = capsys.readouterr().out

    assert output == "Version sync OK: all references match 3.3.6\n"


def test_main_prints_failure_block_with_canonical_source(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_version_sync,
        "parse_args",
        lambda: argparse.Namespace(require_tag=True),
    )
    monkeypatch.setattr(check_version_sync, "get_canonical_version", lambda: "3.3.6")
    monkeypatch.setattr(
        check_version_sync,
        "check_all",
        lambda _canonical: ["  CHANGELOG.md: expected 3.3.6, found 3.3.5 (CHANGELOG.md latest release entry)"],
    )
    monkeypatch.setattr(
        check_version_sync,
        "check_release_tag",
        lambda _canonical: "Missing required release tag: v3.3.6. Create it with: git tag v3.3.6",
    )
    monkeypatch.setattr(check_version_sync, "check_ci_tag_ref_match", lambda _canonical: None)

    with pytest.raises(SystemExit) as excinfo:
        check_version_sync.main()
    output = capsys.readouterr().out

    assert excinfo.value.code == 1
    assert "Version sync check FAILED (canonical: 3.3.6)" in output
    assert "CHANGELOG.md: expected 3.3.6, found 3.3.5" in output
    assert "Missing required release tag: v3.3.6. Create it with: git tag v3.3.6" in output
    assert f"Canonical source: {check_version_sync.VERSION_FILE.relative_to(check_version_sync.ROOT)}" in output


def test_main_includes_ci_tag_mismatch_in_failure_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_version_sync,
        "parse_args",
        lambda: argparse.Namespace(require_tag=True),
    )
    monkeypatch.setattr(check_version_sync, "get_canonical_version", lambda: "3.3.6")
    monkeypatch.setattr(check_version_sync, "check_all", lambda _canonical: [])
    monkeypatch.setattr(check_version_sync, "check_release_tag", lambda _canonical: None)
    monkeypatch.setattr(
        check_version_sync,
        "check_ci_tag_ref_match",
        lambda _canonical: "GITHUB_REF tag mismatch: expected refs/tags/v3.3.6, found refs/tags/v3.3.5",
    )

    with pytest.raises(SystemExit):
        check_version_sync.main()
    output = capsys.readouterr().out

    assert "GITHUB_REF tag mismatch: expected refs/tags/v3.3.6, found refs/tags/v3.3.5" in output
