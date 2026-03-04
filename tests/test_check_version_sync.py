from __future__ import annotations

from types import SimpleNamespace

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
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    error = check_version_sync.check_release_tag("3.3.6")

    assert error is not None
    assert "v3.3.6" in error
