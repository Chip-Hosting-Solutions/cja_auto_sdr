"""Tests for discovery normalization helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cja_auto_sdr.core.discovery_normalization import (
    extract_owner_name,
    extract_owner_name_from_record,
    extract_tags,
    is_missing_value,
    normalize_display_text,
    pick_first_present_text,
)


def test_is_missing_value_handles_pandas_scalars() -> None:
    assert is_missing_value(None) is True
    assert is_missing_value(float("nan")) is True
    assert is_missing_value(pd.NA) is True


def test_is_missing_value_handles_blank_and_null_like_strings() -> None:
    assert is_missing_value("", treat_blank_string=True) is True
    assert is_missing_value("   ", treat_blank_string=True) is True
    assert is_missing_value(" NaN ", treat_null_like_strings=True) is True
    assert is_missing_value("value", treat_blank_string=True, treat_null_like_strings=True) is False


def test_is_missing_value_handles_pd_isna_type_error(monkeypatch) -> None:
    def raising_isna(_value: Any) -> bool:
        raise TypeError("simulated isna failure")

    monkeypatch.setattr("cja_auto_sdr.core.discovery_normalization.pd.isna", raising_isna)
    assert is_missing_value(object()) is False


def test_is_missing_value_handles_pd_isna_all_type_error(monkeypatch) -> None:
    class FailingAllResult:
        def all(self) -> bool:
            raise TypeError("simulated all failure")

    monkeypatch.setattr(
        "cja_auto_sdr.core.discovery_normalization.pd.isna",
        lambda _value: FailingAllResult(),
    )
    assert is_missing_value(object()) is False


def test_is_missing_value_handles_pd_isna_non_bool_without_all(monkeypatch) -> None:
    class NonBoolNoAllResult:
        pass

    monkeypatch.setattr(
        "cja_auto_sdr.core.discovery_normalization.pd.isna",
        lambda _value: NonBoolNoAllResult(),
    )
    assert is_missing_value(object()) is False


def test_normalize_display_text_treats_null_like_strings_as_missing() -> None:
    assert normalize_display_text(pd.NA, default="N/A", treat_null_like_strings=True) == "N/A"
    assert normalize_display_text(" nan ", default="N/A", treat_null_like_strings=True) == "N/A"
    assert normalize_display_text("  Alice  ", default="N/A", treat_null_like_strings=True) == "Alice"


def test_pick_first_present_text_skips_missing_candidates() -> None:
    candidates: list[Any] = [pd.NA, float("nan"), "", "none", "Owner A"]
    assert pick_first_present_text(candidates, default="N/A", treat_null_like_strings=True) == "Owner A"


def test_extract_owner_name_falls_back_through_owner_fields() -> None:
    owner = {"name": pd.NA, "fullName": " ", "email": "alias@example.com"}
    assert extract_owner_name(owner) == "alias@example.com"


def test_extract_owner_name_from_record_uses_alias_when_owner_missing_like() -> None:
    record = {"owner": float("nan"), "ownerFullName": "Alias Owner"}
    assert extract_owner_name_from_record(record) == "Alias Owner"


def test_extract_owner_name_mapping_with_only_missing_values_returns_default() -> None:
    owner = {
        "name": pd.NA,
        "fullName": " ",
        "email": " none ",
        "id": "nan",
    }
    assert extract_owner_name(owner) == "N/A"


def test_extract_owner_name_from_record_prefers_owner_object_when_present() -> None:
    record = {"owner": {"name": "Primary Owner"}, "ownerFullName": "Alias Owner"}
    assert extract_owner_name_from_record(record) == "Primary Owner"


def test_extract_owner_name_from_record_returns_default_when_owner_and_alias_missing() -> None:
    record = {
        "owner": {"name": " ", "fullName": pd.NA},
        "ownerFullName": " ",
        "ownerName": "none",
        "owner_name": pd.NA,
        "owner_full_name": "",
    }
    assert extract_owner_name_from_record(record) == "N/A"


def test_extract_tags_handles_mixed_and_missing_values() -> None:
    raw_tags: list[Any] = [
        {"name": "kpi"},
        pd.NA,
        float("nan"),
        {"name": " nan "},
        "finance",
        "   ",
    ]
    assert extract_tags(raw_tags) == ["kpi", "finance"]


def test_extract_tags_handles_scalar_missing_values() -> None:
    assert extract_tags(float("nan")) == []
    assert extract_tags(pd.NA) == []
    assert extract_tags("nan") == []
    assert extract_tags("prod") == ["prod"]


# ---------------------------------------------------------------------------
# extract_tags — Mapping input (line 161)
# ---------------------------------------------------------------------------


def test_extract_tags_from_single_mapping() -> None:
    """A dict passed as tags_data → wrapped in list, name extracted."""
    assert extract_tags({"name": "finance"}) == ["finance"]


def test_extract_tags_from_mapping_falls_through_fields() -> None:
    """Mapping with missing 'name' → falls through to 'label'."""
    assert extract_tags({"name": pd.NA, "label": "accounting"}) == ["accounting"]


# ---------------------------------------------------------------------------
# normalize_display_text — redundant null-like guard (line 68)
# ---------------------------------------------------------------------------


def test_normalize_display_text_non_string_that_stringifies_to_null_like() -> None:
    """A non-string value whose str() repr is null-like, with treat_null_like_strings."""

    # is_missing_value sees an int (not missing), but str(value) == "0" (not null-like)
    # so the line-68 branch only fires when str(value).casefold() ∈ _NULL_LIKE_TEXT_VALUES.
    # Use a custom object whose repr is "null".
    class NullStr:
        def __str__(self) -> str:
            return "null"

    result = normalize_display_text(NullStr(), default="N/A", treat_null_like_strings=True)
    assert result == "N/A"
