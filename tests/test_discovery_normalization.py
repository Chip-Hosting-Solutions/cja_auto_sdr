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

