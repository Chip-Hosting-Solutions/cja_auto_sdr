"""Discovery payload classification for data-view inspection commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from cja_auto_sdr.core.discovery_normalization import is_missing_value

EXPLICIT_ERROR_KEYS = frozenset({"error", "errorcode", "errordescription", "error_description"})
STATUS_KEYS = frozenset({"statuscode", "status_code", "status"})
ERROR_TEXT_KEYS = frozenset({"message", "detail", "title"})
COMPONENT_SEQUENCE_KEYS = ("content", "items", "results", "data", "rows")


class PayloadKind(Enum):
    """Normalized classification for discovery API payloads."""

    DATA = "data"
    EMPTY = "empty"
    ERROR = "error"
    INVALID = "invalid"


@dataclass(frozen=True)
class PayloadAssessment:
    """Normalized payload assessment used by discovery list/describe commands."""

    kind: PayloadKind
    rows: list[dict[str, Any]]
    reason: str
    raw_type: str


def normalized_payload_keys(payload: Mapping[str, Any]) -> set[str]:
    """Return normalized dictionary keys for payload-shape checks."""
    return {str(key).strip().casefold() for key in payload}


def has_identity_value(payload: Mapping[str, Any], identity_keys: tuple[str, ...]) -> bool:
    """Return True when any identity key has a non-missing value."""
    for key in identity_keys:
        value = payload.get(key)
        if is_missing_value(value, treat_blank_string=True, treat_null_like_strings=True):
            continue
        return True
    return False


def _schema_indicates_error(keys: set[str], *, has_identity: bool) -> bool:
    if not keys:
        return True
    if keys & EXPLICIT_ERROR_KEYS:
        return True
    if has_identity:
        return False
    return bool(keys & STATUS_KEYS) and bool(keys & ERROR_TEXT_KEYS)


def looks_like_error_payload(payload: Mapping[str, Any], *, identity_keys: tuple[str, ...] = ("id", "name")) -> bool:
    """Return True when object keys match an API-error-like shape."""
    keys = normalized_payload_keys(payload)
    has_identity = has_identity_value(payload, identity_keys)
    return _schema_indicates_error(keys, has_identity=has_identity)


def is_dataview_error_payload(payload: Mapping[str, Any]) -> bool:
    """Return True when getDataView payload appears to be an API error object."""
    return looks_like_error_payload(payload, identity_keys=("id", "name"))


def extract_component_sequence(payload: Mapping[str, Any]) -> list[Any] | None:
    """Extract list-like component rows from common API envelope shapes."""
    for key in COMPONENT_SEQUENCE_KEYS:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            return []
        if isinstance(value, pd.DataFrame):
            return value.to_dict("records")
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, Mapping):
            nested = extract_component_sequence(value)
            if nested is not None:
                return nested
        return []
    return None


def looks_like_component_error_payload(payload: Mapping[str, Any]) -> bool:
    """Return True when a component row payload matches an API error shape."""
    return looks_like_error_payload(payload, identity_keys=("id", "name"))


def _empty_dataframe_schema_is_error(frame: pd.DataFrame) -> bool:
    return _schema_indicates_error({str(column).strip().casefold() for column in frame.columns}, has_identity=False)


def assess_component_payload(raw_payload: Any) -> PayloadAssessment:
    """Classify mixed component payload shapes into DATA/EMPTY/ERROR/INVALID."""
    raw_type = type(raw_payload).__name__

    if raw_payload is None:
        return PayloadAssessment(PayloadKind.EMPTY, [], "none_payload", raw_type)

    if isinstance(raw_payload, pd.DataFrame):
        if raw_payload.empty:
            if len(raw_payload.columns) == 0:
                return PayloadAssessment(PayloadKind.EMPTY, [], "empty_dataframe_no_columns", raw_type)
            if _empty_dataframe_schema_is_error(raw_payload):
                return PayloadAssessment(PayloadKind.ERROR, [], "empty_dataframe_error_schema", raw_type)
            return PayloadAssessment(PayloadKind.EMPTY, [], "empty_dataframe_rows", raw_type)
        return assess_component_payload(raw_payload.to_dict("records"))

    if isinstance(raw_payload, Mapping):
        if normalized_payload_keys(raw_payload) & EXPLICIT_ERROR_KEYS:
            return PayloadAssessment(PayloadKind.ERROR, [], "object_explicit_error_shape", raw_type)
        extracted_rows = extract_component_sequence(raw_payload)
        if extracted_rows is not None:
            return assess_component_payload(extracted_rows)
        if looks_like_component_error_payload(raw_payload):
            return PayloadAssessment(PayloadKind.ERROR, [], "object_error_shape", raw_type)
        return PayloadAssessment(PayloadKind.DATA, [dict(raw_payload)], "single_object_row", raw_type)

    if isinstance(raw_payload, (list, tuple, set)):
        rows = list(raw_payload)
        if not rows:
            return PayloadAssessment(PayloadKind.EMPTY, [], "empty_sequence", raw_type)

        dict_rows = [dict(row) for row in rows if isinstance(row, Mapping)]

        if not dict_rows:
            return PayloadAssessment(PayloadKind.INVALID, [], "non_mapping_sequence_rows", raw_type)

        if any(looks_like_component_error_payload(row) for row in dict_rows):
            return PayloadAssessment(PayloadKind.ERROR, [], "error_row_detected", raw_type)

        return PayloadAssessment(PayloadKind.DATA, dict_rows, "mapping_rows", raw_type)

    return PayloadAssessment(PayloadKind.INVALID, [], "unsupported_payload_type", raw_type)


def coerce_component_rows_or_none(raw_payload: Any) -> list[dict[str, Any]] | None:
    """Normalize component payloads into row dicts, or None when invalid/error."""
    assessment = assess_component_payload(raw_payload)
    if assessment.kind in {PayloadKind.DATA, PayloadKind.EMPTY}:
        return assessment.rows
    return None


def is_component_error_payload(raw_payload: Any) -> bool:
    """Return True when a component payload is assessed as an API error."""
    return assess_component_payload(raw_payload).kind is PayloadKind.ERROR


def count_component_items_or_na(raw_payload: Any) -> int | str:
    """Return component count or 'N/A' when payload is unavailable/invalid."""
    assessment = assess_component_payload(raw_payload)
    if assessment.kind in {PayloadKind.ERROR, PayloadKind.INVALID}:
        return "N/A"
    return len(assessment.rows)
