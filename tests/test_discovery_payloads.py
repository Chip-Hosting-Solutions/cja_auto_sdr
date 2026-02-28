"""Tests for discovery payload classification contracts."""

import pandas as pd

from cja_auto_sdr.core.discovery_payloads import (
    PayloadKind,
    assess_component_payload,
    assess_dataview_lookup_payload,
    coerce_component_rows_or_none,
    count_component_items_or_na,
    extract_component_sequence,
    is_component_error_payload,
    is_dataview_error_payload,
    looks_like_error_payload,
)


def test_assess_component_empty_typed_dataframe_is_empty() -> None:
    frame = pd.DataFrame(columns=["id", "name", "type"])
    assessment = assess_component_payload(frame)
    assert assessment.kind is PayloadKind.EMPTY
    assert assessment.rows == []


def test_assess_component_empty_dataframe_without_columns_is_empty() -> None:
    frame = pd.DataFrame()
    assessment = assess_component_payload(frame)
    assert assessment.kind is PayloadKind.EMPTY
    assert assessment.reason == "empty_dataframe_no_columns"
    assert assessment.rows == []


def test_assess_component_empty_error_dataframe_is_error() -> None:
    frame = pd.DataFrame(columns=["statusCode", "message"])
    assessment = assess_component_payload(frame)
    assert assessment.kind is PayloadKind.ERROR


def test_assess_component_non_empty_dataframe_recurses_to_records() -> None:
    frame = pd.DataFrame([{"id": "metrics/revenue", "name": "Revenue"}])
    assessment = assess_component_payload(frame)
    assert assessment.kind is PayloadKind.DATA
    assert assessment.reason == "mapping_rows"
    assert assessment.rows == [{"id": "metrics/revenue", "name": "Revenue"}]


def test_assess_component_sequence_payload_prefers_embedded_rows() -> None:
    payload = {"statusCode": 200, "message": "ok", "data": []}
    assessment = assess_component_payload(payload)
    assert assessment.kind is PayloadKind.EMPTY
    assert assessment.rows == []


def test_assess_component_explicit_error_wins_over_envelope_rows() -> None:
    payload = {"errorCode": "forbidden", "data": []}
    assessment = assess_component_payload(payload)
    assert assessment.kind is PayloadKind.ERROR


def test_assess_component_regular_rows_are_data() -> None:
    payload = [
        {"id": "metrics/revenue", "name": "Revenue", "type": "currency"},
        {"id": "metrics/pageviews", "name": "Page Views", "type": "decimal"},
    ]
    assessment = assess_component_payload(payload)
    assert assessment.kind is PayloadKind.DATA
    assert len(assessment.rows) == 2


def test_count_component_items_or_na_for_empty_typed_dataframe() -> None:
    frame = pd.DataFrame(columns=["id", "name", "type"])
    assert count_component_items_or_na(frame) == 0


def test_count_component_items_or_na_for_error_payload() -> None:
    payload = {"statusCode": 500, "message": "backend timeout"}
    assert count_component_items_or_na(payload) == "N/A"


def test_dataview_payload_error_shape_detected() -> None:
    assert is_dataview_error_payload({"statusCode": 404, "errorCode": "not_found"}) is True


def test_dataview_payload_with_identity_not_error() -> None:
    payload = {"id": "dv_1", "name": "Test View", "status": "active"}
    assert is_dataview_error_payload(payload) is False


def test_dataview_payload_with_na_identity_values_is_treated_as_error() -> None:
    payload = {"statusCode": 404, "message": "missing", "id": pd.NA, "name": pd.NA}
    assert is_dataview_error_payload(payload) is True


def test_dataview_payload_with_na_id_and_present_name_is_not_error() -> None:
    payload = {"statusCode": 200, "message": "ok", "id": pd.NA, "name": "Test View"}
    assert is_dataview_error_payload(payload) is False


def test_dataview_payload_with_identity_and_error_field_is_not_error() -> None:
    payload = {"id": "dv_1", "name": "Test View", "error": "non-fatal detail"}
    assert is_dataview_error_payload(payload) is False


def test_assess_dataview_lookup_payload_accepts_valid_payload() -> None:
    payload = {"id": "dv_1", "name": "Valid View", "owner": {"name": "Owner"}}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.DATA
    assert assessment.is_valid is True


def test_assess_dataview_lookup_payload_accepts_valid_payload_with_error_field() -> None:
    payload = {"id": "dv_1", "name": "Valid View", "error": "temporary warning"}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.DATA
    assert assessment.reason == "valid_lookup_payload"
    assert assessment.is_valid is True


def test_assess_dataview_lookup_payload_rejects_explicit_lookup_failed_marker() -> None:
    payload = {
        "id": "dv_1",
        "name": "Unknown",
        "lookup_failed": True,
        "lookup_failure_reason": "exception",
        "error": "network timeout",
    }
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "failure_flag:lookup_failed"


def test_assess_dataview_lookup_payload_rejects_circuit_breaker_marker() -> None:
    payload = {"id": "dv_1", "name": "Unknown", "circuit_breaker_open": "true"}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "failure_flag:circuit_breaker_open"


def test_assess_dataview_lookup_payload_rejects_lookup_failure_reason_detail() -> None:
    payload = {"id": "dv_1", "name": "Unknown", "lookup_failure_reason": "exception"}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "failure_detail:lookup_failure_reason"


def test_assess_dataview_lookup_payload_rejects_legacy_unknown_placeholder() -> None:
    payload = {"id": "dv_1", "name": "Unknown"}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "legacy_unknown_placeholder"


def test_assess_dataview_lookup_payload_rejects_id_mismatch() -> None:
    payload = {"id": "dv_other", "name": "Valid View"}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "id_mismatch"


def test_assess_dataview_lookup_payload_handles_non_stringifiable_name_value() -> None:
    class _ExplodingName:
        def __str__(self) -> str:  # pragma: no cover - explicit failure path
            raise RuntimeError("boom")

    payload = {"id": "dv_1", "name": _ExplodingName()}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.DATA
    assert assessment.is_valid is True


def test_assess_dataview_lookup_payload_rejects_invalid_id_type() -> None:
    class _ExplodingId:
        def __str__(self) -> str:  # pragma: no cover - explicit failure path
            raise RuntimeError("boom")

    payload = {"id": _ExplodingId(), "name": "Valid View"}
    assessment = assess_dataview_lookup_payload(payload, expected_data_view_id="dv_1")
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "invalid_id_type"


# ---------------------------------------------------------------------------
# looks_like_error_payload — empty dict (line 56)
# ---------------------------------------------------------------------------


def test_empty_dict_looks_like_error() -> None:
    """Empty keys → _schema_indicates_error returns True."""
    assert looks_like_error_payload({}) is True


# ---------------------------------------------------------------------------
# extract_component_sequence branches (lines 83, 85, 88-92)
# ---------------------------------------------------------------------------


def test_extract_component_sequence_none_value_returns_empty() -> None:
    """Component key present but value is None → empty list."""
    assert extract_component_sequence({"data": None}) == []


def test_extract_component_sequence_dataframe_value() -> None:
    """Component key holds a DataFrame → converted to records."""
    df = pd.DataFrame({"id": ["m1", "m2"], "name": ["Revenue", "Views"]})
    result = extract_component_sequence({"data": df})
    assert result == [{"id": "m1", "name": "Revenue"}, {"id": "m2", "name": "Views"}]


def test_extract_component_sequence_nested_mapping() -> None:
    """Component key holds a nested dict with its own component key → recurse."""
    payload = {"results": {"data": [{"id": "1"}]}}
    assert extract_component_sequence(payload) == [{"id": "1"}]


def test_extract_component_sequence_nested_mapping_no_match() -> None:
    """Nested mapping with no component keys → empty list."""
    payload = {"items": {"randomKey": "value"}}
    assert extract_component_sequence(payload) == []


# ---------------------------------------------------------------------------
# assess_component_payload branches (lines 110, 129, 139, 142)
# ---------------------------------------------------------------------------


def test_assess_none_payload_is_empty() -> None:
    """None input → EMPTY with reason 'none_payload'."""
    assessment = assess_component_payload(None)
    assert assessment.kind is PayloadKind.EMPTY
    assert assessment.reason == "none_payload"


def test_assess_single_object_row() -> None:
    """Mapping without error/sequence keys and with identity → DATA single row."""
    payload = {"id": "obj_1", "name": "Object", "description": "Details"}
    assessment = assess_component_payload(payload)
    assert assessment.kind is PayloadKind.DATA
    assert assessment.reason == "single_object_row"
    assert assessment.rows == [payload]


def test_assess_non_mapping_sequence_is_invalid() -> None:
    """List of non-Mapping items → INVALID."""
    assessment = assess_component_payload([1, 2, 3])
    assert assessment.kind is PayloadKind.INVALID
    assert assessment.reason == "non_mapping_sequence_rows"


def test_assess_unsupported_payload_type_is_invalid() -> None:
    assessment = assess_component_payload(42)
    assert assessment.kind is PayloadKind.INVALID
    assert assessment.reason == "unsupported_payload_type"


def test_assess_error_row_in_sequence() -> None:
    """List containing an error-shaped dict → ERROR 'error_row_detected'."""
    payload = [
        {"id": "obj_1", "name": "Valid"},
        {"statusCode": 500, "message": "Server error"},
    ]
    assessment = assess_component_payload(payload)
    assert assessment.kind is PayloadKind.ERROR
    assert assessment.reason == "error_row_detected"


# ---------------------------------------------------------------------------
# coerce_component_rows_or_none (lines 151-154)
# ---------------------------------------------------------------------------


def test_coerce_data_payload_returns_rows() -> None:
    result = coerce_component_rows_or_none([{"id": "1"}])
    assert result == [{"id": "1"}]


def test_coerce_empty_payload_returns_empty_list() -> None:
    result = coerce_component_rows_or_none([])
    assert result == []


def test_coerce_error_payload_returns_none() -> None:
    result = coerce_component_rows_or_none({"statusCode": 500, "message": "fail"})
    assert result is None


def test_coerce_invalid_payload_returns_none() -> None:
    result = coerce_component_rows_or_none([1, 2, 3])
    assert result is None


# ---------------------------------------------------------------------------
# is_component_error_payload (line 159)
# ---------------------------------------------------------------------------


def test_is_component_error_payload_true() -> None:
    assert is_component_error_payload({"statusCode": 500, "message": "Error"}) is True


def test_is_component_error_payload_false() -> None:
    assert is_component_error_payload([{"id": "1", "name": "OK"}]) is False
