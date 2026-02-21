"""Tests for discovery payload classification contracts."""

import pandas as pd

from cja_auto_sdr.core.discovery_payloads import (
    PayloadKind,
    assess_component_payload,
    count_component_items_or_na,
    is_dataview_error_payload,
)


def test_assess_component_empty_typed_dataframe_is_empty() -> None:
    frame = pd.DataFrame(columns=["id", "name", "type"])
    assessment = assess_component_payload(frame)
    assert assessment.kind is PayloadKind.EMPTY
    assert assessment.rows == []


def test_assess_component_empty_error_dataframe_is_error() -> None:
    frame = pd.DataFrame(columns=["statusCode", "message"])
    assessment = assess_component_payload(frame)
    assert assessment.kind is PayloadKind.ERROR


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
