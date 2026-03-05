"""Tests for discovery exception classification contracts."""

from cja_auto_sdr.core.discovery_exceptions import is_dataview_lookup_not_found_error
from cja_auto_sdr.core.exceptions import APIError


class _RuntimeLookupError(RuntimeError):
    """Custom runtime error used to simulate non-whitelisted cjapy wrappers."""


def test_non_api_error_with_status_code_404_maps_to_not_found() -> None:
    error = _RuntimeLookupError("wrapped failure")
    error.status_code = 404  # type: ignore[attr-defined]

    assert is_dataview_lookup_not_found_error(error) is True


def test_non_api_error_with_nested_response_status_403_maps_to_not_found() -> None:
    error = _RuntimeLookupError("wrapped failure")
    error.response = {"error": {"statusCode": "403"}}  # type: ignore[attr-defined]

    assert is_dataview_lookup_not_found_error(error) is True


def test_os_error_wrapper_with_404_metadata_maps_to_not_found() -> None:
    error = OSError("wrapped failure")
    error.response = {"error": {"statusCode": "404"}}  # type: ignore[attr-defined]

    assert is_dataview_lookup_not_found_error(error) is True


def test_non_api_error_5xx_status_remains_non_not_found() -> None:
    error = _RuntimeLookupError("backend unavailable")
    error.status_code = 503  # type: ignore[attr-defined]

    assert is_dataview_lookup_not_found_error(error) is False


def test_api_error_message_markers_map_to_not_found_without_status_code() -> None:
    error = APIError("resource_not_found while resolving data view")

    assert is_dataview_lookup_not_found_error(error) is True


def test_non_api_error_message_marker_does_not_map_without_status_signal() -> None:
    error = _RuntimeLookupError("resource_not_found while resolving data view")

    assert is_dataview_lookup_not_found_error(error) is False
