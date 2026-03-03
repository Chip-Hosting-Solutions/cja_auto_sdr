"""Discovery exception classification for data-view inspection commands."""

from __future__ import annotations

import contextlib
import re
from typing import Any

from cja_auto_sdr.core.exceptions import APIError

_DATAVIEW_LOOKUP_NOT_FOUND_STATUS_CODES = frozenset({403, 404})
_DATAVIEW_LOOKUP_STATUS_ATTRS = ("status_code", "statusCode", "status", "http_status", "httpStatus", "code")
_DATAVIEW_LOOKUP_NOT_FOUND_MESSAGE_MARKERS = (
    "not found",
    "not_found",
    "resource_not_found",
    "forbidden",
    "no access",
    "access denied",
)
_HTTP_STATUS_CODE_RE = re.compile(r"\b([1-5]\d{2})\b")


def coerce_http_status_code(value: Any) -> int | None:
    """Best-effort coercion of status-like values into an HTTP status code."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 100 <= value <= 599 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            numeric_code = int(stripped)
            return numeric_code if 100 <= numeric_code <= 599 else None
        match = _HTTP_STATUS_CODE_RE.search(stripped)
        if match:
            numeric_code = int(match.group(1))
            return numeric_code if 100 <= numeric_code <= 599 else None
    return None


def iter_error_chain_nodes(error: Exception) -> list[Any]:
    """Return error/cause/response nodes for robust status-code extraction."""
    pending: list[Any] = [error]
    seen: set[int] = set()
    nodes: list[Any] = []
    while pending:
        node = pending.pop()
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        nodes.append(node)

        if isinstance(node, dict):
            nested = node.get("error")
            if nested is not None:
                pending.append(nested)
            continue

        for attr_name in ("original_error", "__cause__", "__context__", "response"):
            with contextlib.suppress(Exception):
                nested = getattr(node, attr_name)
                if nested is not None:
                    pending.append(nested)
    return nodes


def extract_http_status_codes(error: Exception) -> set[int]:
    """Extract candidate HTTP status codes from nested error objects."""
    status_codes: set[int] = set()
    for node in iter_error_chain_nodes(error):
        if isinstance(node, dict):
            for key in _DATAVIEW_LOOKUP_STATUS_ATTRS:
                code = coerce_http_status_code(node.get(key))
                if code is not None:
                    status_codes.add(code)
            continue

        for attr_name in _DATAVIEW_LOOKUP_STATUS_ATTRS:
            with contextlib.suppress(Exception):
                code = coerce_http_status_code(getattr(node, attr_name))
                if code is not None:
                    status_codes.add(code)
    return status_codes


def is_dataview_lookup_not_found_error(error: Exception) -> bool:
    """Return True when getDataView failures should map to not_found."""
    status_codes = extract_http_status_codes(error)
    if any(code in _DATAVIEW_LOOKUP_NOT_FOUND_STATUS_CODES for code in status_codes):
        return True

    # Some APIError paths omit status_code but still include a stable
    # not-found/forbidden marker in the message body.
    if isinstance(error, APIError):
        normalized_message = str(error).casefold()
        return any(marker in normalized_message for marker in _DATAVIEW_LOOKUP_NOT_FOUND_MESSAGE_MARKERS)

    return False
