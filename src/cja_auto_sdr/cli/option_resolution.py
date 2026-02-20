"""Shared helpers for argparse-compatible long-option token resolution."""

from __future__ import annotations

from typing import NamedTuple


class LongOptionResolution(NamedTuple):
    """Resolution outcome for a long-option token."""

    canonical_option: str | None
    is_ambiguous: bool


def resolve_long_option_token(option_text: str, known_long_options: frozenset[str]) -> LongOptionResolution:
    """Resolve a token to a canonical long option if argparse would accept it.

    Returns ``is_ambiguous=True`` when an abbreviation matches multiple options.
    """
    option_name = option_text.split("=", 1)[0]
    if not option_name.startswith("--") or option_name == "--":
        return LongOptionResolution(canonical_option=None, is_ambiguous=False)
    if option_name in known_long_options:
        return LongOptionResolution(canonical_option=option_name, is_ambiguous=False)

    matches = [option for option in known_long_options if option.startswith(option_name)]
    if len(matches) == 1:
        return LongOptionResolution(canonical_option=matches[0], is_ambiguous=False)
    if len(matches) > 1:
        return LongOptionResolution(canonical_option=None, is_ambiguous=True)
    return LongOptionResolution(canonical_option=None, is_ambiguous=False)
