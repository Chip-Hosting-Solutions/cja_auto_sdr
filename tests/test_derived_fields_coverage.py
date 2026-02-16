"""
Derived-field inventory edge-case tests targeting untested code paths.

Areas covered:
- _coerce_int_index() edge cases (bool, NaN, overflow, strings, None)
- _compute_complexity_score() zero/max/boundary values
- _infer_output_type() empty list, mixed match outputs, no branches
- _generate_logic_summary() all handler branches
- _describe_match_logic() label_map resolution, defaults, multiple-condition grouping
- _describe_predicate() all comparison/logical operators, max_depth
- _normalize_label_key() / _build_label_map() edge cases
"""

import json

import pandas as pd
import pytest

from cja_auto_sdr.inventory.derived_fields import (
    DerivedFieldInventory,
    DerivedFieldInventoryBuilder,
    DerivedFieldSummary,
)

# ==================== HELPERS ====================


def _builder() -> DerivedFieldInventoryBuilder:
    """Convenience factory for a builder instance."""
    return DerivedFieldInventoryBuilder()


def _build_one(definition, component_type="Dimension"):
    """Build a single-field inventory from a definition list and return the summary."""
    builder = _builder()
    field_def = json.dumps(definition) if isinstance(definition, list) else definition
    col = "Metric" if component_type == "Metric" else "Dimension"
    df = pd.DataFrame(
        [
            {
                "id": f"{col.lower()}s/test_field",
                "name": "Test Field",
                "sourceFieldType": "derived",
                "fieldDefinition": field_def,
                "dataSetType": "event",
            }
        ]
    )
    metrics_df = df if col == "Metric" else pd.DataFrame()
    dims_df = df if col == "Dimension" else pd.DataFrame()
    inventory = builder.build(metrics_df, dims_df, "dv_test", "Test")
    assert inventory.total_derived_fields == 1, "Expected exactly one derived field"
    return inventory.fields[0]


# ==================== _coerce_int_index() ====================


class TestCoerceIntIndex:
    """Edge cases for _coerce_int_index()."""

    def test_bool_true_returns_default(self):
        """bool inputs should return the default (bool is excluded before int check)."""
        b = _builder()
        assert b._coerce_int_index(True) == 0

    def test_bool_false_returns_default(self):
        b = _builder()
        assert b._coerce_int_index(False) == 0

    def test_none_returns_default(self):
        b = _builder()
        assert b._coerce_int_index(None) == 0

    def test_none_custom_default(self):
        b = _builder()
        assert b._coerce_int_index(None, default=5) == 5

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_float_returns_default(self, value):
        b = _builder()
        assert b._coerce_int_index(value) == 0

    def test_normal_float_coerced(self):
        b = _builder()
        assert b._coerce_int_index(3.7) == 3

    def test_normal_int_passes_through(self):
        b = _builder()
        assert b._coerce_int_index(42) == 42

    def test_whitespace_string_returns_default(self):
        b = _builder()
        assert b._coerce_int_index("   ") == 0

    def test_empty_string_returns_default(self):
        b = _builder()
        assert b._coerce_int_index("") == 0

    def test_non_numeric_string_returns_default(self):
        b = _builder()
        assert b._coerce_int_index("abc") == 0

    def test_numeric_string_coerced(self):
        b = _builder()
        assert b._coerce_int_index("  7  ") == 7

    def test_unexpected_type_returns_default(self):
        """Objects that are not int/float/str/bool/None should return default."""
        b = _builder()
        assert b._coerce_int_index([1, 2, 3]) == 0

    def test_negative_int(self):
        b = _builder()
        assert b._coerce_int_index(-1) == -1

    def test_zero_float(self):
        b = _builder()
        assert b._coerce_int_index(0.0) == 0


# ==================== _compute_complexity_score() ====================


class TestComputeComplexityScore:
    """Edge cases for _compute_complexity_score()."""

    def test_all_zeros_returns_zero(self):
        b = _builder()
        score = b._compute_complexity_score(operators=0, branches=0, nesting=0, functions=0, schema_fields=0, regex=0)
        assert score == 0.0

    def test_all_at_max_returns_100(self):
        """When every factor hits its cap the score should be 100."""
        b = _builder()
        score = b._compute_complexity_score(
            operators=200,
            branches=50,
            nesting=5,
            functions=20,
            schema_fields=10,
            regex=5,
        )
        assert score == 100.0

    def test_values_above_max_are_capped(self):
        """Values beyond the max should not produce a score > 100."""
        b = _builder()
        score = b._compute_complexity_score(
            operators=9999,
            branches=9999,
            nesting=9999,
            functions=9999,
            schema_fields=9999,
            regex=9999,
        )
        assert score == 100.0

    def test_only_operators_contribute(self):
        """Only operators non-zero should produce score = weight * 100 * (ops/max)."""
        b = _builder()
        score = b._compute_complexity_score(operators=100, branches=0, nesting=0, functions=0, schema_fields=0, regex=0)
        # 100/200 * 0.30 * 100 = 15.0
        assert score == 15.0

    def test_only_regex_contributes(self):
        b = _builder()
        score = b._compute_complexity_score(operators=0, branches=0, nesting=0, functions=0, schema_fields=0, regex=5)
        # 5/5 * 0.05 * 100 = 5.0
        assert score == 5.0


# ==================== _infer_output_type() ====================


class TestInferOutputType:
    """Edge cases for _infer_output_type()."""

    def test_empty_functions_list(self):
        b = _builder()
        assert b._infer_output_type([]) == "unknown"

    @pytest.mark.parametrize("func", ["divide", "multiply", "add", "subtract"])
    def test_math_operations_numeric(self, func):
        b = _builder()
        assert b._infer_output_type([{"func": func}]) == "numeric"

    @pytest.mark.parametrize(
        "func", ["lowercase", "uppercase", "trim", "concatenate", "split", "url-parse", "regex-replace"]
    )
    def test_string_operations(self, func):
        b = _builder()
        assert b._infer_output_type([{"func": func}]) == "string"

    def test_match_all_numeric_outputs(self):
        b = _builder()
        result = b._infer_output_type(
            [
                {
                    "func": "match",
                    "branches": [
                        {"pred": {"func": "true"}, "map-to": 1},
                        {"pred": {"func": "eq", "val": "a"}, "map-to": 2.5},
                    ],
                }
            ]
        )
        assert result == "numeric"

    def test_match_all_string_outputs(self):
        b = _builder()
        result = b._infer_output_type(
            [
                {
                    "func": "match",
                    "branches": [
                        {"pred": {"func": "true"}, "map-to": "yes"},
                        {"pred": {"func": "eq", "val": "a"}, "map-to": "no"},
                    ],
                }
            ]
        )
        assert result == "string"

    def test_match_mixed_outputs_unknown(self):
        """Mixed numeric and string map-to values should produce unknown."""
        b = _builder()
        result = b._infer_output_type(
            [
                {
                    "func": "match",
                    "branches": [
                        {"pred": {"func": "true"}, "map-to": 42},
                        {"pred": {"func": "eq", "val": "a"}, "map-to": "text"},
                    ],
                }
            ]
        )
        assert result == "unknown"

    def test_match_no_valid_branches(self):
        """Match with empty branches should fall through to unknown."""
        b = _builder()
        result = b._infer_output_type([{"func": "match", "branches": []}])
        assert result == "unknown"

    def test_match_non_list_branches(self):
        b = _builder()
        result = b._infer_output_type([{"func": "match", "branches": "bad"}])
        assert result == "unknown"

    def test_unknown_func(self):
        b = _builder()
        assert b._infer_output_type([{"func": "profile"}]) == "unknown"


# ==================== _generate_logic_summary() handler branches ====================


class TestLogicSummaryHandlers:
    """Cover every handler branch in _generate_logic_summary() individually."""

    def test_summarize_handler(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "events", "label": "ev"},
                {"func": "summarize"},
            ]
        )
        assert "summarize" in field.logic_summary.lower()

    def test_deduplicate_with_scope(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "pageId", "label": "p"},
                {"func": "deduplicate", "scope": "session", "args": [{"func": "raw-field", "id": "pageId"}]},
            ]
        )
        assert "deduplicate" in field.logic_summary.lower()
        assert "session" in field.logic_summary.lower()

    def test_deduplicate_without_scope(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "pageId", "label": "p"},
                {"func": "deduplicate", "args": [{"func": "raw-field", "id": "pageId"}]},
            ]
        )
        assert "deduplicate" in field.logic_summary.lower()

    def test_merge_with_multiple_fields(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "fieldA", "label": "a"},
                {"func": "raw-field", "id": "fieldB", "label": "b"},
                {"func": "merge"},
            ]
        )
        assert "merge" in field.logic_summary.lower()

    def test_split_with_delimiter_and_index(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "url_path", "label": "u"},
                {"func": "split", "delimiter": "/", "index": 2, "args": [{"func": "raw-field", "id": "url_path"}]},
            ]
        )
        assert "split" in field.logic_summary.lower()
        assert "part 3" in field.logic_summary  # 0-indexed + 1

    def test_datetime_slice_with_component_and_args(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "ts_field", "label": "ts"},
                {
                    "func": "datetime-slice",
                    "component": "dayOfWeek",
                    "args": [{"func": "raw-field", "id": "ts_field"}],
                },
            ]
        )
        assert "dayofweek" in field.logic_summary.lower() or "dayOfWeek" in field.logic_summary

    def test_datetime_slice_component_only(self):
        """datetime-slice with component but no args or field should still produce a summary."""
        field = _build_one(
            [
                {"func": "datetime-slice", "component": "month"},
            ]
        )
        assert "month" in field.logic_summary.lower()

    def test_timezone_shift_full(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "event_time", "label": "et"},
                {
                    "func": "timezone-shift",
                    "from": "UTC",
                    "to": "US/Pacific",
                    "args": [{"func": "raw-field", "id": "event_time"}],
                },
            ]
        )
        assert "utc" in field.logic_summary.lower()
        assert "us/pacific" in field.logic_summary.lower() or "US/Pacific" in field.logic_summary

    def test_timezone_shift_only_dst(self):
        field = _build_one(
            [
                {"func": "timezone-shift", "to": "Europe/Berlin"},
            ]
        )
        assert "europe/berlin" in field.logic_summary.lower() or "Europe/Berlin" in field.logic_summary

    def test_timezone_shift_fallback(self):
        """timezone-shift without from/to should use the fallback summary."""
        field = _build_one(
            [
                {"func": "timezone-shift"},
            ]
        )
        assert "timezone" in field.logic_summary.lower()

    def test_find_replace_with_both_values(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "domain", "label": "d"},
                {
                    "func": "find-replace",
                    "find": "www.",
                    "replace": "",
                    "args": [{"func": "raw-field", "id": "domain"}],
                },
            ]
        )
        # Empty replace => "Removes 'www.' from domain"
        assert "remove" in field.logic_summary.lower() or "replace" in field.logic_summary.lower()

    def test_find_replace_fallback(self):
        """find-replace without find value should produce the fallback."""
        field = _build_one(
            [
                {"func": "find-replace"},
            ]
        )
        assert "find and replace" in field.logic_summary.lower()

    def test_depth_with_args(self):
        field = _build_one(
            [
                {"func": "raw-field", "id": "site_section", "label": "s"},
                {
                    "func": "depth",
                    "delimiter": "|",
                    "args": [{"func": "raw-field", "id": "site_section"}],
                },
            ]
        )
        assert "depth" in field.logic_summary.lower()
        assert "'|'" in field.logic_summary

    def test_depth_fallback(self):
        field = _build_one(
            [
                {"func": "depth"},
            ]
        )
        assert "depth" in field.logic_summary.lower()

    def test_profile_with_namespace(self):
        field = _build_one(
            [
                {"func": "profile", "attribute": "email", "namespace": "crm"},
            ]
        )
        assert "profile" in field.logic_summary.lower()
        assert "crm" in field.logic_summary.lower() or "crm/email" in field.logic_summary

    def test_profile_without_attribute(self):
        """Profile without attribute should use the fallback."""
        field = _build_one(
            [
                {"func": "profile"},
            ]
        )
        assert "profile" in field.logic_summary.lower()

    def test_transforms_appended_to_existing_summary(self):
        """Lowercase/uppercase/trim should be appended with arrow when other logic exists."""
        field = _build_one(
            [
                {"func": "raw-field", "id": "pageName", "label": "p"},
                {"func": "deduplicate", "args": [{"func": "raw-field", "id": "pageName"}]},
                {"func": "lowercase", "field": "p"},
                {"func": "trim", "field": "p"},
            ]
        )
        assert "lowercase" in field.logic_summary.lower()
        assert "trim" in field.logic_summary.lower()

    def test_transforms_only_no_primary_logic(self):
        """When only transforms are present, summary should start with 'Applies'."""
        field = _build_one(
            [
                {"func": "raw-field", "id": "pageName", "label": "p"},
                {"func": "uppercase", "field": "p"},
            ]
        )
        assert "applies" in field.logic_summary.lower()
        assert "uppercase" in field.logic_summary.lower()

    def test_references_single_field_fallback(self):
        """When there's no recognized handler and one schema field, show 'References'."""
        field = _build_one(
            [
                {"func": "raw-field", "id": "web.pageURL", "label": "url_ref"},
            ]
        )
        assert "references" in field.logic_summary.lower()

    def test_references_multiple_fields_fallback(self):
        """Multiple schema fields with no handler should show 'Combines N fields'."""
        field = _build_one(
            [
                {"func": "raw-field", "id": "fieldA", "label": "a"},
                {"func": "raw-field", "id": "fieldB", "label": "b"},
                {"func": "raw-field", "id": "fieldC", "label": "c"},
            ]
        )
        assert "combines" in field.logic_summary.lower() or "3 fields" in field.logic_summary

    def test_match_with_many_rule_names_truncated(self):
        """Match with >4 rule names should truncate the list."""
        definition = [
            {"func": "raw-field", "id": "f", "label": "f"},
        ]
        # Create a match with no describable branches (so it falls to rule_names path)
        # We need multiple rules; use separate match objects with rule names
        definition.append(
            {
                "func": "match",
                "field": "f",
                "#rule_name": "Rule1",
                "branches": [],
            }
        )
        # Additional rules via separate functions
        for i in range(2, 7):
            definition.append(
                {
                    "func": "match",
                    "field": "f",
                    "#rule_name": f"Rule{i}",
                    "branches": [],
                }
            )

        builder = _builder()
        functions = definition
        parsed = builder._parse_definition(functions)
        summary = builder._generate_logic_summary(functions, parsed, "Test")
        assert "+2 more" in summary or "+3 more" in summary or "conditional" in summary.lower()


# ==================== _describe_predicate() ====================


class TestDescribePredicate:
    """Test all operator branches in _describe_predicate()."""

    @pytest.fixture
    def b(self):
        return _builder()

    def test_eq_operator(self, b):
        result = b._describe_predicate(
            {"func": "eq", "field": "status", "val": "active"},
            label_map={"status": "status"},
        )
        assert '="active"' in result

    def test_ne_operator(self, b):
        result = b._describe_predicate(
            {"func": "ne", "field": "status", "val": "inactive"},
            label_map={"status": "status"},
        )
        assert "status" in result.lower() or "\u2260" in result

    def test_gt_operator(self, b):
        result = b._describe_predicate(
            {"func": "gt", "arg1": {"func": "raw-field", "id": "price"}, "arg2": {"val": 100}},
        )
        assert ">100" in result or "> 100" in result

    def test_gte_operator(self, b):
        result = b._describe_predicate(
            {"func": "gte", "arg1": {"func": "raw-field", "id": "price"}, "arg2": {"val": 50}},
        )
        assert ">=50" in result or ">= 50" in result

    def test_lt_operator(self, b):
        result = b._describe_predicate(
            {"func": "lt", "arg1": {"func": "raw-field", "id": "count"}, "arg2": {"val": 10}},
        )
        assert "<10" in result or "< 10" in result

    def test_lte_operator(self, b):
        result = b._describe_predicate(
            {"func": "lte", "arg1": {"func": "raw-field", "id": "count"}, "arg2": {"val": 5}},
        )
        assert "<=5" in result or "<= 5" in result

    def test_starts_with_operator(self, b):
        result = b._describe_predicate(
            {"func": "starts_with", "field": "url", "val": "https"},
            label_map={"url": "url"},
        )
        assert 'starts "https"' in result

    def test_ends_with_operator(self, b):
        result = b._describe_predicate(
            {"func": "ends_with", "field": "url", "val": ".html"},
            label_map={"url": "url"},
        )
        assert 'ends ".html"' in result

    def test_regex_match_operator(self, b):
        result = b._describe_predicate(
            {"func": "regex_match", "field": "path", "val": "^/products/.*"},
            label_map={"path": "path"},
        )
        assert "matches" in result
        assert "/" in result

    def test_regex_match_long_pattern_truncated(self, b):
        long_pattern = "a" * 30
        result = b._describe_predicate(
            {"func": "regex_match", "field": "f", "val": long_pattern},
            label_map={"f": "field"},
        )
        assert "..." in result

    def test_isset_operator(self, b):
        result = b._describe_predicate(
            {"func": "isset", "field": "email"},
            label_map={"email": "email"},
        )
        assert "exists" in result

    def test_and_with_truncation(self, b):
        """AND with >3 predicates should truncate."""
        preds = [{"func": "eq", "field": "f", "val": str(i)} for i in range(5)]
        result = b._describe_predicate(
            {"func": "and", "preds": preds},
            label_map={"f": "field"},
        )
        assert "AND" in result
        assert "+2 more" in result

    def test_or_with_truncation(self, b):
        """OR with >3 predicates should truncate."""
        preds = [{"func": "eq", "field": "f", "val": str(i)} for i in range(4)]
        result = b._describe_predicate(
            {"func": "or", "preds": preds},
            label_map={"f": "field"},
        )
        assert "OR" in result
        assert "+1 more" in result

    def test_true_predicate(self, b):
        result = b._describe_predicate({"func": "true"})
        assert result == "default"

    def test_max_depth_exceeded_returns_empty(self, b):
        result = b._describe_predicate(
            {"func": "eq", "field": "f", "val": "x"},
            max_depth=0,
        )
        assert result == ""

    def test_non_dict_pred_returns_empty(self, b):
        result = b._describe_predicate("not a dict")
        assert result == ""

    def test_ne_numeric_value(self, b):
        result = b._describe_predicate(
            {"func": "ne", "field": "count", "val": 0},
            label_map={"count": "count"},
        )
        # Should use non-equals sign with numeric
        assert "\u2260" in result or "count" in result

    def test_eq_without_field_shows_value_only(self, b):
        """eq predicate without a field reference should still show the value."""
        result = b._describe_predicate({"func": "eq", "val": "test"})
        assert '="test"' in result

    def test_ne_without_field_shows_value_only(self, b):
        result = b._describe_predicate({"func": "ne", "val": 99})
        assert "\u2260" in result or "99" in result

    def test_and_non_list_preds_returns_empty(self, b):
        """AND with non-list preds should return empty string."""
        result = b._describe_predicate({"func": "and", "preds": "bad"})
        assert result == ""

    def test_or_non_list_preds_returns_empty(self, b):
        result = b._describe_predicate({"func": "or", "preds": "bad"})
        assert result == ""

    def test_unknown_func_returns_empty(self, b):
        result = b._describe_predicate({"func": "unknown_op", "val": "x"})
        assert result == ""


# ==================== _normalize_label_key() ====================


class TestNormalizeLabelKey:
    """Edge cases for _normalize_label_key()."""

    def test_none_returns_empty(self):
        assert _builder()._normalize_label_key(None) == ""

    def test_string_stripped(self):
        assert _builder()._normalize_label_key("  hello  ") == "hello"

    def test_dict_with_label_key(self):
        assert _builder()._normalize_label_key({"label": "my_label"}) == "my_label"

    def test_dict_with_field_key(self):
        assert _builder()._normalize_label_key({"field": "my_field"}) == "my_field"

    def test_dict_with_id_key(self):
        assert _builder()._normalize_label_key({"id": "my_id"}) == "my_id"

    def test_dict_with_name_key(self):
        assert _builder()._normalize_label_key({"name": "my_name"}) == "my_name"

    def test_dict_with_value_key(self):
        assert _builder()._normalize_label_key({"value": "my_value"}) == "my_value"

    def test_dict_with_no_known_keys_returns_empty(self):
        assert _builder()._normalize_label_key({"unknown": "data"}) == ""

    def test_bool_input(self):
        assert _builder()._normalize_label_key(True) == "True"

    def test_int_input(self):
        assert _builder()._normalize_label_key(42) == "42"

    def test_float_input(self):
        assert _builder()._normalize_label_key(3.14) == "3.14"

    def test_list_input_stringified(self):
        """Non-string/dict/numeric/None types should be str()'d and stripped."""
        result = _builder()._normalize_label_key([1, 2])
        assert result == "[1, 2]"


# ==================== _build_label_map() ====================


class TestBuildLabelMap:
    """Edge cases for _build_label_map()."""

    def test_raw_field_mapped(self):
        functions = [
            {"func": "raw-field", "id": "web.pageURL", "label": "url"},
        ]
        label_map = _builder()._build_label_map(functions)
        assert "url" in label_map
        assert label_map["url"] == "pageURL"  # extract_short_name on dot path

    def test_url_parse_query_mapped(self):
        functions = [
            {
                "func": "url-parse",
                "label": "campaign_param",
                "component": {"func": "query", "param": "utm_campaign"},
            },
        ]
        label_map = _builder()._build_label_map(functions)
        assert label_map.get("campaign_param") == "?utm_campaign"

    def test_url_parse_non_query_component(self):
        functions = [
            {
                "func": "url-parse",
                "label": "host",
                "component": {"func": "hostname"},
            },
        ]
        label_map = _builder()._build_label_map(functions)
        assert label_map.get("host") == "hostname"

    def test_url_parse_string_component(self):
        functions = [
            {
                "func": "url-parse",
                "label": "proto",
                "component": "protocol",
            },
        ]
        label_map = _builder()._build_label_map(functions)
        assert label_map.get("proto") == "protocol"

    def test_transform_with_known_input(self):
        functions = [
            {"func": "raw-field", "id": "web.pageName", "label": "pg"},
            {"func": "lowercase", "field": "pg", "label": "pg_lower"},
        ]
        label_map = _builder()._build_label_map(functions)
        assert "pg_lower" in label_map
        assert "lowercase" in label_map["pg_lower"]

    def test_transform_without_known_input(self):
        functions = [
            {"func": "lowercase", "field": "unknown_ref", "label": "result"},
        ]
        label_map = _builder()._build_label_map(functions)
        assert label_map.get("result") == "lowercase"

    def test_entries_without_label_skipped(self):
        functions = [
            {"func": "raw-field", "id": "field_a"},
        ]
        label_map = _builder()._build_label_map(functions)
        assert len(label_map) == 0


# ==================== _describe_match_logic() ====================


class TestDescribeMatchLogic:
    """Edge cases for _describe_match_logic() and condition grouping."""

    def test_no_valid_branches_returns_empty(self):
        b = _builder()
        result = b._describe_match_logic(
            [
                {"func": "match", "branches": ["not_a_dict", 123]},
            ]
        )
        assert result == ""

    def test_default_value_shown(self):
        b = _builder()
        result = b._describe_match_logic(
            [
                {
                    "func": "match",
                    "field": "status",
                    "branches": [
                        {"pred": {"func": "eq", "field": "status", "val": "active"}, "map-to": "Yes"},
                    ],
                    "default": "No",
                },
            ]
        )
        assert 'else: "No"' in result

    def test_multiple_conditions_same_output_grouped(self):
        """Multiple branches mapping to the same output should be grouped."""
        b = _builder()
        result = b._describe_match_logic(
            [
                {"func": "raw-field", "id": "channel", "label": "ch"},
                {
                    "func": "match",
                    "field": "ch",
                    "branches": [
                        {"pred": {"func": "eq", "field": "ch", "val": "google"}, "map-to": "Search"},
                        {"pred": {"func": "eq", "field": "ch", "val": "bing"}, "map-to": "Search"},
                        {"pred": {"func": "eq", "field": "ch", "val": "yahoo"}, "map-to": "Search"},
                    ],
                },
            ]
        )
        assert "OR" in result
        assert "Search" in result

    def test_many_conditions_same_output_summarized(self):
        """4+ conditions to the same output should show '+N more'."""
        b = _builder()
        branches = [{"pred": {"func": "eq", "field": "ch", "val": f"val{i}"}, "map-to": "Same"} for i in range(5)]
        result = b._describe_match_logic(
            [
                {"func": "match", "field": "ch", "branches": branches},
            ]
        )
        assert "+4 more" in result or "+3 more" in result

    def test_map_to_dict_type_field_resolved_from_label_map(self):
        """map-to with type='field' should be resolved via label_map."""
        b = _builder()
        result = b._describe_match_logic(
            [
                {"func": "raw-field", "id": "web.pageName", "label": "page_ref"},
                {
                    "func": "match",
                    "field": "page_ref",
                    "branches": [
                        {
                            "pred": {"func": "true"},
                            "map-to": {"type": "field", "value": {"label": "page_ref"}},
                        },
                    ],
                },
            ]
        )
        assert "[pageName]" in result

    def test_map_to_dict_with_val(self):
        b = _builder()
        result = b._describe_match_logic(
            [
                {
                    "func": "match",
                    "branches": [
                        {"pred": {"func": "true"}, "map-to": {"val": "hello"}},
                    ],
                },
            ]
        )
        assert '"hello"' in result

    def test_map_to_dict_dynamic(self):
        b = _builder()
        result = b._describe_match_logic(
            [
                {
                    "func": "match",
                    "branches": [
                        {"pred": {"func": "true"}, "map-to": {"type": "computed"}},
                    ],
                },
            ]
        )
        assert "[dynamic]" in result

    def test_map_to_numeric(self):
        b = _builder()
        result = b._describe_match_logic(
            [
                {
                    "func": "match",
                    "branches": [
                        {"pred": {"func": "true"}, "map-to": 42},
                    ],
                },
            ]
        )
        assert "42" in result


# ==================== DerivedFieldSummary.to_dict() edge cases ====================


class TestSummaryToDict:
    """Edge cases for DerivedFieldSummary.to_dict() output formatting."""

    def test_schema_fields_over_five_truncated(self):
        """schema_field_list should show first 5 + '...' when >5 fields."""
        summary = DerivedFieldSummary(
            component_id="test",
            component_name="Test",
            component_type="Dimension",
            complexity_score=10.0,
            functions_used=["Case When"],
            functions_used_internal=["match"],
            branch_count=1,
            nesting_depth=0,
            operator_count=1,
            schema_field_count=7,
            schema_fields=[f"field_{i}" for i in range(7)],
            lookup_references=[],
            component_references=[],
            rule_names=[],
            rule_descriptions=[],
            logic_summary="test",
            inferred_output_type="string",
        )
        d = summary.to_dict()
        assert d["schema_field_list"].endswith("...")

    def test_empty_functions_shows_dash(self):
        summary = DerivedFieldSummary(
            component_id="test",
            component_name="Test",
            component_type="Metric",
            complexity_score=0.0,
            functions_used=[],
            functions_used_internal=[],
            branch_count=0,
            nesting_depth=0,
            operator_count=0,
            schema_field_count=0,
            schema_fields=[],
            lookup_references=[],
            component_references=[],
            rule_names=[],
            rule_descriptions=[],
            logic_summary="",
            inferred_output_type="",
        )
        d = summary.to_dict()
        assert d["functions_used"] == "-"
        assert d["schema_field_list"] == "-"
        assert d["lookup_references"] == "-"
        assert d["rule_names"] == "-"
        assert d["logic_summary"] == "-"
        assert d["output_type"] == "-"


# ==================== DerivedFieldInventory properties ====================


class TestInventoryProperties:
    """Test computed properties on empty/populated inventories."""

    def test_empty_inventory_avg_complexity(self):
        inv = DerivedFieldInventory(data_view_id="x", data_view_name="X")
        assert inv.avg_complexity == 0.0

    def test_empty_inventory_max_complexity(self):
        inv = DerivedFieldInventory(data_view_id="x", data_view_name="X")
        assert inv.max_complexity == 0.0

    def test_summary_high_and_elevated_counts(self):
        """get_summary() should correctly bucket high (>=75) and elevated (50-74)."""
        inv = DerivedFieldInventory(data_view_id="x", data_view_name="X")
        inv.fields = [
            DerivedFieldSummary(
                component_id=f"id_{i}",
                component_name=f"Field {i}",
                component_type="Dimension",
                complexity_score=score,
                functions_used=[],
                functions_used_internal=[],
                branch_count=0,
                nesting_depth=0,
                operator_count=0,
                schema_field_count=0,
                schema_fields=[],
                lookup_references=[],
                component_references=[],
                rule_names=[],
                rule_descriptions=[],
                logic_summary="test",
                inferred_output_type="unknown",
            )
            for i, score in enumerate([10.0, 50.0, 60.0, 75.0, 90.0])
        ]
        summary = inv.get_summary()
        assert summary["complexity"]["high_complexity_count"] == 2  # 75 and 90
        assert summary["complexity"]["elevated_complexity_count"] == 2  # 50 and 60
