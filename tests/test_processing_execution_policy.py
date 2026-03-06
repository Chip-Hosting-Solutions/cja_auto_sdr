"""Unit tests for process_single_dataview execution-policy derivation."""

import pytest

from cja_auto_sdr.generator import _build_processing_execution_policy


@pytest.mark.parametrize(
    (
        "output_format",
        "inventory_only",
        "include_derived_inventory",
        "metrics_only",
        "dimensions_only",
        "quality_report_only",
        "skip_validation",
        "expected_required_components",
        "expected_inventory_omits_standard",
        "expected_validation_required",
        "expected_run_validation",
    ),
    [
        ("excel", False, False, False, False, False, False, {"metrics", "dimensions"}, False, True, True),
        ("excel", False, False, True, False, False, False, {"metrics"}, False, True, True),
        ("excel", False, False, False, True, False, False, {"dimensions"}, False, True, True),
        ("json", True, False, False, False, False, False, set(), True, False, False),
        ("html", True, False, False, False, False, False, set(), True, False, False),
        ("markdown", True, False, False, False, False, False, set(), True, False, False),
        ("all", True, False, False, False, False, False, set(), True, False, False),
        ("json", True, True, False, False, False, False, {"metrics", "dimensions"}, True, False, False),
        ("json", False, False, False, False, True, False, {"metrics", "dimensions"}, False, True, True),
        ("json", False, False, False, False, True, True, {"metrics", "dimensions"}, False, True, False),
    ],
)
def test_build_processing_execution_policy_matrix(
    output_format,
    inventory_only,
    include_derived_inventory,
    metrics_only,
    dimensions_only,
    quality_report_only,
    skip_validation,
    expected_required_components,
    expected_inventory_omits_standard,
    expected_validation_required,
    expected_run_validation,
):
    policy = _build_processing_execution_policy(
        output_format=output_format,
        inventory_only=inventory_only,
        include_derived_inventory=include_derived_inventory,
        metrics_only=metrics_only,
        dimensions_only=dimensions_only,
        quality_report_only=quality_report_only,
        skip_validation=skip_validation,
    )

    assert policy.required_component_endpoints == frozenset(expected_required_components)
    assert policy.inventory_only_omits_standard_sections is expected_inventory_omits_standard
    assert policy.validation_required_for_output is expected_validation_required
    assert policy.run_data_quality_validation is expected_run_validation
