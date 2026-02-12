"""
Test suite for CJA Calculated Metrics Inventory

Tests cover:
- Parsing calculated metric formulas
- Inventory generation
- Complexity score calculation
- Formula summary generation
- DataFrame output
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from cja_auto_sdr.inventory.calculated_metrics import (
    CalculatedMetricsInventory,
    CalculatedMetricsInventoryBuilder,
    CalculatedMetricSummary,
)

# ==================== FIXTURES ====================


@pytest.fixture
def sample_simple_calc_metric():
    """Simple calculated metric: Revenue per Order (Revenue / Orders)"""
    return {
        "id": "cm_revenue_per_order",
        "name": "Revenue per Order",
        "description": "Average revenue per order",
        "owner": {"name": "Test Owner"},
        "polarity": "positive",
        "type": "currency",
        "precision": 2,
        "definition": {
            "func": "calc-metric",
            "version": [1, 0, 0],
            "formula": {
                "func": "divide",
                "col1": {"func": "metric", "name": "metrics/revenue"},
                "col2": {"func": "metric", "name": "metrics/orders"},
            },
        },
    }


@pytest.fixture
def sample_complex_calc_metric():
    """Complex calculated metric with segment and nested operations"""
    return {
        "id": "cm_mobile_conversion_rate",
        "name": "Mobile Conversion Rate",
        "description": "Conversion rate for mobile visitors",
        "owner": {"name": "Analytics Team"},
        "polarity": "positive",
        "type": "percent",
        "precision": 2,
        "definition": {
            "func": "calc-metric",
            "version": [1, 0, 0],
            "formula": {
                "func": "segment",
                "segment_id": "s_mobile_visitors",
                "metric": {
                    "func": "divide",
                    "col1": {"func": "metric", "name": "metrics/orders"},
                    "col2": {"func": "metric", "name": "metrics/visits"},
                },
            },
        },
    }


@pytest.fixture
def sample_nested_calc_metric():
    """Deeply nested calculated metric for complexity testing"""
    return {
        "id": "cm_complex_ratio",
        "name": "Complex Profit Ratio",
        "description": "Complex nested calculation",
        "owner": {"name": "Finance Team"},
        "polarity": "positive",
        "type": "decimal",
        "precision": 4,
        "definition": {
            "func": "calc-metric",
            "version": [1, 0, 0],
            "formula": {
                "func": "divide",
                "col1": {
                    "func": "subtract",
                    "col1": {"func": "metric", "name": "metrics/revenue"},
                    "col2": {
                        "func": "add",
                        "col1": {"func": "metric", "name": "metrics/cost"},
                        "col2": {
                            "func": "multiply",
                            "col1": {"func": "metric", "name": "metrics/overhead"},
                            "col2": {"func": "number", "val": 1.2},
                        },
                    },
                },
                "col2": {"func": "metric", "name": "metrics/revenue"},
            },
        },
    }


@pytest.fixture
def sample_conditional_calc_metric():
    """Calculated metric with conditional logic"""
    return {
        "id": "cm_conditional",
        "name": "Conditional Metric",
        "description": "Metric with if condition",
        "owner": {"name": "Test User"},
        "polarity": "neutral",
        "type": "decimal",
        "precision": 0,
        "definition": {
            "func": "calc-metric",
            "version": [1, 0, 0],
            "formula": {
                "func": "if",
                "condition": {
                    "func": "gt",
                    "left": {"func": "metric", "name": "metrics/visits"},
                    "right": {"func": "number", "val": 100},
                },
                "then": {"func": "metric", "name": "metrics/orders"},
                "else": {"func": "number", "val": 0},
            },
        },
    }


@pytest.fixture
def sample_addition_calc_metric():
    """Simple addition calculated metric"""
    return {
        "id": "cm_total_events",
        "name": "Total Events",
        "description": "Sum of click and view events",
        "owner": {"name": "Test"},
        "polarity": "positive",
        "type": "integer",
        "precision": 0,
        "definition": {
            "func": "calc-metric",
            "version": [1, 0, 0],
            "formula": {
                "func": "add",
                "col1": {"func": "metric", "name": "metrics/clicks"},
                "col2": {"func": "metric", "name": "metrics/views"},
            },
        },
    }


@pytest.fixture
def mock_cja_instance(sample_simple_calc_metric, sample_complex_calc_metric):
    """Create a mock CJA instance with calculated metrics"""
    mock_cja = Mock()
    mock_cja.getCalculatedMetrics.return_value = pd.DataFrame([sample_simple_calc_metric, sample_complex_calc_metric])
    return mock_cja


@pytest.fixture
def mock_cja_instance_list_response(sample_simple_calc_metric, sample_complex_calc_metric):
    """Create a mock CJA instance that returns a list (not DataFrame)"""
    mock_cja = Mock()
    mock_cja.getCalculatedMetrics.return_value = [sample_simple_calc_metric, sample_complex_calc_metric]
    return mock_cja


@pytest.fixture
def mock_cja_instance_empty():
    """Create a mock CJA instance with no calculated metrics"""
    mock_cja = Mock()
    mock_cja.getCalculatedMetrics.return_value = pd.DataFrame()
    return mock_cja


# ==================== BUILDER TESTS ====================


class TestCalculatedMetricsInventoryBuilder:
    """Tests for CalculatedMetricsInventoryBuilder class"""

    def test_build_basic(self, mock_cja_instance):
        """Test basic inventory building"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test View")

        assert isinstance(inventory, CalculatedMetricsInventory)
        assert inventory.data_view_id == "dv_test"
        assert inventory.data_view_name == "Test View"
        assert inventory.total_calculated_metrics == 2

    def test_build_from_list_response(self, mock_cja_instance_list_response):
        """Test building when API returns a list instead of DataFrame"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance_list_response, "dv_test", "Test")

        assert inventory.total_calculated_metrics == 2

    def test_build_empty(self, mock_cja_instance_empty):
        """Test building with no calculated metrics"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance_empty, "dv_empty", "Empty")

        assert inventory.total_calculated_metrics == 0

    def test_api_called_with_correct_params(self, mock_cja_instance):
        """Test that API is called with correct parameters"""
        builder = CalculatedMetricsInventoryBuilder()
        builder.build(mock_cja_instance, "dv_abc123", "Test")

        mock_cja_instance.getCalculatedMetrics.assert_called_once_with(dataIds="dv_abc123", full=True)

    def test_complexity_score_calculated(self, mock_cja_instance):
        """Test that complexity scores are calculated"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")

        for metric in inventory.metrics:
            assert 0 <= metric.complexity_score <= 100

    def test_metric_references_extracted(self, sample_simple_calc_metric):
        """Test that metric references are extracted correctly"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_simple_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert len(inventory.metrics) == 1
        metric = inventory.metrics[0]
        assert "revenue" in metric.metric_references
        assert "orders" in metric.metric_references

    def test_segment_references_extracted(self, sample_complex_calc_metric):
        """Test that segment references are extracted correctly"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_complex_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert len(inventory.metrics) == 1
        metric = inventory.metrics[0]
        assert "s_mobile_visitors" in metric.segment_references

    def test_functions_extracted(self, sample_simple_calc_metric):
        """Test that functions are extracted correctly"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_simple_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        assert "Division" in metric.functions_used

    def test_owner_extracted(self, sample_simple_calc_metric):
        """Test that owner is extracted from nested structure"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_simple_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        assert metric.owner == "Test Owner"

    def test_missing_definition_skipped(self):
        """Test that metrics without definitions are skipped"""
        metric_no_def = {
            "id": "cm_no_def",
            "name": "No Definition",
            "description": "",
            "owner": {"name": "Test"},
        }
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [metric_no_def]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert inventory.total_calculated_metrics == 0

    def test_nan_id_skipped(self):
        """Calculated metrics with NaN IDs should be skipped."""
        metric_nan_id = {
            "id": float("nan"),
            "name": "NaN ID Metric",
            "description": "",
            "owner": {"name": "Test"},
            "definition": {"formula": {"func": "metric", "name": "metrics/revenue"}},
        }
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [metric_nan_id]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert inventory.total_calculated_metrics == 0


# ==================== INVENTORY TESTS ====================


class TestCalculatedMetricsInventory:
    """Tests for CalculatedMetricsInventory class"""

    def test_get_dataframe(self, mock_cja_instance):
        """Test DataFrame output"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")

        df = inventory.get_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == inventory.total_calculated_metrics
        assert "name" in df.columns
        assert "complexity_score" in df.columns
        assert "functions_used" in df.columns
        assert "formula_summary" in df.columns

    def test_get_dataframe_sorted_by_complexity(self, mock_cja_instance):
        """Test that DataFrame is sorted by complexity score descending"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")

        df = inventory.get_dataframe()

        scores = df["complexity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_get_summary(self, mock_cja_instance):
        """Test summary statistics"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")

        summary = inventory.get_summary()

        assert "data_view_id" in summary
        assert "data_view_name" in summary
        assert "total_calculated_metrics" in summary
        assert "complexity" in summary
        assert "function_usage" in summary

    def test_to_json(self, mock_cja_instance):
        """Test JSON export"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")

        json_data = inventory.to_json()

        assert "summary" in json_data
        assert "metrics" in json_data
        assert len(json_data["metrics"]) == inventory.total_calculated_metrics

    def test_empty_inventory_dataframe(self, mock_cja_instance_empty):
        """Test DataFrame output for empty inventory"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance_empty, "dv_test", "Test")

        df = inventory.get_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "name" in df.columns  # Columns should still be present


# ==================== COMPLEXITY SCORE TESTS ====================


class TestComplexityScore:
    """Tests for complexity score calculation"""

    def test_simple_metric_low_complexity(self, sample_simple_calc_metric):
        """Test that simple metrics have low complexity"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_simple_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        assert metric.complexity_score < 30  # Simple division should be low

    def test_nested_metric_higher_complexity(self, sample_nested_calc_metric):
        """Test that nested metrics have higher complexity"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_nested_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        assert metric.complexity_score > 20  # Nested should be more complex

    def test_segment_metric_adds_complexity(self, sample_complex_calc_metric):
        """Test that segment filters add complexity"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_complex_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        # Should have segment contribution
        assert metric.complexity_score > 0
        assert len(metric.segment_references) > 0

    def test_conditional_metric_adds_complexity(self, sample_conditional_calc_metric):
        """Test that conditional logic adds complexity"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_conditional_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        assert metric.conditional_count > 0


# ==================== FORMULA SUMMARY TESTS ====================


class TestFormulaSummary:
    """Tests for formula summary generation"""

    def test_division_summary(self, sample_simple_calc_metric):
        """Test formula summary for division"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_simple_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        assert "revenue" in metric.formula_summary.lower()
        assert "orders" in metric.formula_summary.lower()

    def test_addition_summary(self, sample_addition_calc_metric):
        """Test formula summary for addition"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_addition_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        # Should mention the referenced metrics or sum
        assert (
            "sum" in metric.formula_summary.lower()
            or "clicks" in metric.formula_summary.lower()
            or "addition" in metric.formula_summary.lower()
        )

    def test_segment_summary(self, sample_complex_calc_metric):
        """Test formula summary for segmented metric"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_complex_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        # Should mention segment/filter OR show segment in bracket notation [segment]
        assert (
            "segment" in metric.formula_summary.lower()
            or "filter" in metric.formula_summary.lower()
            or "[" in metric.formula_summary
        )  # bracket notation for segments

    def test_conditional_summary(self, sample_conditional_calc_metric):
        """Test formula summary for conditional metric"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [sample_conditional_calc_metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        metric = inventory.metrics[0]
        # Should show conditional logic either as word or IF(...) function
        assert "conditional" in metric.formula_summary.lower() or "if(" in metric.formula_summary.lower()


# ==================== EDGE CASE TESTS ====================


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_missing_owner(self):
        """Test handling of missing owner"""
        metric = {
            "id": "cm_test",
            "name": "Test",
            "description": "",
            "definition": {"func": "calc-metric", "formula": {"func": "metric", "name": "metrics/visits"}},
        }
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert len(inventory.metrics) == 1
        assert inventory.metrics[0].owner == ""

    def test_owner_as_string(self):
        """Test handling of owner as string instead of dict"""
        metric = {
            "id": "cm_test",
            "name": "Test",
            "description": "",
            "owner": "string_owner",
            "definition": {"func": "calc-metric", "formula": {"func": "metric", "name": "metrics/visits"}},
        }
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert inventory.metrics[0].owner == "string_owner"

    def test_empty_formula(self):
        """Test handling of empty formula"""
        metric = {
            "id": "cm_test",
            "name": "Test",
            "description": "",
            "definition": {"func": "calc-metric", "formula": {}},
        }
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        # Empty formula should be skipped
        assert inventory.total_calculated_metrics == 0

    def test_api_error_handling(self):
        """Test handling of API errors"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.side_effect = Exception("API Error")

        builder = CalculatedMetricsInventoryBuilder()

        with pytest.raises(Exception) as exc_info:
            builder.build(mock_cja, "dv_test", "Test")

        assert "API Error" in str(exc_info.value)

    def test_none_response(self):
        """Test handling of None API response"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = None

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert inventory.total_calculated_metrics == 0

    def test_unexpected_response_type(self):
        """Test handling of unexpected response type"""
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = "unexpected"

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert inventory.total_calculated_metrics == 0

    def test_scalar_formula_is_supported(self):
        """Scalar formulas should not crash summary generation."""
        metric = {
            "id": "cm_scalar",
            "name": "Scalar Metric",
            "description": "",
            "definition": {"func": "calc-metric", "formula": 1.0},
        }
        mock_cja = Mock()
        mock_cja.getCalculatedMetrics.return_value = [metric]

        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja, "dv_test", "Test")

        assert inventory.total_calculated_metrics == 1
        assert inventory.metrics[0].formula_summary == "1.0"


# ==================== DATA CLASS TESTS ====================


class TestCalculatedMetricSummary:
    """Tests for CalculatedMetricSummary data class"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        summary = CalculatedMetricSummary(
            metric_id="cm_test",
            metric_name="Test Metric",
            description="Test description",
            owner="Test Owner",
            complexity_score=45.5,
            functions_used=["Division", "Metric Reference"],
            functions_used_internal=["divide", "metric"],
            nesting_depth=2,
            operator_count=1,
            metric_references=["revenue", "orders"],
            segment_references=[],
            conditional_count=0,
            formula_summary="revenue divided by orders",
            polarity="positive",
            metric_type="currency",
            precision=2,
        )

        d = summary.to_dict()

        assert d["name"] == "Test Metric"
        assert d["description"] == "Test description"
        assert d["owner"] == "Test Owner"
        assert d["complexity_score"] == 45.5
        assert "Division" in d["functions_used"]
        assert "revenue" in d["metric_references"]
        assert d["polarity"] == "Positive"

    def test_to_full_dict(self):
        """Test conversion to full dictionary"""
        summary = CalculatedMetricSummary(
            metric_id="cm_test",
            metric_name="Test",
            description="",
            owner="",
            complexity_score=10.0,
            functions_used=["Division"],
            functions_used_internal=["divide"],
            nesting_depth=1,
            operator_count=1,
            metric_references=["a", "b"],
            segment_references=["seg1"],
            conditional_count=0,
            formula_summary="a / b",
            polarity="positive",
            metric_type="decimal",
            precision=4,
        )

        d = summary.to_full_dict()

        assert d["metric_id"] == "cm_test"
        assert d["nesting_depth"] == 1
        assert d["segment_references"] == ["seg1"]
        assert d["precision"] == 4


class TestCalculatedMetricsInventoryProperties:
    """Tests for CalculatedMetricsInventory properties"""

    def test_avg_complexity_empty(self):
        """Test average complexity with no metrics"""
        inventory = CalculatedMetricsInventory(data_view_id="dv_test", data_view_name="Test")
        assert inventory.avg_complexity == 0.0

    def test_max_complexity_empty(self):
        """Test max complexity with no metrics"""
        inventory = CalculatedMetricsInventory(data_view_id="dv_test", data_view_name="Test")
        assert inventory.max_complexity == 0.0


# ==================== SUMMARY COLUMN ALIAS TESTS ====================


class TestSummaryColumnAlias:
    """Tests for standardized summary column alias"""

    def test_summary_column_exists(self, mock_cja_instance):
        """Test that summary column exists in DataFrame"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")
        df = inventory.get_dataframe()

        assert "summary" in df.columns

    def test_summary_equals_formula_summary(self, mock_cja_instance):
        """Test that summary column equals formula_summary column"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance, "dv_test", "Test")
        df = inventory.get_dataframe()

        assert all(df["summary"] == df["formula_summary"])

    def test_empty_dataframe_has_summary_column(self, mock_cja_instance_empty):
        """Test that empty DataFrame still has summary column"""
        builder = CalculatedMetricsInventoryBuilder()
        inventory = builder.build(mock_cja_instance_empty, "dv_test", "Test")
        df = inventory.get_dataframe()

        assert "summary" in df.columns
