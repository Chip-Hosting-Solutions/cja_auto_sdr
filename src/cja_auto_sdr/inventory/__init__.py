"""
CJA Inventory Modules

Provides inventory builders for CJA components:
- Calculated Metrics
- Derived Fields
- Segments
"""

from cja_auto_sdr.inventory.calculated_metrics import (
    CalculatedMetricSummary,
    CalculatedMetricsInventory,
    CalculatedMetricsInventoryBuilder,
)
from cja_auto_sdr.inventory.derived_fields import (
    DerivedFieldSummary,
    DerivedFieldInventory,
    DerivedFieldInventoryBuilder,
)
from cja_auto_sdr.inventory.segments import (
    SegmentSummary,
    SegmentsInventory,
    SegmentsInventoryBuilder,
)
from cja_auto_sdr.inventory.utils import (
    BatchProcessingStats,
    compute_complexity_score,
    extract_owner,
    extract_short_name,
    extract_tags,
    format_iso_date,
    normalize_api_response,
    validate_required_id,
)

__all__ = [
    # Calculated Metrics
    "CalculatedMetricSummary",
    "CalculatedMetricsInventory",
    "CalculatedMetricsInventoryBuilder",
    # Derived Fields
    "DerivedFieldSummary",
    "DerivedFieldInventory",
    "DerivedFieldInventoryBuilder",
    # Segments
    "SegmentSummary",
    "SegmentsInventory",
    "SegmentsInventoryBuilder",
    # Utilities
    "BatchProcessingStats",
    "compute_complexity_score",
    "extract_owner",
    "extract_short_name",
    "extract_tags",
    "format_iso_date",
    "normalize_api_response",
    "validate_required_id",
]


from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(
    __name__,
    ["display_inventory_summary"],
    mapping={"display_inventory_summary": "cja_auto_sdr.inventory.summary"},
)
