"""
CJA Org-Wide Analysis Module.

Provides comprehensive org-wide component analysis including:
- Component distribution (core/common/limited/isolated)
- Pairwise similarity matrix using Jaccard similarity
- Hierarchical clustering of related data views
- Governance recommendations and threshold checking
- Naming convention audits
- Owner-based summaries

Example usage:
    from cja_auto_sdr.org import OrgComponentAnalyzer, OrgReportConfig

    config = OrgReportConfig(
        filter_pattern="Prod.*",
        core_threshold=0.5,
        enable_clustering=True,
    )
    analyzer = OrgComponentAnalyzer(cja, config, logger)
    result = analyzer.run_analysis()
"""

from cja_auto_sdr.org.models import (
    ComponentDistribution,
    ComponentInfo,
    DataViewCluster,
    DataViewSummary,
    OrgReportComparison,
    OrgReportConfig,
    OrgReportResult,
    SimilarityPair,
)
from cja_auto_sdr.org.cache import OrgReportCache, OrgReportLock

__all__ = [
    # Models
    "ComponentDistribution",
    "ComponentInfo",
    "DataViewCluster",
    "DataViewSummary",
    "OrgReportComparison",
    "OrgReportConfig",
    "OrgReportResult",
    "SimilarityPair",
    # Cache & Lock
    "OrgReportCache",
    "OrgReportLock",
    # Analyzer (lazy)
    "OrgComponentAnalyzer",
]


from cja_auto_sdr.core.lazy import make_getattr

__getattr__ = make_getattr(
    __name__,
    [
        "OrgComponentAnalyzer",
    ],
    mapping={
        "OrgComponentAnalyzer": "cja_auto_sdr.org.analyzer",
    },
)
