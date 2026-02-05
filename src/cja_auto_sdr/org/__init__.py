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
from cja_auto_sdr.org.analyzer import OrgComponentAnalyzer

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
    # Analyzer
    "OrgComponentAnalyzer",
]
