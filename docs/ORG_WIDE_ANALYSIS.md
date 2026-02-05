# Org-Wide Component Analysis Guide

> **New in v3.2.0** - Comprehensive org-wide analysis with governance features

Analyze component usage patterns across all data views in your CJA organization. This feature provides governance insights, identifies duplication, and helps standardize your analytics implementation.

## Overview

The org-wide analysis feature allows you to:
- **Analyze all data views** in your organization simultaneously
- **Identify core components** used across multiple data views
- **Detect duplicate data views** using Jaccard similarity
- **Generate governance recommendations** based on usage patterns
- **Export reports** in all supported output formats (console, JSON, Excel, HTML, Markdown, CSV)
- **Track trends** by comparing reports over time
- **Audit naming conventions** for consistency
- **Group data views** into clusters based on component similarity
- **Integrate with CI/CD** using governance threshold exit codes

## Quick Start

```bash
# Basic org-wide report (console output)
cja_auto_sdr --org-report

# With filtering
cja_auto_sdr --org-report --filter "Prod.*" --exclude "Test|Dev"

# Export to Excel
cja_auto_sdr --org-report --format excel

# Export all formats
cja_auto_sdr --org-report --format all
```

## How It Works

### 1. Data View Discovery

The analyzer fetches all accessible data views from your CJA organization, applying optional filters:

```bash
# Include only data views matching pattern
cja_auto_sdr --org-report --filter "Production.*"

# Exclude data views matching pattern
cja_auto_sdr --org-report --exclude "Test|Sandbox|Dev"

# Limit to first N data views (useful for testing)
cja_auto_sdr --org-report --limit 10
```

### 2. Component Indexing

For each data view, the analyzer fetches all metrics and dimensions, building a global index that tracks:
- **Component ID** - Unique identifier (e.g., `metrics/pageviews`)
- **Component Type** - Metric or Dimension
- **Name** - Display name (with `--include-names`)
- **Data Views** - Which data views contain this component

### 3. Distribution Classification

Components are classified into four buckets based on how many data views contain them:

| Bucket | Criteria | Description |
|--------|----------|-------------|
| **Core** | >= 50% of DVs | Foundation components used organization-wide |
| **Common** | 25-49% of DVs | Shared across multiple teams/use cases |
| **Limited** | 2+ DVs, < 25% | Used in a subset of data views |
| **Isolated** | 1 DV only | Unique to a single data view |

Customize the core threshold:

```bash
# Components in 70%+ of data views are "core"
cja_auto_sdr --org-report --core-threshold 0.7

# Or use absolute count: components in 5+ DVs are "core"
cja_auto_sdr --org-report --core-min-count 5
```

### 4. Similarity Matrix

The analyzer computes pairwise Jaccard similarity between data views:

```
Jaccard Similarity = |Components in Both| / |Components in Either|
```

This identifies:
- **Duplicate data views** (90%+ similarity)
- **Prod/staging pairs** (high overlap with minor differences)
- **Related data views** that share common components

Note: For governance checks, pairs with >= 90% similarity are always included.
If `--overlap-threshold` is set above 0.9, the effective similarity threshold is capped at 0.9
and reports will note the configured vs. effective threshold.

```bash
# Flag pairs with 90%+ similarity (default: 80%)
cja_auto_sdr --org-report --overlap-threshold 0.9

# Skip similarity calculation for large orgs
cja_auto_sdr --org-report --skip-similarity
```

### 5. Recommendations

Based on the analysis, the tool generates governance recommendations:

| Type | Severity | Description |
|------|----------|-------------|
| `review_isolated` | Medium | Data views with many unique components |
| `review_overlap` | High | Near-duplicate data view pairs (90%+ similarity) |
| `standardization_opportunity` | Low | Components in 70-99% of DVs (near-universal) |
| `fetch_errors` | High | Data views that couldn't be analyzed |
| `high_derived_ratio` | Low | Data views with >50% derived components |
| `stale_data_view` | Low | Data views not modified in 6+ months |
| `missing_descriptions` | Low | Data views lacking descriptions |

## Command Reference

### Basic Options

| Option | Description |
|--------|-------------|
| `--org-report` | Enable org-wide analysis mode |
| `--filter PATTERN` | Include data views matching regex |
| `--exclude PATTERN` | Exclude data views matching regex |
| `--limit N` | Analyze only first N data views |
| `--include-names` | Fetch and display component names |
| `--quiet` | Suppress progress output |

### Threshold Options

| Option | Default | Description |
|--------|---------|-------------|
| `--core-threshold` | 0.5 | Fraction of DVs for "core" classification |
| `--core-min-count` | - | Absolute count for "core" (overrides threshold) |
| `--overlap-threshold` | 0.8 | Minimum similarity to flag as "high overlap" (capped at 0.9 for governance checks) |

### Component Type Breakdown

| Option | Default | Description |
|--------|---------|-------------|
| `--no-component-types` | Off | Disable standard/derived field breakdown |

When enabled (default), tracks:
- **Standard metrics/dimensions** - Native XDM-based components
- **Derived metrics/dimensions** - Components created via derived fields

```bash
# View component type distribution
cja_auto_sdr --org-report --format excel
# Columns: Standard Metrics, Derived Metrics, Standard Dimensions, Derived Dimensions
```

### Metadata Options

| Option | Description |
|--------|-------------|
| `--include-metadata` | Include owner, dates, descriptions for each data view |

When enabled:
- Shows data view owner name and ID
- Shows creation and modification dates
- Tracks description completeness
- Generates `stale_data_view` and `missing_descriptions` recommendations

```bash
# Include metadata in report
cja_auto_sdr --org-report --include-metadata --format excel
```

### Drift Detection

| Option | Description |
|--------|-------------|
| `--include-drift` | Show exact component differences between similar DV pairs |

When enabled, high-overlap pairs include:
- Components only in DV1
- Components only in DV2
- Component names (when `--include-names` also used)

```bash
# Detailed drift analysis between similar data views
cja_auto_sdr --org-report --include-drift --include-names --format excel
```

### Sampling Options

For very large organizations, use sampling to analyze a representative subset:

| Option | Description |
|--------|-------------|
| `--sample N` | Randomly sample N data views |
| `--sample-seed SEED` | Random seed for reproducible sampling |
| `--sample-stratified` | Stratify sample by data view name prefix |

```bash
# Analyze a random sample of 20 data views
cja_auto_sdr --org-report --sample 20 --sample-seed 42

# Stratified sampling (proportional by name prefix)
cja_auto_sdr --org-report --sample 30 --sample-stratified
```

### Caching Options

Cache data view components for faster repeat runs:

| Option | Default | Description |
|--------|---------|-------------|
| `--use-cache` | Off | Enable caching of fetched components |
| `--cache-max-age HOURS` | 24 | Maximum cache age before refresh |
| `--refresh-cache` | Off | Clear cache and fetch fresh data |

```bash
# First run - fetches and caches
cja_auto_sdr --org-report --use-cache

# Second run - uses cache (much faster)
cja_auto_sdr --org-report --use-cache

# Force refresh
cja_auto_sdr --org-report --use-cache --refresh-cache
```

> **Note:** Smart cache validation (`--validate-cache`) compares modification
> timestamps. If the CJA API doesn't return a modification timestamp for a
> data view, the cached entry is treated as valid (optimistic caching). Use
> `--refresh-cache` to force a full refresh when in doubt.

Cache is stored in `~/.cja_auto_sdr/cache/org_report_cache.json`.

### Clustering Options

Group related data views into clusters using hierarchical clustering:

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster` | Off | Enable hierarchical clustering |
| `--cluster-method METHOD` | average | Linkage method: `average`, `complete`, or `ward` |

```bash
# Identify data view families
cja_auto_sdr --org-report --cluster --format excel

# Use complete linkage for tighter clusters
cja_auto_sdr --org-report --cluster --cluster-method complete
```

Generates a "Clusters" sheet showing:
- Cluster ID and inferred name
- Member data views
- Cohesion score (within-cluster similarity)

**Note:** Clustering requires the optional `scipy` dependency. Install with:

```bash
uv pip install 'cja-auto-sdr[clustering]'
```

If `scipy` is not installed and `--cluster` is used, a warning is logged and clustering is skipped gracefully.

**Important:** The default method is `average` because it works correctly with Jaccard distances. The `ward` method assumes Euclidean distances and may produce incorrect results with similarity-based distances.

### Performance Options

| Option | Description |
|--------|-------------|
| `--skip-similarity` | Skip O(n²) pairwise similarity calculation |
| `--similarity-max-dvs N` | Guardrail to skip similarity when data views exceed N (default: 250) |
| `--force-similarity` | Force similarity even if guardrails would skip it |

### Concurrency Options

| Option | Description |
|--------|-------------|
| `--org-shared-client` | Use a single shared cjapy client across threads (faster, but may be unsafe if cjapy is not thread-safe) |

### Output Options

| Option | Description |
|--------|-------------|
| `--format FORMAT` | Output format: `console`, `json`, `excel`, `markdown`, `html`, `csv`, `all`, or aliases |
| `--output PATH` | Specific output file path |
| `--output-dir DIR` | Output directory (default: current) |

**Format Aliases:**
| Alias | Expands To | Use Case |
|-------|------------|----------|
| `reports` | excel + markdown | Documentation and sharing |
| `data` | csv + json | Data pipelines and analysis |
| `ci` | json + markdown | CI/CD integration |

```bash
# Generate Excel + Markdown for documentation
cja_auto_sdr --org-report --format reports

# Generate CSV + JSON for data pipelines
cja_auto_sdr --org-report --format data
```

## Output Formats

### Console (Default)

ASCII-formatted report with distribution bars:

```
==============================================================================================================
ORG-WIDE COMPONENT ANALYSIS REPORT: ABC123DEF456GHI789@AdobeOrg
==============================================================================================================
Generated: 2024-01-15T10:30:00
Data Views Analyzed: 12 / 12
Analysis Duration: 8.45s

----------------------------------------------------------------------
DISTRIBUTION
----------------------------------------------------------------------
Metrics by data view coverage:
  Core:     ████████░░░░░░░░░░░░░░░░░░░░░░  27% (45)
  Common:   ██████░░░░░░░░░░░░░░░░░░░░░░░░  19% (32)
  Limited:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░  14% (24)
  Isolated: ████████████░░░░░░░░░░░░░░░░░░  40% (67)
```

### JSON

Structured JSON for programmatic processing:

```json
{
  "report_type": "org_analysis",
  "org_id": "ABC123DEF456GHI789@AdobeOrg",
  "summary": {
    "data_views_total": 12,
    "data_views_analyzed": 12,
    "total_unique_metrics": 168,
    "total_unique_dimensions": 94
  },
  "distribution": {
    "core": {"metrics_count": 45, "dimensions_count": 28},
    "common": {...},
    "limited": {...},
    "isolated": {...}
  },
  "component_index": {...},
  "similarity_pairs": [...],
  "recommendations": [...]
}
```

### Excel

Multi-sheet workbook with:
- **Summary** - Overview statistics
- **Data Views** - List of all analyzed data views
- **Core Components** - Components in 50%+ of data views
- **Isolated by DV** - Isolated component counts per data view
- **Similarity** - High-overlap pairs
- **Recommendations** - Actionable governance items

### Markdown

GitHub-flavored markdown with tables:

```markdown
# Org-Wide Component Analysis Report

## Summary
| Metric | Value |
|--------|-------|
| Data Views Analyzed | 12 / 12 |
| Total Unique Metrics | 168 |
...
```

### HTML

Styled HTML report with:
- Interactive statistics cards
- Progress bar visualizations
- Sortable tables
- Color-coded severity badges

### CSV

Multiple CSV files in a directory:
- `org_report_summary.csv`
- `org_report_data_views.csv`
- `org_report_components.csv`
- `org_report_distribution.csv`
- `org_report_similarity.csv`
- `org_report_recommendations.csv`

## Use Cases

### 1. Governance Audit

Identify data view sprawl and standardization opportunities:

```bash
cja_auto_sdr --org-report --format excel --output-dir ./audit
```

Review the Excel report for:
- Data views with many isolated components (specialized or orphaned?)
- Near-duplicate pairs that could be consolidated
- Components used in 70-99% of DVs (candidates for standardization)

### 2. Pre-Migration Analysis

Before migrating or consolidating data views:

```bash
cja_auto_sdr --org-report --include-names --format json
```

The JSON output provides a complete component inventory for planning.

If you want to stream JSON to another tool, send it to stdout and pipe it:

```bash
cja_auto_sdr --org-report --format json --output - | jq '.summary'
```

### 3. Environment Validation

Verify prod/staging parity:

```bash
cja_auto_sdr --org-report --filter "Prod|Staging" --overlap-threshold 0.95
```

High-overlap pairs (>95%) indicate proper synchronization.

### 4. Quick Health Check

Fast summary without detailed analysis:

```bash
cja_auto_sdr --org-report --skip-similarity --org-summary
```

### 5. CI/CD Integration

Automated governance checks:

```bash
cja_auto_sdr --org-report --format json --quiet --output ./reports/org-audit.json
```

## Performance Considerations

### Large Organizations

For organizations with many data views (50+):

1. **Use `--skip-similarity`** to avoid O(n²) pairwise comparison
2. **Use `--limit`** to test with a subset first
3. **Use filters** to focus on specific data view groups

```bash
# Analyze only production data views, skip similarity
cja_auto_sdr --org-report --filter "^Prod" --skip-similarity
```

### API Rate Limits

The analyzer fetches components in parallel (up to 10 concurrent requests). For very large orgs, you may see rate limiting. The tool handles this gracefully with automatic retries.

### Memory Usage

Component indices are held in memory. For orgs with 100+ data views and 10,000+ unique components, ensure adequate memory (4GB+ recommended).

## Troubleshooting

### "No data views found matching criteria"

- Check your filter/exclude patterns (they're case-insensitive regex)
- Verify API credentials have access to data views
- Run `cja_auto_sdr --list-dataviews` to see available data views

### Some data views show ERROR

- Check the `fetch_errors` recommendation for details
- Verify permissions for those specific data views
- Data views may be in an invalid state

### Similarity matrix is slow

For N data views, similarity requires N×(N-1)/2 comparisons. With 50 DVs, that's 1,225 comparisons. Use `--skip-similarity` for faster results.

### "Another --org-report is already running"

The tool prevents concurrent `--org-report` runs for the same organization to avoid:
- Duplicate API calls and wasted resources
- Rate limit exhaustion
- Inconsistent results

If you see this error:
1. Check if another terminal/process is running an org-report
2. Wait for the existing run to complete
3. If the previous run crashed, the lock will automatically expire after 1 hour

The lock file is stored in `~/.cja_auto_sdr/locks/`.

To force bypass the lock (for testing only):
```bash
# Use skip_lock in your config (not recommended for production)
# The lock exists to protect against accidental concurrent runs
```

## Advanced Features

### Governance Exit Codes

Integrate org-report into CI/CD pipelines with threshold-based exit codes.

```bash
# Exit with code 2 if more than 5 high-similarity pairs exist
cja_auto_sdr --org-report --duplicate-threshold 5 --fail-on-threshold
echo $?  # 0 = pass, 2 = threshold exceeded

# Exit with code 2 if isolated components exceed 30%
cja_auto_sdr --org-report --isolated-threshold 0.3 --fail-on-threshold

# Combine thresholds for comprehensive governance checks
cja_auto_sdr --org-report \
  --duplicate-threshold 3 \
  --isolated-threshold 0.4 \
  --fail-on-threshold \
  --format json --output governance-report.json
```

| Option | Description |
|--------|-------------|
| `--duplicate-threshold N` | Max allowed high-similarity pairs (>=90%) |
| `--isolated-threshold PERCENT` | Max isolated component percentage (0.0-1.0) |
| `--fail-on-threshold` | Enable exit code 2 when thresholds exceeded |

### Org Summary Stats Mode

Quick stats-only mode for fast health checks - skips similarity and clustering computation.

```bash
# Fast summary (no similarity matrix or clustering)
cja_auto_sdr --org-report --org-stats

# Output:
# ORG STATS: ABC123@AdobeOrg
# Data Views: 25 analyzed
# Components: 450 unique
#   Metrics:    280
#   Dimensions: 170
# Distribution:
#   Core:      85 (18.9%)
#   Common:   120 (26.7%)
#   Limited:   95 (21.1%)
#   Isolated: 150 (33.3%)
```

### Naming Convention Audit

Detect inconsistent naming patterns across components.

```bash
# Run naming audit
cja_auto_sdr --org-report --audit-naming

# Detects:
# - Mixed case styles (snake_case vs camelCase vs PascalCase)
# - Common prefix groupings
# - Stale patterns (test, old, temp, dates)
```

### Trending/Drift Report

Compare org-reports over time to detect changes.

```bash
# Save baseline
cja_auto_sdr --org-report --format json --output baseline.json

# Later, compare to baseline
cja_auto_sdr --org-report --compare-org-report baseline.json

# Shows:
# - Data views added/removed
# - Component count changes (↑ / ↓)
# - New high-similarity pairs
# - Resolved pairs
```

### Owner/Team Summary

Group statistics by data view owner.

```bash
# Requires --include-metadata
cja_auto_sdr --org-report --include-metadata --owner-summary

# Shows:
# Owner              DVs    Metrics    Dimensions    Avg/DV
# Alice              5      450        280           146.0
# Bob                3      210        150           120.0
# Unknown            2       80         50            65.0
```

### Stale Component Heuristics

Flag components with cleanup candidate naming patterns.

```bash
cja_auto_sdr --org-report --flag-stale

# Detects:
# - Stale keywords: test, old, temp, backup, copy, deprecated
# - Version suffixes: _v1, _v2
# - Date patterns: _20240101, _2024-01-01
```

### Complete Governance CI/CD Example

```bash
#!/bin/bash
# governance-check.sh

cja_auto_sdr --org-report \
  --duplicate-threshold 5 \
  --isolated-threshold 0.35 \
  --fail-on-threshold \
  --audit-naming \
  --flag-stale \
  --format json \
  --output governance-report.json \
  --quiet

EXIT_CODE=$?

if [ $EXIT_CODE -eq 2 ]; then
  echo "GOVERNANCE CHECK FAILED: Thresholds exceeded"
  exit 1
elif [ $EXIT_CODE -ne 0 ]; then
  echo "GOVERNANCE CHECK ERROR"
  exit 1
fi

echo "GOVERNANCE CHECK PASSED"
```

## Related Documentation

- [CLI Reference](CLI_REFERENCE.md) - Complete command-line options
- [Output Formats](OUTPUT_FORMATS.md) - Format-specific details
- [Configuration Guide](CONFIGURATION.md) - Credential and profile setup
- [Batch Processing Guide](BATCH_PROCESSING_GUIDE.md) - Processing multiple data views
- [Use Cases & Best Practices](USE_CASES.md) - Governance workflows and automation examples
- [Quick Reference](QUICK_REFERENCE.md) - Single-page command cheat sheet
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
