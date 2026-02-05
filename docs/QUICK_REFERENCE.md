# Quick Reference Card

Single-page command cheat sheet for CJA SDR Generator v3.2.0.

## Three Main Modes

| Mode | Purpose | Output |
|------|---------|--------|
| **SDR Generation** | Document a data view's dimensions, metrics, and calculated metrics | Excel, CSV, JSON, HTML, Markdown reports |
| **Diff Comparison** | Compare two data views or snapshots to identify changes | Side-by-side comparison showing added, removed, and modified components |
| **Org-Wide Analysis** | Analyze component usage across all data views in an organization | Distribution reports, similarity matrix, governance recommendations |

**SDR Generation** creates a Solution Design Reference—a comprehensive inventory of all components in a data view. Use this for documentation, audits, and onboarding.

**Diff Comparison** identifies what changed between two data views or between a current state and a saved snapshot. Use this for change tracking, QA validation, and migration verification.

**Org-Wide Analysis** examines all accessible data views to identify core components, detect duplicates, and provide governance insights. Use this for audits, standardization, and understanding your analytics landscape.

## Running Commands

You have three equivalent options:

| Method | Command | Notes |
|--------|---------|-------|
| **uv run** | `uv run cja_auto_sdr ...` | Works immediately on macOS/Linux, may have issues on Windows |
| **Activated venv** | `cja_auto_sdr ...` | After activating: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows) |
| **Direct script** | `cja_auto_sdr ...` | Most reliable on Windows |

This guide uses `uv run`. Windows users should substitute with `cja_auto_sdr`. The command examples below omit the prefix for brevity.

## Interactive Mode (Recommended for New Users)

```bash
# Launch interactive mode - walks through all options step by step
cja_auto_sdr --interactive
# or
cja_auto_sdr -i
```

Interactive mode guides you through:
1. **Data View Selection** - Shows list of accessible data views
2. **Output Format** - Excel, JSON, CSV, HTML, Markdown, or all
3. **Inventory Options** - Segments, calculated metrics, derived fields

## Essential Commands (SDR Generation)

```bash
# Generate SDR for a single data view
cja_auto_sdr dv_12345

# Generate and open file immediately
cja_auto_sdr dv_12345 --open

# Process multiple data views in parallel
cja_auto_sdr dv_12345 dv_67890 dv_abcde

# Use data view names instead of IDs
cja_auto_sdr "Production Analytics"

# Quick stats (no full report)
cja_auto_sdr dv_12345 --stats

# List all accessible data views
cja_auto_sdr --list-dataviews

# List data views as JSON (for scripting)
cja_auto_sdr --list-dataviews --format json

# Interactively select data views from a list
cja_auto_sdr --interactive

# Validate config without processing
cja_auto_sdr --validate-config
```

## Diff Comparison Commands (Diff Mode)

```bash
# Compare two data views
cja_auto_sdr --diff dv_12345 dv_67890

# Compare using names
cja_auto_sdr --diff "Production" "Staging"

# Save snapshot for later comparison
cja_auto_sdr dv_12345 --snapshot ./baseline.json

# Compare current state to snapshot
cja_auto_sdr dv_12345 --diff-snapshot ./baseline.json

# Compare against most recent snapshot (auto-finds it)
cja_auto_sdr dv_12345 --compare-with-prev

# Compare two snapshots (no API calls)
cja_auto_sdr --compare-snapshots ./old.json ./new.json

# Auto-save snapshots during diff
cja_auto_sdr --diff dv_12345 dv_67890 --auto-snapshot

# Show only changes (hide unchanged)
cja_auto_sdr --diff dv_12345 dv_67890 --changes-only

# Custom labels in diff output
cja_auto_sdr --diff dv_12345 dv_67890 --diff-labels "Before" "After"
```

## Org-Wide Analysis Commands

```bash
# Analyze all data views in organization (console output)
cja_auto_sdr --org-report

# Filter data views by name pattern
cja_auto_sdr --org-report --filter "Prod.*"

# Exclude test/dev data views
cja_auto_sdr --org-report --exclude "Test|Dev|Sandbox"

# Combine filter and exclude
cja_auto_sdr --org-report --filter "Analytics" --exclude "Test"

# Limit to first N data views (for testing)
cja_auto_sdr --org-report --limit 10

# Include component names (slower but more readable)
cja_auto_sdr --org-report --include-names

# Skip similarity matrix (faster for large orgs)
cja_auto_sdr --org-report --skip-similarity

# Custom thresholds
cja_auto_sdr --org-report --core-threshold 0.7 --overlap-threshold 0.9

# Export to Excel
cja_auto_sdr --org-report --format excel

# Export all formats
cja_auto_sdr --org-report --format all --output-dir ./reports

# Quick stats mode (fast overview, no similarity/clustering)
cja_auto_sdr --org-report --org-stats

# Quick health check with sampling
cja_auto_sdr --org-report --org-stats --sample 10
```

### Advanced Org-Wide Options

```bash
# --- Clustering & Similarity ---
# Requires: uv pip install 'cja-auto-sdr[clustering]'

# Cluster similar data views into families
cja_auto_sdr --org-report --cluster

# Cluster with specific method (average recommended, complete also valid)
cja_auto_sdr --org-report --cluster --cluster-method complete

# --- Caching (for large orgs) ---
# Use cached data (refresh if stale)
cja_auto_sdr --org-report --use-cache

# Force cache refresh
cja_auto_sdr --org-report --use-cache --refresh-cache

# Custom cache max age (seconds)
cja_auto_sdr --org-report --use-cache --cache-max-age 7200

# --- Sampling (for very large orgs) ---
# Random sample 50 data views
cja_auto_sdr --org-report --sample 50

# Reproducible sample with seed
cja_auto_sdr --org-report --sample 50 --sample-seed 42

# Stratified sampling by owner
cja_auto_sdr --org-report --sample 50 --sample-stratified

# --- Trending & Comparison ---
# Compare against previous org report
cja_auto_sdr --org-report --compare-org-report ./previous.json

# --- Naming Audit ---
# Audit naming conventions
cja_auto_sdr --org-report --audit-naming

# Flag potentially stale components
cja_auto_sdr --org-report --flag-stale

# --- Governance & CI/CD ---
# CI/CD with governance thresholds
cja_auto_sdr --org-report --fail-on-threshold

# Custom thresholds
cja_auto_sdr --org-report --fail-on-threshold \
  --duplicate-threshold 0.15 \
  --isolated-threshold 0.25

# --- Component Types & Metadata ---
# Filter to specific component types
cja_auto_sdr --org-report --component-types dimensions,metrics

# Include owner/team summary
cja_auto_sdr --org-report --owner-summary

# Include metadata in output
cja_auto_sdr --org-report --include-metadata
```

## Profile Management (Multi-Org)

```bash
# Create a profile for each organization
cja_auto_sdr --profile-add client-a
cja_auto_sdr --profile-add client-b

# List all profiles
cja_auto_sdr --profile-list

# Use a specific profile
cja_auto_sdr --profile client-a --list-dataviews
cja_auto_sdr -p client-b "Main Data View" --format excel

# Test profile connectivity
cja_auto_sdr --profile-test client-a

# Set default profile
export CJA_PROFILE=client-a
cja_auto_sdr --list-dataviews  # Uses client-a
```

## Common Options

| Option | Purpose | Mode |
|--------|---------|------|
| `--profile NAME`, `-p` | Use named profile from `~/.cja/orgs/` | Both |
| `--output-dir PATH` | Save output to specific directory | Both |
| `--output PATH` | Output file path; use `-` for stdout (JSON/CSV) | Both |
| `--format FORMAT` | Output format (see note below) | Both |
| `--open` | Open generated file(s) in default application | SDR |
| `--stats` | Quick statistics only (no full report) | SDR |
| `--interactive`, `-i` | Interactively select data views from a numbered list | Both |
| `--config-file PATH` | Use custom config file (default: config.json) | Both |
| `--log-level LEVEL` | Set logging: `DEBUG`, `INFO`, `WARNING`, `ERROR` | Both |
| `--log-format FORMAT` | Log output: `text` (default) or `json` (structured) | Both |
| `--workers N` | Parallel workers: `auto` (default) or `1-256` | SDR only |
| `--skip-validation` | Skip data quality checks (faster) | SDR only |
| `--continue-on-error` | Don't stop on failures in batch mode | SDR only |
| `--include-segments` | Add segments inventory sheet/section | SDR + Snapshot Diff |
| `--include-derived` | Add derived field inventory sheet/section | SDR only |
| `--include-calculated` | Add calculated metrics inventory sheet/section | SDR + Snapshot Diff |
| `--include-all-inventory` | Enable all inventory options (smart mode detection) | SDR + Snapshot Diff |
| `--inventory-only` | Output only inventory sheets (requires `--include-*`) | SDR only |
| `--inventory-summary` | Quick stats without full output (requires `--include-*`) | SDR only |

> **Note:** `--include-derived` is for SDR generation only. Derived fields are already included in the standard metrics/dimensions output, so changes are captured in the Metrics/Dimensions diff.

### Diff-Specific Options

| Option | Purpose |
|--------|---------|
| `--changes-only` | Hide unchanged components, show only differences |
| `--compare-with-prev` | Compare against most recent snapshot in --snapshot-dir |
| `--diff-labels A B` | Custom labels for comparison columns (default: data view names) |
| `--auto-snapshot` | Automatically save snapshots during diff for future comparisons |
| `--warn-threshold PERCENT` | Exit with code 3 if change % exceeds threshold (for CI/CD) |
| `--no-color` | Disable ANSI color codes in console output |
| `--format-pr-comment` | Output in GitHub/GitLab PR comment format |

### Format Support by Mode

| Format | SDR | Diff | Org-Report | Description |
|--------|-----|------|------------|-------------|
| `excel` | ✅ (default) | ✅ | ✅ | Excel workbook |
| `csv` | ✅ | ✅ | ✅ | Comma-separated values |
| `json` | ✅ | ✅ | ✅ | JSON for integrations |
| `html` | ✅ | ✅ | ✅ | Browser-viewable |
| `markdown` | ✅ | ✅ | ✅ | Documentation-ready |
| `console` | ❌ | ✅ (default) | ✅ (default) | Terminal output |
| `all` | ✅ | ✅ | ✅ | All formats |

### Format Aliases (Shortcuts)

| Alias | Generates | Use Case |
|-------|-----------|----------|
| `reports` | excel + markdown | Documentation and sharing |
| `data` | csv + json | Data pipelines and integrations |
| `ci` | json + markdown | CI/CD logs and PR comments |

## Quick Recipes

```bash
# Fast processing (skip validation)
cja_auto_sdr dv_12345 --skip-validation

# Generate and open immediately
cja_auto_sdr dv_12345 --open

# Quick stats check before full generation
cja_auto_sdr dv_12345 --stats

# Stats as JSON to stdout (for scripting)
cja_auto_sdr dv_12345 --stats --output -

# List data views and pipe to jq
cja_auto_sdr --list-dataviews --output - | jq '.dataViews[].name'

# All output formats
cja_auto_sdr dv_12345 --format all

# Debug mode (verbose logging)
cja_auto_sdr dv_12345 --log-level DEBUG

# JSON logging (for Splunk, ELK, CloudWatch)
cja_auto_sdr dv_12345 --log-format json

# Dry run (validate only, no output)
cja_auto_sdr dv_12345 --dry-run

# Batch with custom parallelism (default: auto-detect)
cja_auto_sdr dv_* --workers 8 --continue-on-error

# Production mode (minimal logging)
cja_auto_sdr dv_12345 --production

# Custom output directory (macOS/Linux)
cja_auto_sdr dv_12345 --output-dir ./reports/$(date +%Y%m%d)

# Custom output directory (Windows PowerShell)
cja_auto_sdr dv_12345 --output-dir ./reports/$(Get-Date -Format "yyyyMMdd")

# Include segments inventory
cja_auto_sdr dv_12345 --include-segments

# Include derived field inventory
cja_auto_sdr dv_12345 --include-derived

# Include calculated metrics inventory
cja_auto_sdr dv_12345 --include-calculated

# All three inventories (longhand)
cja_auto_sdr dv_12345 --include-segments --include-derived --include-calculated

# All inventories (shorthand - same as above)
cja_auto_sdr dv_12345 --include-all-inventory

# All inventories with inventory-only mode
cja_auto_sdr dv_12345 --include-all-inventory --inventory-only

# All inventories with quick summary
cja_auto_sdr dv_12345 --include-all-inventory --inventory-summary

# Inventory-only mode (no standard SDR sheets)
cja_auto_sdr dv_12345 --include-segments --inventory-only

# Multiple inventories only
cja_auto_sdr dv_12345 --include-segments --include-calculated --inventory-only

# Quick inventory stats (no file output)
cja_auto_sdr dv_12345 --include-segments --include-calculated --inventory-summary

# Save summary to JSON
cja_auto_sdr dv_12345 --include-segments --inventory-summary --format json

# --- Inventory Diff (same data view over time) ---

# Create snapshot with inventory
cja_auto_sdr dv_12345 --snapshot ./baseline.json --include-calculated --include-segments

# Compare against snapshot with inventory
cja_auto_sdr dv_12345 --diff-snapshot ./baseline.json --include-calculated --include-segments
```

## Environment Variables

```bash
# Credentials (override config file)
export ORG_ID=your_org_id@AdobeOrg
export CLIENT_ID=your_client_id
export SECRET=your_client_secret
export SCOPES="your_scopes_from_developer_console"

# Optional settings
export OUTPUT_DIR=./reports
export LOG_LEVEL=INFO
```

## Setup Commands

```bash
# Generate sample config file
cja_auto_sdr --sample-config

# Validate configuration and API connection
cja_auto_sdr --validate-config

# Test specific data views without generating
cja_auto_sdr dv_12345 --dry-run
```

See [CONFIGURATION.md](CONFIGURATION.md) for detailed setup of `config.json` and environment variables.

## Output Files

| Format | File Pattern | Description |
|--------|--------------|-------------|
| Excel | `SDR_<name>_<date>.xlsx` | Full report with Data Quality sheet |
| CSV | `SDR_<name>_<date>.csv` | Flat component list |
| JSON | `SDR_<name>_<date>.json` | Machine-readable format |
| HTML | `SDR_<name>_<date>.html` | Browser-viewable report |
| Markdown | `SDR_<name>_<date>.md` | Documentation-ready format |

**Excel Sheet Order:**
1. Metadata
2. Data Quality
3. DataView
4. Metrics
5. Dimensions
6. Segments (if `--include-segments`)
7. Derived Fields (if `--include-derived`)
8. Calculated Metrics (if `--include-calculated`)

> **Note:** Inventory sheets appear at the end in CLI argument order. With `--inventory-only`, only sheets 6-8 are generated.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (diff: no changes found) |
| 1 | Error (config, API, or processing failure) |
| 2 | Diff: changes found |
| 3 | Diff: changes exceeded threshold |
| 10 | Org-report: governance threshold exceeded (any) |
| 11 | Org-report: duplicate threshold exceeded |
| 12 | Org-report: isolated threshold exceeded |
| 13 | Org-report: naming audit failed |

> **CI/CD Tip:** Use exit codes with `--fail-on-threshold` for automated governance checks.

## More Information

- Configuration: [CONFIGURATION.md](CONFIGURATION.md) - config.json, environment variables
- Full CLI docs: [CLI_REFERENCE.md](CLI_REFERENCE.md)
- Diff comparison: [DIFF_COMPARISON.md](DIFF_COMPARISON.md)
- Org-wide analysis: [ORG_WIDE_ANALYSIS.md](ORG_WIDE_ANALYSIS.md) - Cross-DV governance
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Batch processing: [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md)
- Data quality: [DATA_QUALITY.md](DATA_QUALITY.md)
- Derived fields: [DERIVED_FIELDS_INVENTORY.md](DERIVED_FIELDS_INVENTORY.md)
- Calculated metrics: [CALCULATED_METRICS_INVENTORY.md](CALCULATED_METRICS_INVENTORY.md)
