# Troubleshooting Guide

Comprehensive solutions for issues with the CJA SDR Generator.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Exit Codes Reference](#exit-codes-reference)
- [Configuration Errors](#configuration-errors)
- [Authentication & Connection Errors](#authentication--connection-errors)
- [Data View Errors](#data-view-errors)
- [API & Network Errors](#api--network-errors)
- [Retry Mechanism & Rate Limiting](#retry-mechanism--rate-limiting)
- [Data Quality Issues](#data-quality-issues)
- [Output & File Errors](#output--file-errors)
- [Batch Processing Issues](#batch-processing-issues)
- [Validation Cache Issues](#validation-cache-issues)
- [Performance Issues](#performance-issues)
- [Dependency Issues](#dependency-issues)
- [Debug Mode & Logging](#debug-mode--logging)
- [Common Error Messages Reference](#common-error-messages-reference)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Run this script to gather system information:

```bash
#!/bin/bash
echo "=== System Information ==="
python --version
uv --version
echo ""
echo "=== Project Dependencies ==="
uv pip list | grep -E "cjapy|pandas|openpyxl"
echo ""
echo "=== Configuration Check ==="
if [ -f myconfig.json ]; then
    echo "myconfig.json exists"
    python -c "import json; json.load(open('myconfig.json')); print('JSON syntax: valid')" 2>&1 || echo "JSON syntax: INVALID"
else
    echo "myconfig.json NOT FOUND"
fi
echo ""
echo "=== Environment Variables ==="
[ -n "$ORG_ID" ] && echo "ORG_ID: set" || echo "ORG_ID: not set"
[ -n "$CLIENT_ID" ] && echo "CLIENT_ID: set" || echo "CLIENT_ID: not set"
[ -n "$SECRET" ] && echo "SECRET: set" || echo "SECRET: not set"
[ -n "$SCOPES" ] && echo "SCOPES: set" || echo "SCOPES: not set"
echo ""
echo "=== Recent Logs ==="
ls -lh logs/ 2>/dev/null | tail -5 || echo "No logs directory"
```

Save as `diagnose.sh` and run:
```bash
chmod +x diagnose.sh
./diagnose.sh > diagnostic_report.txt
```

### Quick Validation Commands

```bash
# Validate configuration without processing
uv run cja_auto_sdr --validate-config

# Test with dry run (validates but doesn't generate output)
uv run cja_auto_sdr dv_12345 --dry-run

# List accessible data views
uv run cja_auto_sdr --list-dataviews

# Generate sample config
uv run cja_auto_sdr --sample-config
```

---

## Exit Codes Reference

| Exit Code | Meaning | Common Causes |
|-----------|---------|---------------|
| `0` | Success | Command completed successfully |
| `1` | General Error | Configuration errors, missing arguments, validation failures |
| `1` | Missing Data View | No data view ID provided (except special modes) |
| `1` | Invalid Arguments | Invalid worker count, cache settings, or retry parameters |

---

## Configuration Errors

### Configuration File Not Found

**Symptoms:**
```
CRITICAL - Configuration file not found: myconfig.json
FileNotFoundError: Config file not found: myconfig.json
```

**Solutions:**
1. Generate a sample configuration:
   ```bash
   uv run cja_auto_sdr --sample-config
   ```
2. Or use environment variables (see [Environment Variable Configuration](#environment-variable-configuration))

### Invalid JSON Syntax

**Symptoms:**
```
CRITICAL - Configuration file is not valid JSON: Expecting ',' delimiter: line 5 column 3
```

**Solutions:**
1. Validate JSON syntax:
   ```bash
   python -c "import json; json.load(open('myconfig.json'))"
   ```
2. Common JSON issues:
   - Missing commas between fields
   - Trailing commas after last field (not allowed in JSON)
   - Unquoted strings
   - Single quotes instead of double quotes

### Missing Required Fields

**Symptoms:**
```
CRITICAL - Missing required field: 'org_id'
CRITICAL - Empty value for required field: 'client_id'
```

**Required fields in myconfig.json:**
| Field | Type | Description |
|-------|------|-------------|
| `org_id` | string | Adobe Organization ID (ends with @AdobeOrg) |
| `client_id` | string | OAuth Client ID from Adobe Developer Console |
| `secret` | string | Client Secret from Adobe Developer Console |

**Optional fields:**
| Field | Type | Description |
|-------|------|-------------|
| `scopes` | string | OAuth scopes (recommended) |
| `sandbox` | string | Sandbox name |

**Solution:**
```json
{
  "org_id": "YOUR_ORG_ID@AdobeOrg",
  "client_id": "YOUR_CLIENT_ID",
  "secret": "YOUR_CLIENT_SECRET",
  "scopes": "openid, AdobeID, additional_info.projectedProductContext"
}
```

### Unknown Fields Warning

**Symptoms:**
```
WARNING - Unknown fields in config (possible typos): ['orgid', 'clientid']
```

**Common typos:**
| Wrong | Correct |
|-------|---------|
| `orgid` | `org_id` |
| `clientid` | `client_id` |
| `client_secret` | `secret` |
| `scope` | `scopes` |

### Configuration File Permission Error

**Symptoms:**
```
PermissionError: Cannot read configuration file: Permission denied
```

**Solutions:**
```bash
# Check permissions
ls -la myconfig.json

# Fix permissions
chmod 600 myconfig.json
```

### Environment Variable Configuration

**Supported environment variables:**

| Environment Variable | Maps To | Required |
|---------------------|---------|----------|
| `ORG_ID` | org_id | Yes |
| `CLIENT_ID` | client_id | Yes |
| `SECRET` | secret | Yes |
| `SCOPES` | scopes | No (recommended) |
| `SANDBOX` | sandbox | No |
| `LOG_LEVEL` | log_level | No (default: INFO) |
| `OUTPUT_DIR` | output_dir | No (default: current dir) |

**Symptoms when missing:**
```
ERROR - Missing required environment variable: ORG_ID
WARNING - Environment credentials missing OAuth scopes - recommend setting SCOPES
```

**Solutions:**

Option 1: Export directly
```bash
export ORG_ID=your_org_id@AdobeOrg
export CLIENT_ID=your_client_id
export SECRET=your_client_secret
export SCOPES='openid, AdobeID, additional_info.projectedProductContext'
```

Option 2: Use .env file (copy from .env.example)
```bash
cp .env.example .env
# Edit .env with your values
```

> **Note:** Environment variables take precedence over myconfig.json

---

## Authentication & Connection Errors

### CJA Initialization Failed

**Symptoms:**
```
CRITICAL - CJA INITIALIZATION FAILED
CRITICAL - Failed to initialize CJA connection
Configuration error: Authentication failed
```

**Troubleshooting steps displayed:**
```
1. Verify your configuration file exists and is valid JSON
2. Check that all authentication credentials are correct
3. Ensure your API credentials have the necessary permissions
4. Verify you have network connectivity to Adobe services
5. Check if cjapy library is up to date: pip install --upgrade cjapy
```

**Solutions:**
1. Verify credentials match Adobe Developer Console exactly
2. Ensure the integration has CJA API enabled
3. Check network connectivity:
   ```bash
   ping adobe.io
   curl -I https://analytics.adobe.io
   ```
4. Upgrade cjapy:
   ```bash
   uv add --upgrade cjapy
   ```

### API Connection Test Failed

**Symptoms:**
```
WARNING - API connection test returned None
WARNING - Could not verify connection with test call
```

**Causes:**
- Network issues
- Invalid credentials
- API permissions not configured

**Solutions:**
1. Run with debug logging:
   ```bash
   uv run cja_auto_sdr dv_12345 --log-level DEBUG
   ```
2. Check Adobe Status: https://status.adobe.com
3. Verify product profile includes CJA access

### Import Error for cjapy

**Symptoms:**
```
CRITICAL - Failed to import cjapy module: No module named 'cjapy'
ImportError: cjapy not found
```

**Solutions:**
```bash
# Install cjapy
uv add cjapy

# Or upgrade existing
uv add --upgrade cjapy

# Verify installation
uv pip list | grep cjapy
```

---

## Data View Errors

### Data View Not Found

**Symptoms:**
```
ERROR - Data view 'dv_12345' returned empty response
ERROR - Data view returned empty response
INFO - You have access to 5 data view(s):
INFO -   1. Production Analytics (ID: dv_abc123)
```

**Solutions:**
1. List available data views:
   ```bash
   uv run cja_auto_sdr --list-dataviews
   ```
2. Copy the exact ID from the output
3. Verify you have access permissions in CJA

### Invalid Data View ID Format

**Symptoms:**
```
ERROR: Invalid data view ID format: invalid_id, test123
WARNING - Data view ID does not follow standard format (dv_...)
```

**Requirements:**
- Data view IDs must start with `dv_`
- Must be non-empty strings

**Solutions:**
```bash
# Wrong
uv run cja_auto_sdr 12345
uv run cja_auto_sdr invalid_id

# Correct
uv run cja_auto_sdr dv_12345
```

### No Access to Data Views

**Symptoms:**
```
WARNING - No data views found - no access to any data views
ERROR - Could not list available data views
```

**Solutions:**
1. Verify API credentials have CJA read permissions
2. Check product profile in Adobe Admin Console
3. Contact your Adobe administrator

### API Method Not Available

**Symptoms:**
```
ERROR - API method 'getDataView' not available
AttributeError: API method error - getMetrics may not be available
```

**Cause:** Outdated cjapy library

**Solution:**
```bash
uv add --upgrade cjapy
```

---

## API & Network Errors

### HTTP Status Code Errors

**Retryable errors (automatic retry):**

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 408 | Request Timeout | Auto-retry with backoff |
| 429 | Too Many Requests | Auto-retry with backoff (rate limited) |
| 500 | Internal Server Error | Auto-retry with backoff |
| 502 | Bad Gateway | Auto-retry with backoff |
| 503 | Service Unavailable | Auto-retry with backoff |
| 504 | Gateway Timeout | Auto-retry with backoff |

**Non-retryable errors:**

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Check credentials |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Verify data view ID |

### Connection and Timeout Errors

**Symptoms:**
```
ConnectionError: Failed to connect to Adobe API
TimeoutError: Request timed out
OSError: Network unreachable
```

**These are automatically retried.** If all retries fail:

```
ERROR - All 3 attempts failed for fetch_metrics: Connection timed out
```

**Solutions:**
1. Check network connectivity
2. Increase retry parameters:
   ```bash
   uv run cja_auto_sdr dv_12345 --max-retries 5 --retry-base-delay 2.0 --retry-max-delay 60.0
   ```

---

## Retry Mechanism & Rate Limiting

### Understanding the Retry System

The tool automatically retries failed API calls with exponential backoff:

**Default retry configuration:**
| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Max retries | 3 | `--max-retries` |
| Base delay | 1.0s | `--retry-base-delay` |
| Max delay | 30.0s | `--retry-max-delay` |

**Backoff formula:**
```
delay = min(base_delay * (2 ^ attempt), max_delay) * random(0.5, 1.5)
```

**Example progression:**
- Attempt 1 fails → wait ~1s (0.5-1.5s with jitter)
- Attempt 2 fails → wait ~2s (1-3s with jitter)
- Attempt 3 fails → wait ~4s (2-6s with jitter)
- All attempts exhausted → error raised

### Retry Log Messages

**During retries:**
```
WARNING - fetch_metrics attempt 1/3 failed: Connection timed out. Retrying in 1.2s...
WARNING - fetch_metrics attempt 2/3 failed: HTTP 503. Retrying in 2.8s...
```

**After successful retry:**
```
INFO - fetch_metrics succeeded on attempt 3/3
```

**All retries failed:**
```
ERROR - All 3 attempts failed for fetch_metrics: HTTP 503: Service Unavailable
```

### Rate Limiting (HTTP 429)

**Symptoms:**
```
WARNING - fetch_metrics attempt 1/3 failed: HTTP 429. Retrying in 1.5s...
```

**Solutions:**
1. Reduce parallel workers in batch mode:
   ```bash
   uv run cja_auto_sdr dv_1 dv_2 dv_3 --workers 2
   ```
2. Increase delays:
   ```bash
   uv run cja_auto_sdr dv_12345 --retry-base-delay 2.0 --retry-max-delay 60.0
   ```

### Customizing Retry Behavior

```bash
# More aggressive retrying for flaky networks
uv run cja_auto_sdr dv_12345 --max-retries 5 --retry-base-delay 2.0 --retry-max-delay 120.0

# Minimal retries for fast-fail scenarios
uv run cja_auto_sdr dv_12345 --max-retries 1

# No retries (fail immediately)
uv run cja_auto_sdr dv_12345 --max-retries 0
```

---

## Data Quality Issues

### No Metrics or Dimensions

**Symptoms:**
```
ERROR - No metrics or dimensions fetched. Cannot generate SDR.
WARNING - No metrics returned from API
WARNING - No dimensions returned from API
```

**Causes:**
- Data view has no components configured
- API permissions don't include component read access
- Data view is newly created and empty

**Solutions:**
1. Verify data view has components in CJA UI
2. Check API permissions include read access
3. Run dry-run to validate:
   ```bash
   uv run cja_auto_sdr dv_12345 --dry-run
   ```

### Data Quality Validation Issues

**Severity levels:**

| Severity | Color | Meaning |
|----------|-------|---------|
| CRITICAL | Red | Blocking issues (empty data, missing required fields) |
| HIGH | Orange | Serious issues (missing IDs, duplicates) |
| MEDIUM | Yellow | Notable issues (null values in critical fields) |
| LOW | Blue | Minor issues (missing descriptions) |
| INFO | Gray | Informational |

**Common validation issues:**

| Issue | Severity | Message |
|-------|----------|---------|
| Empty data | CRITICAL | "No {item_type} found in data view" |
| Missing fields | CRITICAL | "Missing required fields: {fields}" |
| Invalid IDs | HIGH | "{count} items with missing IDs" |
| Duplicates | HIGH | "Item '{name}' appears {count} times" |
| Null values | MEDIUM | "{count} items missing {field}" |
| No descriptions | LOW | "{count} items without descriptions" |

**Skip validation if not needed:**
```bash
# 20-30% faster processing
uv run cja_auto_sdr dv_12345 --skip-validation
```

---

## Output & File Errors

### Permission Denied Writing Output

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'SDR_Analytics_2024-01-15.xlsx'
ERROR - Permission denied writing to SDR_Analytics_2024-01-15.xlsx
```

**Causes:**
- File is open in Excel or another program
- Insufficient write permissions to directory

**Solutions:**
1. Close the Excel file
2. Specify a different output directory:
   ```bash
   uv run cja_auto_sdr dv_12345 --output-dir ./reports
   ```

### Output Directory Does Not Exist

**Symptoms:**
```
ERROR - Permission denied creating output directory: /nonexistent/path
ERROR - Cannot create output directory '/path/to/dir': Permission denied
```

**Solutions:**
```bash
# Create directory first
mkdir -p ./reports

# Then run
uv run cja_auto_sdr dv_12345 --output-dir ./reports
```

### Excel Generation Failures

**Symptoms:**
```
ERROR - Failed to generate Excel file: {error}
ERROR - Error formatting JSON cell: {error}
```

**Solutions:**
1. Try a different format:
   ```bash
   uv run cja_auto_sdr dv_12345 --format csv
   uv run cja_auto_sdr dv_12345 --format json
   ```
2. Check disk space:
   ```bash
   df -h .
   ```

### Empty Output File

**Possible Causes:**
- Data view has no components configured
- API permissions don't include read access

**Solutions:**
1. Check log file for "No metrics returned from API"
2. Verify data view has components in CJA UI
3. Run with debug logging:
   ```bash
   uv run cja_auto_sdr dv_12345 --log-level DEBUG
   ```

---

## Batch Processing Issues

### Batch Initialization Errors

**Symptoms:**
```
CRITICAL - Permission denied creating output directory: ./reports
CRITICAL - Cannot create output directory './reports': {error}
```

**Solution:** Ensure you have write permissions to the output directory.

### Individual Data View Failures

**Symptoms:**
```
[batch_abc123] dv_12345: FAILED - Data view not found
[batch_abc123] dv_67890: EXCEPTION - Connection timeout
```

**Continue processing despite failures:**
```bash
uv run cja_auto_sdr dv_1 dv_2 dv_3 --continue-on-error
```

### Batch Processing Slower Than Expected

**Causes:**
- Too many workers causing rate limiting
- Network bottleneck
- Large data views

**Solutions:**
```bash
# Reduce workers (default: 4)
uv run cja_auto_sdr dv_1 dv_2 dv_3 --workers 2

# Check logs for rate limiting
grep "429\|rate limit" logs/*.log
```

### Worker Count Validation Errors

**Symptoms:**
```
ERROR: --workers must be at least 1
ERROR: --workers cannot exceed 256
```

**Valid range:** 1-256 workers

---

## Validation Cache Issues

### Understanding the Cache

The validation cache stores data quality check results to avoid redundant processing:

**Cache parameters:**
| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Enable cache | Off | `--enable-cache` |
| Cache size | 1000 entries | `--cache-size` |
| Cache TTL | 3600s (1 hour) | `--cache-ttl` |

### Cache Log Messages

**Debug-level cache messages:**
```
DEBUG - Cache HIT: metrics (age: 45s)
DEBUG - Cache MISS: dimensions
DEBUG - Cache EXPIRED: metrics (age: 3700s)
```

### Cache Issues

**Cache not helping performance:**
- TTL too short
- Cache size too small
- Data changing frequently

**Solutions:**
```bash
# Increase cache TTL for stable data
uv run cja_auto_sdr dv_12345 --enable-cache --cache-ttl 7200

# Increase cache size for many data views
uv run cja_auto_sdr dv_1 dv_2 dv_3 --enable-cache --cache-size 5000
```

**Clear cache before processing:**
```bash
uv run cja_auto_sdr dv_12345 --enable-cache --clear-cache
```

### Cache Parameter Validation Errors

**Symptoms:**
```
ERROR: --cache-size must be at least 1
ERROR: --cache-ttl must be at least 1 second
```

---

## Performance Issues

### Normal Processing Times

| Data View Size | Expected Time |
|----------------|---------------|
| Small (<50 components) | 15-30 seconds |
| Medium (50-200 components) | 30-60 seconds |
| Large (200+ components) | 60-120 seconds |

### Slow Processing Solutions

```bash
# Skip validation (20-30% faster)
uv run cja_auto_sdr dv_12345 --skip-validation

# Use production mode (reduces logging)
uv run cja_auto_sdr dv_12345 --production

# Enable caching for repeated runs (50-90% faster on cache hits)
uv run cja_auto_sdr dv_12345 --enable-cache

# Use quiet mode (minimal output)
uv run cja_auto_sdr dv_12345 --quiet
```

### Batch Processing Performance

```bash
# Optimal batch processing
uv run cja_auto_sdr dv_1 dv_2 dv_3 --workers 4 --enable-cache

# Balance speed vs. rate limiting
uv run cja_auto_sdr dv_1 dv_2 dv_3 --workers 2 --retry-base-delay 1.5
```

---

## Dependency Issues

### Module Not Found

**Symptoms:**
```
ModuleNotFoundError: No module named 'cjapy'
ModuleNotFoundError: No module named 'pandas'
ModuleNotFoundError: No module named 'openpyxl'
```

**Solutions:**
```bash
# Sync all dependencies
uv sync

# Or reinstall everything
uv sync --reinstall

# Verify installation
uv pip list | grep -E "cjapy|pandas|openpyxl"
```

### Version Conflicts

**Solutions:**
```bash
# Check for conflicts
uv pip check

# Update specific package
uv add --upgrade cjapy

# Regenerate lock file
rm uv.lock
uv sync
```

### uv Command Not Found

**Solutions:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version

# Then sync project
uv sync
```

### Wrong Python Version

**Solution:**
```bash
# Remove and recreate venv
rm -rf .venv
uv venv --python 3.14
uv sync
```

---

## Debug Mode & Logging

### Log Levels

| Level | Flag | Description |
|-------|------|-------------|
| DEBUG | `--log-level DEBUG` | Detailed operation tracking |
| INFO | (default) | General progress information |
| WARNING | `--production` | Important notices only |
| ERROR | `--quiet` | Errors only |

### Enabling Debug Mode

```bash
# Maximum verbosity
uv run cja_auto_sdr dv_12345 --log-level DEBUG
```

**Debug mode shows:**
- Cache operations (hits, misses, expirations)
- Individual validation checks
- API call details
- Performance timing

### Log File Locations

| Mode | Log File Pattern |
|------|------------------|
| Single data view | `logs/SDR_Generation_dv_{id}_{timestamp}.log` |
| Batch processing | `logs/SDR_Batch_Generation_{timestamp}.log` |

### Searching Logs

```bash
# Find all errors
grep -i "error\|critical" logs/*.log

# Find warnings
grep -i warning logs/*.log

# Find specific data view
grep "dv_12345" logs/*.log

# Find rate limiting issues
grep -i "429\|rate limit\|retry" logs/*.log

# View latest log
cat logs/$(ls -t logs/ | head -1)
```

---

## Common Error Messages Reference

### Configuration Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Configuration file not found: {path}` | Config file missing | Run `--sample-config` |
| `Configuration file is not valid JSON: {error}` | Invalid JSON syntax | Check JSON formatting |
| `Configuration file must contain a JSON object` | Config is array, not object | Wrap in `{}` |
| `Missing required field: '{field}'` | Required field absent | Add field to config |
| `Empty value for required field: '{field}'` | Field is empty/whitespace | Provide value |
| `Invalid type for '{field}': expected {type}` | Wrong data type | Fix field type |
| `Unknown fields in config (possible typos)` | Unexpected fields | Check field names |

### API Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `API method 'getDataView' not available` | Outdated cjapy | Upgrade cjapy |
| `API call failed: {error}` | General API error | Check logs |
| `All {N} attempts failed for {operation}` | Retries exhausted | Check network/credentials |
| `HTTP 429: Too Many Requests` | Rate limited | Reduce workers, increase delays |

### Data View Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Invalid data view ID format` | ID doesn't start with `dv_` | Use correct format |
| `Data view returned empty response` | Not found or no access | Use `--list-dataviews` |
| `No data views found` | No access to any data views | Check permissions |

### Output Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Permission denied writing to {path}` | File locked or no permissions | Close file, check permissions |
| `No metrics or dimensions fetched` | Empty data view | Check CJA configuration |
| `Cannot create output directory` | Permission issue | Create directory manually |

### CLI Argument Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `At least one data view ID is required` | No data view provided | Add data view ID |
| `--workers must be at least 1` | Invalid worker count | Use 1-256 |
| `--cache-size must be at least 1` | Invalid cache size | Use positive integer |
| `--cache-ttl must be at least 1 second` | Invalid TTL | Use positive integer |
| `--max-retries cannot be negative` | Invalid retry count | Use 0 or positive |
| `--retry-max-delay must be >= --retry-base-delay` | Invalid delay config | Ensure max >= base |

---

## Getting Help

If you encounter issues not covered here:

1. **Enable debug logging:**
   ```bash
   uv run cja_auto_sdr dv_12345 --log-level DEBUG
   ```

2. **Check the log file** in `logs/` directory

3. **Run diagnostics:**
   ```bash
   ./diagnose.sh > diagnostic_report.txt
   ```

4. **Validate configuration:**
   ```bash
   uv run cja_auto_sdr --validate-config
   ```

5. **When reporting issues, include:**
   - Complete error message
   - Relevant log entries (anonymize credentials)
   - Python version: `python --version`
   - uv version: `uv --version`
   - cjapy version: `uv pip show cjapy`

---

## See Also

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [CLI Reference](CLI_REFERENCE.md) - Complete command options
- [Performance Guide](PERFORMANCE.md) - Optimization options
- [Data Quality Guide](DATA_QUALITY.md) - Understanding validation
- [Batch Processing Guide](BATCH_PROCESSING_GUIDE.md) - Multi-data view processing
