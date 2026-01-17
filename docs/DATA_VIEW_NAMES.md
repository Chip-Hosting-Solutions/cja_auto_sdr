# Using Data View Names

You can now specify data views by their **name** instead of just their ID.

## Overview

Instead of Data View IDs like `dv_677ea9291244fd082f02dd42`, you can now use the Data View name directly:

```bash
# Before (still works)
cja_auto_sdr dv_677ea9291244fd082f02dd42

# Now (new feature)
cja_auto_sdr "Production Analytics"
```

## Benefits

- **Easier to remember** - Names are more meaningful than long IDs
- **More readable** - Scripts and documentation are clearer
- **Flexible** - Mix IDs and names in the same command

## How It Works

When you provide a data view name:

1. The tool connects to CJA and fetches all accessible data views
2. It searches for data views with an **exact name match** (case-sensitive)
3. If one data view matches, it's processed
4. If multiple data views match, **all are processed**
5. If no data views match, an error is shown

## Usage Examples

### Single Data View by Name

```bash
# macOS/Linux
uv run cja_auto_sdr "Production Analytics"

# Windows
python cja_sdr_generator.py "Production Analytics"
```

### Multiple Data Views by Name

```bash
cja_auto_sdr "Production" "Staging" "Test Environment"
```

### Mix IDs and Names

```bash
cja_auto_sdr dv_12345 "Production Analytics" dv_67890 "Test"
```

### Names with Special Characters

```bash
# Use quotes for names with spaces or special characters
cja_auto_sdr "Production - North America"
cja_auto_sdr "Test (v2.0)"
```

## Handling Duplicate Names

If multiple data views share the same name, all matching views will be processed:

```bash
$ cja_auto_sdr "Production"

Resolving 1 data view name(s)...
INFO - Name 'Production' matched 3 data views: ['dv_prod001', 'dv_prod002', 'dv_prod003']

Data view name resolution:
  ✓ 'Production' → 3 matching data views:
      - dv_prod001
      - dv_prod002
      - dv_prod003

Processing 3 data view(s) total...
```

This is useful when you have multiple environments with the same name (e.g., different sandboxes) and want to process all of them at once.

## Listing Available Names

To see all accessible data view names and IDs:

```bash
# macOS/Linux
uv run cja_auto_sdr --list-dataviews

# Windows
python cja_sdr_generator.py --list-dataviews
```

Output:
```
Found 5 accessible data view(s):

ID                                            Name                                     Owner
----------------------------------------------------------------------------------------------------
dv_677ea9291244fd082f02dd42                   Production Analytics                     admin@company.com
dv_789bcd123456ef7890ab                       Test Environment                         admin@company.com
dv_abc123def456789                            Staging                                  admin@company.com
```

## Important Notes

### Case Sensitivity

Name matching is **case-sensitive**. These are different:
- `Production Analytics` ✓
- `production analytics` ✗ (will not match)
- `PRODUCTION ANALYTICS` ✗ (will not match)

### Exact Match Required

Names must match **exactly**:
- `Production Analytics` ✓
- `Production` ✗ (will not match "Production Analytics")
- `Analytics` ✗ (will not match "Production Analytics")

### Quotes Recommended

Always use quotes around names to avoid shell interpretation issues:

```bash
# Good
cja_auto_sdr "Production Analytics"

# May fail if name has spaces
cja_auto_sdr Production Analytics  # Shell interprets as two separate arguments
```

## Error Handling

### Name Not Found

```bash
$ cja_auto_sdr "NonexistentView"

ERROR: No valid data views found

Possible issues:
  - Data view ID(s) or name(s) not found or you don't have access
  - Data view name is not an EXACT match (names are case-sensitive)
  - Configuration issue preventing data view lookup

Tips for using Data View Names:
  • Names must match EXACTLY: 'Production Analytics' ≠ 'production analytics'
  • Use quotes around names: cja_auto_sdr "Production Analytics"
  • IDs start with 'dv_': cja_auto_sdr dv_12345

Try running: python cja_sdr_generator.py --list-dataviews
  to see all accessible data view IDs and names
```

**Common Causes & Solutions:**

1. **Case Mismatch** - Names are case-sensitive
   - ❌ Wrong: `cja_auto_sdr "production analytics"`
   - ✅ Right: `cja_auto_sdr "Production Analytics"`

2. **Partial Name** - Must match the complete name
   - ❌ Wrong: `cja_auto_sdr "Production"` (when name is "Production Analytics")
   - ✅ Right: `cja_auto_sdr "Production Analytics"`

3. **Missing Quotes** - Names with spaces need quotes
   - ❌ Wrong: `cja_auto_sdr Production Analytics` (shell interprets as two arguments)
   - ✅ Right: `cja_auto_sdr "Production Analytics"`

4. **Typos** - Verify spelling matches exactly
   - Run `--list-dataviews` to copy the exact name

### Case Sensitivity Errors

Data view names are **strictly case-sensitive**. Common mistakes:

```bash
# Example: Actual name is "Production Analytics"

# ❌ These will NOT work:
cja_auto_sdr "production analytics"     # lowercase
cja_auto_sdr "PRODUCTION ANALYTICS"     # uppercase
cja_auto_sdr "Production analytics"     # mixed case
cja_auto_sdr "production Analytics"     # mixed case

# ✅ This WILL work:
cja_auto_sdr "Production Analytics"     # exact match
```

**How to avoid:** Always copy the name exactly from `--list-dataviews` output.

### Partial Name Match Errors

Names must match **completely and exactly**:

```bash
# Example: Actual name is "Production Analytics - North America"

# ❌ These will NOT work:
cja_auto_sdr "Production"                           # partial match
cja_auto_sdr "Production Analytics"                 # missing suffix
cja_auto_sdr "Analytics"                            # partial match
cja_auto_sdr "North America"                        # partial match

# ✅ This WILL work:
cja_auto_sdr "Production Analytics - North America" # exact match
```

### Configuration Error

If the tool can't connect to CJA to resolve names, you'll see connection errors:

```bash
ERROR - Failed to resolve data view names: Authentication failed
```

**Solutions:**
1. Verify your `config.json` has correct credentials
2. Run `cja_auto_sdr --validate-config` to test the configuration
3. Check network connectivity to Adobe services
4. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#authentication--connection-errors) for detailed steps

### Access Permission Errors

If you don't have access to the data view:

```bash
WARNING - Data view name 'Restricted View' not found in accessible data views
ERROR: No valid data views found
```

**This means:**
- The data view exists, but your account doesn't have permission to access it
- The data view name is spelled correctly, but it's not in your accessible list

**Solutions:**
1. Run `--list-dataviews` to see which data views you can access
2. Contact your Adobe administrator to grant access
3. Verify you're using the correct Adobe organization credentials

## Batch Processing with Names

Names work seamlessly with batch processing:

```bash
# Process multiple environments by name
cja_auto_sdr "Production" "Staging" "Test" --workers 4

# Mix with IDs
cja_auto_sdr dv_prod123 "Staging Analytics" dv_test456 --batch
```

## Dry Run with Names

Test name resolution without generating reports:

```bash
cja_auto_sdr "Production Analytics" --dry-run
```

Output:
```
Resolving 1 data view name(s)...
INFO - Name 'Production Analytics' resolved to ID: dv_677ea9291244fd082f02dd42

Data view name resolution:
  ✓ 'Production Analytics' → dv_677ea9291244fd082f02dd42

Processing 1 data view(s) total...

============================================================
DRY RUN MODE - No files will be generated
============================================================
✓ Configuration valid
✓ API connection successful
✓ Data view "Production Analytics" found and accessible
✓ All pre-flight checks passed

Dry run complete. Remove --dry-run to generate the SDR.
```

## API Considerations

Name resolution requires an additional API call to fetch all data views. This adds minimal overhead:
- ~1-2 seconds for the initial lookup
- Results can be cached with `--enable-cache` for repeated runs

## Backward Compatibility

Using data view IDs still works exactly as before. This feature is purely additive:

```bash
# Old way - still works perfectly
cja_auto_sdr dv_677ea9291244fd082f02dd42

# New way - also works
cja_auto_sdr "Production Analytics"
```

## Use Cases

### Scheduled Reports

More readable cron jobs:

```bash
# Before
0 2 * * * cd /path/to/cja_auto_sdr && uv run cja_auto_sdr dv_677ea9291244fd082f02dd42

# After (more maintainable)
0 2 * * * cd /path/to/cja_auto_sdr && uv run cja_auto_sdr "Production Analytics"
```

### CI/CD Pipelines

Clearer pipeline configurations:

```yaml
# .github/workflows/sdr-generation.yml
jobs:
  generate:
    steps:
      - name: Generate SDRs
        run: |
          cja_auto_sdr "Production" "Staging"
```

### Documentation

More understandable scripts:

```bash
#!/bin/bash
# Generate SDRs for all environments
cja_auto_sdr "Production - North America" \
             "Production - Europe" \
             "Production - APAC"
```

## See Also

- [CLI Reference](CLI_REFERENCE.md) - Complete command-line options
- [Batch Processing Guide](BATCH_PROCESSING_GUIDE.md) - Processing multiple data views
- [Quick Start Guide](QUICKSTART_GUIDE.md) - Getting started
