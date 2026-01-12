# Adobe Customer Journey Analytics SDR Generator

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/54a43474-3fc6-4379-909c-452c19cdeac2" />

**Version 3.0.7** - Generate Solution Design Reference (SDR) documents from your CJA implementation with automated data quality validation.

## Quick Start

```bash
# 1. Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
cd cja-auto-sdr-2026 && uv sync

# 2. Configure credentials (see docs/INSTALLATION.md)
# Create myconfig.json with your Adobe I/O credentials

# 3. Run
uv run python cja_sdr_generator.py dv_YOUR_DATA_VIEW_ID

# Or batch process multiple data views
uv run python cja_sdr_generator.py dv_ID1 dv_ID2 dv_ID3
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/INSTALLATION.md) | Setup, authentication, configuration |
| [CLI Reference](docs/CLI_REFERENCE.md) | All command-line options and examples |
| [Data Quality](docs/DATA_QUALITY.md) | Validation checks and severity levels |
| [Performance](docs/PERFORMANCE.md) | Optimization, caching, batch processing |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common errors and solutions |
| [Use Cases](docs/USE_CASES.md) | Automation, scheduling, best practices |
| [Output Formats](docs/OUTPUT_FORMATS.md) | Excel, CSV, JSON, HTML specifications |
| [Changelog](CHANGELOG.md) | Version history |

## License

[MIT](LICENSE)
