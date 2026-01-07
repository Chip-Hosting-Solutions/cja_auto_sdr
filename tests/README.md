# CJA SDR Generator - Test Suite

This directory contains automated tests for the CJA SDR Generator.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_cli.py              # Command-line interface tests
├── test_data_quality.py     # Data quality validation tests
├── test_utils.py            # Utility function tests
└── README.md                # This file
```

## Running Tests

### Run All Tests

```bash
# Using uv (recommended)
uv run pytest

# Or with activated virtual environment
pytest
```

### Run Specific Test Files

```bash
# Test CLI functionality
uv run pytest tests/test_cli.py

# Test data quality validation
uv run pytest tests/test_data_quality.py

# Test utility functions
uv run pytest tests/test_utils.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
uv run pytest tests/test_cli.py::TestCLIArguments

# Run a specific test function
uv run pytest tests/test_cli.py::TestCLIArguments::test_parse_single_data_view
```

### Run Tests with Verbose Output

```bash
uv run pytest -v
```

### Run Tests with Coverage Report

```bash
# Install pytest-cov first
uv add --dev pytest-cov

# Run with coverage
uv run pytest --cov=cja_sdr_generator --cov-report=html --cov-report=term
```

## Test Categories

### Unit Tests (`test_utils.py`)
- **Logging setup**: Tests log file creation and configuration
- **Config validation**: Tests configuration file validation
- **Filename sanitization**: Tests filename helper functions
- **Performance tracking**: Tests performance measurement utilities

### CLI Tests (`test_cli.py`)
- **Argument parsing**: Tests command-line argument parsing
- **Data view validation**: Tests data view ID format validation
- **Error handling**: Tests error cases and edge conditions

### Data Quality Tests (`test_data_quality.py`)
- **Duplicate detection**: Tests detection of duplicate component names
- **Missing field detection**: Tests detection of missing required fields
- **Null value detection**: Tests detection of null values in critical fields
- **Severity classification**: Tests proper severity level assignment
- **Clean data handling**: Tests that clean data produces minimal issues

## Test Fixtures

Test fixtures are defined in `conftest.py`:

- **`mock_config_file`**: Creates a temporary mock configuration file
- **`mock_cja_instance`**: Provides a mocked CJA API instance
- **`sample_metrics_df`**: Sample metrics DataFrame for testing
- **`sample_dimensions_df`**: Sample dimensions DataFrame with test data
- **`temp_output_dir`**: Temporary directory for test outputs

## Writing New Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
def test_example_functionality():
    """Test description"""
    # Arrange
    input_data = "test_input"

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output
```

### Using Fixtures

```python
def test_with_fixture(mock_config_file):
    """Test using a fixture"""
    result = validate_config_file(mock_config_file)
    assert result is not None
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.14'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: uv run pytest
```

## Test Coverage Goals

- **Unit tests**: 80%+ coverage of utility functions
- **CLI tests**: 100% coverage of argument parsing
- **Data quality tests**: 90%+ coverage of validation logic
- **Integration tests**: Key workflows tested end-to-end

## Troubleshooting

### Tests Failing Locally

1. **Ensure dependencies are installed**:

   ```bash
   uv sync
   ```

2. **Check Python version**:

   ```bash
   python --version  # Should be 3.14+
   ```

3. **Clear pytest cache**:

   ```bash
   rm -rf .pytest_cache __pycache__ tests/__pycache__
   ```

### Import Errors

If you see import errors, ensure the project root is in Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv run pytest
```

## Best Practices

1. **Isolated tests**: Each test should be independent
2. **Mock external dependencies**: Use fixtures to mock API calls
3. **Clear assertions**: Use descriptive assertion messages
4. **Fast execution**: Keep unit tests fast (< 1 second each)
5. **Descriptive names**: Use clear, descriptive test function names

## Future Enhancements

- [ ] Add integration tests with actual CJA API (optional)
- [ ] Add performance benchmarking tests
- [ ] Add tests for Excel generation logic
- [ ] Add tests for batch processing functionality
- [ ] Increase test coverage to 90%+
