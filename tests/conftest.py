"""Pytest configuration and fixtures for CJA SDR Generator tests"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a temporary mock configuration file"""
    config_data = {
        "org_id": "test_org@AdobeOrg",
        "client_id": "test_client_id",
        "tech_id": "test_tech_id@techacct.adobe.com",
        "secret": "test_secret",
        "private_key": "test_private.key"
    }
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config_data))
    return str(config_file)


@pytest.fixture
def mock_cja_instance():
    """Create a mock CJA instance"""
    mock_cja = Mock()

    # Mock data views
    mock_cja.getDataViews.return_value = [
        {"id": "dv_test_12345", "name": "Test Data View 1"},
        {"id": "dv_test_67890", "name": "Test Data View 2"}
    ]

    # Mock single data view
    mock_cja.getDataView.return_value = {
        "id": "dv_test_12345",
        "name": "Test Data View 1",
        "owner": {"name": "Test Owner"}
    }

    # Mock metrics
    mock_cja.getMetrics.return_value = [
        {
            "id": "metric1",
            "name": "Test Metric 1",
            "type": "calculated",
            "title": "Test Metric 1 Title",
            "description": "Test metric description"
        },
        {
            "id": "metric2",
            "name": "Test Metric 2",
            "type": "standard",
            "title": "Test Metric 2 Title",
            "description": None  # Missing description for testing
        }
    ]

    # Mock dimensions
    mock_cja.getDimensions.return_value = [
        {
            "id": "dim1",
            "name": "Test Dimension 1",
            "type": "string",
            "title": "Test Dimension 1 Title",
            "description": "Test dimension description"
        },
        {
            "id": "dim2",
            "name": "Test Dimension 2",
            "type": "string",
            "title": "Test Dimension 2 Title",
            "description": ""
        },
        {
            "id": "dim3",
            "name": "Test Dimension 1",  # Duplicate name for testing
            "type": "string",
            "title": "Test Dimension Duplicate",
            "description": "Duplicate dimension"
        }
    ]

    return mock_cja


@pytest.fixture
def sample_metrics_df():
    """Create a sample metrics DataFrame for testing"""
    return pd.DataFrame([
        {
            "id": "metric1",
            "name": "Test Metric 1",
            "type": "calculated",
            "title": "Test Metric 1 Title",
            "description": "Test metric description"
        },
        {
            "id": "metric2",
            "name": "Test Metric 2",
            "type": "standard",
            "title": "Test Metric 2 Title",
            "description": None
        }
    ])


@pytest.fixture
def sample_dimensions_df():
    """Create a sample dimensions DataFrame for testing"""
    return pd.DataFrame([
        {
            "id": "dim1",
            "name": "Test Dimension 1",
            "type": "string",
            "title": "Test Dimension 1 Title",
            "description": "Test dimension description"
        },
        {
            "id": "dim2",
            "name": "Test Dimension 2",
            "type": "string",
            "title": "Test Dimension 2 Title",
            "description": ""
        },
        {
            "id": "dim3",
            "name": "Test Dimension 1",  # Duplicate
            "type": "string",
            "title": "Test Dimension Duplicate",
            "description": "Duplicate dimension"
        }
    ])


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
