"""Tests for environment variable credential loading"""
import pytest
import os
import json
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cja_auto_sdr.generator import (
    load_credentials_from_env,
    validate_env_credentials,
    _config_from_env,
    ENV_VAR_MAPPING
)


class TestLoadCredentialsFromEnv:
    """Test environment variable credential loading"""

    def test_load_all_oauth_credentials(self, mock_env_credentials, clean_env):
        """Test loading complete OAuth S2S credentials from environment"""
        with patch.dict(os.environ, mock_env_credentials, clear=False):
            credentials = load_credentials_from_env()
            assert credentials is not None
            assert credentials['org_id'] == 'test_org@AdobeOrg'
            assert credentials['client_id'] == 'test_client_id'
            assert credentials['secret'] == 'test_secret'
            assert credentials['scopes'] == 'openid, AdobeID, additional_info.projectedProductContext'

    def test_returns_none_when_no_env_vars(self, clean_env):
        """Test that None is returned when no credential env vars are set"""
        credentials = load_credentials_from_env()
        assert credentials is None

    def test_partial_credentials(self, clean_env):
        """Test loading partial credentials returns what's available"""
        env_vars = {
            'ORG_ID': 'test_org@AdobeOrg',
            'CLIENT_ID': 'test_client_id'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            credentials = load_credentials_from_env()
            assert credentials is not None
            assert 'org_id' in credentials
            assert 'client_id' in credentials
            assert 'secret' not in credentials

    def test_strips_whitespace(self, clean_env):
        """Test that whitespace is stripped from values"""
        env_vars = {
            'ORG_ID': '  test_org@AdobeOrg  ',
            'CLIENT_ID': '\ttest_client_id\n',
            'SECRET': ' test_secret '
        }
        with patch.dict(os.environ, env_vars, clear=False):
            credentials = load_credentials_from_env()
            assert credentials['org_id'] == 'test_org@AdobeOrg'
            assert credentials['client_id'] == 'test_client_id'
            assert credentials['secret'] == 'test_secret'

    def test_ignores_empty_values(self, clean_env):
        """Test that empty or whitespace-only values are ignored"""
        env_vars = {
            'ORG_ID': 'test_org@AdobeOrg',
            'CLIENT_ID': '',
            'SECRET': '   '
        }
        with patch.dict(os.environ, env_vars, clear=False):
            credentials = load_credentials_from_env()
            assert credentials is not None
            assert 'org_id' in credentials
            assert 'client_id' not in credentials
            assert 'secret' not in credentials


class TestValidateEnvCredentials:
    """Test environment credential validation"""

    def test_valid_oauth_credentials(self):
        """Test validation passes for complete OAuth credentials"""
        credentials = {
            'org_id': 'test_org@AdobeOrg',
            'client_id': 'test_client_id',
            'secret': 'test_secret',
            'scopes': 'openid, AdobeID'
        }
        logger = MagicMock()
        assert validate_env_credentials(credentials, logger) is True

    def test_missing_required_field(self):
        """Test validation fails when required field is missing"""
        credentials = {
            'org_id': 'test_org@AdobeOrg',
            'client_id': 'test_client_id'
            # missing 'secret'
        }
        logger = MagicMock()
        assert validate_env_credentials(credentials, logger) is False

    def test_empty_required_field(self):
        """Test validation fails when required field is empty"""
        credentials = {
            'org_id': 'test_org@AdobeOrg',
            'client_id': 'test_client_id',
            'secret': '   '  # whitespace only
        }
        logger = MagicMock()
        assert validate_env_credentials(credentials, logger) is False

    def test_base_credentials_only_warns(self):
        """Test validation passes with warning when only base credentials present"""
        credentials = {
            'org_id': 'test_org@AdobeOrg',
            'client_id': 'test_client_id',
            'secret': 'test_secret'
            # No scopes specified
        }
        logger = MagicMock()
        result = validate_env_credentials(credentials, logger)
        assert result is True
        # Should log a warning about missing auth method
        logger.warning.assert_called()


class TestConfigFromEnv:
    """Test temporary config file creation from env credentials"""

    def test_creates_temp_config(self):
        """Test that _config_from_env creates a valid temp config"""
        credentials = {
            'org_id': 'test_org@AdobeOrg',
            'client_id': 'test_client_id',
            'secret': 'test_secret',
            'scopes': 'openid'
        }
        logger = MagicMock()

        with patch('cja_auto_sdr.generator.cjapy') as mock_cjapy:
            _config_from_env(credentials, logger)

            # Verify importConfigFile was called
            mock_cjapy.importConfigFile.assert_called_once()

            # Get the temp file path that was used
            config_path = mock_cjapy.importConfigFile.call_args[0][0]

            # Verify it's a valid JSON file with correct content
            with open(config_path, 'r') as f:
                saved_config = json.load(f)

            assert saved_config['org_id'] == 'test_org@AdobeOrg'
            assert saved_config['client_id'] == 'test_client_id'
            assert saved_config['secret'] == 'test_secret'
            assert saved_config['scopes'] == 'openid'


class TestEnvVarMapping:
    """Test the environment variable mapping constant"""

    def test_all_required_fields_mapped(self):
        """Test that all credential fields have env var mappings"""
        required_fields = ['org_id', 'client_id', 'secret']
        for field in required_fields:
            assert field in ENV_VAR_MAPPING

    def test_optional_fields_mapped(self):
        """Test that optional fields have env var mappings"""
        optional_fields = ['scopes', 'sandbox']
        for field in optional_fields:
            assert field in ENV_VAR_MAPPING

    def test_mapping_follows_naming_convention(self):
        """Test that env var names follow uppercase convention"""
        for config_key, env_var in ENV_VAR_MAPPING.items():
            # Env var should be uppercase version of config key
            assert env_var == config_key.upper()
