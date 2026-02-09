"""CJA client initialization for CJA Auto SDR."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

import cjapy

from cja_auto_sdr.api.resilience import make_api_call_with_retry
from cja_auto_sdr.core.constants import BANNER_WIDTH
from cja_auto_sdr.core.credentials import CredentialResolver
from cja_auto_sdr.core.exceptions import (
    CredentialSourceError,
    ProfileConfigError,
    ProfileNotFoundError,
)


def _bootstrap_dotenv(logger: logging.Logger) -> None:
    """Load .env variables if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed (.env files will not be auto-loaded)")
        return

    try:
        dotenv_loaded = load_dotenv()
        if dotenv_loaded:
            logger.debug(".env file found and loaded")
        else:
            logger.debug(".env file not found (python-dotenv available but no .env file)")
    except Exception as e:
        logger.debug(f"Failed to load .env via python-dotenv: {e}")


def _config_from_env(credentials: dict[str, str], logger: logging.Logger):
    """
    Configure cjapy using environment credentials.

    Creates a temporary JSON config file that is cleaned up on exit.

    Args:
        credentials: Dictionary of credentials from environment
        logger: Logger instance
    """
    # cjapy.importConfigFile expects a JSON file, so we create a temporary one
    # This is cleaned up on exit. Use restrictive permissions (0o600) since it contains credentials.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="cja_env_config_") as temp_config:
        json.dump(credentials, temp_config)
        temp_config_path = temp_config.name
    os.chmod(temp_config_path, 0o600)

    logger.debug(f"Created temporary config file: {temp_config_path}")

    # Register cleanup
    def cleanup_temp_config():
        with contextlib.suppress(OSError):
            os.unlink(temp_config_path)

    atexit.register(cleanup_temp_config)

    # Import the temporary config
    cjapy.importConfigFile(temp_config_path)


def configure_cjapy(
    profile: str | None = None, config_file: str = "config.json", logger: logging.Logger | None = None
) -> tuple[bool, str, dict[str, str] | None]:
    """
    Configure cjapy with credentials using priority: profile > env > config file.

    This is a lightweight configuration function that sets up cjapy credentials
    without creating a CJA instance. Use this for commands that need credential
    configuration but manage their own CJA lifecycle.

    Uses CredentialResolver internally to resolve credentials following the
    standard priority order.

    Args:
        profile: Optional profile name to use for credentials
        config_file: Path to fallback config file
        logger: Optional logger instance

    Returns:
        Tuple of (success, source_description, credentials_dict)
        - success: True if cjapy was configured successfully
        - source_description: Description of credential source used
        - credentials_dict: The credentials that were used (for display purposes)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

    try:
        # Ensure direct api.client consumers get .env loading parity with generator path.
        _bootstrap_dotenv(logger)

        # Use CredentialResolver for unified credential resolution
        resolver = CredentialResolver(logger)
        credentials, source = resolver.resolve(profile=profile, config_file=config_file)

        # Configure cjapy with the resolved credentials
        # Source format from resolver: "profile:name", "environment", "config:filename"
        if source.startswith("profile:") or source == "environment":
            # Profile or environment credentials need temp file for cjapy
            _config_from_env(credentials, logger)
        else:
            # Config file can be imported directly
            cjapy.importConfigFile(config_file)

        # Format source for display (convert resolver format to user-friendly format)
        if source.startswith("profile:"):
            display_source = f"Profile: {source.split(':', 1)[1]}"
        elif source == "environment":
            display_source = "Environment variables"
        elif source.startswith("config:"):
            display_source = f"Config file: {Path(config_file).resolve()}"
        else:
            display_source = source

        return True, display_source, credentials

    except CredentialSourceError as e:
        logger.error(f"Credential error: {e}")
        return False, str(e), None
    except (ProfileNotFoundError, ProfileConfigError) as e:
        logger.error(f"Profile error: {e}")
        return False, f"Profile error: {e}", None


def initialize_cja(
    config_file: str | Path = "config.json", logger: logging.Logger | None = None, profile: str | None = None
) -> cjapy.CJA | None:
    """Initialize CJA connection with comprehensive error handling.

    Uses CredentialResolver for unified credential loading with priority:
        1. Profile credentials (if --profile or CJA_PROFILE specified)
        2. Environment variables (ORG_ID, CLIENT_ID, SECRET, etc.)
        3. Configuration file (config.json)

    Args:
        config_file: Path to CJA configuration file
        logger: Logger instance (uses module logger if None)
        profile: Profile name to load credentials from (optional)

    Returns:
        Initialized CJA instance, or None if initialization fails

    Raises:
        ConfigurationError: If credentials are invalid or missing
        APIError: If API connection fails
    """
    # Import here to avoid circular dependency — these are still in generator.py
    from cja_auto_sdr.generator import (
        _DOTENV_AVAILABLE,
        _DOTENV_LOADED,
        resolve_active_profile,
    )

    logger = logger or logging.getLogger(__name__)
    try:
        logger.info("=" * BANNER_WIDTH)
        logger.info("INITIALIZING CJA CONNECTION")
        logger.info("=" * BANNER_WIDTH)

        # Log dotenv status for debugging
        if _DOTENV_AVAILABLE:
            if _DOTENV_LOADED:
                logger.debug(".env file found and loaded")
            else:
                logger.debug(".env file not found (python-dotenv available but no .env file)")
        else:
            logger.debug("python-dotenv not installed (.env files will not be auto-loaded)")

        # Resolve active profile (--profile > CJA_PROFILE > None)
        active_profile = resolve_active_profile(profile)

        # Use CredentialResolver for unified credential loading
        resolver = CredentialResolver(logger)
        try:
            credentials, source = resolver.resolve(profile=active_profile, config_file=config_file)
            logger.info(f"Credentials loaded from: {source}")
        except CredentialSourceError as e:
            logger.critical("=" * BANNER_WIDTH)
            logger.critical("CREDENTIAL LOADING FAILED")
            logger.critical("=" * BANNER_WIDTH)
            logger.critical(str(e))
            if e.reason:
                logger.critical(f"Reason: {e.reason}")
            if e.details:
                logger.critical(e.details)
            logger.critical("")
            logger.critical("Options to provide credentials:")
            logger.critical("  1. Profile:   cja_auto_sdr --profile <name> ...")
            logger.critical("  2. Env vars:  export ORG_ID=... CLIENT_ID=... SECRET=... SCOPES=...")
            logger.critical("  3. Config:    Create config.json with credentials")
            return None

        # Configure cjapy with resolved credentials
        if source.startswith("config:"):
            # Config file - use cjapy's native loading
            logger.info(f"Loading CJA configuration from {config_file}...")
            cjapy.importConfigFile(config_file)
        else:
            # Profile or environment - use temp config file
            logger.info(f"Loading CJA configuration from {source}...")
            _config_from_env(credentials, logger)

        logger.info("Configuration loaded successfully")

        # Attempt to create CJA instance
        logger.info("Creating CJA instance...")
        cja = cjapy.CJA()
        logger.info("CJA instance created successfully")

        # Test connection with a simple API call (with retry)
        logger.info("Testing API connection...")
        try:
            # Attempt to list data views to verify connection with retry
            test_call = make_api_call_with_retry(
                cja.getDataViews, logger=logger, operation_name="getDataViews (connection test)"
            )
            if test_call is not None:
                logger.info(
                    f"\u2713 API connection successful! Found {len(test_call) if hasattr(test_call, '__len__') else 'multiple'} data view(s)"
                )
            else:
                logger.warning("API connection test returned None - connection may be unstable")
        except Exception as test_error:
            logger.warning(f"Could not verify connection with test call: {test_error!s}")
            logger.warning("Proceeding anyway - errors may occur during data fetching")

        logger.info("CJA initialization complete")
        return cja

    except FileNotFoundError:
        logger.critical("=" * BANNER_WIDTH)
        logger.critical("CONFIGURATION FILE ERROR")
        logger.critical("=" * BANNER_WIDTH)
        logger.critical(f"Config file not found: {config_file}")
        logger.critical(f"Current working directory: {Path.cwd()}")
        logger.critical("Please ensure the configuration file exists in the correct location")
        return None

    except ImportError as e:
        logger.critical("=" * BANNER_WIDTH)
        logger.critical("DEPENDENCY ERROR")
        logger.critical("=" * BANNER_WIDTH)
        logger.critical(f"Failed to import cjapy module: {e!s}")
        logger.critical("Please ensure cjapy is installed: pip install cjapy")
        return None

    except AttributeError as e:
        logger.critical("=" * BANNER_WIDTH)
        logger.critical("CJA CONFIGURATION ERROR")
        logger.critical("=" * BANNER_WIDTH)
        logger.critical(f"Configuration error: {e!s}")
        logger.critical("This usually indicates an issue with the authentication credentials")
        logger.critical("Please verify all fields in your configuration file are correct")
        return None

    except PermissionError as e:
        logger.critical("=" * BANNER_WIDTH)
        logger.critical("PERMISSION ERROR")
        logger.critical("=" * BANNER_WIDTH)
        logger.critical(f"Cannot read configuration file: {e!s}")
        logger.critical("Please check file permissions")
        return None

    except Exception as e:
        logger.critical("=" * BANNER_WIDTH)
        logger.critical("CJA INITIALIZATION FAILED")
        logger.critical("=" * BANNER_WIDTH)
        logger.critical(f"Unexpected error: {e!s}")
        logger.critical(f"Error type: {type(e).__name__}")
        logger.exception("Full error details:")
        logger.critical("")
        logger.critical("Troubleshooting steps:")
        logger.critical("1. Verify your configuration file exists and is valid JSON")
        logger.critical("2. Check that all authentication credentials are correct")
        logger.critical("3. Ensure your API credentials have the necessary permissions")
        logger.critical("4. Verify you have network connectivity to Adobe services")
        logger.critical("5. Check if cjapy library is up to date: pip install --upgrade cjapy")
        return None
