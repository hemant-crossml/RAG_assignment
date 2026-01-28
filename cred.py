"""
This module loads and validates environment-based credentials required for
accessing the Google Gemini API.

It retrieves the Gemini API key from a `.env` file using `python-dotenv`,
ensuring the key is present before allowing the application to proceed.
If the required credential is missing, the module logs a critical error
and raises an exception to prevent the application from running in an
invalid state.

All credential-loading steps are logged for observability and debugging.
"""

import os

from dotenv import load_dotenv

from logger_config import get_logger

logger = get_logger(__name__)

try:
    logger.info("Loading environment variables from .env file")
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    if not gemini_api_key:
        logger.critical("GEMINI_API_KEY not found in environment")
        raise EnvironmentError("GEMINI_API_KEY not found in .env file")

    logger.info("GEMINI_API_KEY loaded successfully")

except Exception:
    # critical because app cannot run without credentials
    logger.exception("Failed to load Gemini API credentials")
    raise
