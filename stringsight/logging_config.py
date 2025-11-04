"""
Logging configuration for StringSight.

This module provides a centralized logging configuration that can be controlled
via environment variables:
- STRINGSIGHT_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- STRINGSIGHT_LOG_FORMAT: Custom log format (optional)

Usage:
    from stringsight.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
"""

import logging
import os
import sys
from typing import Optional


# Default log format - simple format without timestamp/level for cleaner output
DEFAULT_LOG_FORMAT = "%(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_log_level() -> int:
    """Get the logging level from environment variable or default to INFO."""
    level_name = os.environ.get("STRINGSIGHT_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _get_log_format() -> str:
    """Get the log format from environment variable or use default."""
    return os.environ.get("STRINGSIGHT_LOG_FORMAT", DEFAULT_LOG_FORMAT)


def configure_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> None:
    """
    Configure the root logger for StringSight.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string: Custom format string for log messages
        date_format: Custom date format for timestamps
    """
    if level is None:
        level = _get_log_level()
    
    if format_string is None:
        format_string = _get_log_format()
    
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT
    
    # Configure the root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        stream=sys.stdout,
        force=True  # Override any existing configuration
    )
    
    # Suppress noisy third-party library logs
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Suppress Numba debug output at the environment level
    os.environ.setdefault("NUMBA_DISABLE_PERFORMANCE_WARNINGS", "1")
    os.environ.setdefault("NUMBA_WARNINGS", "0")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        A configured logger instance
    """
    # Ensure logging is configured
    if not logging.getLogger().handlers:
        configure_logging()
    
    return logging.getLogger(name)


# Configure logging on module import
configure_logging()


