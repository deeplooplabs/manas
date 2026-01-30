"""Logging configuration for ManasRAG.

Provides a TRACE level (5) below DEBUG (10) for detailed execution tracing.
"""

import logging
import sys
from typing import Optional

# Custom TRACE level below DEBUG
TRACE = 5

# Map of level names to values
LOG_LEVELS = {
    "TRACE": TRACE,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _setup_component_loggers(level: str | int) -> None:
    """Configure loggers for ManasRAG components.

    Args:
        level: Logging level as string or int
    """
    # Convert string to int if needed
    if isinstance(level, str):
        level_upper = level.upper()
        if level_upper not in LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}. Valid: {list(LOG_LEVELS.keys())}")
        level = LOG_LEVELS[level_upper]

    # Configure the root manasrag logger
    root_logger = logging.getLogger("manasrag")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    # Format: timestamp [level] name: message
    formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    root_logger.propagate = False


def setup_logging(
    level: str = "INFO",
    format: Optional[str] = None,
) -> None:
    """Configure global logging for the manasrag package.

    Args:
        level: Logging level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Optional custom format string. If None, uses default format.
    """
    _setup_component_loggers(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name under the manasrag namespace.

    Args:
        name: Logger name (will be prefixed with 'manasrag.')

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"manasrag.{name}")


def get_tracer() -> logging.Logger:
    """Get a logger specifically for trace-level operations.

    Returns:
        Logger configured for trace output
    """
    return logging.getLogger("manasrag.trace")


# Add TRACE level to logging module if not already present
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE, "TRACE")


def _log_with_level(logger: logging.Logger, level: int, msg: str, *args, **kwargs) -> None:
    """Log a message with a custom level.

    Args:
        logger: Logger instance
        level: Level to log at
        msg: Message format string
        *args: Positional arguments for format string
        **kwargs: Keyword arguments for logging
    """
    logger.log(level, msg, *args, **kwargs)


def trace(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log at TRACE level (5).

    Args:
        logger: Logger instance
        msg: Message format string
        *args: Positional arguments for format string
        **kwargs: Keyword arguments for logging
    """
    logger.log(TRACE, msg, *args, **kwargs)
