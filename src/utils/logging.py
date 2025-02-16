"""
Logging utilities for the document analysis system.
Provides consistent logging setup and decorators for timing operations.
"""

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union, cast

# Type variable for generic function type
F = TypeVar("F", bound=Callable[..., Any])

# Configure logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LoggerMixin:
    """Mixin to add logging capabilities to any class."""

    logger: logging.Logger

    def __init_subclass__(cls) -> None:
        """Initialize logger when subclass is created."""
        super().__init_subclass__()
        cls.logger = get_logger(cls.__name__)


def get_logger(
    name: str,
    level: Optional[Union[str, int]] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Get a logger with specified configuration.

    Args:
        name: Name for the logger
        level: Logging level (default: None, uses root logger level)
        log_file: Optional file path to log to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    # Set level if specified
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

    return logger


def setup_logger(
    name: str,
    *,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    file_level: Optional[Union[str, int]] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Name for the logger
        level: Logging level for console output
        log_file: Optional file path to log to
        file_level: Optional separate logging level for file output
        format_string: Optional custom format string
        date_format: Optional custom date format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Convert string level to integer if necessary
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        format_string or DEFAULT_FORMAT,
        date_format or DEFAULT_DATE_FORMAT
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        if file_level:
            if isinstance(file_level, str):
                file_level = getattr(logging, file_level.upper())
            file_handler.setLevel(file_level)
        else:
            file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    message: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use (if None, creates one)
        level: Logging level for the time log
        message: Optional custom message format

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        # Get logger if not provided
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time

                # Use % formatting instead of f-strings for logging
                log_msg = message or "%s executed in %.3f seconds"
                logger.log(
                    level,
                    log_msg if message else log_msg,
                    func.__name__,
                    execution_time
                )

                return result

            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    "%s failed after %.3f seconds: %s",
                    func.__name__,
                    execution_time,
                    str(e)
                )
                raise

        return cast(F, wrapper)

    return decorator
