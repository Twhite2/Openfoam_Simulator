#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging utilities for Openfoam_Simulator application.

This module provides functions for setting up and retrieving loggers
for various components of the application, with configurable logging
levels, formats, and output destinations (console, file, etc.).
"""

import os
import sys
import logging
import logging.handlers
import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List


# Default log format strings
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_LOG_FORMAT = '%(levelname)s - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# Log levels mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Global registry of loggers created by this module
# This avoids creating multiple loggers for the same component
_loggers = {}


def setup_logger(log_level: str = None, log_file: str = None, file_log_level: str = None,
                console: bool = True, console_log_level: str = None,
                log_format: str = DEFAULT_LOG_FORMAT) -> logging.Logger:
    """
    Set up the root logger for the application.

    Args:
        log_level (str, optional): Overall log level (default from config or INFO)
        log_file (str, optional): Path to log file (None means no file logging)
        file_log_level (str, optional): Log level for file handler (defaults to log_level)
        console (bool, optional): Whether to log to console (default True)
        console_log_level (str, optional): Log level for console handler (defaults to log_level)
        log_format (str, optional): Format string for log messages

    Returns:
        logging.Logger: The configured root logger
    """
    # Try to get log level from config
    if log_level is None:
        try:
            from ..config import get_value
            log_level_str = get_value('app.log_level', 'INFO')
            log_level = LOG_LEVELS.get(log_level_str, logging.INFO)
        except (ImportError, KeyError):
            log_level = logging.INFO
    elif isinstance(log_level, str):
        log_level = LOG_LEVELS.get(log_level, logging.INFO)

    # Use the same level for file and console if not specified
    if file_log_level is None:
        file_log_level = log_level
    elif isinstance(file_log_level, str):
        file_log_level = LOG_LEVELS.get(file_log_level, log_level)

    if console_log_level is None:
        console_log_level = log_level
    elif isinstance(console_log_level, str):
        console_log_level = LOG_LEVELS.get(console_log_level, log_level)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_log_level if console else logging.CRITICAL,
                         file_log_level if log_file else logging.CRITICAL))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Set up rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Register the root logger
    _loggers['root'] = root_logger
    return root_logger


def get_logger(name: str = None, log_level: Union[str, int] = None) -> logging.Logger:
    """
    Get a logger for the specified component.

    Args:
        name (str, optional): The name of the logger/component
                             (None for the root logger)
        log_level (str or int, optional): Override the default log level

    Returns:
        logging.Logger: The logger for the specified component
    """
    global _loggers

    # Use root logger if no name specified
    if name is None:
        logger_name = 'root'
    else:
        logger_name = name

    # Check if logger already exists in registry
    if logger_name in _loggers:
        logger = _loggers[logger_name]
    else:
        # Create new logger
        logger = logging.getLogger(name)

        # Set default level if not inheriting from parent
        if log_level is not None:
            if isinstance(log_level, str):
                level = LOG_LEVELS.get(log_level, logging.INFO)
            else:
                level = log_level
            logger.setLevel(level)

        # Store in registry
        _loggers[logger_name] = logger

    return logger


def set_log_level(level: Union[str, int], logger_name: str = None) -> None:
    """
    Set the log level for a specific logger.

    Args:
        level (str or int): The log level to set
        logger_name (str, optional): The name of the logger to update
                                   (None for the root logger)
    """
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level, logging.INFO)

    # Update the specified logger
    logger = get_logger(logger_name)
    logger.setLevel(level)


def get_log_file_path() -> Optional[str]:
    """
    Get the path to the current log file.

    Returns:
        str or None: The path to the log file, or None if file logging is not configured
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Look for file handlers
    for handler in root_logger.handlers:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            return handler.baseFilename

    return None


def configure_from_config() -> None:
    """
    Configure the logger based on application configuration.
    """
    try:
        from ..config import get_value

        # Get log configuration
        log_level = get_value('app.log_level', 'INFO')
        log_dir = get_value('app.log_dir', None)

        # Set up log file path if directory is specified
        log_file = None
        if log_dir:
            # Create log directory if it doesn't exist
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Create log filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(log_path / f'openfoam_simulator_{timestamp}.log')

        # Configure root logger
        setup_logger(
            log_level=log_level,
            log_file=log_file,
            console=True
        )

    except (ImportError, KeyError) as e:
        # Fall back to basic configuration if config module isn't available
        print(f"Warning: Could not configure logger from config: {e}")
        setup_logger(log_level='INFO', console=True)


def log_exception(logger: logging.Logger, exception: Exception, message: str = None) -> None:
    """
    Log an exception with traceback.

    Args:
        logger (logging.Logger): The logger to use
        exception (Exception): The exception to log
        message (str, optional): Additional message to log with the exception
    """
    import traceback
    if message:
        logger.error(f"{message}: {str(exception)}")
    else:
        logger.error(str(exception))
    logger.debug(traceback.format_exc())


def create_console_handler(level: Union[str, int] = 'INFO',
                         format_str: str = DEFAULT_LOG_FORMAT) -> logging.Handler:
    """
    Create a console handler for logging.

    Args:
        level (str or int): Log level for the handler
        format_str (str): Format string for log messages

    Returns:
        logging.Handler: The configured console handler
    """
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level, logging.INFO)

    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_str))
    return handler


def create_file_handler(filename: str, level: Union[str, int] = 'INFO',
                      format_str: str = DEFAULT_LOG_FORMAT,
                      max_bytes: int = 10*1024*1024,
                      backup_count: int = 5) -> logging.Handler:
    """
    Create a rotating file handler for logging.

    Args:
        filename (str): Path to the log file
        level (str or int): Log level for the handler
        format_str (str): Format string for log messages
        max_bytes (int): Maximum size of the log file before rotation
        backup_count (int): Number of backup files to keep

    Returns:
        logging.Handler: The configured file handler
    """
    # Convert string level to numeric if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level, logging.INFO)

    # Ensure directory exists
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create handler
    handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_str))
    return handler


# Simple module initialization
if __name__ != "__main__":
    # Create the root logger when the module is imported
    # Use a simple console logger initially, this will be overridden
    # when the application properly initializes
    setup_logger(console=True, log_level='INFO')


# Example usage if run directly
if __name__ == "__main__":
    # Configure logging
    root_logger = setup_logger(log_level='DEBUG', console=True,
                               log_file='openfoam_simulator.log')
    root_logger.info("Root logger initialized")

    # Get a module logger
    module_logger = get_logger("test_module")
    module_logger.info("Module logger initialized")
    module_logger.debug("This is a debug message")
    module_logger.warning("This is a warning message")
    module_logger.error("This is an error message")

    # Test exception logging
    try:
        x = 1 / 0
    except Exception as e:
        log_exception(module_logger, e, "Error in calculation")