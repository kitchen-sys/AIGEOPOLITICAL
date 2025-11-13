"""
Logging utilities for GeoBotv1
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = 'geobot',
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with consistent formatting.

    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level
    log_file : str, optional
        Log file path

    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'geobot') -> logging.Logger:
    """
    Get existing logger.

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
