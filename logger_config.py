"""
This module provides a centralized logger configuration utility for the
application.

It defines a helper function to create and return a configured `logging.Logger`
instance with:
- A rotating file handler for persistent logs
- A console handler for real-time output
- A consistent, timestamped log format
- Protection against duplicate handler registration

The logger is intended to be imported and reused across modules to ensure
uniform logging behavior throughout the application.
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "app.log",
):
    """
    Create and return a configured logger instance.
    """

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate logs
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Console handler (stderr-safe)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
