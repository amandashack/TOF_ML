import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import yaml


def setup_logger(name: str, config_path: str = "config/logging.yaml") -> logging.Logger:
    """
    Sets up logging from a YAML config and returns a named logger.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Logging config file not found: {config_path}")

    with open(config_path, 'r') as f:
        log_config = yaml.safe_load(f)

    # dictConfig will configure the entire logging system:
    logging.config.dictConfig(log_config)

    # Return the logger with the specified name (e.g. "trainer")
    return logging.getLogger(name)

