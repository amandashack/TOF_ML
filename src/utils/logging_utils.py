import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_file: str = "reports/logs/app.log", level: int = logging.INFO):
    """
    Sets up the logging configuration.
    """

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Suppress Matplotlib debug and info logs
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)

    # Suppress specific sub-loggers if necessary
    matplotlib_font_manager_logger = logging.getLogger("matplotlib.font_manager")
    matplotlib_font_manager_logger.setLevel(logging.WARNING)

    # Prevent logging from propagating to the root logger multiple times
    logger.propagate = False
