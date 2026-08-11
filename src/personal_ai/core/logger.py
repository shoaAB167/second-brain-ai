import logging
import sys

from personal_ai.config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Configure and return a standard logger for the application.

    Args:
        name: The name of the module or component requesting the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if get_logger is called multiple times for the same logger
    if not logger.handlers:
        logger.setLevel(settings.log_level.upper())

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(settings.log_level.upper())

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Do not propagate to root logger to avoid duplicate log outputs
        logger.propagate = False

    return logger
