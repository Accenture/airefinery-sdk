import logging
import logging.config

import sys


class ColoredFormatter(logging.Formatter):
    """Color codes for the logs"""

    COLOR_CODES = {
        "reset": "\033[0m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
    }

    LEVEL_COLOR = {
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red",
    }

    def format(self, record):
        if sys.stdout.isatty():
            # Format time
            asctime = self.formatTime(record, self.datefmt)
            asctime_colored = (
                f"{self.COLOR_CODES['cyan']}{asctime}{self.COLOR_CODES['reset']}"
            )

            # Colorize levelname
            color = self.LEVEL_COLOR.get(record.levelno, "reset")
            levelname_colored = f"{self.COLOR_CODES[color]}{record.levelname}{self.COLOR_CODES['reset']}"

            # Get the message
            message = record.getMessage()

            # Construct the formatted message
            formatted = (
                f"{asctime_colored} - {levelname_colored} - {record.name} - {message}"
            )

            # Include exception info if present
            if record.exc_info:
                if not record.exc_text:
                    record.exc_text = self.formatException(record.exc_info)
                formatted += f"\n{record.exc_text}"

            return formatted
        return super().format(record)


def configure_logging(log_level="INFO"):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}. Using default 'INFO'.")
        numeric_level = logging.INFO

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": ColoredFormatter,
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colored",
            },
        },
        "loggers": {
            "azure": {
                "level": logging.WARNING,  # Set as integer
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": numeric_level,  # Already an integer
        },
    }

    logging.config.dictConfig(logging_config)
