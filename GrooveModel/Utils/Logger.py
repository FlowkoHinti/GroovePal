import logging
import sys

from colorlog import ColoredFormatter


def setup_logger(name='TrainerLogger', level=logging.DEBUG):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent adding multiple handlers

    logger.setLevel(level)

    # stdout handler for < WARNING
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)

    # stderr handler for >= WARNING
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(levelname)s - %(message)s",
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger
