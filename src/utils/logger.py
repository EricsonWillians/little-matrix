# logger.py

"""
Logger module for the little-matrix simulation.

This module defines the `setup_logging` function, which configures the logging settings for the simulation.
It ensures that logs are formatted consistently and can be easily adjusted for different verbosity levels.

Functions:
    setup_logging(): Configures the logging settings for the simulation.
"""

import logging
import sys

def setup_logging(log_level: int = logging.INFO):
    """
    Configures the logging settings for the simulation.

    Args:
        log_level (int): The logging level threshold. Default is `logging.INFO`.

    Returns:
        None
    """
    # Define the logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Silence less severe logs from external libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("pygame").setLevel(logging.WARNING)
