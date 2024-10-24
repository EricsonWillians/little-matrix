# config.py

"""
Configuration module for the little-matrix simulation.

This module defines the `load_config` function, which loads configuration settings from a YAML file.
It provides a centralized way to manage simulation parameters and settings.

Functions:
    load_config(config_file: str) -> dict: Loads configuration settings from a YAML file.
"""

import yaml
import logging

def load_config(config_file: str) -> dict:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration settings as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    logger = logging.getLogger(__name__)
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from '{config_file}'.")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{config_file}': {e}")
        raise
