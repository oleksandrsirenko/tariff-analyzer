"""Input validation utilities for the tariff analyzer."""

import os
import json
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from .logger import logger


def validate_json_input(data: Dict[str, Any]) -> bool:
    """
    Validate tariff data input.

    Args:
        data: The input data to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        logger.error("Input data must be a dictionary")
        return False

    if "events" not in data:
        logger.error("Input data must contain 'events' key")
        return False

    if not isinstance(data["events"], list):
        logger.error("'events' must be a list")
        return False

    # Check the first few events
    for i, event in enumerate(data["events"][:5]):
        if not isinstance(event, dict):
            logger.error(f"Event {i} is not a dictionary")
            return False

        if "tariffs_v2" not in event:
            logger.error(f"Event {i} does not contain 'tariffs_v2' key")
            return False

    logger.info("Input data validation passed")
    return True


def validate_country_code_file(file_path: str) -> bool:
    """
    Validate country code reference file.

    Args:
        file_path: Path to the country code file

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"Country code file not found: {file_path}")
        return False

    # Check file extension
    if not file_path.endswith(".csv"):
        logger.error(f"Country code file must be a CSV file: {file_path}")
        return False

    try:
        # Try to read the file
        df = pd.read_csv(file_path)

        # Check required columns
        required_columns = ["Country", "Alpha-2 code"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Country code file missing required column: {col}")
                return False

        # Check for empty values in required columns
        if df[required_columns].isnull().any().any():
            logger.warning(
                "Country code file contains missing values in required columns"
            )

        logger.info("Country code file validation passed")
        return True
    except Exception as e:
        logger.error(f"Error validating country code file: {e}")
        return False


def validate_output_path(file_path: str) -> bool:
    """
    Validate output file path.

    Args:
        file_path: Path to the output file

    Returns:
        True if valid, False otherwise
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        # Check if file can be written
        with open(file_path, "a") as f:
            pass

        logger.info(f"Output path validation passed: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Invalid output path: {file_path} - {e}")
        return False
