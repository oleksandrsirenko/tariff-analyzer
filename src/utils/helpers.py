"""Helper utilities for the tariff analyzer."""

import os
import json
from typing import Dict, List, Any, Union, Optional

from .logger import logger


def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        The loaded JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise


def save_json_data(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: The data to save
        file_path: Path to the output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list.

    Args:
        nested_list: A list that may contain lists

    Returns:
        A flat list
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists.

    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
