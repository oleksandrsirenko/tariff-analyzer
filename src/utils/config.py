"""Configuration utilities for the tariff analyzer."""

import os
import yaml
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "preprocessing": {
        "batch_size": 1000,
        "deduplication": True,
        "standardize_countries": True,
        "standardize_dates": True,
        "extract_tariff_rates": True,
    },
    "analysis": {
        "trade_relationship_threshold": 3,
        "time_series_min_events": 5,
        "outlier_threshold": 3.0,
    },
    "visualization": {
        "map_projection": "mercator",
        "color_palette": "viridis",
        "network_layout": "spring",
    },
    "logging": {"directory": "logs", "level": "INFO"},
}


class Config:
    """Configuration manager for the tariff analyzer."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to YAML configuration file
        """
        self._config = DEFAULT_CONFIG.copy()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_file: Path to YAML configuration file
        """
        with open(config_file, "r") as f:
            loaded_config = yaml.safe_load(f)

        if loaded_config:
            # Merge with default config (don't overwrite default keys not in file)
            self._update_dict(self._config, loaded_config)

    def _update_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with values from another dictionary.

        Args:
            target: Dictionary to update
            source: Dictionary with updates
        """
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._update_dict(target[key], value)
            else:
                target[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., 'preprocessing.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        current = self._config

        if not key_path:
            return current

        keys = key_path.split(".")

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., 'preprocessing.batch_size')
            value: Value to set
        """
        if not key_path:
            return

        keys = key_path.split(".")
        current = self._config

        # Navigate to the nested dict containing the key to set
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value


# Singleton instance
config = Config("config/config.yaml")
