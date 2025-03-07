"""
Tests for utility components.

This file contains tests for the utility modules.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

# Import directly from src modules (this will work with conftest.py fix)
from src.utils.helpers import (
    load_json_data,
    save_json_data,
    flatten_list,
    ensure_directory,
)
from src.utils.validators import (
    validate_json_input,
    validate_country_code_file,
    validate_output_path,
)


class TestHelpers:
    """Tests for helper functions."""

    def test_load_json_data(self):
        """Test loading JSON data."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"test": "data"}, f)
            temp_file = f.name

        try:
            # Test loading the file
            data = load_json_data(temp_file)
            assert isinstance(data, dict)
            assert data["test"] == "data"

            # Test with non-existent file
            with pytest.raises(FileNotFoundError):
                load_json_data("non_existent_file.json")

            # Test with invalid JSON
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w"
            ) as f:
                f.write("This is not valid JSON")
                invalid_file = f.name

            try:
                with pytest.raises(json.JSONDecodeError):
                    load_json_data(invalid_file)
            finally:
                os.unlink(invalid_file)
        finally:
            os.unlink(temp_file)

    def test_save_json_data(self):
        """Test saving JSON data."""
        # Create test data
        test_data = {"test": "data", "nested": {"value": 123}}

        # Create a temporary file path
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name
        os.unlink(temp_file)  # Delete it so save_json_data creates it

        try:
            # Test saving to file
            save_json_data(test_data, temp_file)

            # Verify file exists and contains the data
            assert os.path.exists(temp_file)
            with open(temp_file, "r") as f:
                loaded_data = json.load(f)
                assert loaded_data == test_data
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_flatten_list(self):
        """Test flattening nested lists."""
        # Test with various list structures
        assert flatten_list([1, 2, 3]) == [1, 2, 3]
        assert flatten_list([1, [2, 3]]) == [1, 2, 3]
        assert flatten_list([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
        assert flatten_list([]) == []

        # Test with non-list elements
        assert flatten_list([1, "a", [2, "b"]]) == [1, "a", 2, "b"]

    def test_ensure_directory(self):
        """Test ensuring directory exists."""
        # Create a temporary directory path
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_dir")

            # Test creating directory
            ensure_directory(test_dir)
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

            # Test with existing directory (should not raise error)
            ensure_directory(test_dir)


class TestValidators:
    """Tests for validator functions."""

    def test_validate_json_input(self):
        """Test JSON input validation."""
        # Valid input
        valid_input = {
            "events": [
                {"tariffs_v2": {"data": "value"}},
                {"tariffs_v2": {"more": "data"}},
            ]
        }
        assert validate_json_input(valid_input) is True

        # Invalid inputs
        assert validate_json_input({"no_events": []}) is False
        assert validate_json_input({"events": "not a list"}) is False
        assert validate_json_input({"events": [{"no_tariffs": {}}]}) is False
