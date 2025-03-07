"""
Simple test to verify imports are working.

Run this first to verify the import system is correctly set up.
"""

import os
import sys


def test_import_paths():
    """Test that Python paths are set up correctly."""
    # Print the current Python path for debugging
    print("\nPython path:")
    for p in sys.path:
        print(f"  - {p}")

    success = True

    # Try importing modules one by one
    modules_to_test = [
        "src",
        "src.utils.config",
        "src.utils.helpers",
        "src.utils.validators",
        "src.utils.logger",
        "src.preprocessing.processor",
        "src.preprocessing.normalizer",
        "src.preprocessing.deduplicator",
        "src.preprocessing.extractor",
    ]

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"Successfully imported: {module}")
        except ImportError as e:
            print(f"Failed to import: {module} - {e}")
            success = False

    assert success, "One or more imports failed"


def test_module_structure():
    """Test the basic module structure exists."""
    # Get the project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check essential directories and files
    paths_to_check = [
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/helpers.py",
        "src/utils/logger.py",
        "src/utils/validators.py",
        "src/preprocessing/__init__.py",
        "src/analysis/__init__.py",
        "src/visualization/__init__.py",
    ]

    for path in paths_to_check:
        full_path = os.path.join(root_dir, path)
        assert os.path.exists(full_path), f"Missing: {path}"
