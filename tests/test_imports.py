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

    # Try importing from different modules
    try:
        # Core modules
        import src
        from src.utils import helpers, validators
        from src.preprocessing import processor

        # Print success message
        print("\nSuccessfully imported core modules")
        assert True
    except ImportError as e:
        print(f"\nImport error: {e}")
        assert False, f"Import failed: {e}"


def test_module_structure():
    """Test the basic module structure exists."""
    # Get the project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check essential directories and files
    paths_to_check = [
        "src/utils/__init__.py",
        "src/preprocessing/__init__.py",
        "src/analysis/__init__.py",
        "src/visualization/__init__.py",
    ]

    for path in paths_to_check:
        full_path = os.path.join(root_dir, path)
        assert os.path.exists(full_path), f"Missing: {path}"
