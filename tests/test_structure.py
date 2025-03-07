"""
Tests for project structure.

This file contains tests to validate the project structure is set up correctly.
"""

import os
import importlib.util
from pathlib import Path
import pytest


def test_project_structure():
    """Test that the project structure is correct."""
    # Get project root directory
    root_dir = Path(__file__).parent.parent

    # Check main directories exist
    assert (root_dir / "src").is_dir()
    assert (root_dir / "tests").is_dir()
    assert (root_dir / "data").is_dir()
    assert (root_dir / "notebooks").exists()

    # Check src subdirectories exist
    assert (root_dir / "src" / "preprocessing").is_dir()
    assert (root_dir / "src" / "analysis").is_dir()
    assert (root_dir / "src" / "visualization").is_dir()
    assert (root_dir / "src" / "utils").is_dir()

    # Check config directory exists
    assert (root_dir / "config").is_dir()
    assert (root_dir / "config" / "config.yaml").is_file()


def test_module_imports():
    """Test that all main modules can be imported."""
    # List of key modules to check
    modules = [
        "src.preprocessing.processor",
        "src.preprocessing.normalizer",
        "src.preprocessing.deduplicator",
        "src.preprocessing.extractor",
        "src.analysis.statistics",
        "src.analysis.time_series",
        "src.analysis.network",
        "src.analysis.impact",
        "src.visualization.dashboard",
        "src.visualization.geo_viz",
        "src.visualization.network_viz",
        "src.visualization.time_viz",
        "src.utils.helpers",
        "src.utils.validators",
    ]

    # Try importing each module
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


def test_package_init_files():
    """Test that all packages have __init__.py files."""
    # Get project root directory
    root_dir = Path(__file__).parent.parent

    # Check src/__init__.py
    assert (root_dir / "src" / "__init__.py").is_file()

    # Check package __init__.py files
    packages = [
        "src/preprocessing",
        "src/analysis",
        "src/visualization",
        "src/utils",
    ]

    for package in packages:
        init_path = root_dir / package / "__init__.py"
        assert init_path.is_file(), f"Missing __init__.py in {package}"

        # Check that the __init__.py is not empty (should expose modules)
        content = init_path.read_text()
        assert content.strip(), f"Empty __init__.py in {package}"
        assert "__all__" in content, f"__init__.py in {package} should define __all__"


def test_environment_files():
    """Test that environment files exist and are consistent."""
    # Get project root directory
    root_dir = Path(__file__).parent.parent

    # Check that environment files exist
    assert (root_dir / "requirements.txt").is_file()
    assert (root_dir / "setup.py").is_file()
    assert (root_dir / "environment.yml").is_file()

    # Check that requirements.txt has key packages
    with open(root_dir / "requirements.txt", "r") as f:
        requirements = f.read()

    key_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "networkx",
        "pyyaml",
        "pytest",
    ]

    for package in key_packages:
        assert package in requirements, f"{package} should be in requirements.txt"

    # Check that setup.py has the same key packages
    with open(root_dir / "setup.py", "r") as f:
        setup_content = f.read()

    for package in key_packages:
        assert package in setup_content, f"{package} should be in setup.py"

    # Check that environment.yml has the same key packages
    with open(root_dir / "environment.yml", "r") as f:
        env_content = f.read()

    for package in key_packages:
        assert package in env_content, f"{package} should be in environment.yml"


def test_documentation_files():
    """Test that essential documentation files exist."""
    # Get project root directory
    root_dir = Path(__file__).parent.parent

    # Check README exists
    assert (root_dir / "README.md").is_file()

    # Check README content
    with open(root_dir / "README.md", "r") as f:
        readme = f.read()

    assert "Tariff Analyzer" in readme
    assert "Installation" in readme
    assert "Usage" in readme
