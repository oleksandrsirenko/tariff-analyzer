"""
Configuration file for pytest.

This file is automatically recognized by pytest and used to configure
the test environment.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to Python's module search path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
