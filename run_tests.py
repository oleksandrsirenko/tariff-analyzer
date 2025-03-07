#!/usr/bin/env python
"""
Run all tests for the tariff analyzer.

This script runs all tests using pytest and provides a summary of the results.
"""

import os
import sys
import subprocess
import time


def main():
    """Run all tests and display results."""
    print("Running tests for Tariff Analyzer...\n")
    start_time = time.time()

    # Run pytest with appropriate options
    result = subprocess.run(
        ["pytest", "-v", "--color=yes"], capture_output=True, text=True
    )

    # Print output
    print(result.stdout)

    if result.stderr:
        print("Errors:\n", result.stderr)

    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nTest execution completed in {execution_time:.2f} seconds")

    # Return the appropriate exit code
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
