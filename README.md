# Tariff Analyzer

A system for processing and analyzing tariff policy data.

## Overview

This project provides tools for:
- Processing and normalizing tariff event data
- Analyzing patterns in trade policy
- Visualizing tariff impacts and relationships

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```python
from tariff_analyzer.preprocessing import TariffProcessor

# Initialize processor
processor = TariffProcessor("data/reference/country_codes.csv")

# Process tariff data
processed_data = processor.process("data/raw/tariffs.json")

# Export to CSV
processor.export_to_csv(processed_data, "data/processed/processed_tariffs.csv")
```

## Project Structure

- `src/`: Source code
  - `preprocessing/`: Data preprocessing
  - `analysis/`: Data analysis
  - `visualization/`: Data visualization
  - `utils/`: Utilities
- `notebooks/`: Jupyter notebooks for examples
- `tests/`: Unit tests
- `data/`: Data files
- `docs/`: Documentation