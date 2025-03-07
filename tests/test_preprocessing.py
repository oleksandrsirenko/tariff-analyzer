"""
Tests for preprocessing components.

This file contains tests for the preprocessing module components.
"""

import os
import json
import pytest
from pathlib import Path
import pandas as pd

from src.preprocessing.processor import TariffProcessor
from src.preprocessing.normalizer import CountryNormalizer, DateNormalizer
from src.preprocessing.deduplicator import EventDeduplicator
from src.preprocessing.extractor import TariffRateExtractor, TariffProductExtractor


@pytest.fixture
def sample_data():
    """Fixture to load sample data."""
    # Get paths relative to this test file
    base_dir = Path(__file__).parent.parent
    sample_data_path = (
        base_dir / "data" / "reference" / "event_api_response_example.json"
    )

    # Load sample data
    with open(sample_data_path, "r") as f:
        return json.load(f)


@pytest.fixture
def country_codes_path():
    """Fixture to get path to country codes file."""
    base_dir = Path(__file__).parent.parent
    return base_dir / "data" / "reference" / "country_codes_iso_3166_1_alpha_2_code.csv"


class TestTariffProcessor:
    """Tests for the TariffProcessor class."""

    def test_process(self, sample_data, country_codes_path):
        """Test the main processing function."""
        processor = TariffProcessor(str(country_codes_path))
        result = processor.process(sample_data)

        # Check result structure
        assert "events" in result
        assert "metadata" in result
        assert len(result["events"]) > 0

        # Check that events have been processed
        event = result["events"][0]
        assert "tariffs_v2" in event

        # Check that derived fields have been added
        tariff = event["tariffs_v2"]
        assert "rate_category" in tariff
        assert "impact_score" in tariff

    def test_to_dataframe(self, sample_data, country_codes_path):
        """Test conversion to DataFrame."""
        processor = TariffProcessor(str(country_codes_path))
        processed_data = processor.process(sample_data)

        df = processor.to_dataframe(processed_data)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "imposing_country_code" in df.columns
        assert "measure_type" in df.columns


class TestCountryNormalizer:
    """Tests for the CountryNormalizer class."""

    def test_normalize_country_code(self, country_codes_path):
        """Test country code normalization."""
        normalizer = CountryNormalizer(str(country_codes_path))

        # Test valid codes
        assert normalizer.normalize_country_code("US") == "US"
        assert normalizer.normalize_country_code("us") == "US"
        assert normalizer.normalize_country_code(" US ") == "US"

        # Test invalid codes
        assert normalizer.normalize_country_code("XX") is None
        assert normalizer.normalize_country_code("") is None

    def test_get_country_name(self, country_codes_path):
        """Test getting country name from code."""
        normalizer = CountryNormalizer(str(country_codes_path))

        # Test valid codes
        assert normalizer.get_country_name("US") == "United States"
        assert normalizer.get_country_name("FR") == "France"

        # Test invalid codes
        assert normalizer.get_country_name("XX") is None


class TestDateNormalizer:
    """Tests for the DateNormalizer class."""

    def test_normalize_date(self):
        """Test date normalization."""
        normalizer = DateNormalizer()

        # Test various date formats
        assert normalizer.normalize_date("2025/01/01") == "2025-01-01"
        assert normalizer.normalize_date("2025-01-01") == "2025-01-01"
        assert normalizer.normalize_date("2025/01") == "2025-01-01"
        assert normalizer.normalize_date("2025") == "2025-01-01"

        # Test invalid dates
        assert normalizer.normalize_date("") is None
        assert normalizer.normalize_date("not a date") is None

    def test_calculate_date_difference(self):
        """Test date difference calculation."""
        normalizer = DateNormalizer()

        # Test difference calculation
        assert normalizer.calculate_date_difference("2025-01-01", "2025-01-31") == 30
        assert normalizer.calculate_date_difference("2025-01-01", "2025-02-01") == 31

        # Test with invalid dates
        assert normalizer.calculate_date_difference(None, "2025-01-01") is None
        assert normalizer.calculate_date_difference("2025-01-01", None) is None


class TestTariffRateExtractor:
    """Tests for the TariffRateExtractor class."""

    def test_extract_rates(self, country_codes_path):
        """Test tariff rate extraction."""
        normalizer = CountryNormalizer(str(country_codes_path))
        extractor = TariffRateExtractor(normalizer)

        # Test with sample rate strings
        rate_strings = [
            "25% on steel",
            "10% on automobiles from Japan",
            "Increase by 15% on electronics",
        ]

        rates = extractor.extract_rates(rate_strings)

        # Check extracted rates
        assert len(rates) == 3
        assert rates[0]["rate"] == 25
        assert rates[0]["target"] == "steel"
        assert rates[1]["rate"] == 10
        assert rates[1]["product"] == "automobiles"
        assert rates[1]["country"] == "Japan"

    def test_extract_main_rate(self, country_codes_path):
        """Test main rate extraction."""
        normalizer = CountryNormalizer(str(country_codes_path))
        extractor = TariffRateExtractor(normalizer)

        # Test with sample rate strings
        rate_strings = ["25% on steel", "10% on automobiles", "35% on electronics"]

        main_rate = extractor.extract_main_rate(rate_strings)

        # Check main rate (should be the highest)
        assert main_rate == 35
