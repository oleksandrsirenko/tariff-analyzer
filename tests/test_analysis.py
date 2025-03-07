"""
Tests for analysis components.

This file contains tests for the analysis module components.
"""

import os
import json
import pytest
import pandas as pd
from pathlib import Path

from src.preprocessing.processor import TariffProcessor
from src.analysis.statistics import TariffStatistics
from src.analysis.time_series import TariffTimeSeriesAnalyzer
from src.analysis.network import TariffNetworkAnalyzer
from src.analysis.impact import TariffImpactAnalyzer


@pytest.fixture
def processed_data():
    """Fixture to load and process sample data."""
    # Get paths relative to this test file
    base_dir = Path(__file__).parent.parent
    sample_data_path = (
        base_dir / "data" / "reference" / "event_api_response_example.json"
    )
    country_codes_path = (
        base_dir / "data" / "reference" / "country_codes_iso_3166_1_alpha_2_code.csv"
    )

    # Load sample data
    with open(sample_data_path, "r") as f:
        raw_data = json.load(f)

    # Process the data
    processor = TariffProcessor(str(country_codes_path))
    return processor.process(raw_data)


@pytest.fixture
def processed_df(processed_data):
    """Fixture to convert processed data to DataFrame."""
    processor = TariffProcessor()
    return processor.to_dataframe(processed_data)


class TestTariffStatistics:
    """Tests for the TariffStatistics class."""

    def test_init_with_dict(self, processed_data):
        """Test initialization with dictionary data."""
        stats = TariffStatistics(processed_data)

        # Check that the data was converted to DataFrame
        assert hasattr(stats, "df")
        assert isinstance(stats.df, pd.DataFrame)
        assert not stats.df.empty

    def test_init_with_df(self, processed_df):
        """Test initialization with DataFrame."""
        stats = TariffStatistics(processed_df)

        # Check that the DataFrame was used
        assert hasattr(stats, "df")
        assert id(stats.df) == id(processed_df)

    def test_get_summary_statistics(self, processed_data):
        """Test summary statistics generation."""
        stats = TariffStatistics(processed_data)
        summary = stats.get_summary_statistics()

        # Check summary structure
        assert isinstance(summary, dict)
        assert "total_events" in summary
        assert "rate_statistics" in summary
        assert "top_imposing_countries" in summary

        # Check summary values
        assert summary["total_events"] > 0
        assert isinstance(summary["top_imposing_countries"], list)

    def test_get_rate_distribution(self, processed_data):
        """Test rate distribution calculation."""
        stats = TariffStatistics(processed_data)
        distribution = stats.get_rate_distribution()

        # Check distribution structure
        assert isinstance(distribution, dict)
        assert "bins" in distribution
        assert "counts" in distribution

        # Check distribution values
        assert len(distribution["bins"]) > 0
        assert len(distribution["counts"]) > 0
        assert len(distribution["bins"]) == len(distribution["counts"])


class TestTariffTimeSeriesAnalyzer:
    """Tests for the TariffTimeSeriesAnalyzer class."""

    def test_get_monthly_event_counts(self, processed_data):
        """Test monthly event counts calculation."""
        analyzer = TariffTimeSeriesAnalyzer(processed_data)
        counts = analyzer.get_monthly_event_counts()

        # Check that counts were generated
        assert isinstance(counts, dict)

        # Either we have results or a clear error message
        if "error" in counts:
            assert "No" in counts["error"]
        else:
            assert "dates" in counts
            assert "counts" in counts
            assert len(counts["dates"]) == len(counts["counts"])

    def test_get_yearly_event_counts(self, processed_data):
        """Test yearly event counts calculation."""
        analyzer = TariffTimeSeriesAnalyzer(processed_data)
        counts = analyzer.get_yearly_event_counts()

        # Check that counts were generated
        assert isinstance(counts, dict)

        # Either we have results or a clear error message
        if "error" in counts:
            assert "No" in counts["error"]
        else:
            assert "years" in counts
            assert "counts" in counts
            assert len(counts["years"]) == len(counts["counts"])


class TestTariffNetworkAnalyzer:
    """Tests for the TariffNetworkAnalyzer class."""

    def test_get_network_summary(self, processed_data):
        """Test network summary generation."""
        analyzer = TariffNetworkAnalyzer(processed_data)
        summary = analyzer.get_network_summary()

        # Check summary structure
        assert isinstance(summary, dict)

        # Either we have results or a clear error message
        if "error" in summary:
            assert "Graph" in summary["error"]
        else:
            assert "nodes" in summary
            assert "edges" in summary
            assert summary["nodes"] > 0
            assert summary["edges"] > 0

    def test_get_relationship_analysis(self, processed_data):
        """Test relationship analysis generation."""
        analyzer = TariffNetworkAnalyzer(processed_data)
        analysis = analyzer.get_relationship_analysis()

        # Check analysis structure
        assert isinstance(analysis, dict)

        # Either we have results or a clear error message
        if "error" in analysis:
            assert "No" in analysis["error"]
        else:
            assert "top_relationships" in analysis
            assert "total_relationships" in analysis
            assert isinstance(analysis["top_relationships"], list)


class TestTariffImpactAnalyzer:
    """Tests for the TariffImpactAnalyzer class."""

    def test_get_industry_impact_assessment(self, processed_data):
        """Test industry impact assessment."""
        analyzer = TariffImpactAnalyzer(processed_data)
        assessment = analyzer.get_industry_impact_assessment()

        # Check assessment structure
        assert isinstance(assessment, dict)

        # Either we have results or a clear error message
        if "error" in assessment:
            assert "No" in assessment["error"]
        else:
            assert "industries" in assessment
            assert "total_industries" in assessment
            assert isinstance(assessment["industries"], list)

    def test_get_country_impact_assessment(self, processed_data):
        """Test country impact assessment."""
        analyzer = TariffImpactAnalyzer(processed_data)
        assessment = analyzer.get_country_impact_assessment()

        # Check assessment structure
        assert isinstance(assessment, dict)

        # Either we have results or a clear error message
        if "error" in assessment:
            assert "No" in assessment["error"]
        else:
            assert "countries" in assessment
            assert "total_countries" in assessment
            assert isinstance(assessment["countries"], list)

    def test_get_aggregate_impact_metrics(self, processed_data):
        """Test aggregate impact metrics."""
        analyzer = TariffImpactAnalyzer(processed_data)
        metrics = analyzer.get_aggregate_impact_metrics()

        # Check metrics structure
        assert isinstance(metrics, dict)
        assert "total_events" in metrics
        assert metrics["total_events"] > 0
