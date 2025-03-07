"""
Tests for visualization components.

This file contains tests for the visualization module components.
"""

import os
import json
import pytest
from pathlib import Path

from src.preprocessing.processor import TariffProcessor
from src.visualization.dashboard import TariffDashboard
from src.visualization.geo_viz import GeoVisualizer
from src.visualization.network_viz import NetworkVisualizer
from src.visualization.time_viz import TimeVisualizer


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
def output_dir(tmp_path):
    """Fixture to create a temporary output directory."""
    return tmp_path / "visualization_output"


class TestDashboard:
    """Tests for the TariffDashboard component."""

    def test_generate_overview_stats(self, processed_data):
        """Test that overview statistics are generated correctly."""
        dashboard = TariffDashboard(processed_data)
        stats = dashboard.generate_overview_stats()

        # Check basic stats structure
        assert isinstance(stats, dict)
        assert "total_events" in stats
        assert "avg_tariff_rate" in stats

        # Check stats values
        assert stats["total_events"] > 0

    def test_generate_dashboard_html(self, processed_data, output_dir):
        """Test dashboard HTML generation."""
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / "dashboard.html"

        dashboard = TariffDashboard(processed_data)
        html = dashboard.generate_dashboard_html(str(output_file))

        # Check that file was created
        assert output_file.exists()

        # Check that HTML contains key dashboard elements
        with open(output_file, "r") as f:
            content = f.read()
            assert "<html" in content
            assert "Tariff Analysis Dashboard" in content
            assert "Number of Events" in content


class TestGeoVisualizer:
    """Tests for the GeoVisualizer component."""

    def test_plot_tariff_world_map(self, processed_data):
        """Test world map visualization."""
        base_dir = Path(__file__).parent.parent
        country_codes_path = (
            base_dir
            / "data"
            / "reference"
            / "country_codes_iso_3166_1_alpha_2_code.csv"
        )

        geo_viz = GeoVisualizer(processed_data, str(country_codes_path))
        world_map = geo_viz.plot_tariff_world_map()

        # Check that image was generated
        assert isinstance(world_map, str)
        assert len(world_map) > 0

    def test_plot_regional_tariff_intensity(self, processed_data):
        """Test regional tariff intensity visualization."""
        base_dir = Path(__file__).parent.parent
        country_codes_path = (
            base_dir
            / "data"
            / "reference"
            / "country_codes_iso_3166_1_alpha_2_code.csv"
        )

        geo_viz = GeoVisualizer(processed_data, str(country_codes_path))
        regional_chart = geo_viz.plot_regional_tariff_intensity()

        # Check that image was generated
        assert isinstance(regional_chart, str)
        assert len(regional_chart) > 0


class TestNetworkVisualizer:
    """Tests for the NetworkVisualizer component."""

    def test_build_graph(self, processed_data):
        """Test network graph building."""
        network_viz = NetworkVisualizer(processed_data)
        graph = network_viz.graph

        # Check that graph was created with nodes and edges
        assert graph is not None
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_plot_tariff_network(self, processed_data):
        """Test tariff network visualization."""
        network_viz = NetworkVisualizer(processed_data)
        network_image = network_viz.plot_tariff_network()

        # Check that image was generated
        assert isinstance(network_image, str)
        assert len(network_image) > 0


class TestTimeVisualizer:
    """Tests for the TimeVisualizer component."""

    def test_plot_monthly_trend(self, processed_data):
        """Test monthly trend visualization."""
        time_viz = TimeVisualizer(processed_data)
        monthly_trend = time_viz.plot_monthly_trend()

        # Check that image was generated
        assert isinstance(monthly_trend, str)
        assert len(monthly_trend) > 0

    def test_plot_seasonal_pattern(self, processed_data):
        """Test seasonal pattern visualization."""
        time_viz = TimeVisualizer(processed_data)
        seasonal_pattern = time_viz.plot_seasonal_pattern()

        # Check that image was generated
        assert isinstance(seasonal_pattern, str)
        assert len(seasonal_pattern) > 0
