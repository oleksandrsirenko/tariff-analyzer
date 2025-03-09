"""
Example usage of the Tariff Analyzer system.

This script demonstrates how to use the main components of the system
to process, analyze and visualize tariff data.
"""

import os
import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to make imports work when running directly
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.processor import TariffProcessor
from src.analysis.statistics import TariffStatistics
from src.analysis.time_series import TariffTimeSeriesAnalyzer
from src.analysis.network import TariffNetworkAnalyzer
from src.analysis.impact import TariffImpactAnalyzer
from src.visualization.dashboard import TariffDashboard
from src.visualization.geo_viz import GeoVisualizer
from src.visualization.network_viz import NetworkVisualizer
from src.visualization.time_viz import TimeVisualizer


def main():
    """Run the example tariff analysis."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Input files
    sample_data_path = data_dir / "reference" / "event_api_response_example.json"
    country_codes_path = (
        data_dir / "reference" / "country_codes_iso_3166_1_alpha_2_code.csv"
    )

    # Output files
    processed_csv = output_dir / "processed_tariffs.csv"
    dashboard_html = output_dir / "dashboard.html"
    stats_json = output_dir / "statistics.json"
    impact_json = output_dir / "impact_analysis.json"

    print(f"Loading data from {sample_data_path}")

    # Load sample data
    with open(sample_data_path, "r") as f:
        data = json.load(f)

    # 1. PREPROCESSING
    print("\nStep 1: Processing tariff data...")

    processor = TariffProcessor(str(country_codes_path))
    processed_data = processor.process(data)

    print(f"Processed {len(processed_data['events'])} events")
    print(f"Removed {processed_data['metadata']['duplicates_removed']} duplicates")

    # Export to CSV
    processor.export_to_csv(processed_data, str(processed_csv))
    print(f"Exported processed data to {processed_csv}")

    # 2. ANALYSIS
    print("\nStep 2: Analyzing tariff data...")

    # Statistical analysis
    print("- Generating statistical summary...")
    stats = TariffStatistics(processed_data)
    summary = stats.get_summary_statistics()

    # Convert numbers to make JSON serializable
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save statistics
    with open(stats_json, "w") as f:
        json.dump(summary, f, indent=2, default=convert_numpy_types)

    # Impact analysis
    print("- Analyzing economic impact...")
    impact = TariffImpactAnalyzer(processed_data)
    impact_data = {
        "industry_impact": impact.get_industry_impact_assessment(),
        "country_impact": impact.get_country_impact_assessment(),
        "aggregate_metrics": impact.get_aggregate_impact_metrics(),
    }

    # Save impact analysis
    with open(impact_json, "w") as f:
        json.dump(impact_data, f, indent=2)
    print(f"Saved impact analysis to {impact_json}")

    # Time series analysis
    print("- Analyzing temporal patterns...")
    time_analyzer = TariffTimeSeriesAnalyzer(processed_data)
    monthly_counts = time_analyzer.get_monthly_event_counts()
    yearly_counts = time_analyzer.get_yearly_event_counts()

    # Network analysis
    print("- Analyzing trade relationships...")
    network_analyzer = TariffNetworkAnalyzer(processed_data)
    network_summary = network_analyzer.get_network_summary()
    relationship_analysis = network_analyzer.get_relationship_analysis()

    # 3. VISUALIZATION
    print("\nStep 3: Creating visualizations...")

    # Dashboard visualization
    print("- Generating dashboard...")
    dashboard = TariffDashboard(processed_data)
    dashboard.generate_dashboard_html(str(dashboard_html))
    print(f"Created dashboard at {dashboard_html}")

    # Geographic visualizations
    print("- Creating geographic visualizations...")
    geo_viz = GeoVisualizer(processed_data, str(country_codes_path))

    world_map = geo_viz.plot_tariff_world_map()
    with open(output_dir / "world_map.html", "w") as f:
        f.write(
            f"<html><body><h1>Tariff World Map</h1><img src='data:image/png;base64,{world_map}'></body></html>"
        )

    # Network visualization
    print("- Creating network visualizations...")
    network_viz = NetworkVisualizer(processed_data)

    network_graph = network_viz.plot_tariff_network()
    with open(output_dir / "network_graph.html", "w") as f:
        f.write(
            f"<html><body><h1>Tariff Network Graph</h1><img src='data:image/png;base64,{network_graph}'></body></html>"
        )

    # Time series visualization
    print("- Creating time series visualizations...")
    time_viz = TimeVisualizer(processed_data)

    monthly_trend = time_viz.plot_monthly_trend()
    with open(output_dir / "monthly_trend.html", "w") as f:
        f.write(
            f"<html><body><h1>Monthly Trend</h1><img src='data:image/png;base64,{monthly_trend}'></body></html>"
        )

    print("\nAnalysis complete! All outputs saved to:", output_dir)
    print("Open dashboard.html in a web browser to view the results")


if __name__ == "__main__":
    main()
