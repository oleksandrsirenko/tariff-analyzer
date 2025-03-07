"""
Integration tests.

This file contains tests that demonstrate the integration between modules.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

from src.preprocessing.processor import TariffProcessor
from src.analysis.statistics import TariffStatistics
from src.analysis.impact import TariffImpactAnalyzer
from src.visualization.dashboard import TariffDashboard


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


def test_end_to_end_pipeline(sample_data, country_codes_path, tmp_path):
    """Test the complete data processing and analysis pipeline."""
    # 1. Preprocessing
    processor = TariffProcessor(str(country_codes_path))
    processed_data = processor.process(sample_data)

    # Verify processing results
    assert "events" in processed_data
    assert len(processed_data["events"]) > 0
    assert "metadata" in processed_data

    # 2. Export to CSV
    csv_output = tmp_path / "processed_tariffs.csv"
    processor.export_to_csv(processed_data, str(csv_output))

    # Verify CSV export
    assert csv_output.exists()

    # 3. Statistical Analysis
    stats = TariffStatistics(processed_data)
    summary = stats.get_summary_statistics()

    # Verify statistics
    assert "total_events" in summary
    assert summary["total_events"] > 0

    # 4. Impact Analysis
    impact = TariffImpactAnalyzer(processed_data)
    industry_impact = impact.get_industry_impact_assessment()
    country_impact = impact.get_country_impact_assessment()

    # Verify impact analysis
    if "error" not in industry_impact:
        assert "industries" in industry_impact

    if "error" not in country_impact:
        assert "countries" in country_impact

    # 5. Visualization
    dashboard = TariffDashboard(processed_data)
    dashboard_html = dashboard.generate_dashboard_html(str(tmp_path / "dashboard.html"))

    # Verify dashboard generation
    assert (tmp_path / "dashboard.html").exists()

    # 6. Verify that the entire pipeline ran without errors
    assert True


def test_csv_roundtrip(sample_data, country_codes_path, tmp_path):
    """Test roundtrip conversion from raw data to CSV and back."""
    # 1. Process the data
    processor = TariffProcessor(str(country_codes_path))
    processed_data = processor.process(sample_data)

    # 2. Export to CSV
    csv_output = tmp_path / "processed_tariffs.csv"
    processor.export_to_csv(processed_data, str(csv_output))

    # 3. Read back the CSV
    df = processor.to_dataframe(processed_data)
    df_from_csv = pd.read_csv(str(csv_output))

    # 4. Verify key columns match
    assert len(df) == len(df_from_csv)
    for column in ["imposing_country_code", "measure_type", "main_tariff_rate"]:
        if column in df.columns and column in df_from_csv.columns:
            # Compare non-null values
            df_col = df[column].dropna()
            df_csv_col = df_from_csv[column].dropna()

            if not df_col.empty and not df_csv_col.empty:
                # Check first value is equivalent
                assert df_col.iloc[0] == df_csv_col.iloc[0] or str(
                    df_col.iloc[0]
                ) == str(df_csv_col.iloc[0])


def test_visualization_with_analysis(sample_data, country_codes_path, tmp_path):
    """Test combining analysis results with visualization."""
    # 1. Process the data
    processor = TariffProcessor(str(country_codes_path))
    processed_data = processor.process(sample_data)

    # 2. Perform impact analysis
    impact = TariffImpactAnalyzer(processed_data)
    country_impact = impact.get_country_impact_assessment()

    # 3. Convert impact assessment to HTML table
    from src.visualization.dashboard import TariffDashboard

    dashboard = TariffDashboard(processed_data)

    # Create a simple HTML report combining dashboard and impact analysis
    html_output = tmp_path / "impact_report.html"

    with open(html_output, "w") as f:
        f.write(
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tariff Impact Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Tariff Impact Analysis</h1>
        """
        )

        # Add country impact table if available
        if "error" not in country_impact and "countries" in country_impact:
            f.write(
                "<h2>Country Impact Analysis</h2><table><tr><th>Country</th><th>Targeted Count</th><th>Imposing Count</th><th>Net Position</th></tr>"
            )

            for country in country_impact["countries"][:5]:  # Top 5 countries
                f.write(
                    f"<tr><td>{country['country']}</td><td>{country['targeted_count']}</td><td>{country['imposing_count']}</td><td>{country['net_position']}</td></tr>"
                )

            f.write("</table>")

        # Add a visualization
        monthly_trend = dashboard.plot_monthly_trend()
        if monthly_trend:
            f.write("<h2>Monthly Trend</h2>")
            f.write(f'<img src="data:image/png;base64,{monthly_trend}" width="100%">')

        f.write("</body></html>")

    # Verify the report was created
    assert html_output.exists()


import pandas as pd  # Import needed for the CSV test
