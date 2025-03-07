"""Dashboard components for tariff data visualization."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import io
import base64
from datetime import datetime

from ..utils import logger, config


class TariffDashboard:
    """
    Main dashboard for tariff data visualization.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Initialize the dashboard with data.

        Args:
            data: DataFrame or dictionary with tariff data
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, dict) and "events" in data:
            # Convert events to DataFrame
            from ..preprocessing.processor import TariffProcessor

            processor = TariffProcessor()
            self.df = processor.to_dataframe(data)
        else:
            raise ValueError(
                "Data must be a DataFrame or a dictionary with 'events' key"
            )

        # Set default style
        self._set_plot_style()

    def _set_plot_style(self) -> None:
        """Set the default plot style for all visualizations."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        palette = config.get("visualization.color_palette", "viridis")
        sns.set_palette(palette)

    def generate_overview_stats(self) -> Dict[str, Any]:
        """
        Generate overview statistics for dashboard.

        Returns:
            Dictionary with overview statistics
        """
        stats = {
            "total_events": len(self.df),
            "time_period": None,
            "total_countries_imposing": 0,
            "total_countries_targeted": 0,
            "total_industries": 0,
            "avg_tariff_rate": 0.0,
            "top_measure_types": [],
            "retaliatory_percentage": 0.0,
        }

        # Time period
        if "announcement_date_std" in self.df.columns:
            dates = pd.to_datetime(self.df["announcement_date_std"], errors="coerce")
            if not dates.empty:
                min_date = dates.min()
                max_date = dates.max()
                if pd.notna(min_date) and pd.notna(max_date):
                    stats["time_period"] = (
                        f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                    )

        # Country counts
        if "imposing_country_code" in self.df.columns:
            stats["total_countries_imposing"] = self.df[
                "imposing_country_code"
            ].nunique()

        if "targeted_country_codes" in self.df.columns:
            # Handle list in string format
            if isinstance(
                self.df["targeted_country_codes"].iloc[0] if not self.df.empty else "",
                str,
            ):
                all_targets = set()
                for targets in self.df["targeted_country_codes"].dropna():
                    all_targets.update([t.strip() for t in targets.split(";")])
                stats["total_countries_targeted"] = len(all_targets)

        # Industries count
        if "affected_industries" in self.df.columns:
            # Handle list in string format
            if isinstance(
                self.df["affected_industries"].iloc[0] if not self.df.empty else "", str
            ):
                all_industries = set()
                for industries in self.df["affected_industries"].dropna():
                    all_industries.update([i.strip() for i in industries.split(";")])
                stats["total_industries"] = len(all_industries)

        # Average tariff rate
        if "main_tariff_rate" in self.df.columns:
            avg_rate = self.df["main_tariff_rate"].mean()
            if pd.notna(avg_rate):
                stats["avg_tariff_rate"] = round(avg_rate, 2)

        # Top measure types
        if "measure_type" in self.df.columns:
            top_types = self.df["measure_type"].value_counts().head(3)
            stats["top_measure_types"] = [
                {"type": t, "count": c} for t, c in top_types.items()
            ]

        # Retaliatory percentage
        if "is_retaliatory" in self.df.columns:
            total = len(self.df)
            if total > 0:
                retaliatory = self.df["is_retaliatory"].sum()
                stats["retaliatory_percentage"] = round((retaliatory / total) * 100, 2)

        return stats

    def plot_measure_type_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Plot distribution of measure types.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if "measure_type" not in self.df.columns:
            return ""

        plt.figure(figsize=figsize)

        # Count measure types
        measure_counts = self.df["measure_type"].value_counts()

        # Plot horizontal bar chart
        ax = sns.barplot(x=measure_counts.values, y=measure_counts.index, orient="h")

        # Add counts as text
        for i, count in enumerate(measure_counts.values):
            ax.text(count + 0.5, i, str(count), va="center")

        plt.title("Distribution of Tariff Measure Types", fontsize=15)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Measure Type", fontsize=12)
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 encoded string.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64 encoded PNG image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return img_str

    def generate_dashboard_html(self, output_path: Optional[str] = None) -> str:
        """
        Generate HTML dashboard with all visualizations.

        Args:
            output_path: Path to save HTML file (optional)

        Returns:
            HTML content as string
        """
        # Generate all plots
        measure_plot = self.plot_measure_type_distribution()
        rate_plot = self.plot_tariff_rate_histogram()
        industry_plot = self.plot_industry_impact()
        trend_plot = self.plot_monthly_trend()

        # Get overview stats
        stats = self.generate_overview_stats()

        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tariff Analysis Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .stats-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 15px;
                    flex: 1;
                    min-width: 200px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stat-card h3 {{
                    margin-top: 0;
                    color: #2c3e50;
                }}
                .plot-container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .plot-row {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .plot {{
                    flex: 1;
                    min-width: 45%;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Tariff Analysis Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="container">
                <div class="stats-container">
                    <div class="stat-card">
                        <h3>Total Events</h3>
                        <p>{stats.get('total_events', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Time Period</h3>
                        <p>{stats.get('time_period', 'N/A')}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Imposing Countries</h3>
                        <p>{stats.get('total_countries_imposing', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Targeted Countries</h3>
                        <p>{stats.get('total_countries_targeted', 0)}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Avg Tariff Rate</h3>
                        <p>{stats.get('avg_tariff_rate', 0)}%</p>
                    </div>
                    <div class="stat-card">
                        <h3>Retaliatory %</h3>
                        <p>{stats.get('retaliatory_percentage', 0)}%</p>
                    </div>
                </div>
                
                <div class="plot-row">
                    <div class="plot-container plot">
                        <h2>Measure Types</h2>
                        {'<img src="data:image/png;base64,' + measure_plot + '" width="100%">' if measure_plot else '<p>No data available</p>'}
                    </div>
                    <div class="plot-container plot">
                        <h2>Tariff Rate Distribution</h2>
                        {'<img src="data:image/png;base64,' + rate_plot + '" width="100%">' if rate_plot else '<p>No data available</p>'}
                    </div>
                </div>
                
                <div class="plot-row">
                    <div class="plot-container plot">
                        <h2>Industry Impact</h2>
                        {'<img src="data:image/png;base64,' + industry_plot + '" width="100%">' if industry_plot else '<p>No data available</p>'}
                    </div>
                    <div class="plot-container plot">
                        <h2>Monthly Trend</h2>
                        {'<img src="data:image/png;base64,' + trend_plot + '" width="100%">' if trend_plot else '<p>No data available</p>'}
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Tariff Analyzer — Generated using Python data visualization tools</p>
            </div>
        </body>
        </html>
        """

        # Save to file if path is provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html)
                logger.info(f"Dashboard saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving dashboard: {e}")

        return html.close()

        return img

    def plot_tariff_rate_histogram(self, figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Plot histogram of tariff rates.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if "main_tariff_rate" not in self.df.columns:
            return ""

        plt.figure(figsize=figsize)

        # Plot histogram
        ax = sns.histplot(
            self.df["main_tariff_rate"].dropna(), kde=True, bins=15, color="steelblue"
        )

        # Add median line
        median = self.df["main_tariff_rate"].median()
        if pd.notna(median):
            plt.axvline(
                median,
                color="darkred",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median:.1f}%",
            )
            plt.legend()

        plt.title("Distribution of Tariff Rates", fontsize=15)
        plt.xlabel("Tariff Rate (%)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_industry_impact(self, figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Plot impact by industry.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if (
            "affected_industries" not in self.df.columns
            or "main_tariff_rate" not in self.df.columns
        ):
            return ""

        # Extract industries and calculate average tariff rate by industry
        industry_tariffs = {}

        # Handle list in string format
        for _, row in self.df.dropna(
            subset=["affected_industries", "main_tariff_rate"]
        ).iterrows():
            industries = (
                [ind.strip() for ind in row["affected_industries"].split(";")]
                if isinstance(row["affected_industries"], str)
                else row["affected_industries"]
            )

            for industry in industries:
                if not industry:
                    continue

                if industry not in industry_tariffs:
                    industry_tariffs[industry] = {"rates": [], "count": 0}

                industry_tariffs[industry]["rates"].append(row["main_tariff_rate"])
                industry_tariffs[industry]["count"] += 1

        # Calculate averages
        for industry in industry_tariffs:
            industry_tariffs[industry]["avg_rate"] = (
                sum(industry_tariffs[industry]["rates"])
                / len(industry_tariffs[industry]["rates"])
                if industry_tariffs[industry]["rates"]
                else 0
            )

        # Prepare data for plotting
        industries = []
        avg_rates = []
        counts = []

        for industry, data in sorted(
            industry_tariffs.items(), key=lambda x: x[1]["avg_rate"], reverse=True
        ):
            industries.append(industry)
            avg_rates.append(data["avg_rate"])
            counts.append(data["count"])

        # Limit to top 10
        if len(industries) > 10:
            industries = industries[:10]
            avg_rates = avg_rates[:10]
            counts = counts[:10]

        plt.figure(figsize=figsize)

        # Create a color map based on count
        colors = plt.cm.viridis([count / max(counts) for count in counts])

        # Plot bars with color mapping
        bars = plt.barh(industries, avg_rates, color=colors)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(min(counts), max(counts))
        )
        sm._A = []  # Hack to make it work
        cbar = plt.colorbar(sm)
        cbar.set_label("Number of Tariff Events")

        plt.title("Average Tariff Rate by Industry", fontsize=15)
        plt.xlabel("Average Tariff Rate (%)", fontsize=12)
        plt.ylabel("Industry", fontsize=12)
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_monthly_trend(self, figsize: Tuple[int, int] = (12, 6)) -> str:
        """
        Plot monthly trend of tariff events.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if "announcement_date_std" not in self.df.columns:
            return ""

        # Convert to datetime
        self.df["date"] = pd.to_datetime(
            self.df["announcement_date_std"], errors="coerce"
        )

        if self.df["date"].isna().all():
            return ""

        # Extract month and year
        self.df["month_year"] = self.df["date"].dt.to_period("M")

        # Count events by month
        monthly_counts = self.df.groupby("month_year").size()

        # Fill in missing months
        if len(monthly_counts) >= 2:
            date_range = pd.period_range(
                start=monthly_counts.index.min(),
                end=monthly_counts.index.max(),
                freq="M",
            )
            monthly_counts = monthly_counts.reindex(date_range, fill_value=0)

        plt.figure(figsize=figsize)

        # Plot line chart
        ax = plt.plot(
            [str(period) for period in monthly_counts.index],
            monthly_counts.values,
            marker="o",
            markersize=5,
            linewidth=2,
            color="steelblue",
        )

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        # Set x-axis tick label frequency based on number of periods
        if len(monthly_counts) > 12:
            # Show every 3 months if more than a year
            plt.gca().xaxis.set_major_locator(
                plt.MaxNLocator(int(len(monthly_counts) / 3))
            )

        plt.title("Monthly Trend of Tariff Events", fontsize=15)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Number of Events", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt
