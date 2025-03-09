"""Time series visualizations for tariff data."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import io
import base64
from datetime import datetime
import sys

sys.path.append("..")

from ..utils import logger, config


class TimeVisualizer:
    """
    Time series visualizations for tariff data.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Initialize the time visualizer.

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

        # Process dates
        self._process_dates()

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")

    def _process_dates(self) -> None:
        """Process dates in the DataFrame for time series analysis."""
        # Convert date strings to datetime objects
        date_columns = [
            "announcement_date_std",
            "implementation_date_std",
            "expiration_date_std",
        ]

        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        # Create period columns if they don't exist
        if "announcement_date_std" in self.df.columns:
            if "announcement_year" not in self.df.columns:
                self.df["announcement_year"] = self.df["announcement_date_std"].dt.year

            if "announcement_month" not in self.df.columns:
                self.df["announcement_month"] = self.df[
                    "announcement_date_std"
                ].dt.month

            if "announcement_quarter" not in self.df.columns:
                self.df["announcement_quarter"] = self.df[
                    "announcement_date_std"
                ].dt.quarter

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

        # Drop rows with missing announcement date
        df_valid = self.df.dropna(subset=["announcement_date_std"])

        if df_valid.empty:
            return ""

        # Convert to period for grouping
        df_valid["period"] = df_valid["announcement_date_std"].dt.to_period("M")

        # Count events by period
        monthly_counts = df_valid.groupby("period").size()

        # Convert index to datetime for proper plotting
        monthly_dates = monthly_counts.index.to_timestamp()

        plt.figure(figsize=figsize)

        # Plot line
        plt.plot(
            monthly_dates,
            monthly_counts,
            marker="o",
            linestyle="-",
            color="steelblue",
            linewidth=2,
            markersize=6,
        )

        # Fill area under the curve
        plt.fill_between(monthly_dates, monthly_counts, alpha=0.3, color="steelblue")

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        # Add grid lines
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add labels and title
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Number of Tariff Events", fontsize=12)
        plt.title("Monthly Trend of Tariff Events", fontsize=15)

        # Add annotations for peaks
        threshold = monthly_counts.mean() + monthly_counts.std()
        for date, count in zip(monthly_dates, monthly_counts):
            if count > threshold:
                plt.annotate(
                    f"{count}",
                    xy=(date, count),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_tariff_rate_trend(self, figsize: Tuple[int, int] = (12, 6)) -> str:
        """
        Plot trend of average tariff rates over time.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if (
            "announcement_date_std" not in self.df.columns
            or "main_tariff_rate" not in self.df.columns
        ):
            return ""

        # Drop rows with missing values
        df_valid = self.df.dropna(subset=["announcement_date_std", "main_tariff_rate"])

        if df_valid.empty:
            return ""

        # Convert to period for grouping
        df_valid["quarter"] = df_valid["announcement_date_std"].dt.to_period("Q")

        # Calculate average rate by quarter
        quarterly_rates = df_valid.groupby("quarter")["main_tariff_rate"].agg(
            ["mean", "median", "std", "count"]
        )

        # Convert index to datetime for proper plotting
        quarterly_dates = quarterly_rates.index.to_timestamp()

        plt.figure(figsize=figsize)

        # Plot mean rate with error band
        plt.plot(
            quarterly_dates,
            quarterly_rates["mean"],
            marker="o",
            linestyle="-",
            color="steelblue",
            linewidth=2,
            label="Mean Rate",
        )

        # Add error band (standard deviation)
        plt.fill_between(
            quarterly_dates,
            quarterly_rates["mean"] - quarterly_rates["std"],
            quarterly_rates["mean"] + quarterly_rates["std"],
            alpha=0.2,
            color="steelblue",
            label="±1 Std Dev",
        )

        # Add median line
        plt.plot(
            quarterly_dates,
            quarterly_rates["median"],
            marker="^",
            linestyle="--",
            color="forestgreen",
            linewidth=1.5,
            label="Median Rate",
        )

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y Q%q"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha="right")

        # Create twin axis for count
        ax2 = plt.gca().twinx()
        ax2.bar(
            quarterly_dates,
            quarterly_rates["count"],
            alpha=0.15,
            color="gray",
            width=90,  # Width in days
        )
        ax2.set_ylabel("Number of Events", color="gray")

        # Add grid lines
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add labels and title
        plt.xlabel("Quarter", fontsize=12)
        plt.ylabel("Average Tariff Rate (%)", fontsize=12)
        plt.title("Trend of Tariff Rates Over Time", fontsize=15)

        # Add legend
        plt.legend()

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_retaliatory_trend(self, figsize: Tuple[int, int] = (12, 6)) -> str:
        """
        Plot trend of retaliatory vs. non-retaliatory tariffs over time.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if (
            "announcement_date_std" not in self.df.columns
            or "is_retaliatory" not in self.df.columns
        ):
            return ""

        # Drop rows with missing values
        df_valid = self.df.dropna(subset=["announcement_date_std"])

        if df_valid.empty:
            return ""

        # Convert to period for grouping
        df_valid["quarter"] = df_valid["announcement_date_std"].dt.to_period("Q")

        # Ensure is_retaliatory is boolean
        if df_valid["is_retaliatory"].dtype != bool:
            df_valid["is_retaliatory"] = df_valid["is_retaliatory"].astype(bool)

        # Group by quarter and retaliatory status
        quarterly_retaliation = (
            df_valid.groupby(["quarter", "is_retaliatory"]).size().unstack(fill_value=0)
        )

        # Rename columns for clarity
        quarterly_retaliation.columns = ["Regular", "Retaliatory"]

        # Calculate percentage of retaliatory
        quarterly_retaliation["Total"] = quarterly_retaliation.sum(axis=1)
        quarterly_retaliation["Retaliatory_Pct"] = (
            quarterly_retaliation["Retaliatory"] / quarterly_retaliation["Total"] * 100
        )

        # Sort by time
        quarterly_retaliation = quarterly_retaliation.sort_index()

        # Convert index to datetime for proper plotting
        quarterly_dates = quarterly_retaliation.index.to_timestamp()

        plt.figure(figsize=figsize)

        # Create subplot grid
        gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 0.5])

        # Plot stacked bars for counts
        ax1 = plt.subplot(gs[0])
        quarterly_retaliation[["Regular", "Retaliatory"]].plot(
            kind="bar",
            stacked=True,
            ax=ax1,
            color=["steelblue", "firebrick"],
            width=0.8,
        )

        ax1.set_xlabel("")
        ax1.set_ylabel("Number of Events", fontsize=12)
        ax1.set_title("Retaliatory vs. Regular Tariffs Over Time", fontsize=15)
        ax1.legend(loc="upper left")

        # Rotate x-axis labels
        plt.setp(ax1.get_xticklabels(), rotation=0)

        # Plot retaliatory percentage trend
        ax2 = plt.subplot(gs[1])
        ax2.plot(
            range(len(quarterly_dates)),
            quarterly_retaliation["Retaliatory_Pct"],
            marker="o",
            linestyle="-",
            color="darkorange",
            linewidth=2,
        )

        # Add threshold line at 50%
        ax2.axhline(50, color="gray", linestyle="--", alpha=0.7, label="50% threshold")

        # Highlight periods with high retaliation
        for i, pct in enumerate(quarterly_retaliation["Retaliatory_Pct"]):
            if pct > 50:
                ax2.axvspan(i - 0.4, i + 0.4, color="lightsalmon", alpha=0.3)

        ax2.set_ylabel("% Retaliatory", fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Use same x-ticks as the top plot
        ax2.set_xticks(range(len(quarterly_dates)))
        ax2.set_xticklabels([str(d) for d in quarterly_retaliation.index])
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        # Add colorful periods explanation
        ax3 = plt.subplot(gs[2])
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            "⚠ Highlighted periods show quarters with majority retaliatory tariffs (>50%)",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            bbox=dict(facecolor="lightsalmon", alpha=0.3, boxstyle="round,pad=0.5"),
        )

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_seasonal_pattern(self, figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Plot seasonal pattern of tariff events by month.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if "announcement_date_std" not in self.df.columns:
            return ""

        # Drop rows with missing values
        df_valid = self.df.dropna(subset=["announcement_date_std"])

        if df_valid.empty:
            return ""

        # Extract month from date
        df_valid["month"] = df_valid["announcement_date_std"].dt.month

        # Count events by month
        monthly_counts = df_valid.groupby("month").size()

        # Ensure all months are represented
        all_months = pd.Series(index=range(1, 13), data=0)
        monthly_counts = monthly_counts.add(all_months, fill_value=0)

        # Month names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        plt.figure(figsize=figsize)

        # Create bar chart
        bars = plt.bar(
            monthly_counts.index,
            monthly_counts.values,
            color=plt.cm.viridis(np.linspace(0, 0.9, 12)),
            width=0.7,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom",
            )

        # Set x-axis labels
        plt.xticks(range(1, 13), month_names)

        # Add trend line (polynomial fit)
        x = np.array(range(1, 13))
        y = monthly_counts.values
        z = np.polyfit(x, y, 4)
        p = np.poly1d(z)

        plt.plot(
            x, p(x), linestyle="--", color="firebrick", linewidth=2, label="Trend Line"
        )

        # Find and mark peak months
        peak_threshold = monthly_counts.mean() + 0.5 * monthly_counts.std()
        peak_months = [
            i for i, count in enumerate(monthly_counts, 1) if count > peak_threshold
        ]

        if peak_months:
            for month in peak_months:
                plt.annotate(
                    "Peak",
                    xy=(month, monthly_counts[month]),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"),
                )

        # Add labels and title
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Average Number of Events", fontsize=12)
        plt.title("Seasonal Pattern of Tariff Announcements", fontsize=15)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
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

    def plot_quarterly_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Plot quarterly comparison of tariff events by year.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if (
            "announcement_year" not in self.df.columns
            or "announcement_quarter" not in self.df.columns
        ):
            return ""

        # Group by year and quarter
        quarterly = (
            self.df.groupby(["announcement_year", "announcement_quarter"])
            .size()
            .unstack()
        )

        if quarterly.empty:
            return ""

        plt.figure(figsize=figsize)

        # Plot quarters as grouped bars
        quarterly.plot(
            kind="bar", ax=plt.gca(), width=0.8, color=sns.color_palette("viridis", 4)
        )

        # Add value labels on bars
        for container in plt.gca().containers:
            plt.bar_label(container, fmt="%d", padding=3)

        # Add labels and title
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Number of Tariff Events", fontsize=12)
        plt.title("Quarterly Comparison of Tariff Events by Year", fontsize=15)

        # Improve legend
        plt.legend(
            title="Quarter",
            labels=[f"Q{int(q)}" for q in quarterly.columns],
            loc="upper left",
        )

        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_measure_type_evolution(self, figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Plot evolution of measure types over time.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if (
            "announcement_date_std" not in self.df.columns
            or "measure_type" not in self.df.columns
        ):
            return ""

        # Drop rows with missing values
        df_valid = self.df.dropna(subset=["announcement_date_std", "measure_type"])

        if df_valid.empty:
            return ""

        # Convert to period for grouping
        df_valid["quarter"] = df_valid["announcement_date_std"].dt.to_period("Q")

        # Count events by quarter and measure type
        quarterly_types = (
            df_valid.groupby(["quarter", "measure_type"]).size().unstack(fill_value=0)
        )

        # Sort by time
        quarterly_types = quarterly_types.sort_index()

        # Plot data
        plt.figure(figsize=figsize)

        # Select color palette based on the number of measure types
        num_types = len(quarterly_types.columns)
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_types))

        # Plot stacked area chart
        quarterly_types.plot(
            kind="area", stacked=True, color=colors, alpha=0.7, ax=plt.gca()
        )

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, pos: (
                    str(quarterly_types.index[int(x)])
                    if int(x) < len(quarterly_types.index)
                    else ""
                )
            )
        )
        plt.xticks(rotation=45, ha="right")

        # Add labels and title
        plt.xlabel("Quarter", fontsize=12)
        plt.ylabel("Number of Tariff Events", fontsize=12)
        plt.title("Evolution of Tariff Measure Types", fontsize=15)

        # Improve legend
        plt.legend(title="Measure Type", loc="upper left", bbox_to_anchor=(1, 1))

        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img

    def plot_implementation_delay_histogram(
        self, figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """
        Plot histogram of delay between announcement and implementation.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Base64 encoded PNG image
        """
        if "days_to_implementation" not in self.df.columns:
            return ""

        # Drop rows with missing values
        delay_data = self.df.dropna(subset=["days_to_implementation"])

        if delay_data.empty:
            return ""

        plt.figure(figsize=figsize)

        # Plot histogram with KDE
        sns.histplot(
            delay_data["days_to_implementation"], kde=True, bins=20, color="steelblue"
        )

        # Add vertical line for mean
        mean_delay = delay_data["days_to_implementation"].mean()
        plt.axvline(
            mean_delay,
            color="firebrick",
            linestyle="--",
            label=f"Mean: {mean_delay:.1f} days",
        )

        # Add vertical line for median
        median_delay = delay_data["days_to_implementation"].median()
        plt.axvline(
            median_delay,
            color="darkgreen",
            linestyle="-.",
            label=f"Median: {median_delay:.1f} days",
        )

        # Add labels and title
        plt.xlabel("Days from Announcement to Implementation", fontsize=12)
        plt.ylabel("Number of Tariff Events", fontsize=12)
        plt.title("Distribution of Implementation Delay", fontsize=15)

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        # Convert plot to base64 string
        img = self._fig_to_base64(plt.gcf())
        plt.close()

        return img
