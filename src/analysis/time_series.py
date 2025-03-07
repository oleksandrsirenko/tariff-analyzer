"""Time series analysis for tariff data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
from datetime import datetime

from ..utils import logger


class TariffTimeSeriesAnalyzer:
    """
    Analyzes temporal patterns in tariff data.
    """

    def __init__(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        """
        Initialize with either a DataFrame or processed tariff data dictionary.

        Args:
            data: Processed tariff data as DataFrame or dictionary
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, dict) and "events" in data:
            # Convert events to DataFrame
            self.df = self._convert_events_to_df(data["events"])
        else:
            raise ValueError(
                "Data must be a DataFrame or a dictionary with 'events' key"
            )

        # Ensure proper date format for time series analysis
        self._prepare_data()

    def _convert_events_to_df(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert events to DataFrame.

        Args:
            events: List of event dictionaries

        Returns:
            DataFrame with flattened event data
        """
        flattened = []

        for event in events:
            tariff = event.get("tariffs_v2", {})

            # Base record with key time-related fields
            record = {
                "id": event.get("id"),
                "announcement_date": tariff.get("announcement_date"),
                "announcement_date_std": tariff.get("announcement_date_std"),
                "implementation_date": tariff.get("implementation_date"),
                "implementation_date_std": tariff.get("implementation_date_std"),
                "expiration_date": tariff.get("expiration_date"),
                "expiration_date_std": tariff.get("expiration_date_std"),
                "announcement_year": tariff.get("announcement_year"),
                "announcement_month": tariff.get("announcement_month"),
                "announcement_quarter": tariff.get("announcement_quarter"),
                "days_to_implementation": tariff.get("days_to_implementation"),
                "implementation_duration": tariff.get("implementation_duration"),
                "days_since_announcement": tariff.get("days_since_announcement"),
                "measure_type": tariff.get("measure_type"),
                "main_tariff_rate": tariff.get("main_tariff_rate"),
                "imposing_country_code": tariff.get("imposing_country_code"),
                "targeted_country_codes": tariff.get("targeted_country_codes"),
                "is_retaliatory": tariff.get("is_retaliatory"),
            }

            flattened.append(record)

        return pd.DataFrame(flattened)

    def _prepare_data(self) -> None:
        """Prepare data for time series analysis."""
        # Convert date strings to datetime objects
        date_cols = [
            "announcement_date_std",
            "implementation_date_std",
            "expiration_date_std",
        ]

        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")

        # Make sure numeric columns are numeric
        numeric_cols = [
            "main_tariff_rate",
            "days_to_implementation",
            "implementation_duration",
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def get_monthly_event_counts(self) -> Dict[str, Any]:
        """
        Get monthly counts of tariff events.

        Returns:
            Dictionary with monthly event counts
        """
        if "announcement_date_std" not in self.df.columns:
            return {"error": "No announcement date data available"}

        # Drop missing dates
        date_df = self.df.dropna(subset=["announcement_date_std"])

        if date_df.empty:
            return {"error": "No valid dates found"}

        # Resample by month and count events
        date_df["month"] = date_df["announcement_date_std"].dt.to_period("M")
        monthly_counts = date_df.groupby("month").size()

        # Format as time series data
        result = {
            "dates": [str(period) for period in monthly_counts.index],
            "counts": monthly_counts.tolist(),
        }

        return result

    def get_quarterly_event_counts(self) -> Dict[str, Any]:
        """
        Get quarterly counts of tariff events.

        Returns:
            Dictionary with quarterly event counts
        """
        if "announcement_date_std" not in self.df.columns:
            return {"error": "No announcement date data available"}

        # Drop missing dates
        date_df = self.df.dropna(subset=["announcement_date_std"])

        if date_df.empty:
            return {"error": "No valid dates found"}

        # Resample by quarter and count events
        date_df["quarter"] = date_df["announcement_date_std"].dt.to_period("Q")
        quarterly_counts = date_df.groupby("quarter").size()

        # Format as time series data
        result = {
            "dates": [str(period) for period in quarterly_counts.index],
            "counts": quarterly_counts.tolist(),
        }

        return result

    def get_yearly_event_counts(self) -> Dict[str, Any]:
        """
        Get yearly counts of tariff events.

        Returns:
            Dictionary with yearly event counts
        """
        if "announcement_year" in self.df.columns:
            # Use pre-extracted year
            yearly_counts = self.df.groupby("announcement_year").size()

            # Format as time series data
            result = {
                "years": yearly_counts.index.tolist(),
                "counts": yearly_counts.tolist(),
            }

            return result
        elif "announcement_date_std" in self.df.columns:
            # Extract year from date
            date_df = self.df.dropna(subset=["announcement_date_std"])

            if date_df.empty:
                return {"error": "No valid dates found"}

            date_df["year"] = date_df["announcement_date_std"].dt.year
            yearly_counts = date_df.groupby("year").size()

            # Format as time series data
            result = {
                "years": yearly_counts.index.tolist(),
                "counts": yearly_counts.tolist(),
            }

            return result
        else:
            return {"error": "No year or date data available"}

    def get_implementation_timeline(self) -> Dict[str, Any]:
        """
        Analyze the timeline from announcement to implementation.

        Returns:
            Dictionary with implementation timeline statistics
        """
        if "days_to_implementation" not in self.df.columns:
            return {"error": "No implementation delay data available"}

        # Filter out missing values
        delay_data = self.df["days_to_implementation"].dropna()

        if delay_data.empty:
            return {"error": "No valid implementation delay data found"}

        # Calculate statistics
        stats = {
            "count": int(delay_data.count()),
            "mean": float(delay_data.mean()),
            "median": float(delay_data.median()),
            "min": float(delay_data.min()),
            "max": float(delay_data.max()),
            "std": float(delay_data.std()),
        }

        # Create histogram of implementation delays
        hist, bin_edges = np.histogram(delay_data, bins=10)

        # Format bin labels
        bin_labels = [
            f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            for i in range(len(bin_edges) - 1)
        ]

        result = {
            "statistics": stats,
            "histogram": {"bins": bin_labels, "counts": hist.tolist()},
        }

        return result

    def get_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in tariff announcements.

        Returns:
            Dictionary with seasonal pattern data
        """
        if "announcement_date_std" not in self.df.columns:
            return {"error": "No announcement date data available"}

        # Drop missing dates
        date_df = self.df.dropna(subset=["announcement_date_std"])

        if date_df.empty:
            return {"error": "No valid dates found"}

        # Extract month and count events
        date_df["month"] = date_df["announcement_date_std"].dt.month
        monthly_counts = date_df.groupby("month").size()

        # Ensure all months are represented
        all_months = pd.Series(index=range(1, 13), data=0)
        monthly_counts = monthly_counts.add(all_months, fill_value=0)

        # Month names for labeling
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

        result = {"months": month_names, "counts": monthly_counts.tolist()}

        return result

    def get_measure_type_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in measure types over time.

        Returns:
            Dictionary with measure type trends
        """
        if (
            "announcement_date_std" not in self.df.columns
            or "measure_type" not in self.df.columns
        ):
            return {"error": "Missing required columns"}

        # Drop missing values
        trend_df = self.df.dropna(subset=["announcement_date_std", "measure_type"])

        if trend_df.empty:
            return {"error": "No valid data found"}

        # Extract year and quarter
        trend_df["year"] = trend_df["announcement_date_std"].dt.year
        trend_df["quarter"] = trend_df["announcement_date_std"].dt.quarter
        trend_df["year_quarter"] = (
            trend_df["year"].astype(str) + "-Q" + trend_df["quarter"].astype(str)
        )

        # Count measure types by time period
        pivot_table = pd.pivot_table(
            trend_df,
            index="year_quarter",
            columns="measure_type",
            aggfunc="size",
            fill_value=0,
        )

        # Convert to time series format
        result = {
            "time_periods": pivot_table.index.tolist(),
            "measure_types": pivot_table.columns.tolist(),
            "data": pivot_table.values.tolist(),
        }

        return result

    def get_retaliation_timeline(self) -> Dict[str, Any]:
        """
        Analyze the timeline of retaliatory tariffs.

        Returns:
            Dictionary with retaliation timeline data
        """
        if (
            "announcement_date_std" not in self.df.columns
            or "is_retaliatory" not in self.df.columns
        ):
            return {"error": "Missing required columns"}

        # Drop missing values
        retaliation_df = self.df.dropna(
            subset=["announcement_date_std", "is_retaliatory"]
        )

        if retaliation_df.empty:
            return {"error": "No valid data found"}

        # Extract month and count retaliatory vs. non-retaliatory events
        retaliation_df["month"] = retaliation_df["announcement_date_std"].dt.to_period(
            "M"
        )

        # Group by month and retaliatory status
        grouped = (
            retaliation_df.groupby(["month", "is_retaliatory"])
            .size()
            .unstack(fill_value=0)
        )

        # Rename columns for clarity
        if 1 in grouped.columns:
            grouped = grouped.rename(columns={1: "retaliatory", 0: "non_retaliatory"})

        # Format as time series data
        months = [str(period) for period in grouped.index]

        result = {
            "months": months,
            "retaliatory": grouped.get(
                "retaliatory", pd.Series(0, index=grouped.index)
            ).tolist(),
            "non_retaliatory": grouped.get(
                "non_retaliatory", pd.Series(0, index=grouped.index)
            ).tolist(),
        }

        return result

    def get_tariff_rate_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in tariff rates over time.

        Returns:
            Dictionary with tariff rate trend data
        """
        if (
            "announcement_date_std" not in self.df.columns
            or "main_tariff_rate" not in self.df.columns
        ):
            return {"error": "Missing required columns"}

        # Drop missing values
        rate_df = self.df.dropna(subset=["announcement_date_std", "main_tariff_rate"])

        if rate_df.empty:
            return {"error": "No valid data found"}

        # Extract quarter
        rate_df["quarter"] = rate_df["announcement_date_std"].dt.to_period("Q")

        # Calculate average rate by quarter
        quarterly_rates = rate_df.groupby("quarter")["main_tariff_rate"].agg(
            ["mean", "median", "count"]
        )

        # Format as time series data
        quarters = [str(period) for period in quarterly_rates.index]

        result = {
            "quarters": quarters,
            "mean_rates": quarterly_rates["mean"].tolist(),
            "median_rates": quarterly_rates["median"].tolist(),
            "counts": quarterly_rates["count"].tolist(),
        }

        return result
