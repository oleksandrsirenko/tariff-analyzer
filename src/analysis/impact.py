"""Economic impact analysis for tariff data."""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter

from ..utils import logger


class TariffImpactAnalyzer:
    """
    Analyzes economic impact of tariffs across industries, products, and countries.
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

        # Prepare data for impact analysis
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

            # Base record with impact-related fields
            record = {
                "id": event.get("id"),
                "imposing_country_code": tariff.get("imposing_country_code"),
                "targeted_country_codes": tariff.get("targeted_country_codes"),
                "measure_type": tariff.get("measure_type"),
                "main_tariff_rate": tariff.get("main_tariff_rate"),
                "rate_category": tariff.get("rate_category"),
                "impact_score": tariff.get("impact_score"),
                "impact_category": tariff.get("impact_category"),
                "affected_industries": tariff.get("affected_industries"),
                "affected_products": tariff.get("affected_products"),
                "hs_product_categories": tariff.get("hs_product_categories"),
                "estimated_trade_value": tariff.get("estimated_trade_value"),
                "announcement_date_std": tariff.get("announcement_date_std"),
                "parsed_rates": tariff.get("parsed_rates"),
            }

            flattened.append(record)

        return pd.DataFrame(flattened)

    def _prepare_data(self) -> None:
        """Prepare data for impact analysis."""
        # Convert trade value strings to numeric if needed
        if "estimated_trade_value" in self.df.columns:
            if self.df["estimated_trade_value"].dtype == object:
                # Extract numeric values from strings like "$300 billion"
                self.df["trade_value_numeric"] = self.df["estimated_trade_value"].apply(
                    self._extract_trade_value
                )
            else:
                # Already numeric
                self.df["trade_value_numeric"] = self.df["estimated_trade_value"]

        # Make sure impact_score is numeric
        if "impact_score" in self.df.columns:
            self.df["impact_score"] = pd.to_numeric(
                self.df["impact_score"], errors="coerce"
            )

        # Make sure main_tariff_rate is numeric
        if "main_tariff_rate" in self.df.columns:
            self.df["main_tariff_rate"] = pd.to_numeric(
                self.df["main_tariff_rate"], errors="coerce"
            )

    def _extract_trade_value(self, value_str: Optional[str]) -> Optional[float]:
        """
        Extract numeric trade value from string representation.

        Args:
            value_str: String representation of trade value

        Returns:
            Numeric value in billions of dollars or None if not extractable
        """
        if not value_str or not isinstance(value_str, str):
            return None

        # Extract numeric value and unit
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:(million|billion|trillion))?",
            value_str,
            re.IGNORECASE,
        )
        if not match:
            return None

        value = float(match.group(1))
        unit = (
            match.group(2).lower() if match.group(2) else "billion"
        )  # Default to billions

        # Convert to billions
        if unit == "trillion":
            value *= 1000  # Convert trillions to billions
        elif unit == "million":
            value /= 1000  # Convert millions to billions

        return value

    def get_industry_impact_assessment(self) -> Dict[str, Any]:
        """
        Assess the impact of tariffs across industries.

        Returns:
            Dictionary with industry impact data
        """
        if "affected_industries" not in self.df.columns:
            return {"error": "No industry data available"}

        # Process industries from DataFrame
        industries = self._extract_list_column("affected_industries")

        if not industries:
            return {"error": "Could not extract industry data"}

        # Calculate industry metrics
        industry_metrics = defaultdict(
            lambda: {
                "event_count": 0,
                "avg_rate": 0.0,
                "avg_impact": 0.0,
                "total_trade_value": 0.0,
            }
        )

        for _, row in self.df.iterrows():
            row_industries = self._get_list_values(row, "affected_industries")

            if not row_industries:
                continue

            # Extract metrics for this row
            rate = pd.to_numeric(row.get("main_tariff_rate"), errors="coerce")
            impact = pd.to_numeric(row.get("impact_score"), errors="coerce")
            trade_value = pd.to_numeric(row.get("trade_value_numeric"), errors="coerce")

            for industry in row_industries:
                metrics = industry_metrics[industry]
                metrics["event_count"] += 1

                if not pd.isna(rate):
                    metrics["avg_rate"] += rate

                if not pd.isna(impact):
                    metrics["avg_impact"] += impact

                if not pd.isna(trade_value):
                    metrics["total_trade_value"] += trade_value

        # Calculate averages
        for industry, metrics in industry_metrics.items():
            count = metrics["event_count"]
            if count > 0:
                metrics["avg_rate"] /= count
                metrics["avg_impact"] /= count

        # Format as list
        industry_impacts = []
        for industry, metrics in industry_metrics.items():
            impact_data = {
                "industry": industry,
                "event_count": metrics["event_count"],
                "avg_rate": round(metrics["avg_rate"], 2),
                "avg_impact": round(metrics["avg_impact"], 2),
                "total_trade_value": round(metrics["total_trade_value"], 2),
            }

            # Calculate vulnerability score (normalized composite)
            impact_data["vulnerability"] = round(
                (metrics["avg_rate"] / 100 * 0.3)
                + (metrics["avg_impact"] / 10 * 0.5)
                + (min(metrics["event_count"] / 10, 1) * 0.2),
                2,
            )

            industry_impacts.append(impact_data)

        # Sort by vulnerability score
        industry_impacts.sort(key=lambda x: x["vulnerability"], reverse=True)

        return {
            "industries": industry_impacts,
            "total_industries": len(industry_impacts),
        }

    def get_product_impact_assessment(self) -> Dict[str, Any]:
        """
        Assess the impact of tariffs across products.

        Returns:
            Dictionary with product impact data
        """
        if "affected_products" not in self.df.columns:
            return {"error": "No product data available"}

        # Process products from DataFrame
        products = self._extract_list_column("affected_products")

        if not products:
            return {"error": "Could not extract product data"}

        # Calculate product metrics
        product_metrics = defaultdict(
            lambda: {
                "event_count": 0,
                "avg_rate": 0.0,
                "countries": set(),
                "measure_types": set(),
            }
        )

        for _, row in self.df.iterrows():
            row_products = self._get_list_values(row, "affected_products")

            if not row_products:
                continue

            # Extract metrics for this row
            rate = pd.to_numeric(row.get("main_tariff_rate"), errors="coerce")
            imposing_country = row.get("imposing_country_code")
            measure_type = row.get("measure_type")

            for product in row_products:
                metrics = product_metrics[product]
                metrics["event_count"] += 1

                if not pd.isna(rate):
                    metrics["avg_rate"] += rate

                if imposing_country:
                    metrics["countries"].add(imposing_country)

                if measure_type:
                    metrics["measure_types"].add(measure_type)

        # Calculate averages
        for product, metrics in product_metrics.items():
            count = metrics["event_count"]
            if count > 0:
                metrics["avg_rate"] /= count

        # Format as list
        product_impacts = []
        for product, metrics in product_metrics.items():
            product_impacts.append(
                {
                    "product": product,
                    "event_count": metrics["event_count"],
                    "avg_rate": round(metrics["avg_rate"], 2),
                    "country_count": len(metrics["countries"]),
                    "countries": sorted(list(metrics["countries"])),
                    "measure_types": sorted(list(metrics["measure_types"])),
                }
            )

        # Sort by event count
        product_impacts.sort(key=lambda x: x["event_count"], reverse=True)

        return {"products": product_impacts, "total_products": len(product_impacts)}

    def get_country_impact_assessment(self) -> Dict[str, Any]:
        """
        Assess the impact of tariffs across countries.

        Returns:
            Dictionary with country impact data
        """
        if "targeted_country_codes" not in self.df.columns:
            return {"error": "No targeted country data available"}

        # Process targeted countries from DataFrame
        country_metrics = defaultdict(
            lambda: {
                "targeted_count": 0,
                "imposing_count": 0,
                "avg_rate_received": 0.0,
                "avg_rate_imposed": 0.0,
                "total_trade_value_targeted": 0.0,
                "industries_targeted": set(),
                "products_targeted": set(),
            }
        )

        # Count events where country is targeted
        for _, row in self.df.iterrows():
            targeted_countries = self._get_list_values(row, "targeted_country_codes")
            imposing_country = row.get("imposing_country_code")
            rate = pd.to_numeric(row.get("main_tariff_rate"), errors="coerce")
            trade_value = pd.to_numeric(row.get("trade_value_numeric"), errors="coerce")

            # Process as targeted
            for country in targeted_countries:
                metrics = country_metrics[country]
                metrics["targeted_count"] += 1

                if not pd.isna(rate):
                    metrics["avg_rate_received"] += rate

                if not pd.isna(trade_value):
                    metrics["total_trade_value_targeted"] += trade_value

                # Add industries and products
                for industry in self._get_list_values(row, "affected_industries"):
                    metrics["industries_targeted"].add(industry)

                for product in self._get_list_values(row, "affected_products"):
                    metrics["products_targeted"].add(product)

            # Process as imposing
            if imposing_country:
                metrics = country_metrics[imposing_country]
                metrics["imposing_count"] += 1

                if not pd.isna(rate):
                    metrics["avg_rate_imposed"] += rate

        # Calculate averages
        for country, metrics in country_metrics.items():
            if metrics["targeted_count"] > 0:
                metrics["avg_rate_received"] /= metrics["targeted_count"]

            if metrics["imposing_count"] > 0:
                metrics["avg_rate_imposed"] /= metrics["imposing_count"]

        # Format as list
        country_impacts = []
        for country, metrics in country_metrics.items():
            # Calculate net_position (positive means more imposing than targeted)
            net_position = metrics["imposing_count"] - metrics["targeted_count"]

            country_impacts.append(
                {
                    "country": country,
                    "targeted_count": metrics["targeted_count"],
                    "imposing_count": metrics["imposing_count"],
                    "net_position": net_position,
                    "avg_rate_received": round(metrics["avg_rate_received"], 2),
                    "avg_rate_imposed": round(metrics["avg_rate_imposed"], 2),
                    "rate_balance": round(
                        metrics["avg_rate_imposed"] - metrics["avg_rate_received"], 2
                    ),
                    "total_trade_value_targeted": round(
                        metrics["total_trade_value_targeted"], 2
                    ),
                    "industry_count": len(metrics["industries_targeted"]),
                    "product_count": len(metrics["products_targeted"]),
                }
            )

        # Sort by impact (targeted count)
        country_impacts.sort(key=lambda x: x["targeted_count"], reverse=True)

        return {"countries": country_impacts, "total_countries": len(country_impacts)}

    def get_measure_type_impact(self) -> Dict[str, Any]:
        """
        Assess the impact of different measure types.

        Returns:
            Dictionary with measure type impact data
        """
        if "measure_type" not in self.df.columns:
            return {"error": "No measure type data available"}

        # Group by measure type
        measure_metrics = defaultdict(
            lambda: {
                "event_count": 0,
                "avg_rate": 0.0,
                "avg_impact": 0.0,
                "total_trade_value": 0.0,
                "countries": set(),
                "industries": set(),
            }
        )

        for _, row in self.df.iterrows():
            measure_type = row.get("measure_type")

            if not measure_type:
                continue

            # Extract metrics for this row
            rate = pd.to_numeric(row.get("main_tariff_rate"), errors="coerce")
            impact = pd.to_numeric(row.get("impact_score"), errors="coerce")
            trade_value = pd.to_numeric(row.get("trade_value_numeric"), errors="coerce")
            imposing_country = row.get("imposing_country_code")

            metrics = measure_metrics[measure_type]
            metrics["event_count"] += 1

            if not pd.isna(rate):
                metrics["avg_rate"] += rate

            if not pd.isna(impact):
                metrics["avg_impact"] += impact

            if not pd.isna(trade_value):
                metrics["total_trade_value"] += trade_value

            if imposing_country:
                metrics["countries"].add(imposing_country)

            # Add industries
            for industry in self._get_list_values(row, "affected_industries"):
                metrics["industries"].add(industry)

        # Calculate averages
        for measure_type, metrics in measure_metrics.items():
            count = metrics["event_count"]
            if count > 0:
                metrics["avg_rate"] /= count
                metrics["avg_impact"] /= count

        # Format as list
        measure_impacts = []
        for measure_type, metrics in measure_metrics.items():
            measure_impacts.append(
                {
                    "measure_type": measure_type,
                    "event_count": metrics["event_count"],
                    "avg_rate": round(metrics["avg_rate"], 2),
                    "avg_impact": round(metrics["avg_impact"], 2),
                    "total_trade_value": round(metrics["total_trade_value"], 2),
                    "country_count": len(metrics["countries"]),
                    "industry_count": len(metrics["industries"]),
                }
            )

        # Sort by event count
        measure_impacts.sort(key=lambda x: x["event_count"], reverse=True)

        return {"measure_types": measure_impacts, "total_types": len(measure_impacts)}

    def get_aggregate_impact_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregate impact metrics across the dataset.

        Returns:
            Dictionary with aggregate impact metrics
        """
        metrics = {
            "total_events": len(self.df),
            "total_countries_imposing": len(
                self.df["imposing_country_code"].dropna().unique()
            ),
            "total_countries_targeted": len(
                self._extract_list_column("targeted_country_codes")
            ),
            "total_industries": len(self._extract_list_column("affected_industries")),
            "total_products": len(self._extract_list_column("affected_products")),
        }

        # Rate statistics
        if "main_tariff_rate" in self.df.columns:
            rate_data = self.df["main_tariff_rate"].dropna()
            if not rate_data.empty:
                metrics["rate_stats"] = {
                    "avg_rate": round(float(rate_data.mean()), 2),
                    "median_rate": round(float(rate_data.median()), 2),
                    "min_rate": round(float(rate_data.min()), 2),
                    "max_rate": round(float(rate_data.max()), 2),
                }

        # Impact statistics
        if "impact_score" in self.df.columns:
            impact_data = self.df["impact_score"].dropna()
            if not impact_data.empty:
                metrics["impact_stats"] = {
                    "avg_impact": round(float(impact_data.mean()), 2),
                    "median_impact": round(float(impact_data.median()), 2),
                    "min_impact": round(float(impact_data.min()), 2),
                    "max_impact": round(float(impact_data.max()), 2),
                }

        # Trade value statistics
        if "trade_value_numeric" in self.df.columns:
            trade_data = self.df["trade_value_numeric"].dropna()
            if not trade_data.empty:
                metrics["trade_value_stats"] = {
                    "total_trade_value": round(float(trade_data.sum()), 2),
                    "avg_trade_value": round(float(trade_data.mean()), 2),
                    "median_trade_value": round(float(trade_data.median()), 2),
                    "min_trade_value": round(float(trade_data.min()), 2),
                    "max_trade_value": round(float(trade_data.max()), 2),
                }

        return metrics

    def _extract_list_column(self, column_name: str) -> List[str]:
        """
        Extract unique values from a list column.

        Args:
            column_name: Name of the column containing lists

        Returns:
            List of unique values
        """
        if column_name not in self.df.columns:
            return []

        unique_values = set()

        # Check column type (list or string representation)
        sample_value = (
            self.df[column_name].dropna().iloc[0]
            if not self.df[column_name].dropna().empty
            else None
        )

        if isinstance(sample_value, list):
            # Already a list column
            for value_list in self.df[column_name].dropna():
                unique_values.update(value_list)
        else:
            # String representation - try to parse
            try:
                for value_str in self.df[column_name].dropna():
                    values = [v.strip() for v in value_str.split(";")]
                    unique_values.update(v for v in values if v)
            except:
                logger.warning(f"Could not parse list column: {column_name}")

        return list(unique_values)

    # Update the _get_list_values method in src/analysis/impact.py
    def _get_list_values(self, row: pd.Series, column_name: str) -> List[str]:
        """
        Get list values from a DataFrame row.

        Args:
            row: DataFrame row
            column_name: Name of the column containing lists

        Returns:
            List of values
        """
        # Fix the condition to check if column exists and is not null
        if column_name not in row.index or pd.isna(row[column_name]):
            return []

        value = row[column_name]

        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            # Try to parse string representation
            try:
                return [v.strip() for v in value.split(";") if v.strip()]
            except:
                return []
        else:
            return []
