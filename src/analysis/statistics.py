"""Statistical analysis for tariff data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter, defaultdict

from ..utils import logger


class TariffStatistics:
    """
    Provides statistical analysis for processed tariff data.
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
            events = data["events"]
            flattened = []

            for event in events:
                tariff = event.get("tariffs_v2", {})

                # Base record with key fields
                record = {
                    "id": event.get("id"),
                    "imposing_country_code": tariff.get("imposing_country_code"),
                    "standard_imposing_country": tariff.get(
                        "standard_imposing_country"
                    ),
                    "targeted_country_codes": tariff.get("targeted_country_codes"),
                    "standard_targeted_countries": tariff.get(
                        "standard_targeted_countries"
                    ),
                    "measure_type": tariff.get("measure_type"),
                    "main_tariff_rate": tariff.get("main_tariff_rate"),
                    "rate_category": tariff.get("rate_category"),
                    "impact_score": tariff.get("impact_score"),
                    "impact_category": tariff.get("impact_category"),
                    "announcement_date_std": tariff.get("announcement_date_std"),
                    "implementation_date_std": tariff.get("implementation_date_std"),
                    "announcement_year": tariff.get("announcement_year"),
                    "announcement_month": tariff.get("announcement_month"),
                    "announcement_quarter": tariff.get("announcement_quarter"),
                    "affected_industries": tariff.get("affected_industries"),
                    "affected_products": tariff.get("affected_products"),
                    "hs_product_categories": tariff.get("hs_product_categories"),
                    "estimated_trade_value": tariff.get("estimated_trade_value"),
                    "is_retaliatory": tariff.get("is_retaliatory"),
                }

                flattened.append(record)

            self.df = pd.DataFrame(flattened)
        else:
            raise ValueError(
                "Data must be a DataFrame or a dictionary with 'events' key"
            )

        # Precompute common statistics
        self._precompute_stats()

    def _precompute_stats(self) -> None:
        """Precompute common statistics for faster access."""
        self.total_events = len(self.df)

        # Calculate basic rate statistics if available
        if "main_tariff_rate" in self.df.columns:
            # Convert to numeric, coercing errors to NaN
            self.df["main_tariff_rate"] = pd.to_numeric(
                self.df["main_tariff_rate"], errors="coerce"
            )
            self.rate_stats = {
                "count": self.df["main_tariff_rate"].count(),
                "mean": self.df["main_tariff_rate"].mean(),
                "median": self.df["main_tariff_rate"].median(),
                "min": self.df["main_tariff_rate"].min(),
                "max": self.df["main_tariff_rate"].max(),
                "std": self.df["main_tariff_rate"].std(),
            }
        else:
            self.rate_stats = {}

        # Count by imposing country
        if "imposing_country_code" in self.df.columns:
            self.imposing_country_counts = (
                self.df["imposing_country_code"].value_counts().to_dict()
            )
        else:
            self.imposing_country_counts = {}

        # Count by measure type
        if "measure_type" in self.df.columns:
            self.measure_type_counts = self.df["measure_type"].value_counts().to_dict()
        else:
            self.measure_type_counts = {}

        # Count by rate category
        if "rate_category" in self.df.columns:
            self.rate_category_counts = (
                self.df["rate_category"].value_counts().to_dict()
            )
        else:
            self.rate_category_counts = {}

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the tariff data.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_events": self.total_events,
            "rate_statistics": self.rate_stats,
            "top_imposing_countries": self._get_top_items(
                self.imposing_country_counts, 5
            ),
            "measure_types": self.measure_type_counts,
            "rate_categories": self.rate_category_counts,
        }

        # Add targeted countries
        if "targeted_country_codes" in self.df.columns:
            # Handle lists in DataFrame
            if isinstance(self.df["targeted_country_codes"].iloc[0], list):
                targeted_counts = Counter()
                for codes in self.df["targeted_country_codes"].dropna():
                    targeted_counts.update(codes)

                summary["top_targeted_countries"] = self._get_top_items(
                    targeted_counts, 5
                )
            else:
                # Handle string representation
                summary["top_targeted_countries"] = self._get_top_items(
                    self.df["targeted_country_codes"].value_counts().to_dict(), 5
                )

        # Add industry statistics
        if "affected_industries" in self.df.columns:
            # Handle lists in DataFrame
            if isinstance(self.df["affected_industries"].iloc[0], list):
                industry_counts = Counter()
                for industries in self.df["affected_industries"].dropna():
                    industry_counts.update(industries)

                summary["top_industries"] = self._get_top_items(industry_counts, 5)
            else:
                # Try to parse from string representation
                try:
                    industry_counts = Counter()
                    for industry_str in self.df["affected_industries"].dropna():
                        industries = [i.strip() for i in industry_str.split(";")]
                        industry_counts.update(industries)

                    summary["top_industries"] = self._get_top_items(industry_counts, 5)
                except:
                    logger.warning(
                        "Could not parse industries from string representation"
                    )

        # Add product statistics
        if "affected_products" in self.df.columns:
            # Handle lists in DataFrame
            if isinstance(self.df["affected_products"].iloc[0], list):
                product_counts = Counter()
                for products in self.df["affected_products"].dropna():
                    product_counts.update(products)

                summary["top_products"] = self._get_top_items(product_counts, 5)
            else:
                # Try to parse from string representation
                try:
                    product_counts = Counter()
                    for product_str in self.df["affected_products"].dropna():
                        products = [p.strip() for p in product_str.split(";")]
                        product_counts.update(products)

                    summary["top_products"] = self._get_top_items(product_counts, 5)
                except:
                    logger.warning(
                        "Could not parse products from string representation"
                    )

        # Add time-based statistics
        if "announcement_year" in self.df.columns:
            summary["events_by_year"] = (
                self.df["announcement_year"].value_counts().to_dict()
            )

        if "announcement_quarter" in self.df.columns:
            summary["events_by_quarter"] = (
                self.df["announcement_quarter"].value_counts().to_dict()
            )

        if "is_retaliatory" in self.df.columns:
            summary["retaliatory_count"] = self.df["is_retaliatory"].sum()

        return summary

    def _get_top_items(
        self, counter: Dict[str, int], n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get the top N items from a counter.

        Args:
            counter: Counter dictionary
            n: Number of top items to return

        Returns:
            List of dictionaries with name and count
        """
        return [
            {"name": name, "count": count}
            for name, count in sorted(
                counter.items(), key=lambda x: x[1], reverse=True
            )[:n]
        ]

    def get_rate_distribution(self, bins: int = 10) -> Dict[str, Any]:
        """
        Get the distribution of tariff rates.

        Args:
            bins: Number of bins for the histogram

        Returns:
            Dictionary with rate distribution data
        """
        if "main_tariff_rate" not in self.df.columns:
            return {"error": "No tariff rate data available"}

        # Filter out non-numeric values
        rates = self.df["main_tariff_rate"].dropna()

        # Calculate histogram
        hist, bin_edges = np.histogram(rates, bins=bins)

        # Create bins with labels
        bin_labels = [
            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            for i in range(len(bin_edges) - 1)
        ]

        # Return distribution data
        return {
            "bins": bin_labels,
            "counts": hist.tolist(),
            "min_rate": rates.min(),
            "max_rate": rates.max(),
        }

    def get_industry_impact(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate impact metrics by industry.

        Returns:
            Dictionary with industry impact metrics
        """
        if (
            "affected_industries" not in self.df.columns
            or "impact_score" not in self.df.columns
        ):
            return {"error": "Missing required columns"}

        # Initialize industry metrics
        industry_metrics = defaultdict(
            lambda: {"event_count": 0, "avg_impact": 0.0, "avg_rate": 0.0}
        )

        # Handle list column vs string representation
        if isinstance(self.df["affected_industries"].iloc[0], list):
            # Process each event
            for _, row in self.df.dropna(
                subset=["affected_industries", "impact_score"]
            ).iterrows():
                industries = row["affected_industries"]
                impact = row["impact_score"]
                rate = row.get("main_tariff_rate", 0)

                for industry in industries:
                    metrics = industry_metrics[industry]
                    metrics["event_count"] += 1
                    metrics["avg_impact"] += impact
                    if pd.notna(rate):
                        metrics["avg_rate"] += rate
        else:
            # Try to parse from string representation
            try:
                for _, row in self.df.dropna(
                    subset=["affected_industries", "impact_score"]
                ).iterrows():
                    industries = [
                        i.strip() for i in row["affected_industries"].split(";")
                    ]
                    impact = row["impact_score"]
                    rate = row.get("main_tariff_rate", 0)

                    for industry in industries:
                        if not industry:
                            continue

                        metrics = industry_metrics[industry]
                        metrics["event_count"] += 1
                        metrics["avg_impact"] += impact
                        if pd.notna(rate):
                            metrics["avg_rate"] += rate
            except:
                logger.warning("Could not parse industries from string representation")

        # Calculate averages
        for industry, metrics in industry_metrics.items():
            count = metrics["event_count"]
            if count > 0:
                metrics["avg_impact"] /= count
                metrics["avg_rate"] /= count

        return dict(industry_metrics)

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """
        Calculate correlation matrix between numeric variables.

        Returns:
            Dictionary with correlation data
        """
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {"error": "No numeric columns available"}

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().round(2)

        # Convert to dictionary format
        result = {
            "columns": corr_matrix.columns.tolist(),
            "data": corr_matrix.values.tolist(),
        }

        return result

    def get_cross_tabulation(
        self, row_field: str, col_field: str, normalize: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a cross-tabulation of two categorical variables.

        Args:
            row_field: Field name for rows
            col_field: Field name for columns
            normalize: Whether to normalize by row count

        Returns:
            Dictionary with cross-tabulation data
        """
        if row_field not in self.df.columns or col_field not in self.df.columns:
            return {"error": f"Field not found: {row_field} or {col_field}"}

        # Create cross-tabulation
        try:
            cross_tab = pd.crosstab(
                self.df[row_field], self.df[col_field], normalize=normalize
            )

            # Convert to dictionary format
            result = {
                "rows": cross_tab.index.tolist(),
                "columns": cross_tab.columns.tolist(),
                "data": cross_tab.values.tolist(),
                "normalized": normalize,
            }

            return result
        except Exception as e:
            logger.error(f"Error creating cross-tabulation: {e}")
            return {"error": str(e)}
