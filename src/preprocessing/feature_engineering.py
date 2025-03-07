"""Feature engineering for tariff data."""

import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict

from ..utils import logger


class TariffFeatureEngineer:
    """
    Generates derived features from tariff data.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        # Rate category thresholds
        self.rate_thresholds = {
            "zero": 0,
            "low": 10,
            "medium": 25,
            "high": float("inf"),
        }

        # Industry impact multipliers (for calculating impact scores)
        self.industry_impact = {
            "Energy": 5.0,
            "Materials": 4.5,
            "Industrials": 4.0,
            "Consumer Discretionary": 3.5,
            "Consumer Staples": 3.0,
            "Health Care": 2.5,
            "Financials": 2.0,
            "Information Technology": 4.5,
            "Communication Services": 3.0,
            "Utilities": 3.5,
            "Real Estate": 2.0,
        }

    def add_derived_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add derived features to a tariff event.

        Args:
            event: Event to enhance with features

        Returns:
            Event with added features
        """
        # Copy to avoid modifying the input
        enriched_event = event.copy()
        tariff = enriched_event.get("tariffs_v2", {})

        # 1. Categorize tariff rate
        self._add_rate_category(tariff)

        # 2. Add impact score
        self._add_impact_score(tariff)

        # 3. Add relationship features
        self._add_relationship_features(tariff)

        # 4. Add time-based features (if not already present)
        self._ensure_time_features(tariff)

        return enriched_event

    def _add_rate_category(self, tariff: Dict[str, Any]) -> None:
        """
        Categorize tariff rate based on thresholds.

        Args:
            tariff: Tariff data to enhance
        """
        if "main_tariff_rate" not in tariff or tariff["main_tariff_rate"] is None:
            return

        rate = tariff["main_tariff_rate"]

        # Handle non-numeric rates
        if not isinstance(rate, (int, float)):
            if isinstance(rate, str) and rate.lower() == "double":
                tariff["rate_category"] = "high"
            return

        # Categorize based on thresholds
        if rate == 0:
            tariff["rate_category"] = "zero"
        elif rate <= self.rate_thresholds["low"]:
            tariff["rate_category"] = "low"
        elif rate <= self.rate_thresholds["medium"]:
            tariff["rate_category"] = "medium"
        else:
            tariff["rate_category"] = "high"

    def _add_impact_score(self, tariff: Dict[str, Any]) -> None:
        """
        Calculate and add impact score based on industries, rate and trade value.

        Args:
            tariff: Tariff data to enhance
        """
        # Base impact from rate
        base_impact = 0
        if "main_tariff_rate" in tariff and isinstance(
            tariff["main_tariff_rate"], (int, float)
        ):
            base_impact = tariff["main_tariff_rate"] / 10  # Scale to 0-10

        # Industry multiplier
        industry_factor = 1.0
        if "affected_industries" in tariff and tariff["affected_industries"]:
            industry_weights = [
                self.industry_impact.get(industry, 1.0)
                for industry in tariff["affected_industries"]
            ]
            if industry_weights:
                industry_factor = sum(industry_weights) / len(industry_weights)

        # Trade value factor
        trade_factor = 1.0
        if "estimated_trade_value" in tariff and tariff["estimated_trade_value"]:
            # Try to extract numeric value from string like "$300 billion"
            if isinstance(tariff["estimated_trade_value"], str):
                value_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:billion|million|trillion)?",
                    tariff["estimated_trade_value"],
                    re.IGNORECASE,
                )
                if value_match:
                    value = float(value_match.group(1))

                    # Apply multiplier based on unit
                    if "billion" in tariff["estimated_trade_value"].lower():
                        value *= 1
                    elif "trillion" in tariff["estimated_trade_value"].lower():
                        value *= 10
                    elif "million" in tariff["estimated_trade_value"].lower():
                        value *= 0.1

                    # Scale to 0.5-3.0 range
                    trade_factor = min(max(value / 100, 0.5), 3.0)

        # Calculate final impact score
        impact = base_impact * industry_factor * trade_factor

        # Round to one decimal
        tariff["impact_score"] = round(impact, 1)

        # Add impact category
        if impact < 5:
            tariff["impact_category"] = "low"
        elif impact < 15:
            tariff["impact_category"] = "medium"
        else:
            tariff["impact_category"] = "high"

    def _add_relationship_features(self, tariff: Dict[str, Any]) -> None:
        """
        Add features describing the trade relationship.

        Args:
            tariff: Tariff data to enhance
        """
        # Check if we have both imposing and targeted countries
        if not tariff.get("imposing_country_code") or not tariff.get(
            "targeted_country_codes"
        ):
            return

        # Create relationship key
        imposing = tariff["imposing_country_code"]
        targeted = tariff["targeted_country_codes"]

        # Is this a retaliatory tariff?
        is_retaliatory = tariff.get("measure_type") == "retaliatory tariff" or (
            tariff.get("trigger_event")
            and "response" in tariff["trigger_event"].lower()
        )

        tariff["is_retaliatory"] = is_retaliatory

        # Count of targeted countries
        tariff["targeted_country_count"] = len(targeted)

        # Special relationships
        trade_blocs = {
            "EU": [
                "AT",
                "BE",
                "BG",
                "HR",
                "CY",
                "CZ",
                "DK",
                "EE",
                "FI",
                "FR",
                "DE",
                "GR",
                "HU",
                "IE",
                "IT",
                "LV",
                "LT",
                "LU",
                "MT",
                "NL",
                "PL",
                "PT",
                "RO",
                "SK",
                "SI",
                "ES",
                "SE",
            ],
            "NAFTA": ["US", "CA", "MX"],
            "ASEAN": ["BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "VN"],
        }

        # Check if the relationship is within a trade bloc
        for bloc, members in trade_blocs.items():
            if imposing in members:
                tariff["imposing_bloc"] = bloc

            # Check if all targeted countries are in the same bloc
            if all(country in members for country in targeted):
                tariff["targeted_bloc"] = bloc
                break

    def _ensure_time_features(self, tariff: Dict[str, Any]) -> None:
        """
        Ensure all time-based features are present.

        Args:
            tariff: Tariff data to enhance
        """
        # Add time recency
        if "announcement_date_std" in tariff and tariff["announcement_date_std"]:
            try:
                announcement_date = datetime.strptime(
                    tariff["announcement_date_std"], "%Y-%m-%d"
                )
                current_date = datetime.now()
                days_since = (current_date - announcement_date).days

                # Add recency feature
                tariff["days_since_announcement"] = days_since

                if days_since <= 30:
                    tariff["recency"] = "very_recent"
                elif days_since <= 90:
                    tariff["recency"] = "recent"
                elif days_since <= 365:
                    tariff["recency"] = "this_year"
                else:
                    tariff["recency"] = "historical"
            except Exception as e:
                logger.warning(f"Error calculating time features: {e}")


class TradeRelationshipAnalyzer:
    """
    Analyzes trade relationships between countries in tariff data.
    """

    def __init__(self):
        """Initialize the relationship analyzer."""
        # Track relationship counts
        self.relationships = defaultdict(int)
        self.retaliations = defaultdict(int)

    def add_event(self, event: Dict[str, Any]) -> None:
        """
        Add a tariff event to the relationship tracking.

        Args:
            event: Tariff event to analyze
        """
        tariff = event.get("tariffs_v2", {})

        # Need both imposing and targeted countries
        if not tariff.get("imposing_country_code") or not tariff.get(
            "targeted_country_codes"
        ):
            return

        imposing = tariff["imposing_country_code"]

        for targeted in tariff["targeted_country_codes"]:
            # Increment relationship count
            relationship = f"{imposing}_{targeted}"
            self.relationships[relationship] += 1

            # Track retaliations
            is_retaliatory = tariff.get("measure_type") == "retaliatory tariff" or (
                tariff.get("trigger_event")
                and "response" in tariff["trigger_event"].lower()
            )

            if is_retaliatory:
                self.retaliations[relationship] += 1

    def get_relationship_stats(self) -> Dict[str, Any]:
        """
        Get statistics about trade relationships.

        Returns:
            Dictionary with relationship statistics
        """
        return {
            "total_relationships": len(self.relationships),
            "relationships": dict(self.relationships),
            "total_retaliations": sum(self.retaliations.values()),
            "retaliations": dict(self.retaliations),
        }

    def enhance_events(
        self, events: List[Dict[str, Any]], min_threshold: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Enhance events with relationship data.

        Args:
            events: List of events to enhance
            min_threshold: Minimum relationship frequency to classify as 'frequent'

        Returns:
            Enhanced events with relationship features
        """
        enhanced_events = []

        for event in events:
            enhanced = event.copy()
            tariff = enhanced.get("tariffs_v2", {})

            # Add relationship frequency features
            if tariff.get("imposing_country_code") and tariff.get(
                "targeted_country_codes"
            ):
                imposing = tariff["imposing_country_code"]

                for targeted in tariff["targeted_country_codes"]:
                    relationship = f"{imposing}_{targeted}"
                    count = self.relationships.get(relationship, 0)

                    # Add the count
                    if "relationship_counts" not in tariff:
                        tariff["relationship_counts"] = {}

                    tariff["relationship_counts"][targeted] = count

                    # Set frequency category
                    if count >= min_threshold:
                        if "frequent_targets" not in tariff:
                            tariff["frequent_targets"] = []

                        tariff["frequent_targets"].append(targeted)

                # Add retaliation data
                has_retaliation = False
                for targeted in tariff["targeted_country_codes"]:
                    relationship = f"{imposing}_{targeted}"
                    if self.retaliations.get(relationship, 0) > 0:
                        has_retaliation = True

                tariff["has_retaliation"] = has_retaliation

            enhanced_events.append(enhanced)

        return enhanced_events
