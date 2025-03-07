"""Core preprocessing pipeline for tariff data."""

import os
import json
import copy
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd

from ..utils import (
    logger,
    load_json_data,
    validate_json_input,
    validate_country_code_file,
)
from .normalizer import CountryNormalizer, DateNormalizer
from .deduplicator import EventDeduplicator


class TariffProcessor:
    """
    Main processor for tariff data.
    """

    def __init__(self, country_code_file: Optional[str] = None):
        """
        Initialize the tariff processor.

        Args:
            country_code_file: Path to CSV file with country code mappings
        """
        self.country_normalizer = CountryNormalizer(country_code_file)
        self.date_normalizer = DateNormalizer()
        self.deduplicator = EventDeduplicator()

        # Statistics
        self.stats = {
            "processed_events": 0,
            "duplicate_events": 0,
            "missing_fields": {},
            "measure_types": {},
            "start_time": None,
            "end_time": None,
        }

    def process_file(self, file_path: str, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Process tariff data from a file.

        Args:
            file_path: Path to the JSON file with tariff data
            batch_size: Number of events to process in each batch

        Returns:
            Processed data with metadata
        """
        # Load data
        data = load_json_data(file_path)

        # Validate input
        if not validate_json_input(data):
            raise ValueError(f"Invalid input data: {file_path}")

        # Process data
        return self.process(data, batch_size)

    def process(self, data: Dict[str, Any], batch_size: int = 1000) -> Dict[str, Any]:
        """
        Process tariff data.

        Args:
            data: Raw tariff data
            batch_size: Number of events to process in each batch

        Returns:
            Processed data with metadata
        """
        self.stats["start_time"] = time.time()

        # Extract events
        raw_events = data.get("events", [])
        total_events = len(raw_events)
        logger.info(f"Processing {total_events} tariff events")

        # Set up deduplicator
        self.deduplicator.reset()
        self.deduplicator.setup_bit_mappings(raw_events)

        # Process events in batches
        processed_events = []

        for i in range(0, total_events, batch_size):
            batch = raw_events[i : min(i + batch_size, total_events)]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(total_events + batch_size - 1)//batch_size}"
            )

            for event in batch:
                # Check for duplicates
                if self.deduplicator.is_duplicate(event):
                    continue

                # Process event
                processed_event = self._process_event(event)
                processed_events.append(processed_event)

        # Update stats
        self.stats["processed_events"] = len(processed_events)
        self.stats["duplicate_events"] = self.deduplicator.duplicate_count
        self.stats.update(self.deduplicator.get_stats())

        self.stats["end_time"] = time.time()
        self.stats["processing_time"] = (
            self.stats["end_time"] - self.stats["start_time"]
        )

        # Create result
        result = {
            "events": processed_events,
            "metadata": {
                "original_count": total_events,
                "processed_count": len(processed_events),
                "duplicates_removed": self.deduplicator.duplicate_count,
                "processing_time": self.stats["processing_time"],
            },
        }

        logger.info(
            f"Processing complete. {len(processed_events)} events processed, "
            + f"{self.deduplicator.duplicate_count} duplicates removed."
        )

        return result

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single tariff event.

        Args:
            event: Raw event

        Returns:
            Processed event
        """
        # Deep copy to avoid modifying the original
        processed = copy.deepcopy(event)

        # Ensure tariffs_v2 exists
        if "tariffs_v2" not in processed:
            processed["tariffs_v2"] = {}

        tariff = processed["tariffs_v2"]

        # Track missing fields
        for field in [
            "imposing_country_code",
            "targeted_country_codes",
            "measure_type",
            "affected_industries",
            "main_tariff_rate",
            "announcement_date",
        ]:
            if field not in tariff or tariff[field] is None:
                self.stats["missing_fields"][field] = (
                    self.stats["missing_fields"].get(field, 0) + 1
                )

        # Normalize country names and codes
        self._normalize_countries(tariff)

        # Standardize dates
        self._standardize_dates(tariff)

        # Track measure types
        if "measure_type" in tariff and tariff["measure_type"]:
            measure_type = tariff["measure_type"]
            self.stats["measure_types"][measure_type] = (
                self.stats["measure_types"].get(measure_type, 0) + 1
            )

        return processed

    def _normalize_countries(self, tariff: Dict[str, Any]) -> None:
        """
        Normalize country names and codes in a tariff.

        Args:
            tariff: Tariff data to normalize
        """
        # Normalize imposing country
        if "imposing_country_code" in tariff and tariff["imposing_country_code"]:
            code = self.country_normalizer.normalize_country_code(
                tariff["imposing_country_code"]
            )
            if code:
                tariff["imposing_country_code"] = code
                tariff["standard_imposing_country"] = (
                    self.country_normalizer.get_country_name(code)
                )

        # If we have imposing_country_name but no code, try to get code
        elif "imposing_country_name" in tariff and tariff["imposing_country_name"]:
            code = self.country_normalizer.get_country_code(
                tariff["imposing_country_name"]
            )
            if code:
                tariff["imposing_country_code"] = code
                tariff["standard_imposing_country"] = (
                    self.country_normalizer.get_country_name(code)
                )

        # Normalize targeted countries
        if "targeted_country_codes" in tariff and isinstance(
            tariff["targeted_country_codes"], list
        ):
            normalized_codes = []
            for code in tariff["targeted_country_codes"]:
                norm_code = self.country_normalizer.normalize_country_code(code)
                if norm_code:
                    normalized_codes.append(norm_code)

            tariff["targeted_country_codes"] = normalized_codes
            tariff["standard_targeted_countries"] = [
                self.country_normalizer.get_country_name(code)
                for code in normalized_codes
            ]

        # If we have targeted_country_names but no codes, try to get codes
        elif "targeted_country_names" in tariff and isinstance(
            tariff["targeted_country_names"], list
        ):
            codes = []
            names = []

            for name in tariff["targeted_country_names"]:
                code = self.country_normalizer.get_country_code(name)
                if code:
                    codes.append(code)
                    names.append(self.country_normalizer.get_country_name(code))

            if codes:
                tariff["targeted_country_codes"] = codes
                tariff["standard_targeted_countries"] = names

    def _standardize_dates(self, tariff: Dict[str, Any]) -> None:
        """
        Standardize dates in a tariff.

        Args:
            tariff: Tariff data to standardize
        """
        # Standardize announcement date
        if "announcement_date" in tariff and tariff["announcement_date"]:
            std_date = self.date_normalizer.normalize_date(tariff["announcement_date"])
            if std_date:
                tariff["announcement_date_std"] = std_date

                # Extract date components
                components = self.date_normalizer.extract_date_components(std_date)
                tariff.update(
                    {
                        "announcement_year": components["year"],
                        "announcement_month": components["month"],
                        "announcement_quarter": components["quarter"],
                    }
                )

        # Standardize implementation date
        if "implementation_date" in tariff and tariff["implementation_date"]:
            std_date = self.date_normalizer.normalize_date(
                tariff["implementation_date"]
            )
            if std_date:
                tariff["implementation_date_std"] = std_date

        # Standardize expiration date
        if "expiration_date" in tariff and tariff["expiration_date"]:
            std_date = self.date_normalizer.normalize_date(tariff["expiration_date"])
            if std_date:
                tariff["expiration_date_std"] = std_date

        # Calculate time to implementation
        if (
            "announcement_date_std" in tariff
            and tariff["announcement_date_std"]
            and "implementation_date_std" in tariff
            and tariff["implementation_date_std"]
        ):
            days = self.date_normalizer.calculate_date_difference(
                tariff["announcement_date_std"], tariff["implementation_date_std"]
            )
            if days is not None:
                tariff["days_to_implementation"] = days

        # Calculate implementation duration if expiration date exists
        if (
            "implementation_date_std" in tariff
            and tariff["implementation_date_std"]
            and "expiration_date_std" in tariff
            and tariff["expiration_date_std"]
        ):
            days = self.date_normalizer.calculate_date_difference(
                tariff["implementation_date_std"], tariff["expiration_date_std"]
            )
            if days is not None:
                tariff["implementation_duration"] = days

    def export_to_csv(self, processed_data: Dict[str, Any], output_file: str) -> int:
        """
        Export processed data to CSV for analysis.

        Args:
            processed_data: Processed tariff data
            output_file: Path to output CSV file

        Returns:
            Number of records written
        """
        if "events" not in processed_data or not processed_data["events"]:
            logger.warning("No events to export")
            return 0

        try:
            # Convert to DataFrame
            df = self.to_dataframe(processed_data)

            # Write to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} records to {output_file}")

            return len(df)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    def to_dataframe(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert processed data to pandas DataFrame.

        Args:
            processed_data: Processed tariff data

        Returns:
            DataFrame with flattened data
        """
        if "events" not in processed_data or not processed_data["events"]:
            return pd.DataFrame()

        # Flatten nested structure
        flattened = []

        for event in processed_data["events"]:
            tariff = event.get("tariffs_v2", {})

            # Base record
            record = {
                # Event metadata
                "id": event.get("id"),
                "event_type": event.get("event_type"),
                "extraction_date": event.get("extraction_date"),
                "global_event_type": event.get("global_event_type"),
                # Imposing country
                "imposing_country_code": tariff.get("imposing_country_code"),
                "standard_imposing_country": tariff.get("standard_imposing_country"),
                # Targeted countries
                "targeted_country_codes": self._join_list(
                    tariff.get("targeted_country_codes")
                ),
                "standard_targeted_countries": self._join_list(
                    tariff.get("standard_targeted_countries")
                ),
                # Measure details
                "measure_type": tariff.get("measure_type"),
                "relevance_score": tariff.get("relevance_score"),
                "main_tariff_rate": tariff.get("main_tariff_rate"),
                # Dates
                "announcement_date": tariff.get("announcement_date"),
                "announcement_date_std": tariff.get("announcement_date_std"),
                "implementation_date": tariff.get("implementation_date"),
                "implementation_date_std": tariff.get("implementation_date_std"),
                "expiration_date": tariff.get("expiration_date"),
                "expiration_date_std": tariff.get("expiration_date_std"),
                # Time components
                "announcement_year": tariff.get("announcement_year"),
                "announcement_month": tariff.get("announcement_month"),
                "announcement_quarter": tariff.get("announcement_quarter"),
                "days_to_implementation": tariff.get("days_to_implementation"),
                "implementation_duration": tariff.get("implementation_duration"),
                # Arrays
                "affected_industries": self._join_list(
                    tariff.get("affected_industries")
                ),
                "affected_products": self._join_list(tariff.get("affected_products")),
                "hs_product_categories": self._join_list(
                    tariff.get("hs_product_categories")
                ),
                # Additional fields
                "estimated_trade_value": tariff.get("estimated_trade_value"),
                "legal_basis": tariff.get("legal_basis"),
                "policy_objective": tariff.get("policy_objective"),
                "trigger_event": tariff.get("trigger_event"),
                "summary": tariff.get("summary"),
            }

            flattened.append(record)

        return pd.DataFrame(flattened)

    def _join_list(
        self, items: Optional[List[str]], separator: str = "; "
    ) -> Optional[str]:
        """
        Join a list of items into a string.

        Args:
            items: List to join
            separator: Separator to use

        Returns:
            Joined string or None if input is None
        """
        if not items:
            return None
        return separator.join(str(item) for item in items if item)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        return self.stats
