"""Normalization utilities for tariff data."""

import re
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from ..utils import logger


class CountryNormalizer:
    """
    Normalizes country names and codes according to ISO standards.
    """

    def __init__(self, country_code_file: Optional[str] = None):
        """
        Initialize the country normalizer.

        Args:
            country_code_file: Path to CSV file with country code mappings.
                If None, only basic mappings will be available.
        """
        self.code_to_name: Dict[str, str] = {}
        self.name_to_code: Dict[str, str] = {}

        # Load country mappings
        if country_code_file:
            self._load_country_mappings(country_code_file)
            logger.info(f"Loaded {len(self.code_to_name)} country mappings")
        else:
            logger.warning("No country code file provided, using limited mappings")
            self._load_default_mappings()

    def _load_country_mappings(self, file_path: str) -> None:
        """
        Load country code mappings from a CSV file.

        Args:
            file_path: Path to the CSV file with country mappings
        """
        try:
            df = pd.read_csv(file_path)

            # Check required columns
            if "Country" not in df.columns or "Alpha-2 code" not in df.columns:
                logger.error("Country code file missing required columns")
                self._load_default_mappings()
                return

            # Build mappings
            for _, row in df.iterrows():
                try:
                    # Handle possible string or non-string values
                    if pd.notna(row["Alpha-2 code"]) and pd.notna(row["Country"]):
                        code = (
                            str(row["Alpha-2 code"]).strip().strip("\"'")
                        )  # Remove quotes and whitespace
                        name = str(row["Country"]).strip().strip("\"'")

                        if code and name:
                            self.code_to_name[code] = name
                            self.name_to_code[name.lower()] = code

                            # Add variations for common countries
                            self._add_country_variations(name.lower(), code)
                except Exception as e:
                    logger.debug(f"Error processing country row: {e}")
                    continue

            # Add core mappings to ensure important countries are available
            self._ensure_core_mappings()

            logger.info(f"Loaded {len(self.code_to_name)} country mappings")
        except Exception as e:
            logger.error(f"Error loading country mappings: {e}")
            self._load_default_mappings()

    def _ensure_core_mappings(self) -> None:
        """Ensure core country mappings are present."""
        core_mappings = [
            ("US", "United States"),
            ("CN", "China"),
            ("KR", "South Korea"),
            ("GB", "United Kingdom"),
            ("EU", "European Union"),
        ]

        for code, name in core_mappings:
            if code not in self.code_to_name:
                self.code_to_name[code] = name
                self.name_to_code[name.lower()] = code
                self._add_country_variations(name.lower(), code)

    def _load_default_mappings(self) -> None:
        """Load default country mappings for common countries."""
        default_mappings = [
            ("US", "United States"),
            ("CA", "Canada"),
            ("MX", "Mexico"),
            ("CN", "China"),
            ("JP", "Japan"),
            ("DE", "Germany"),
            ("FR", "France"),
            ("GB", "United Kingdom"),
            ("IT", "Italy"),
            ("IN", "India"),
            ("BR", "Brazil"),
            ("RU", "Russia"),
            ("AU", "Australia"),
            ("KR", "South Korea"),
            ("EU", "European Union"),  # Not ISO but common in trade
        ]

        for code, name in default_mappings:
            self.code_to_name[code] = name
            self.name_to_code[name.lower()] = code
            self._add_country_variations(name.lower(), code)

    def _add_country_variations(self, country_name: str, code: str) -> None:
        """
        Add common variations of country names to the mapping.

        Args:
            country_name: Base country name (lowercase)
            code: Country code
        """
        variations = {
            "united states": [
                "usa",
                "united states of america",
                "u.s.",
                "u.s.a.",
                "us",
            ],
            "united kingdom": ["uk", "britain", "great britain"],
            "russia": ["russian federation"],
            "china": ["peoples republic of china", "people's republic of china"],
            "south korea": ["korea, republic of", "republic of korea"],
            "european union": ["eu"],
        }

        if country_name in variations:
            for variation in variations[country_name]:
                self.name_to_code[variation] = code

    def normalize_country_code(self, code: str) -> Optional[str]:
        """
        Normalize a country code to standard ISO format.

        Args:
            code: Country code to normalize

        Returns:
            Normalized country code or None if invalid
        """
        if not code:
            return None

        # Standardize to uppercase
        code = code.strip().upper()

        # Check if it's a valid code
        if code in self.code_to_name:
            return code

        return None

    def normalize_country_name(self, name: str) -> Optional[str]:
        """
        Normalize a country name to its standard form.

        Args:
            name: Country name to normalize

        Returns:
            Normalized country name or None if not recognized
        """
        if not name:
            return None

        # Standardize name
        name = name.strip().lower()

        # Try to find code
        code = self.name_to_code.get(name)
        if code:
            return self.code_to_name.get(code)

        return None

    def get_country_code(self, name: str) -> Optional[str]:
        """
        Get the country code for a country name.

        Args:
            name: Country name

        Returns:
            Country code or None if not found
        """
        if not name:
            return None

        return self.name_to_code.get(name.strip().lower())

    def get_country_name(self, code: str) -> Optional[str]:
        """
        Get the standard country name for a country code.

        Args:
            code: Country code

        Returns:
            Country name or None if not found
        """
        if not code:
            return None

        return self.code_to_name.get(code.strip().upper())


class DateNormalizer:
    """
    Normalizes date formats to ISO standard.
    """

    def __init__(self):
        """Initialize the date normalizer."""
        # Supported date formats
        self.date_formats = [
            "%Y/%m/%d",  # 2025/01/01
            "%Y-%m-%d",  # 2025-01-01
            "%Y/%m",  # 2025/01
            "%Y-%m",  # 2025-01
            "%Y",  # 2025
            "%B %d, %Y",  # January 1, 2025
            "%b %d, %Y",  # Jan 1, 2025
            "%d %B %Y",  # 1 January 2025
            "%d %b %Y",  # 1 Jan 2025
            "%B %Y",  # January 2025
            "%b %Y",  # Jan 2025
        ]

    def normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalize a date string to ISO format (YYYY-MM-DD).

        Args:
            date_str: Date string to normalize

        Returns:
            Normalized date string or None if invalid
        """
        if not date_str:
            return None

        date_str = date_str.strip()

        # Already in YYYY-MM-DD format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        # YYYY/MM/DD format
        if re.match(r"^\d{4}/\d{2}/\d{2}$", date_str):
            return date_str.replace("/", "-")

        # YYYY-MM format
        if re.match(r"^\d{4}-\d{2}$", date_str):
            return f"{date_str}-01"  # Default to first day of month

        # YYYY/MM format
        if re.match(r"^\d{4}/\d{2}$", date_str):
            parts = date_str.split("/")
            return f"{parts[0]}-{parts[1]}-01"  # Default to first day of month

        # YYYY format
        if re.match(r"^\d{4}$", date_str):
            return f"{date_str}-01-01"  # Default to January 1

        # Try parsing with different formats
        for fmt in self.date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue

        logger.warning(f"Could not normalize date: {date_str}")
        return None

    def extract_date_components(self, date_str: str) -> Dict[str, int]:
        """
        Extract components from a date string.

        Args:
            date_str: Date string in ISO format (YYYY-MM-DD)

        Returns:
            Dictionary with year, month, quarter
        """
        components = {"year": None, "month": None, "quarter": None}

        if not date_str:
            return components

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            components["year"] = date_obj.year
            components["month"] = date_obj.month
            components["quarter"] = (date_obj.month - 1) // 3 + 1
        except ValueError:
            # Try with just year-month
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m")
                components["year"] = date_obj.year
                components["month"] = date_obj.month
                components["quarter"] = (date_obj.month - 1) // 3 + 1
            except ValueError:
                # Try with just year
                try:
                    components["year"] = int(date_str)
                except ValueError:
                    logger.warning(
                        f"Could not extract components from date: {date_str}"
                    )

        return components

    def calculate_date_difference(
        self, start_date: str, end_date: str
    ) -> Optional[int]:
        """
        Calculate the number of days between two dates.

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)

        Returns:
            Number of days between dates or None if invalid
        """
        if not start_date or not end_date:
            return None

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            return (end - start).days
        except ValueError:
            logger.warning(
                f"Could not calculate date difference: {start_date} to {end_date}"
            )
            return None
