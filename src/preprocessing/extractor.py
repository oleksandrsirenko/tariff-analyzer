"""Extraction utilities for tariff data."""

import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict

from ..utils import logger
from .normalizer import CountryNormalizer


class TariffRateExtractor:
    """
    Extracts and parses tariff rate information from text descriptions.
    """

    def __init__(self, country_normalizer: Optional[CountryNormalizer] = None):
        """
        Initialize the tariff rate extractor.

        Args:
            country_normalizer: Normalizer for country names (optional)
        """
        self.country_normalizer = country_normalizer

        # Compile common patterns for rate extraction
        self.rate_patterns = {
            # Standard: 25% on steel
            "standard": re.compile(
                r"^(\d+(?:\.\d+)?)%\s+(?:tariffs?\s+)?(?:on|for)\s+(.+)$", re.IGNORECASE
            ),
            # Product from country: 25% on steel from China
            "product_country": re.compile(
                r"^(\d+(?:\.\d+)?)%\s+(?:tariffs?\s+)?(?:on|for)\s+(.+)\s+from\s+(.+)$",
                re.IGNORECASE,
            ),
            # All products from country: 25% on all products from China
            "all_country": re.compile(
                r"^(\d+(?:\.\d+)?)%\s+(?:tariffs?\s+)?(?:on|for)\s+(?:all|most)\s+(?:imports|products|goods)\s+from\s+(.+)$",
                re.IGNORECASE,
            ),
            # Double pattern: double on X
            "double": re.compile(
                r"^double\s+(?:tariffs?\s+)?(?:on|for)\s+(.+)$", re.IGNORECASE
            ),
            # Tariff increase/reduction: Increase/reduce by X% on Y
            "change": re.compile(
                r"^(?:increase|reduce|decrease|cut)\s+(?:by\s+)?(\d+(?:\.\d+)?)%\s+(?:tariffs?\s+)?(?:on|for)\s+(.+)$",
                re.IGNORECASE,
            ),
        }

    def extract_rates(self, rate_strings: List[str]) -> List[Dict[str, Any]]:
        """
        Extract structured rate information from a list of rate strings.

        Args:
            rate_strings: List of tariff rate strings

        Returns:
            List of dictionaries with parsed rate information
        """
        if not rate_strings:
            return []

        parsed_rates = []

        for rate_str in rate_strings:
            if not rate_str:
                continue

            parsed = self._parse_rate_string(rate_str)
            if parsed:
                parsed["original"] = rate_str
                parsed_rates.append(parsed)
            else:
                # If we couldn't parse it, at least keep the original
                parsed_rates.append({"original": rate_str})

        return parsed_rates

    def _parse_rate_string(self, rate_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single rate string into structured data.

        Args:
            rate_str: Tariff rate string

        Returns:
            Dictionary with parsed information or None if unparseable
        """
        # Try each pattern in order

        # Standard pattern: X% on Y
        match = self.rate_patterns["standard"].match(rate_str)
        if match:
            rate = float(match.group(1))
            target = match.group(2).strip()

            # Check if it's actually product_country pattern
            product_country_match = re.search(
                r"(.+)\s+from\s+(.+)$", target, re.IGNORECASE
            )
            if product_country_match:
                product = product_country_match.group(1).strip()
                country = product_country_match.group(2).strip()

                return self._create_product_country_rate(rate, product, country)

            # Check if it's "all from country" pattern
            all_country_match = re.search(
                r"(?:all|most)\s+(?:imports|products|goods)\s+from\s+(.+)$",
                target,
                re.IGNORECASE,
            )
            if all_country_match:
                country = all_country_match.group(1).strip()
                return self._create_all_country_rate(rate, country)

            # It's a standard product rate
            return {
                "rate": rate,
                "rate_type": "percentage",
                "target_type": "product",
                "target": target,
            }

        # Product from country pattern
        match = self.rate_patterns["product_country"].match(rate_str)
        if match:
            rate = float(match.group(1))
            product = match.group(2).strip()
            country = match.group(3).strip()

            return self._create_product_country_rate(rate, product, country)

        # All products from country pattern
        match = self.rate_patterns["all_country"].match(rate_str)
        if match:
            rate = float(match.group(1))
            country = match.group(2).strip()

            return self._create_all_country_rate(rate, country)

        # Double pattern
        match = self.rate_patterns["double"].match(rate_str)
        if match:
            target = match.group(1).strip()

            return {
                "rate": "double",
                "rate_type": "multiplier",
                "target_type": "product",
                "target": target,
            }

        # Change pattern
        match = self.rate_patterns["change"].match(rate_str)
        if match:
            change = float(match.group(1))
            target = match.group(2).strip()

            change_type = "increase" if "increase" in rate_str.lower() else "decrease"

            return {
                "rate": change,
                "rate_type": "change",
                "change_type": change_type,
                "target_type": "product",
                "target": target,
            }

        # Couldn't parse
        return None

    def _create_product_country_rate(
        self, rate: float, product: str, country: str
    ) -> Dict[str, Any]:
        """
        Create a parsed rate for product from country pattern.

        Args:
            rate: Rate value
            product: Product description
            country: Country name

        Returns:
            Parsed rate dictionary
        """
        result = {
            "rate": rate,
            "rate_type": "percentage",
            "target_type": "product_country",
            "product": product,
            "country": country,
        }

        # Add country code if normalizer is available
        if self.country_normalizer:
            country_code = self.country_normalizer.get_country_code(country)
            if country_code:
                result["country_code"] = country_code

        return result

    def _create_all_country_rate(self, rate: float, country: str) -> Dict[str, Any]:
        """
        Create a parsed rate for all products from country pattern.

        Args:
            rate: Rate value
            country: Country name

        Returns:
            Parsed rate dictionary
        """
        result = {
            "rate": rate,
            "rate_type": "percentage",
            "target_type": "country",
            "country": country,
        }

        # Add country code if normalizer is available
        if self.country_normalizer:
            country_code = self.country_normalizer.get_country_code(country)
            if country_code:
                result["country_code"] = country_code

        return result

    def extract_main_rate(
        self, rate_strings: List[str], current_main_rate: Optional[float] = None
    ) -> Optional[float]:
        """
        Extract the main tariff rate from a list of rate strings.

        Args:
            rate_strings: List of tariff rate strings
            current_main_rate: Current main_tariff_rate value if available

        Returns:
            Main tariff rate as a float or None if not found
        """
        # If we already have a main rate, use it
        if current_main_rate is not None:
            return current_main_rate

        if not rate_strings:
            return None

        # Parse all rates
        parsed_rates = self.extract_rates(rate_strings)
        if not parsed_rates:
            return None

        # Only consider percentage rates
        percentage_rates = [
            r["rate"]
            for r in parsed_rates
            if "rate" in r and "rate_type" in r and r["rate_type"] == "percentage"
        ]

        if not percentage_rates:
            return None

        # Use the highest rate as the main rate
        return max(percentage_rates)


class TariffProductExtractor:
    """
    Extracts and normalizes product information from tariff data.
    """

    def __init__(self):
        """Initialize the product extractor."""
        # Common product name variations for normalization
        self.product_mappings = {
            "automobiles": ["cars", "auto imports", "auto", "car imports", "vehicles"],
            "steel": ["steel products", "steel imports", "steel goods"],
            "aluminum": ["aluminium", "aluminum products", "aluminium products"],
            "agricultural products": ["farm products", "agri products", "agriculture"],
            "electronics": ["electronic products", "electronic goods"],
            "textiles": ["textile products", "clothing", "garments", "apparel"],
            "lumber": ["wood", "timber", "forestry products"],
            "energy": ["energy products", "oil and gas", "petroleum"],
        }

        # Create reverse mappings
        self.normalized_products = {}
        for standard, variations in self.product_mappings.items():
            for variant in variations:
                self.normalized_products[variant.lower()] = standard

    def normalize_product(self, product: str) -> str:
        """
        Normalize a product name to a standard form.

        Args:
            product: Product name to normalize

        Returns:
            Normalized product name
        """
        if not product:
            return product

        # Check direct mapping
        norm_product = self.normalized_products.get(product.lower())
        if norm_product:
            return norm_product

        # Check if it contains any known variants
        for variant, standard in self.normalized_products.items():
            if variant in product.lower():
                return standard

        # No mapping found, return original
        return product

    def extract_products_from_rates(
        self, parsed_rates: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract product names from parsed rates.

        Args:
            parsed_rates: List of parsed rate dictionaries

        Returns:
            List of unique product names
        """
        products = set()

        for rate in parsed_rates:
            if "target_type" in rate:
                if rate["target_type"] == "product" and "target" in rate:
                    products.add(self.normalize_product(rate["target"]))
                elif rate["target_type"] == "product_country" and "product" in rate:
                    products.add(self.normalize_product(rate["product"]))

        return sorted(list(products))
