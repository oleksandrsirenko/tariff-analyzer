"""Deduplication utilities for tariff data."""

from typing import Dict, List, Any, Set, Tuple, Optional
import hashlib
import json

from ..utils import logger


class ArrayBitMasker:
    """
    Creates bit mask representations of arrays for efficient comparison.
    """

    def __init__(self):
        """Initialize the array bit masker."""
        self.item_mappings: Dict[str, Dict[str, int]] = {}

    def create_mapping(self, field_name: str, unique_values: List[str]) -> None:
        """
        Create a bit mapping for a specific field.

        Args:
            field_name: Name of the field to create mapping for
            unique_values: List of unique possible values for the field
        """
        mapping = {}
        for i, value in enumerate(sorted(unique_values)):
            if i >= 64:  # Limit to avoid integer overflow in most languages
                logger.warning(
                    f"Too many unique values for field {field_name}, some values will share bits"
                )
                # For values beyond 64, use modulo to reuse bit positions
                mapping[value] = 1 << (i % 64)
            else:
                mapping[value] = 1 << i

        self.item_mappings[field_name] = mapping
        logger.debug(f"Created bit mapping for {field_name} with {len(mapping)} values")

    def array_to_bits(self, field_name: str, values: Optional[List[str]]) -> int:
        """
        Convert an array of values to a bit representation.

        Args:
            field_name: Name of the field this array belongs to
            values: Array of values to convert

        Returns:
            Integer bit representation
        """
        if not values or field_name not in self.item_mappings:
            return 0

        mapping = self.item_mappings[field_name]
        bits = 0

        for value in values:
            if value in mapping:
                bits |= mapping[value]

        return bits

    def bits_contain_value(self, bits: int, field_name: str, value: str) -> bool:
        """
        Check if a bit representation contains a specific value.

        Args:
            bits: Bit representation to check
            field_name: Name of the field
            value: Value to check for

        Returns:
            True if the bits contain the value
        """
        if (
            field_name not in self.item_mappings
            or value not in self.item_mappings[field_name]
        ):
            return False

        value_bit = self.item_mappings[field_name][value]
        return (bits & value_bit) == value_bit

    def bits_to_array(self, bits: int, field_name: str) -> List[str]:
        """
        Convert bits back to an array of values.

        Args:
            bits: Bit representation
            field_name: Name of the field

        Returns:
            Array of values encoded in the bits
        """
        if field_name not in self.item_mappings or bits == 0:
            return []

        result = []
        mapping = self.item_mappings[field_name]

        for value, value_bit in mapping.items():
            if (bits & value_bit) == value_bit:
                result.append(value)

        return result


class EventDeduplicator:
    """
    Handles deduplication of tariff events using efficient comparisons.
    """

    def __init__(self):
        """Initialize the event deduplicator."""
        self.bit_masker = ArrayBitMasker()
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0

    def setup_bit_mappings(self, events: List[Dict[str, Any]]) -> None:
        """
        Set up bit mappings for array fields in events.

        Args:
            events: List of events to extract unique values from
        """
        # Fields to create bit mappings for
        array_fields = [
            ("affected_industries", []),
            ("hs_product_categories", []),
            ("affected_products", []),
        ]

        # Collect unique values
        for field_name, values in array_fields:
            field_values = set()
            for event in events:
                tariff = event.get("tariffs_v2", {})
                if field_name in tariff and isinstance(tariff[field_name], list):
                    for value in tariff[field_name]:
                        if value:  # Skip empty values
                            field_values.add(value)

            # Create mapping for this field
            self.bit_masker.create_mapping(field_name, list(field_values))
            logger.info(
                f"Created bit mapping for {field_name} with {len(field_values)} unique values"
            )

    def create_event_hash(self, event: Dict[str, Any]) -> str:
        """
        Create a hash for an event for deduplication.

        Args:
            event: Event to create hash for

        Returns:
            Hash string representing the event
        """
        tariff = event.get("tariffs_v2", {})

        # Use key fields for hash
        key_parts = [
            tariff.get("imposing_country_code", ""),
            ",".join(sorted(tariff.get("targeted_country_codes", []) or [])),
            tariff.get("measure_type", ""),
            tariff.get("announcement_date", ""),
        ]

        # Include bit representations of arrays
        for field_name in [
            "affected_industries",
            "hs_product_categories",
            "affected_products",
        ]:
            bits = self.bit_masker.array_to_bits(field_name, tariff.get(field_name, []))
            key_parts.append(str(bits))

        # Additional fields that help identify unique events
        if "main_tariff_rate" in tariff:
            key_parts.append(str(tariff["main_tariff_rate"]))

        # Create a hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def is_duplicate(self, event: Dict[str, Any]) -> bool:
        """
        Check if an event is a duplicate.

        Args:
            event: Event to check

        Returns:
            True if this is a duplicate event
        """
        event_hash = self.create_event_hash(event)

        if event_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True

        self.seen_hashes.add(event_hash)
        return False

    def reset(self) -> None:
        """Reset the deduplicator state."""
        self.seen_hashes.clear()
        self.duplicate_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "duplicate_count": self.duplicate_count,
            "unique_count": len(self.seen_hashes),
        }
