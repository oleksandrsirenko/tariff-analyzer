def _extract_rates_and_products(self, tariff: Dict[str, Any]) -> None:
        """
        Extract and parse tariff rates and products.
        
        Args:
            tariff: Tariff data to process
        """
        # Extract rates from tariff_rates strings
        if "tariff_rates" in tariff and tariff["tariff_rates"]:
            # Parse tariff rates
            parsed_rates = self.rate_extractor.extract_rates(tariff["tariff_rates"])
            if parsed_rates:
                tariff["parsed_rates"] = parsed_rates
                self.stats["extracted_rates"] += len(parsed_rates)
            
            # Try to extract main_tariff_rate if not present
            if ("main_tariff_rate" not in tariff or tariff["main_tariff_rate"] is None) and parsed_rates:
                main_rate = self.rate_extractor.extract_main_rate(tariff["tariff_rates"])
                if main_rate is not None:
                    tariff["main_tariff_rate"] = main_rate
            
            # Extract products from rates
            if "parsed_rates" in tariff:
                products = self.product_extractor.extract_products_from_rates(tariff["parsed_rates"])
                
                # Add extracted products if we don't have them
                if ("affected_products" not in tariff or not tariff["affected_products"]) and products:
                    tariff["affected_products"] = products
                    tariff["extracted_products"] = True
                    self.stats["extracted_products"] += len(products)
    
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
                "targeted_country_codes": self._join_list(tariff.get("targeted_country_codes")),
                "standard_targeted_countries": self._join_list(tariff.get("standard_targeted_countries")),
                "targeted_country_count": tariff.get("targeted_country_count"),
                
                # Measure details
                "measure_type": tariff.get("measure_type"),
                "relevance_score": tariff.get("relevance_score"),
                "main_tariff_rate": tariff.get("main_tariff_rate"),
                "rate_category": tariff.get("rate_category"),
                "impact_score": tariff.get("impact_score"),
                "impact_category": tariff.get("impact_category"),
                
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
                "days_since_announcement": tariff.get("days_since_announcement"),
                "recency": tariff.get("recency"),
                
                # Relationship features
                "is_retaliatory": tariff.get("is_retaliatory"),
                "has_retaliation": tariff.get("has_retaliation"),
                "imposing_bloc": tariff.get("imposing_bloc"),
                "targeted_bloc": tariff.get("targeted_bloc"),
                "frequent_targets": self._join_list(tariff.get("frequent_targets")),
                
                # Arrays
                "affected_industries": self._join_list(tariff.get("affected_industries")),
                "affected_products": self._join_list(tariff.get("affected_products")),
                "hs_product_categories": self._join_list(tariff.get("hs_product_categories")),
                
                # Additional fields
                "estimated_trade_value": tariff.get("estimated_trade_value"),
                "legal_basis": tariff.get("legal_basis"),
                "policy_objective": tariff.get("policy_objective"),
                "trigger_event": tariff.get("trigger_event"),
                "summary": tariff.get("summary")
            }
            
            flattened.append(record)
        
        return pd.DataFrame(flattened)"""Core preprocessing pipeline for tariff data."""

import os
import json
import copy
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd

from ..utils import logger, load_json_data, validate_json_input, validate_country_code_file
from .normalizer import CountryNormalizer, DateNormalizer
from .deduplicator import EventDeduplicator
from .extractor import TariffRateExtractor, TariffProductExtractor
from .feature_engineering import TariffFeatureEngineer, TradeRelationshipAnalyzer


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
        self.rate_extractor = TariffRateExtractor(self.country_normalizer)
        self.product_extractor = TariffProductExtractor()
        self.feature_engineer = TariffFeatureEngineer()
        self.relationship_analyzer = TradeRelationshipAnalyzer()
        
        # Statistics
        self.stats = {
            "processed_events": 0,
            "duplicate_events": 0,
            "missing_fields": {},
            "measure_types": {},
            "rate_categories": {},
            "impact_categories": {},
            "extracted_rates": 0,
            "extracted_products": 0,
            "start_time": None,
            "end_time": None
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
            batch = raw_events[i:min(i + batch_size, total_events)]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_events + batch_size - 1)//batch_size}")
            
            for event in batch:
                # Check for duplicates
                if self.deduplicator.is_duplicate(event):
                    continue
                
                # Process event
                processed_event = self._process_event(event)
                
                # Add to relationship analyzer
                self.relationship_analyzer.add_event(processed_event)
                
                processed_events.append(processed_event)
        
        # Enhance events with relationship data
        enhanced_events = self.relationship_analyzer.enhance_events(processed_events)
        
        # Update stats
        self.stats["processed_events"] = len(enhanced_events)
        self.stats["duplicate_events"] = self.deduplicator.duplicate_count
        self.stats.update(self.deduplicator.get_stats())
        
        self.stats["end_time"] = time.time()
        self.stats["processing_time"] = self.stats["end_time"] - self.stats["start_time"]
        
        # Create result
        result = {
            "events": enhanced_events,
            "metadata": {
                "original_count": total_events,
                "processed_count": len(enhanced_events),
                "duplicates_removed": self.deduplicator.duplicate_count,
                "processing_time": self.stats["processing_time"],
                "relationship_stats": self.relationship_analyzer.get_relationship_stats()
            }
        }
        
        logger.info(f"Processing complete. {len(enhanced_events)} events processed, " +
                   f"{self.deduplicator.duplicate_count} duplicates removed.")
        
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
        for field in ["imposing_country_code", "targeted_country_codes", "measure_type", 
                      "affected_industries", "main_tariff_rate", "announcement_date"]:
            if field not in tariff or tariff[field] is None:
                self.stats["missing_fields"][field] = self.stats["missing_fields"].get(field, 0) + 1
        
        # 1. Normalize country names and codes
        self._normalize_countries(tariff)
        
        # 2. Standardize dates
        self._standardize_dates(tariff)
        
        # 3. Extract rates and products
        self._extract_rates_and_products(tariff)
        
        # 4. Add derived features
        processed = self.feature_engineer.add_derived_features(processed)
        tariff = processed["tariffs_v2"]  # Update tariff reference
        
        # Track statistics
        if "measure_type" in tariff and tariff["measure_type"]:
            measure_type = tariff["measure_type"]
            self.stats["measure_types"][measure_type] = self.stats["measure_types"].get(measure_type, 0) + 1
        
        if "rate_category" in tariff:
            category = tariff["rate_category"]
            self.stats["rate_categories"][category] = self.stats["rate_categories"].get(category, 0) + 1
        
        if "impact_category" in tariff:
            category = tariff["impact_category"]
            self.stats["impact_categories"][category] = self.stats["impact_categories"].get(category, 0) + 1
        
        return processed