"""
Property validation stage.

This stage validates and cleans extracted properties.
"""

from pathlib import Path
import json
import pandas as pd
from typing import Optional, List
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin
from ..storage.adapter import StorageAdapter, get_storage_adapter


class PropertyValidator(LoggingMixin, PipelineStage):
    """
    Validate and clean extracted properties.
    
    This stage ensures that all properties have valid data and removes
    any properties that don't meet quality criteria.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        storage: Optional[StorageAdapter] = None,
        **kwargs
    ):
        """Initialize the property validator."""
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.storage = storage or get_storage_adapter()
        
    def run(self, data: PropertyDataset, progress_callback=None) -> PropertyDataset:
        """
        Validate and clean properties.
        
        Args:
            data: PropertyDataset with properties to validate
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            PropertyDataset with validated properties
        """
        self.log(f"Validating {len(data.properties)} properties")
        
        
        valid_properties = []
        invalid_properties = []
        total_props = len(data.properties)
        for i, prop in enumerate(data.properties):
            if progress_callback and i % 100 == 0:
                try:
                    progress_callback(i / total_props)
                except Exception:
                    pass
            is_valid = self._is_valid_property(prop)
            if is_valid:
                valid_properties.append(prop)
            else:
                invalid_properties.append(prop)
                
        self.log(f"Kept {len(valid_properties)} valid properties")
        self.log(f"Filtered out {len(invalid_properties)} invalid properties")
        
        
        # Check for 0 valid properties and provide helpful error message
        if len(valid_properties) == 0:
            raise RuntimeError(
                "ERROR: 0 valid properties after validation. "
                "This typically means: (1) LLM returned empty/invalid responses, "
                "(2) JSON parsing failures, or (3) All properties filtered during validation. "
                "Check logs above for details."
            )
        
        # Auto-save validation results if output_dir is provided
        if self.output_dir:
            self._save_stage_results(data, valid_properties, invalid_properties)
        
        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=valid_properties,
            clusters=data.clusters,
            model_stats=data.model_stats
        )
    
    def _save_stage_results(self, data: PropertyDataset, valid_properties: List[Property], invalid_properties: List[Property]):
        """Save validation results to the specified output directory."""
        # Create output directory if it doesn't exist
        output_path = self.output_dir
        self.storage.ensure_directory(output_path)

        self.log(f"✅ Auto-saving validation results to: {output_path}")

        # 1. Save validated properties as JSONL
        valid_records = [prop.to_dict() for prop in valid_properties]
        valid_path = f"{output_path}/validated_properties.jsonl"
        self.storage.write_jsonl(valid_path, valid_records)
        self.log(f"  • Validated properties: {valid_path}")

        # 2. Save invalid properties as JSONL (for debugging)
        if invalid_properties:
            invalid_records = [prop.to_dict() for prop in invalid_properties]
            invalid_path = f"{output_path}/invalid_properties.jsonl"
            self.storage.write_jsonl(invalid_path, invalid_records)
            self.log(f"  • Invalid properties: {invalid_path}")

        # 3. Save validation statistics
        stats = {
            "total_input_properties": len(data.properties),
            "total_valid_properties": len(valid_properties),
            "total_invalid_properties": len(invalid_properties),
            "validation_success_rate": len(valid_properties) / len(data.properties) if data.properties else 0,
        }

        stats_path = f"{output_path}/validation_stats.json"
        self.storage.write_json(stats_path, stats)
        self.log(f"  • Validation stats: {stats_path}")
    
    def _is_valid_property(self, prop: Property) -> bool:
        """Check if a property is valid."""
        # Basic validation - property description should exist and not be empty
        if not (prop.property_description and prop.property_description.strip()):
            return False

        # Validate behavior_type if present
        if prop.behavior_type:
            allowed_types = {
                "Positive",
                "Negative (critical)",
                "Negative (non-critical)",
                "Style"
            }
            if prop.behavior_type not in allowed_types:
                return False

        return True 