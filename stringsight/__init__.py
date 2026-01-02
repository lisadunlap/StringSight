"""
StringSight: Language Model Model Vibes Analysis

A toolkit for analyzing and understanding model behavior patterns through
property extraction, clustering, and metrics computation.
"""

from .public import (
    explain,
    explain_side_by_side,
    explain_single_model,
    explain_with_custom_pipeline,
    compute_metrics_only,
    label,
    extract_properties_only,
)
from .utils.tau2 import tau2_json_to_stringsight_df


__version__ = "0.3.6"
__all__ = [
    "explain",
    "explain_side_by_side",
    "explain_single_model",
    "explain_with_custom_pipeline",
    "compute_metrics_only",
    "label",
    "extract_properties_only",
    "tau2_json_to_stringsight_df",
] 