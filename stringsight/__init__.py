"""
StringSight: Language Model Model Vibes Analysis

A toolkit for analyzing and understanding model behavior patterns through
property extraction, clustering, and metrics computation.
"""

from .public import explain, explain_side_by_side, explain_single_model, explain_with_custom_pipeline, compute_metrics_only, label, extract_properties_only


__version__ = "0.3.1"
__all__ = [
    "explain",
    "explain_side_by_side",
    "explain_single_model",
    "explain_with_custom_pipeline",
    "compute_metrics_only",
    "label",
    "extract_properties_only",
] 