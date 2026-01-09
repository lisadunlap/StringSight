"""
Public API for StringSight.

This module provides the main explain() function that users will interact with.
"""

# Import all public API functions from submodules
from ._public.sync_api import explain, extract_properties_only
from ._public.async_api import explain_async, extract_properties_only_async
from ._public.label_api import label
from ._public.convenience import (
    explain_side_by_side,
    explain_single_model,
    explain_with_custom_pipeline,
)
from ._public.metrics_only import compute_metrics_only

# Export all public functions
__all__ = [
    # Main sync API
    "explain",
    "extract_properties_only",
    # Async API
    "explain_async",
    "extract_properties_only_async",
    # Label API
    "label",
    # Convenience functions
    "explain_side_by_side",
    "explain_single_model",
    "explain_with_custom_pipeline",
    # Metrics utility
    "compute_metrics_only",
]
