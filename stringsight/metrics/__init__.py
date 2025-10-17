"""Metrics computation modules.

Currently available:

* :pyclass:`stringsight.metrics.side_by_side.SideBySideMetrics` – metrics for the
  Arena‐style side-by-side dataset where each question is answered by multiple
  models.
* :pyclass:`stringsight.metrics.functional_metrics.FunctionalMetrics` – simplified
  functional approach with cleaner outputs for model-cluster analysis (DEFAULT for single_model).
* :pyclass:`stringsight.metrics.single_model.SingleModelMetrics` – legacy single-model
  metrics (available as 'single_model_legacy').
"""

from importlib import import_module as _imp
from typing import Dict, Any

__all__: list[str] = [
    "SideBySideMetrics",
    "SingleModelMetrics",
    "FunctionalMetrics",
    "get_metrics",
]

# Lazy import to keep import time low
SideBySideMetrics = _imp("stringsight.metrics.side_by_side").SideBySideMetrics
SingleModelMetrics = _imp("stringsight.metrics.single_model").SingleModelMetrics
FunctionalMetrics = _imp("stringsight.metrics.functional_metrics").FunctionalMetrics


def get_metrics(method: str, **kwargs) -> "PipelineStage":
    """
    Factory function for metrics stages.
    
    Args:
        method: "side_by_side", "single_model", "functional", or "single_model_legacy"
        **kwargs: Additional configuration for the metrics stage
        
    Returns:
        Configured metrics stage
    """
    # Remap legacy flag name for wandb to the functional parameter
    if "use_wandb" in kwargs and "log_to_wandb" not in kwargs:
        kwargs["log_to_wandb"] = kwargs.pop("use_wandb")

    if method == "side_by_side":
        return SideBySideMetrics(**kwargs)
    elif method == "single_model":
        # NEW: Default to functional metrics for single_model
        return SingleModelMetrics(**kwargs)
    # elif method == "functional":
    #     return FunctionalMetrics(**kwargs)
    else:
        raise ValueError(f"Unknown metrics method: {method}. Available: 'side_by_side', 'single_model', 'functional', 'single_model_legacy'") 
    