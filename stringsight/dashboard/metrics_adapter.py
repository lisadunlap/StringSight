"""Lightweight access helpers for FunctionalMetrics score dictionaries.

The Gradio UI now receives the *raw* FunctionalMetrics output as a
```
metrics = {
    "model_cluster_scores": {...},
    "cluster_scores": {...},
    "model_scores": {...},
}
```
This module centralises the most common look-ups so that the rest of the
codebase does *not* need to know the exact key names.  If the format
changes again we only need to update these helpers.
"""
from typing import Dict, Any, List

__all__ = [
    "get_model_clusters",
    "get_all_models",
    "get_all_clusters",
]

def get_model_clusters(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Return the per-cluster dictionary for a given model.

    Args:
        metrics: The dict returned by ``load_pipeline_results``.
        model_name: Name of the model.
    """
    if model_name == "all":
        # For "all" model, return cluster_scores (aggregated across all models)
        return metrics.get("cluster_scores", {})
    else:
        return metrics.get("model_cluster_scores", {}).get(model_name, {})


def get_all_models(metrics: Dict[str, Any]) -> List[str]:
    """Return the list of model names present in the metrics dict."""
    models = list(metrics.get("model_cluster_scores", {}).keys())
    # Add "all" as the first option to show aggregated metrics across all models
    return ["all"] + models


def get_all_clusters(metrics: Dict[str, Any]) -> List[str]:
    """Return the list of cluster names (across all models)."""
    return list(metrics.get("cluster_scores", {}).keys()) 