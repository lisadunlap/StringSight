"""
Clustering stages for StringSight.

This module contains stages that cluster properties into coherent groups.
"""

from typing import Union
from ..core.stage import PipelineStage


def get_clusterer(
    method: str = "hdbscan",
    min_cluster_size: int | None = None,
    embedding_model: str = "text-embedding-3-small",
    assign_outliers: bool = False,
    include_embeddings: bool = True,
    **kwargs
) -> PipelineStage:
    """
    Factory function to get the appropriate clusterer.
    
    Args:
        method: Clustering method ("hdbscan", "dummy")
        min_cluster_size: Minimum cluster size
        embedding_model: Embedding model to use
        assign_outliers: Whether to assign outliers to nearest clusters
        include_embeddings: Whether to include embeddings in output
        **kwargs: Additional configuration
        
    Returns:
        Configured clusterer stage
    """
    
    if method == "hdbscan":
        from .hdbscan import HDBSCANClusterer
        return HDBSCANClusterer(
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            assign_outliers=assign_outliers,
            include_embeddings=include_embeddings,
            **kwargs
        )
    # 'hdbscan_stratified' alias has been removed; users should pass
    # `method="hdbscan"` and supply `groupby_column` if stratification is
    # desired.
    elif method == "dummy":
        from .dummy_clusterer import DummyClusterer
        return DummyClusterer(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# Import clusterer classes for direct access
from .hdbscan import HDBSCANClusterer
from .dummy_clusterer import DummyClusterer
from .base import BaseClusterer

__all__ = [
    "get_clusterer",
    "HDBSCANClusterer",
    "DummyClusterer",
    "BaseClusterer",
] 