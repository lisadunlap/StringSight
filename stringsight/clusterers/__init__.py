"""
Clustering stages for StringSight.

This module contains stages that cluster properties into coherent groups.
"""

from typing import Union
from ..core.stage import PipelineStage


def get_clusterer(
    method: str = "hdbscan",
    min_cluster_size: int | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    assign_outliers: bool = False,
    include_embeddings: bool = False,
    cluster_positive: bool = True,
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
        use_gpu: Enable GPU acceleration for embeddings and HDBSCAN.
                None (default) = auto-detect based on CUDA availability.
        cluster_positive: If False and groupby_column is "behavior_type", skip clustering positive behaviors.
                         Defaults to True.
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
            cluster_positive=cluster_positive,
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
