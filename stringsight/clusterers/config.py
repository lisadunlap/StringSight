from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Any
import numpy as np


@dataclass
class ClusterConfig:
    """Configuration for clustering operations.

    This mirrors the configuration used by hierarchical_clustering, but is split
    into a lightweight module to avoid importing heavy dependencies at import time.
    """
    # Core clustering
    min_cluster_size: Optional[int] = None
    verbose: bool = True
    include_embeddings: bool = True
    context: Optional[str] = None
    precomputed_embeddings: Optional[Union[np.ndarray, Dict, str]] = None
    disable_dim_reduction: bool = False
    assign_outliers: bool = False
    input_model_name: Optional[str] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cache_embeddings: bool = False
    groupby_column: Optional[str] = None # if not None, the data will be grouped by this column before clustering

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    summary_model: str = "gpt-4.1"
    cluster_assignment_model: str = "gpt-4.1-mini"

    # Dimension reduction settings
    dim_reduction_method: str = "adaptive"  # "adaptive", "umap", "pca", "none"
    umap_n_components: int = 100
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # wandb configuration
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self) -> None:
        # Keep min_samples as provided (None means let HDBSCAN use its default = min_cluster_size)
        pass

    @classmethod
    def from_args(cls, args: Any) -> "ClusterConfig":
        """Create a ClusterConfig from argparse-style args.

        Mirrors the previous behavior in hierarchical_clustering.
        """
        use_wandb = not args.no_wandb if hasattr(args, "no_wandb") else True
        return cls(
            min_cluster_size=args.min_cluster_size,
            embedding_model=args.embedding_model,
            verbose=not hasattr(args, "quiet") or not args.quiet,
            include_embeddings=not args.no_embeddings,
            context=getattr(args, "context", None),
            precomputed_embeddings=getattr(args, "precomputed_embeddings", None),
            disable_dim_reduction=getattr(args, "disable_dim_reduction", False),
            assign_outliers=getattr(args, "assign_outliers", False),
            input_model_name=getattr(args, "input_model_name", None),
            min_samples=getattr(args, "min_samples", None),
            cluster_selection_epsilon=getattr(args, "cluster_selection_epsilon", 0.0),
            groupby_column=getattr(args, "groupby_column", None),
            # Dimension reduction settings
            dim_reduction_method=getattr(args, "dim_reduction_method", "adaptive"),
            umap_n_components=getattr(args, "umap_n_components", 100),
            umap_n_neighbors=getattr(args, "umap_n_neighbors", 30),
            umap_min_dist=getattr(args, "umap_min_dist", 0.1),
            umap_metric=getattr(args, "umap_metric", "cosine"),
            # wandb
            use_wandb=use_wandb,
            wandb_project=getattr(args, "wandb_project", None),
            wandb_entity=getattr(args, "wandb_entity", None),
            wandb_run_name=getattr(args, "wandb_run_name", None),
        ) 