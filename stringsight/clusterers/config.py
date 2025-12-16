from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Any
import numpy as np


def _cuda_available() -> bool:
    """Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import cupy as cp
        # Try to access a GPU device
        cp.cuda.Device(0).compute_capability
        return True
    except (ImportError, Exception):
        pass
    
    return False


@dataclass
class ClusterConfig:
    """Configuration for clustering operations.

    This mirrors the configuration used by hierarchical_clustering, but is split
    into a lightweight module to avoid importing heavy dependencies at import time.
    """
    # Core clustering
    min_cluster_size: Optional[int] = 5  # Smaller = fewer outliers, more clusters
    verbose: bool = True
    include_embeddings: bool = False
    context: Optional[str] = None
    precomputed_embeddings: Optional[Union[np.ndarray, Dict, str]] = None
    disable_dim_reduction: bool = False
    assign_outliers: bool = True
    input_model_name: Optional[str] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.05  # Small epsilon to merge very similar clusters
    cache_embeddings: bool = True
    groupby_column: Optional[str] = None # if not None, the data will be grouped by this column before clustering
    parallel_clustering: bool = False  # if True, parallelize clustering when groupby_column is set
    cluster_positive: bool = True  # if False and groupby_column is "behavior_type", skip clustering positive behaviors

    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    summary_model: str = "gpt-4.1"
    cluster_assignment_model: str = "gpt-4.1-mini"
    # Parallelism for LLM calls used during clustering (summaries, matching, prettify)
    llm_max_workers: int = 64

    # GPU acceleration (auto-detected by default)
    use_gpu: Optional[bool] = None  # None means auto-detect; will be set in __post_init__

    # Dimension reduction settings
    dim_reduction_method: str = "adaptive"  # "adaptive", "pca", "none"

    # wandb configuration
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self) -> None:
        # Auto-detect GPU availability if not explicitly set
        if self.use_gpu is None:
            self.use_gpu = _cuda_available()
        # Keep min_samples as provided (None means let HDBSCAN use its default = min_cluster_size)

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
            input_model_name=getattr(args, "input_model_name", None),
            min_samples=getattr(args, "min_samples", None),
            cluster_selection_epsilon=getattr(args, "cluster_selection_epsilon", 0.0),
            groupby_column=getattr(args, "groupby_column", None),
            cluster_positive=getattr(args, "cluster_positive", True),
            # Dimension reduction settings
            dim_reduction_method=getattr(args, "dim_reduction_method", "adaptive"),
            # wandb
            use_wandb=use_wandb,
            wandb_project=getattr(args, "wandb_project", None),
            wandb_entity=getattr(args, "wandb_entity", None),
            wandb_run_name=getattr(args, "wandb_run_name", None),
        ) 