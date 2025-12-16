"""
HDBSCAN-based clustering stages.

This module migrates the clustering logic from clustering/hierarchical_clustering.py
into pipeline stages.
"""

from typing import Optional
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base import BaseClusterer
from ..core.data_objects import PropertyDataset
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin
from ..logging_config import get_logger

logger = get_logger(__name__)

# Unified config
try:
    from .config import ClusterConfig
except ImportError:
    from config import ClusterConfig

try:
    from stringsight.clusterers.hierarchical_clustering import (
        hdbscan_cluster_categories,
    )
except ImportError:
    from .hierarchical_clustering import (  # type: ignore
        hdbscan_cluster_categories,
    )

class HDBSCANClusterer(BaseClusterer):
    """
    HDBSCAN clustering stage.

    This stage migrates the hdbscan_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    """

    def __init__(
        self,
        min_cluster_size: int | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        include_embeddings: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        output_dir: Optional[str] = None,
        # Additional explicit configuration parameters\
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: float = 0.0,
        disable_dim_reduction: bool = False,
        dim_reduction_method: str = "adaptive",
        context: Optional[str] = None,
        groupby_column: Optional[str] = None,
        parallel_clustering: bool = True,
        cluster_positive: bool = True,
        precomputed_embeddings: Optional[object] = None,
        cache_embeddings: bool = True,
        input_model_name: Optional[str] = None,
        summary_model: str = "gpt-4.1",
        cluster_assignment_model: str = "gpt-4.1-mini",
        verbose: bool = True,
        llm_max_workers: int = 64,
        **kwargs,
    ):
        """Initialize the HDBSCAN clusterer with explicit, overridable parameters.
        
        """
        super().__init__(
            output_dir=output_dir,
            include_embeddings=include_embeddings,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            **kwargs,
        )

        # Build a unified ClusterConfig (no hardcoded values)
        self.config = ClusterConfig(
            # core
            min_cluster_size=min_cluster_size,
            verbose=verbose,
            include_embeddings=include_embeddings,
            context=context,
            precomputed_embeddings=precomputed_embeddings,
            disable_dim_reduction=disable_dim_reduction,
            input_model_name=input_model_name,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cache_embeddings=cache_embeddings,
            # models
            embedding_model=embedding_model,
            summary_model=summary_model,
            cluster_assignment_model=cluster_assignment_model,
            llm_max_workers=llm_max_workers,
            # dim reduction
            dim_reduction_method=dim_reduction_method,
            # groupby
            groupby_column=groupby_column,
            parallel_clustering=parallel_clustering,
            cluster_positive=cluster_positive,
            # wandb
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )


    async def cluster(self, data: PropertyDataset, column_name: str, progress_callback=None) -> pd.DataFrame:
        """Cluster the dataset.

        If ``self.config.groupby_column`` is provided and present in the data, the
        input DataFrame is first partitioned by that column and each partition is
        clustered independently (stratified clustering). Results are then
        concatenated back together. Otherwise, the entire dataset is clustered
        at once.
        """

        df = data.to_dataframe(type="properties")

        # `to_dataframe(type="properties")` may include conversation rows without any extracted
        # properties (i.e., missing/NaN `column_name`). Those rows cannot be clustered and can
        # also coerce cluster id dtypes to float downstream, which breaks group metadata mapping.
        if column_name in df.columns:
            df = df[df[column_name].notna()].copy()
        
        if getattr(self, "verbose", False):
            logger.debug(f"DataFrame shape after to_dataframe: {df.shape}")
            logger.debug(f"DataFrame columns: {list(df.columns)}")
            logger.debug(f"DataFrame head:")
            logger.debug(df.head())
        
        if column_name in df.columns:
            if getattr(self, "verbose", False):
                logger.debug(f"{column_name} unique values: {df[column_name].nunique()}")
                logger.debug(f"{column_name} value counts:")
                logger.debug(df[column_name].value_counts())
                logger.debug(f"Sample {column_name} values: {df[column_name].head().tolist()}")
        else:
            logger.error(f"Column '{column_name}' not found in DataFrame!")

        group_col = getattr(self.config, "groupby_column", None)
        cluster_positive = getattr(self.config, "cluster_positive", False)

        # Filter out positive behaviors if cluster_positive is False and groupby_column is behavior_type
        positive_mask = None
        positive_df = None
        if group_col == "behavior_type" and not cluster_positive and "behavior_type" in df.columns:
            positive_mask = df["behavior_type"] == "Positive"
            positive_df = df[positive_mask].copy()
            df = df[~positive_mask].copy()
            if len(positive_df) > 0:
                self.log(f"Filtering out {len(positive_df)} positive behaviors from clustering (cluster_positive=False)")
            if len(df) == 0:
                self.log("All behaviors are positive and cluster_positive=False - skipping clustering")

        # Handle case where all behaviors were filtered out
        if len(df) == 0:
            # If we have positive behaviors, return them with special cluster assignment
            if positive_df is not None and len(positive_df) > 0:
                id_col = f"{column_name}_cluster_id"
                label_col = f"{column_name}_cluster_label"
                positive_df[id_col] = -2
                positive_df[label_col] = "Positive (not clustered)"
                if "meta" not in positive_df.columns:
                    positive_df["meta"] = [{} for _ in range(len(positive_df))]
                return positive_df
            # Otherwise return empty DataFrame with required columns
            id_col = f"{column_name}_cluster_id"
            label_col = f"{column_name}_cluster_label"
            df_empty = pd.DataFrame(columns=[column_name, id_col, label_col, "question_id", "id", "meta"])
            return df_empty

        if group_col is not None and group_col in df.columns:
            parallel_clustering = getattr(self.config, "parallel_clustering", False)

            if parallel_clustering:
                # Parallelize clustering per group for better performance (using async)
                groups = list(df.groupby(group_col))

                async def _cluster_group_async(group_info):
                    group, group_df = group_info
                    if getattr(self, "verbose", False):
                        logger.info(f"--------------------------------\nClustering group {group}\n--------------------------------")
                    part = await hdbscan_cluster_categories(
                        group_df,
                        column_name=column_name,
                        config=self.config,
                    )
                    return group, part

                # Process groups in parallel using async
                clustered_parts = []
                max_workers = min(len(groups), getattr(self.config, 'llm_max_workers', 64))
                
                # Create coroutines (not tasks yet - we're not in event loop)
                coros = [_cluster_group_async(group_info) for group_info in groups]
                
                # Use gather to run them all - this works even if not in event loop yet
                # We'll iterate with as_completed for progress tracking
                tasks = [asyncio.ensure_future(coro) for coro in coros]

                # Add progress bar for parallel clustering
                total_groups = len(groups)
                completed_groups = 0
                with tqdm(total=total_groups, desc=f"Clustering {total_groups} groups in parallel", disable=not getattr(self, "verbose", False)) as pbar:
                    for task in asyncio.as_completed(tasks):
                        group, part = await task
                        clustered_parts.append(part)
                        pbar.update(1)
                        completed_groups += 1
                        if progress_callback:
                            try:
                                progress_callback(completed_groups / total_groups)
                            except Exception:
                                pass
                clustered_df = pd.concat(clustered_parts, ignore_index=True)
            else:
                # Process groups sequentially (default behavior)
                clustered_parts = []
                groups = list(df.groupby(group_col))

                # Add progress bar for sequential clustering
                total_groups = len(groups)
                for i, (group, group_df) in enumerate(tqdm(groups, desc=f"Clustering {len(groups)} groups sequentially", disable=not getattr(self, "verbose", False))):
                    if getattr(self, "verbose", False):
                        logger.info(f"--------------------------------\nClustering group {group}\n--------------------------------")
                    part = await hdbscan_cluster_categories(
                        group_df,
                        column_name=column_name,
                        config=self.config,
                    )
                    clustered_parts.append(part)
                    if progress_callback:
                        try:
                            progress_callback((i + 1) / total_groups)
                        except Exception:
                            pass
                clustered_df = pd.concat(clustered_parts, ignore_index=True)
        else:
            clustered_df = await hdbscan_cluster_categories(
                df,
                column_name=column_name,
                config=self.config,
            )

        # Add back positive behaviors with special cluster assignment if they were filtered out
        if positive_df is not None and len(positive_df) > 0:
            id_col = f"{column_name}_cluster_id"
            label_col = f"{column_name}_cluster_label"
            
            # Assign special cluster ID and label for positive behaviors that weren't clustered
            positive_df[id_col] = -2
            positive_df[label_col] = "Positive (not clustered)"
            if "meta" not in positive_df.columns:
                positive_df["meta"] = [{} for _ in range(len(positive_df))]
            
            # Concatenate back with clustered results
            clustered_df = pd.concat([clustered_df, positive_df], ignore_index=True)
            self.log(f"Added back {len(positive_df)} positive behaviors with cluster_id=-2 (not clustered)")

        return clustered_df

    async def postprocess_clustered_df(self, df: pd.DataFrame, column_name: str, prettify_labels: bool = False) -> pd.DataFrame:
        """Standard post-processing plus stratified ID re-assignment when needed."""

        label_col = f"{column_name}_cluster_label"
        id_col = f"{column_name}_cluster_id"

        # Ensure no NaN values in label column (causes downstream issues in metrics)
        if label_col in df.columns and df[label_col].isna().any():
            n_nans = df[label_col].isna().sum()
            self.log(f"Found {n_nans} properties with NaN cluster labels. Assigning to 'Outliers'.")
            df[label_col] = df[label_col].fillna("Outliers")
            df.loc[df[label_col] == "Outliers", id_col] = -1

        df = await super().postprocess_clustered_df(df, label_col, prettify_labels)

        # 1️⃣  Move tiny clusters to Outliers
        label_counts = df[label_col].value_counts()
        min_size_threshold = int((getattr(self.config, "min_cluster_size", 1) or 1))
        too_small_labels = label_counts[label_counts < min_size_threshold].index
        for label in too_small_labels:
            mask = df[label_col] == label
            cid = df.loc[mask, id_col].iloc[0] if not df.loc[mask].empty else None
            self.log(
                f"Assigning cluster {cid} (label '{label}') to Outliers because it has {label_counts[label]} items"
            )
            
            # Check if we're using groupby and assign group-specific outlier labels
            group_col = getattr(self.config, "groupby_column", None)
            if group_col is not None and group_col in df.columns:
                # Assign group-specific outlier labels
                for group_value in df.loc[mask, group_col].unique():
                    group_mask = mask & (df[group_col] == group_value)
                    outlier_label = f"Outliers - {group_value}"
                    df.loc[group_mask, label_col] = outlier_label
                    df.loc[group_mask, id_col] = -1
            else:
                # Standard outlier assignment
                df.loc[mask, label_col] = "Outliers"
                df.loc[mask, id_col] = -1

        # 2️⃣  For stratified mode: ensure cluster IDs are unique across partitions
        group_col = getattr(self.config, "groupby_column", None)
        if group_col is not None and group_col in df.columns:
            # Handle group-specific outlier labels
            outlier_mask = df[label_col].str.startswith("Outliers - ") if df[label_col].dtype == 'object' else df[label_col] == "Outliers"
            non_outlier_mask = ~outlier_mask
            
            unique_pairs = (
                df.loc[non_outlier_mask, [group_col, label_col]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            pair_to_new_id = {
                (row[group_col], row[label_col]): idx for idx, row in unique_pairs.iterrows()
            }
            for (gval, lbl), new_id in pair_to_new_id.items():
                pair_mask = (df[group_col] == gval) & (df[label_col] == lbl) & non_outlier_mask
                df.loc[pair_mask, id_col] = new_id

            # Handle group-specific outliers: assign unique IDs to each outlier group
            if outlier_mask.any():
                # Get unique outlier labels
                unique_outlier_labels = df.loc[outlier_mask, label_col].unique()
                
                # Assign unique IDs to each outlier group, starting from a high negative number
                # to avoid conflicts with regular cluster IDs
                outlier_id_start = -1000
                for i, outlier_label in enumerate(unique_outlier_labels):
                    outlier_label_mask = df[label_col] == outlier_label
                    unique_outlier_id = outlier_id_start - i
                    df.loc[outlier_label_mask, id_col] = unique_outlier_id

        return df

    # ------------------------------------------------------------------
    # 🏷️  Cluster construction helper with group metadata
    # ------------------------------------------------------------------
    def _build_clusters_from_df(self, df: pd.DataFrame, column_name: str):
        """Build clusters and, in stratified mode, add group info to metadata."""

        clusters = super()._build_clusters_from_df(df, column_name)

        group_col = getattr(self.config, "groupby_column", None)
        if group_col is not None and group_col in df.columns:
            id_col = f"{column_name}_cluster_id"

            # Pandas may upcast cluster ids to floats if there are any NaNs in the column.
            # Normalize ids to integers before converting to string keys (Cluster.id is a str).
            id_group_df = df.loc[df[id_col].notna(), [id_col, group_col]].dropna().copy()
            if not id_group_df.empty:
                # If ids are floats like 0.0, cast back to int safely.
                id_group_df[id_col] = id_group_df[id_col].astype(float).astype(int)

            id_to_group = (
                id_group_df.groupby(id_col)[group_col]
                .agg(lambda s: s.iloc[0])
                .to_dict()
            )
            id_to_group = {str(int(k)): v for k, v in id_to_group.items()}
            
            for c in clusters:
                cid = getattr(c, "id", None)
                if cid in id_to_group:
                    c.meta = dict(c.meta or {})
                    group_val = id_to_group[cid]
                    # Frontend expects `metadata.group` for chart categorization.
                    c.meta["group"] = group_val
                    # Also store under the actual grouping column name (e.g. "behavior_type")
                    # for explicitness and easier debugging.
                    c.meta[group_col] = group_val

        return clusters
class LLMOnlyClusterer(HDBSCANClusterer):
    """
    HDBSCAN clustering stage.

    This stage migrates the hdbscan_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    """

    def run(self, data: PropertyDataset, column_name: str = "property_description", progress_callback=None) -> PropertyDataset:
        """Cluster properties using HDBSCAN (delegates to base)."""
        return super().run(data, column_name, progress_callback=progress_callback)


