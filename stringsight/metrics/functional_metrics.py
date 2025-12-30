"""
stringsight.metrics.functional_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplified functional approach to metrics computation that produces cleaner,
more debuggable results with separate outputs for model-cluster, cluster, and model metrics.

This approach is based on the hand-coded implementation that separates concerns
and produces three distinct output files rather than one complex nested structure.

OUTPUT FORMAT DOCUMENTATION
============================

This module produces 3 separate JSON files with the following structure:

1. **model_cluster_scores.json** - Per model-cluster combination metrics
   ```json
   {
     "model_name": {
       "cluster_name": {
         "size": int,                    # Number of conversations for this model in this cluster
         "proportion": float,            # What fraction of this model's conversations are in this cluster (0-1)
         "quality": {                    # Raw quality scores for this model-cluster combination
           "metric_name": float          # e.g., "helpfulness": 4.2, "accuracy": 3.8
         },
         "quality_delta": {              # Raw difference: (Cluster Score - Model Average)
           "metric_name": float          # e.g., "helpfulness": +0.15 (this cluster is 0.15 higher than model's average)
         },
         "proportion_delta": float,      # Salience: how much this model over/under-represents vs average of OTHER models
         "metadata": {},                 # Cluster metadata (e.g., group information from stratified clustering)
         "examples": [                   # Sample conversation IDs and metadata for this model-cluster
           [conversation_id, conversation_metadata, property_metadata], ...
         ],
         
         # Bootstrap confidence intervals (when enabled):
         "proportion_ci": {"lower": float, "upper": float, "mean": float},
         "quality_ci": {"metric_name": {"lower": float, "upper": float, "mean": float}},
         "quality_delta_ci": {"metric_name": {"lower": float, "upper": float, "mean": float}},
         "proportion_delta_ci": {"lower": float, "upper": float, "mean": float},
         
         # Significance testing (when bootstrap enabled):
         "quality_delta_significant": {"metric_name": bool},  # True if quality_delta CI doesn't contain 0
         "proportion_delta_significant": bool                 # True if proportion_delta CI doesn't contain 0
       }
     }
   }
   ```

2. **cluster_scores.json** - Per cluster metrics (aggregated across all models)
   ```json
   {
     "cluster_name": {
       "size": int,                      # Total conversations across all models in this cluster
       "proportion": float,              # What fraction of all conversations are in this cluster
       "quality": {                      # Average quality scores across all models for this cluster
         "metric_name": float
       },
       "quality_delta": {                # Raw difference: (Cluster Score - Global Average)
         "metric_name": float
       },
       "metadata": {},                   # Cluster metadata (e.g., group information from stratified clustering)
       "examples": [...],                # Sample conversations from all models in this cluster
       
       # Bootstrap CIs (when enabled):
       "proportion_ci": {...},
       "quality_ci": {...},
       "quality_delta_ci": {...},
       
       # Significance testing (when bootstrap enabled):
       "quality_delta_significant": {"metric_name": bool}  # True if quality_delta CI doesn't contain 0
     }
   }
   ```

3. **model_scores.json** - Per model metrics (aggregated across all clusters)
   ```json
   {
     "model_name": {
       "size": int,                      # Total conversations for this model across all clusters
       "proportion": float,              # Always 1.0 (model represents 100% of its own conversations)
       "quality": {                      # Average quality scores for this model across all clusters
         "metric_name": float
       },
       "quality_delta": {                # Raw difference: (Model Score - Cross-Model Average)
         "metric_name": float
       },
       "examples": [...],                # Sample conversations for this model across all clusters
       
       # Bootstrap CIs (when enabled):
       "proportion_ci": {...},
       "quality_ci": {...},
       "quality_delta_ci": {...},
       
       # Significance testing (when bootstrap enabled):
       "quality_delta_significant": {"metric_name": bool}  # True if quality_delta CI doesn't contain 0
     }
   }
   ```

KEY CONCEPTS
============

- **proportion**: What fraction of the parent set (model/all) falls into this subset
- **quality**: Raw quality scores (e.g., helpfulness, accuracy ratings)
- **quality_delta**: Raw difference in scores = (Score - Baseline). Shows how much better/worse this cluster/model is compared to baseline.
- **proportion_delta** (salience): How much a model over/under-represents in a cluster compared to OTHER models
  - Positive = model appears more than other models on average in this cluster
  - Negative = model appears less than other models on average in this cluster
- **Bootstrap CIs**: Confidence intervals computed by resampling conversations
  - When bootstrap is enabled, the main metric values are set to bootstrap means

MAPPING TO LEGACY FORMAT
=========================

The legacy `model_stats.json` had a nested structure like:
```json
{
  "model_name": {
    "fine": [
      {
        "property_description": "cluster_name",
        "score": float,           # ← Roughly maps to proportion + proportion_delta
        "quality_score": {...},   # ← Maps to quality
        "proportion": float,      # ← Maps to proportion
        "size": int,             # ← Maps to size
        "score_ci": {...},       # ← Maps to proportion_ci
        "quality_score_ci": {...} # ← Maps to quality_ci
      }
    ]
  }
}
```

The new format is more modular and separates:
- Model-cluster details (model_cluster_scores.json)
- Cluster summaries (cluster_scores.json) 
- Model summaries (model_scores.json)

This makes it easier to analyze data from different perspectives without complex nested navigation.

WANDB LOGGING
=============

When wandb logging is enabled (default), three dataframes are logged as wandb Tables:

1. **model_cluster_scores** - Flattened view of model-cluster combinations
   - Columns: `model`, `property`, `size`, `proportion`, `quality`, `quality_delta`, `proportion_delta`, `examples`, 
     plus confidence intervals (`*_ci`) and significance flags (`*_significant`) when bootstrap is enabled
   - Each row represents one model's performance on one cluster/property
   - Examples are limited to first 5 entries and converted to strings

2. **cluster_scores** - Cluster-level aggregations across all models  
   - Columns: `model` (always "all"), `property`, `size`, `proportion`, `quality`, `quality_delta`, `examples`,
     plus confidence intervals and significance when bootstrap enabled
   - Each row represents one cluster/property's overall statistics

3. **model_scores** - Model-level aggregations across all clusters
   - Columns: `model`, `property` (always "all_clusters"), `size`, `proportion`, `quality`, `quality_delta`, `examples`,
     plus confidence intervals and significance when bootstrap enabled  
   - Each row represents one model's overall performance

All dataframes have dict/list values converted to strings and NA values filled with "None" for wandb compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

import importlib.util

from ..core.stage import PipelineStage
from ..core.mixins import LoggingMixin, TimingMixin
from ..core.data_objects import PropertyDataset
from ..storage.adapter import StorageAdapter, get_storage_adapter
from . import plotting


class FunctionalMetrics(PipelineStage, LoggingMixin, TimingMixin):
    """Simplified functional approach to metrics computation.
    
    Features:
    - Computes model-cluster, cluster, and model-level metrics
    - Optional bootstrap confidence intervals and significance testing
    - Saves results to JSON files (model_cluster_scores.json, cluster_scores.json, model_scores.json)
    - Optional wandb logging of results as tables (enabled by default)
    - Optional comprehensive plot generation with wandb logging (disabled by default)
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        compute_bootstrap: bool = True,
        bootstrap_samples: int = 100,
        bootstrap_seed: int | None = None,
        log_to_wandb: bool = True,
        generate_plots: bool = True,
        storage: Optional[StorageAdapter] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir) if output_dir else None
        self.compute_bootstrap = compute_bootstrap
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed
        self.log_to_wandb = log_to_wandb
        self.generate_plots = generate_plots
        self.storage = storage or get_storage_adapter()

    def run(self, data: PropertyDataset, progress_callback=None) -> PropertyDataset:
        """Main entry point for metrics computation."""
        self.log("⚖️  Computing functional metrics...")

        # Convert to DataFrame and prepare data
        df = self._prepare_data(data)
        if df.empty:
            self.log("No cluster data found; saving empty metrics.")
            if self.output_dir:
                self._save_results({}, {}, {})
            
            # Initialize empty model_stats to avoid AttributeError downstream
            data.model_stats = {
                "model_cluster_scores": pd.DataFrame(),
                "cluster_scores": pd.DataFrame(),
                "model_scores": pd.DataFrame()
            }
            return data

        # Extract cluster names and models, ensuring no NaN values
        cluster_names = [c for c in df["cluster"].unique() if pd.notna(c)]
        model_names = df["model"].unique()

        self.log(f"Computing metrics for {len(model_names)} models and {len(cluster_names)} clusters")

        # Core metrics computation
        model_cluster_scores = self._compute_model_cluster_scores(df, cluster_names, model_names)
        model_cluster_scores = self._compute_salience(model_cluster_scores)
        
        cluster_scores = self._compute_cluster_scores(df, cluster_names, model_names)
        model_scores = self._compute_model_scores(df, cluster_names, model_names)

        # Add bootstrap analysis if enabled and sample count > 0
        if self.compute_bootstrap and self.bootstrap_samples > 0:
            self.log(f"Adding bootstrap confidence intervals with {self.bootstrap_samples} samples...")
            model_cluster_scores, cluster_scores, model_scores = self._add_bootstrap_analysis(
                df,
                model_cluster_scores,
                cluster_scores,
                model_scores,
                cluster_names=cluster_names,
                model_names=list(model_names),
                progress_callback=progress_callback,
            )

        # Save results
        if self.output_dir:
            self._save_results(model_cluster_scores, cluster_scores, model_scores)

        # Log to wandb if enabled
        if self.log_to_wandb:
            self._log_to_wandb(model_cluster_scores, cluster_scores, model_scores)

        # Generate plots if enabled
        if self.generate_plots and self.output_dir:
            self._generate_plots(model_cluster_scores, cluster_scores, model_scores)

        # Create dataframes for return value
        from .data_transformers import (
            flatten_model_cluster_scores,
            flatten_cluster_scores,
            flatten_model_scores
        )
        
        model_cluster_df = flatten_model_cluster_scores(model_cluster_scores)
        cluster_df = flatten_cluster_scores(cluster_scores)
        model_df = flatten_model_scores(model_scores)
        
        # Return dataframes as model_stats
        data.model_stats = {
            "model_cluster_scores": model_cluster_df,
            "cluster_scores": cluster_df,
            "model_scores": model_df
        }

        self.log(f"✅ Functional metrics computed successfully")
        return data

    def _prepare_data(self, data: PropertyDataset) -> pd.DataFrame:
        """Prepare data in the format expected by functional metrics."""
        # Extract clusters and properties data
        if not data.clusters:
            return pd.DataFrame()

        # Create a property_id -> Property object lookup
        property_lookup = {prop.id: prop for prop in data.properties}

        # Build properties dataframe from clusters, preserving model info via property lookups
        cluster_rows = []
        for cluster in data.clusters:
            for prop_id, prop_desc, question_id in zip(
                cluster.property_ids,
                cluster.property_descriptions,
                cluster.question_ids
            ):
                # Look up the full property object to get model info
                prop = property_lookup.get(prop_id)
                if prop:
                    cluster_rows.append({
                        "property_id": prop_id,
                        "property_description": prop_desc,
                        "question_id": question_id,
                        "model": prop.model,  # ← Preserve model info!
                        "cluster": cluster.label,
                        "cluster_metadata": cluster.meta,
                    })
                else:
                    # Fallback if property not found (shouldn't happen)
                    cluster_rows.append({
                        "property_id": prop_id,
                        "property_description": prop_desc,
                        "question_id": question_id,
                        "model": "unknown",
                        "cluster": cluster.label,
                        "cluster_metadata": cluster.meta,
                    })

        properties = pd.DataFrame(cluster_rows)
        if properties.empty:
            return pd.DataFrame()

        properties = properties.drop_duplicates(subset=["property_description", "question_id", "model"])
        properties = properties.dropna(subset=["property_description", "question_id"])

        # Extract conversations data
        conversations = pd.DataFrame([
            {
                "question_id": conv.question_id,
                "scores": conv.scores,
                "conversation_meta": conv.meta,  # Rename to avoid collision with cluster meta
                "model": conv.model if isinstance(conv.model, str) else conv.model[0]  # Handle list case
            }
            for conv in data.conversations
        ])

        # Join conversations with properties on BOTH question_id and model
        # This ensures correct matching when same question_id has multiple models
        properties = properties.merge(conversations, on=["question_id", "model"], how="left")
        properties.rename(
            {"conversation_meta": "conversation_metadata", "question_id": "conversation_id"},
            axis=1,
            inplace=True
        )
        
        # Ensure conversation_metadata exists - fill missing values with empty dict
        if "conversation_metadata" not in properties.columns:
            properties["conversation_metadata"] = {}
        else:
            properties["conversation_metadata"] = properties["conversation_metadata"].fillna({})
        
        # Ensure cluster_metadata exists - fill missing values with empty dict
        if "cluster_metadata" not in properties.columns:
            properties["cluster_metadata"] = {}
        else:
            properties["cluster_metadata"] = properties["cluster_metadata"].fillna({})
        
        properties["property_metadata"] = properties["property_description"].apply(
            lambda x: {"property_description": x}
        )

        # Select important columns
        important_columns = [
            "conversation_id", "conversation_metadata", "property_metadata", 
            "model", "cluster", "property_description", "scores", "cluster_metadata"
        ]
        
        # Ensure all required columns exist before filtering
        for col in important_columns:
            if col not in properties.columns:
                if col == "scores":
                    properties[col] = {}
                elif col == "model":
                    properties[col] = "unknown"
                elif col in ["cluster_metadata", "conversation_metadata"]:
                    properties[col] = {}
                else:
                    properties[col] = ""
        
        properties = properties[important_columns]

        # Ensure "cluster" column has no NaN values
        if "cluster" in properties.columns:
            properties["cluster"] = properties["cluster"].fillna("Outliers")

        return properties

    def compute_quality_scores(self, df: pd.DataFrame, metrics: List[str] = None) -> Dict[str, float]:
        """Compute average score for each quality metric.

        Parameters:
            df: DataFrame with scores column
            metrics: List of metric names to compute. If None, uses all available metrics.
        """
        if df.empty or "scores" not in df.columns:
            return {}

        # Handle case where scores might not all be dicts
        valid_scores = df[df["scores"].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
        if valid_scores.empty:
            return {}

        scores = pd.DataFrame(valid_scores["scores"].tolist())

        # If specific metrics requested, only compute those (fill missing with 0)
        if metrics is not None:
            result = {}
            for metric in metrics:
                if metric in scores.columns:
                    result[metric] = scores[metric].mean()
                else:
                    result[metric] = 0.0
            return result

        return {col: scores[col].mean() for col in scores.columns}

    def compute_size_and_score(self, df: pd.DataFrame, metrics: List[str] = None) -> tuple[int, Dict[str, float]]:
        """Compute size and quality scores for a dataframe subset.

        Parameters:
            df: DataFrame to compute scores for
            metrics: List of metric names to compute. If None, uses all available metrics.
        """
        df = df.drop_duplicates(subset=["conversation_id", "model"])
        size = len(df)
        quality_scores = self.compute_quality_scores(df, metrics=metrics)
        return size, quality_scores

    def empty_metrics(self, metrics: List[str]) -> Dict[str, Any]:
        """Return empty metrics for clusters with no examples."""
        return {
            "size": 0,
            "proportion": 0,
            "quality": {metric: 0 for metric in metrics},
            "quality_delta": {metric: 0 for metric in metrics},
            "metadata": {},
            "examples": [],
        }

    def compute_relative_quality(self, quality_cluster: Dict[str, float], quality_model: Dict[str, float]) -> Dict[str, float]:
        """Compute relative quality scores (cluster vs model baseline)."""
        return {
            metric: (quality_cluster[metric] - quality_model[metric]) if quality_model.get(metric, 0) != 0 else 0
            for metric in quality_cluster.keys()
        }

    def compute_cluster_metrics(self, df: pd.DataFrame, clusters: List[str], models: List[str], *, include_metadata: bool = True) -> Dict[str, Any]:
        """Bulk of the metrics computation for a specific cluster-model combination.
        
        Parameters:
            include_metadata: Whether to include cluster metadata lookup in the result.
        """
        if isinstance(clusters, str):
            clusters = [clusters]
        if isinstance(models, str):
            models = [models]

        model_df = df[df["model"].isin(models)]

        # If the subset contains no rows for these models, return empty metrics
        # (this can happen during bootstrap resampling; callers should not skip the draw).
        if model_df.empty:
            metric_keys = self._infer_metric_keys(df)
            return self.empty_metrics(metric_keys)

        cluster_model_df = model_df[model_df["cluster"].isin(clusters)]

        metrics = self._infer_metric_keys(model_df)

        if len(cluster_model_df) == 0:
            return self.empty_metrics(metrics)

        # Get number of unique conversations for those models across all clusters
        # Pass metrics to ensure both computations use the same metric set
        model_size, model_scores = self.compute_size_and_score(model_df, metrics=metrics)
        cluster_model_size, cluster_model_scores = self.compute_size_and_score(cluster_model_df, metrics=metrics)

        # Extract cluster metadata (take the first non-empty metadata from the cluster)
        cluster_metadata = {}
        if include_metadata:
            if "cluster_metadata" in cluster_model_df.columns:
                non_empty_metadata = cluster_model_df["cluster_metadata"].dropna()
                if not non_empty_metadata.empty:
                    cluster_metadata = non_empty_metadata.iloc[0]

        quality_raw_delta = self.compute_relative_quality(cluster_model_scores, model_scores)
        proportion = cluster_model_size / model_size if model_size != 0 else 0

        # Quality delta is just the raw difference in scores (no proportion weighting)
        quality_delta = quality_raw_delta

        return {
            "size": cluster_model_size,
            "proportion": proportion,
            "quality": cluster_model_scores,
            "quality_delta": quality_delta,
            "metadata": cluster_metadata if include_metadata else {},
            "examples": list(zip(
                cluster_model_df["conversation_id"],
                cluster_model_df["conversation_metadata"],
                cluster_model_df["property_metadata"]
            )),
        }

    def _infer_metric_keys(self, df: pd.DataFrame) -> List[str]:
        """Infer score metric keys from a dataframe.

        Expected input:
            - `df` contains a `scores` column whose entries are dicts mapping metric names (str)
              to numeric values (int/float). Missing metrics are permitted.

        Returns:
            - List of metric keys (strings). Order is deterministic (sorted).
        """
        if df is None or df.empty or "scores" not in df.columns:
            return []
        keys = set()
        for scores in df["scores"]:
            if isinstance(scores, dict):
                keys.update(scores.keys())
        return sorted(keys)

    def _bootstrap_salience_from_proportions(
        self, proportions: "Any", *, n_models: int
    ) -> "Any":
        """Compute per-model proportion deltas (salience) from proportions.

        This is the bootstrap analogue of `_compute_salience()`, but operates on a dense
        array instead of nested dicts.

        Args:
            proportions: Array of shape (n_models, n_clusters) with proportions in [0, 1].
            n_models: Number of models (first dimension).

        Returns:
            Array of shape (n_models, n_clusters) with `proportion_delta`.
        """
        import numpy as np

        if n_models <= 1:
            return np.zeros_like(proportions, dtype=float)

        totals = np.sum(proportions, axis=0, keepdims=True)
        avg_others = (totals - proportions) / float(n_models - 1)
        return proportions - avg_others

    @staticmethod
    def _compute_weighted_means_1d(
        *,
        group_idx: "Any",
        n_groups: int,
        weights: "Any",
        values: "Any",
    ) -> "Any":
        """Compute weighted means per group with NaN-safe handling.

        Args:
            group_idx: int array of shape (n_rows,) mapping each row to a group in [0, n_groups).
            n_groups: number of groups.
            weights: float array of shape (n_rows,) (non-negative weights).
            values: float array of shape (n_rows, n_metrics) with NaN for missing values.

        Returns:
            means: float array of shape (n_groups, n_metrics). If a group has no valid entries
                   for a metric (denominator 0), the mean is 0.0 (matching `empty_metrics` behavior).
        """
        import numpy as np

        n_rows, n_metrics = values.shape
        if n_rows == 0:
            return np.zeros((n_groups, n_metrics), dtype=float)

        means = np.zeros((n_groups, n_metrics), dtype=float)
        for j in range(n_metrics):
            col = values[:, j]
            valid = ~np.isnan(col)
            if not np.any(valid):
                continue
            num = np.bincount(
                group_idx[valid],
                weights=(weights[valid] * col[valid]),
                minlength=n_groups,
            )
            den = np.bincount(group_idx[valid], weights=weights[valid], minlength=n_groups)
            means[:, j] = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
        return means

    @staticmethod
    def _compute_weighted_means_2d(
        *,
        row_idx: "Any",
        col_idx: "Any",
        n_rows: int,
        n_cols: int,
        weights: "Any",
        values: "Any",
    ) -> "Any":
        """Compute weighted means for a 2D grouping (row_idx, col_idx) with NaN-safe handling."""
        import numpy as np

        if len(values) == 0:
            return np.zeros((n_rows, n_cols, values.shape[1]), dtype=float)

        flat = row_idx * n_cols + col_idx
        n_groups = n_rows * n_cols
        means_flat = FunctionalMetrics._compute_weighted_means_1d(
            group_idx=flat,
            n_groups=n_groups,
            weights=weights,
            values=values,
        )
        return means_flat.reshape((n_rows, n_cols, values.shape[1]))

    def _compute_salience(self, model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute salience (proportion deviation from average of OTHER models) for each model-cluster combination."""
        df = pd.DataFrame(model_cluster_scores).reset_index().rename({"index": "cluster"}, axis=1)
        
        # Step 1: Extract proportion values
        model_names = [col for col in df.columns if col not in ['cluster']]

        # Parse the proportion field from the dictionary-like data
        for model in model_names:
            df[f'{model}_proportion'] = df[model].apply(lambda x: x.get('proportion', 0) if isinstance(x, dict) else 0)

        # Step 2 & 3: Compute deviation from average of OTHER models (excluding self)
        for model in model_names:
            # Get all other models' proportion columns
            other_model_cols = [f'{m}_proportion' for m in model_names if m != model]
            if other_model_cols:
                # Average proportion across all OTHER models
                df[f'{model}_avg_others'] = df[other_model_cols].mean(axis=1)
            else:
                # If only one model, deviation is 0
                df[f'{model}_avg_others'] = 0
            # Deviation = this model's proportion - average of others
            df[f'{model}_deviation'] = df[f'{model}_proportion'] - df[f'{model}_avg_others']

        # Step 4: Add deviation into model_cluster_scores
        for i, row in df.iterrows():
            cluster = row['cluster']
            for model in model_names:
                deviation_value = row[f'{model}_deviation']
                if model in model_cluster_scores and cluster in model_cluster_scores[model]:
                    model_cluster_scores[model][cluster]['proportion_delta'] = deviation_value

        return model_cluster_scores

    def _compute_model_cluster_scores(self, df: pd.DataFrame, cluster_names: List[str], model_names: List[str], *, include_metadata: bool = True) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute metrics for each model-cluster combination."""
        model_cluster_scores = {}
        for model in model_names:
            model_cluster_scores[model] = {
                cluster: self.compute_cluster_metrics(df, cluster, [model], include_metadata=include_metadata)
                for cluster in cluster_names
            }
        return model_cluster_scores

    def _compute_cluster_scores(self, df: pd.DataFrame, cluster_names: List[str], model_names: List[str], *, include_metadata: bool = True) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for each cluster across all models."""
        return {
            cluster: self.compute_cluster_metrics(df, cluster, list(model_names), include_metadata=include_metadata)
            for cluster in cluster_names
        }

    def _compute_model_scores(self, df: pd.DataFrame, cluster_names: List[str], model_names: List[str], *, include_metadata: bool = True) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for each model across all clusters."""
        return {
            model: self.compute_cluster_metrics(df, list(cluster_names), [model], include_metadata=include_metadata)
            for model in model_names
        }

    def _bootstrap_scores_to_matrix(
        self,
        *,
        scores_series: pd.Series,
        metric_to_idx: Dict[str, int],
        n_metrics: int,
    ) -> "Any":
        """Convert a `scores` series into a dense matrix.

        Expected input format:
            - `scores_series` entries are `dict[str, number]` or non-dict / empty.
            - `metric_to_idx` maps metric keys to column indices.

        Output:
            - `np.ndarray` of shape (n_rows, n_metrics) with float values.
            - Missing metrics are encoded as NaN (so denominators ignore them).
        """
        import numpy as np

        mat = np.full((len(scores_series), n_metrics), np.nan, dtype=float)
        for i, s in enumerate(scores_series):
            if not isinstance(s, dict):
                continue
            for k, v in s.items():
                j = metric_to_idx.get(k)
                if j is None:
                    continue
                if isinstance(v, (int, float)):
                    mat[i, j] = float(v)
        return mat

    def _bootstrap_prepare(
        self,
        *,
        df: pd.DataFrame,
        cluster_names: List[str],
        model_names: List[str],
    ) -> Dict[str, Any]:
        """Prepare stable indices and dense score matrices for fast bootstrap.

        Returns a dict with:
            - model_names, cluster_names, metric_keys, and mapping dicts
            - conversation index and per-row conversation/model/cluster indices
            - deduplicated frames and dense score matrices
        """
        import numpy as np

        model_names = list(model_names)
        cluster_names = list(cluster_names)
        n_models = len(model_names)
        n_clusters = len(cluster_names)

        metric_keys = self._infer_metric_keys(df)
        metric_to_idx = {k: i for i, k in enumerate(metric_keys)}
        model_to_idx = {m: i for i, m in enumerate(model_names)}
        cluster_to_idx = {c: i for i, c in enumerate(cluster_names)}

        # De-duplicate at the same levels the metric computations implicitly use.
        # - Denominators and model/global scores: unique per (conversation_id, model)
        # - Cluster numerators: unique per (conversation_id, model, cluster)
        df_cm = df.drop_duplicates(subset=["conversation_id", "model"]).copy()
        df_cmc = df.drop_duplicates(subset=["conversation_id", "model", "cluster"]).copy()

        conv_index = pd.Index(df["conversation_id"].unique())
        n_conv = len(conv_index)

        # Precompute row -> conversation/model/cluster indices
        cm_conv_idx = conv_index.get_indexer(df_cm["conversation_id"])
        cm_model_idx = np.array([model_to_idx.get(m, -1) for m in df_cm["model"]], dtype=int)

        cmc_conv_idx = conv_index.get_indexer(df_cmc["conversation_id"])
        cmc_model_idx = np.array([model_to_idx.get(m, -1) for m in df_cmc["model"]], dtype=int)
        cmc_cluster_idx = np.array([cluster_to_idx.get(c, -1) for c in df_cmc["cluster"]], dtype=int)

        # Filter unknown model/cluster rows (should not happen, but keep arrays aligned)
        cm_keep = cm_model_idx >= 0
        df_cm = df_cm.loc[cm_keep].reset_index(drop=True)
        cm_conv_idx = cm_conv_idx[cm_keep]
        cm_model_idx = cm_model_idx[cm_keep]

        cmc_keep = (cmc_model_idx >= 0) & (cmc_cluster_idx >= 0)
        df_cmc = df_cmc.loc[cmc_keep].reset_index(drop=True)
        cmc_conv_idx = cmc_conv_idx[cmc_keep]
        cmc_model_idx = cmc_model_idx[cmc_keep]
        cmc_cluster_idx = cmc_cluster_idx[cmc_keep]

        n_metrics = len(metric_keys)
        cm_scores = self._bootstrap_scores_to_matrix(
            scores_series=df_cm["scores"],
            metric_to_idx=metric_to_idx,
            n_metrics=n_metrics,
        )
        cmc_scores = self._bootstrap_scores_to_matrix(
            scores_series=df_cmc["scores"],
            metric_to_idx=metric_to_idx,
            n_metrics=n_metrics,
        )

        return {
            "model_names": model_names,
            "cluster_names": cluster_names,
            "metric_keys": metric_keys,
            "metric_to_idx": metric_to_idx,
            "model_to_idx": model_to_idx,
            "cluster_to_idx": cluster_to_idx,
            "n_models": n_models,
            "n_clusters": n_clusters,
            "n_metrics": n_metrics,
            "conv_index": conv_index,
            "n_conv": n_conv,
            "cm_conv_idx": cm_conv_idx,
            "cm_model_idx": cm_model_idx,
            "cmc_conv_idx": cmc_conv_idx,
            "cmc_model_idx": cmc_model_idx,
            "cmc_cluster_idx": cmc_cluster_idx,
            "cm_scores": cm_scores,
            "cmc_scores": cmc_scores,
        }

    @staticmethod
    def _bootstrap_allocate_arrays(*, S: int, n_models: int, n_clusters: int, n_metrics: int) -> Dict[str, "Any"]:
        """Allocate bootstrap result arrays."""
        import numpy as np

        return {
            "mc_prop": np.zeros((S, n_models, n_clusters), dtype=float),
            "mc_prop_delta": np.zeros((S, n_models, n_clusters), dtype=float),
            "mc_quality": np.zeros((S, n_models, n_clusters, n_metrics), dtype=float),
            "mc_quality_delta": np.zeros((S, n_models, n_clusters, n_metrics), dtype=float),
            "c_prop": np.zeros((S, n_clusters), dtype=float),
            "c_quality": np.zeros((S, n_clusters, n_metrics), dtype=float),
            "c_quality_delta": np.zeros((S, n_clusters, n_metrics), dtype=float),
            "m_prop": np.zeros((S, n_models), dtype=float),
            "m_quality": np.zeros((S, n_models, n_metrics), dtype=float),
            "m_quality_delta": np.zeros((S, n_models, n_metrics), dtype=float),
        }

    def _bootstrap_compute_one_replicate(
        self,
        *,
        prep: Dict[str, Any],
        arrays: Dict[str, Any],
        i: int,
        conv_weights: "Any",
    ) -> None:
        """Compute all bootstrap metrics for a single replicate and store into `arrays`."""
        import numpy as np

        n_models = prep["n_models"]
        n_clusters = prep["n_clusters"]
        n_metrics = prep["n_metrics"]

        # Row weights (by conversation)
        w_cm = conv_weights[prep["cm_conv_idx"]]
        w_cmc = conv_weights[prep["cmc_conv_idx"]]

        cm_model_idx = prep["cm_model_idx"]
        cmc_model_idx = prep["cmc_model_idx"]
        cmc_cluster_idx = prep["cmc_cluster_idx"]

        # ---- Denominators: by model, and global ----
        model_sizes = np.bincount(cm_model_idx, weights=w_cm, minlength=n_models)
        global_size = float(np.sum(w_cm))

        # Weighted mean scores per model and global
        model_means = self._compute_weighted_means_1d(
            group_idx=cm_model_idx,
            n_groups=n_models,
            weights=w_cm,
            values=prep["cm_scores"],
        )
        global_means = self._compute_weighted_means_1d(
            group_idx=np.zeros(len(prep["cm_scores"]), dtype=int),
            n_groups=1,
            weights=w_cm,
            values=prep["cm_scores"],
        )[0]

        # ---- Cluster-level numerators (across all models) ----
        cluster_sizes = np.bincount(cmc_cluster_idx, weights=w_cmc, minlength=n_clusters)
        cluster_means = self._compute_weighted_means_1d(
            group_idx=cmc_cluster_idx,
            n_groups=n_clusters,
            weights=w_cmc,
            values=prep["cmc_scores"],
        )

        # ---- Model-cluster numerators ----
        flat_mc = cmc_model_idx * n_clusters + cmc_cluster_idx
        mc_sizes_flat = np.bincount(flat_mc, weights=w_cmc, minlength=n_models * n_clusters)
        mc_sizes = mc_sizes_flat.reshape((n_models, n_clusters))
        mc_means = self._compute_weighted_means_2d(
            row_idx=cmc_model_idx,
            col_idx=cmc_cluster_idx,
            n_rows=n_models,
            n_cols=n_clusters,
            weights=w_cmc,
            values=prep["cmc_scores"],
        )

        # ---- Proportions ----
        with np.errstate(divide="ignore", invalid="ignore"):
            proportions = np.divide(
                mc_sizes,
                model_sizes.reshape((n_models, 1)),
                out=np.zeros_like(mc_sizes, dtype=float),
                where=model_sizes.reshape((n_models, 1)) > 0,
            )
        prop_delta = self._bootstrap_salience_from_proportions(proportions, n_models=n_models)

        arrays["mc_prop"][i] = proportions
        arrays["mc_prop_delta"][i] = prop_delta
        arrays["c_prop"][i] = (cluster_sizes / global_size) if global_size > 0 else np.zeros(n_clusters, dtype=float)
        arrays["m_prop"][i] = np.where(model_sizes > 0, 1.0, 0.0)

        # ---- Quality + deltas ----
        arrays["mc_quality"][i] = mc_means
        arrays["c_quality"][i] = cluster_means
        arrays["m_quality"][i] = model_means

        baseline_model = model_means[:, None, :]  # (n_models, 1, n_metrics)
        arrays["mc_quality_delta"][i] = np.where(baseline_model != 0.0, mc_means - baseline_model, 0.0)
        arrays["c_quality_delta"][i] = np.where(global_means[None, :] != 0.0, cluster_means - global_means[None, :], 0.0)

        # model_scores in current implementation are cluster==all clusters, so delta is 0 vs itself
        arrays["m_quality_delta"][i] = np.zeros((n_models, n_metrics), dtype=float)

    def _bootstrap_attach_results(
        self,
        *,
        prep: Dict[str, Any],
        arrays: Dict[str, Any],
        model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]],
        cluster_scores: Dict[str, Dict[str, Any]],
        model_scores: Dict[str, Dict[str, Any]],
    ) -> None:
        """Attach CI dicts + significance flags and replace point-estimates with bootstrap means."""
        import numpy as np

        metric_keys = prep["metric_keys"]
        model_names = prep["model_names"]
        cluster_names = prep["cluster_names"]

        def _ci_dict(arr: "Any") -> Dict[str, float]:
            return {
                "lower": float(np.percentile(arr, 2.5)),
                "upper": float(np.percentile(arr, 97.5)),
                "mean": float(np.mean(arr)),
            }

        # Model-cluster
        for mi, model in enumerate(model_names):
            for ci, cluster in enumerate(cluster_names):
                mc = model_cluster_scores[model][cluster]

                ci_prop = _ci_dict(arrays["mc_prop"][:, mi, ci])
                mc["proportion_ci"] = ci_prop
                mc["proportion"] = ci_prop["mean"]

                ci_pd = _ci_dict(arrays["mc_prop_delta"][:, mi, ci])
                mc["proportion_delta_ci"] = ci_pd
                mc["proportion_delta"] = ci_pd["mean"]
                mc["proportion_delta_significant"] = self._is_significant(ci_pd["lower"], ci_pd["upper"], 0)

                q_ci: Dict[str, Dict[str, float]] = {}
                qd_ci: Dict[str, Dict[str, float]] = {}
                qd_sig: Dict[str, bool] = {}
                for mj, metric in enumerate(metric_keys):
                    ci_q = _ci_dict(arrays["mc_quality"][:, mi, ci, mj])
                    q_ci[metric] = ci_q
                    mc["quality"][metric] = ci_q["mean"]

                    ci_qd = _ci_dict(arrays["mc_quality_delta"][:, mi, ci, mj])
                    qd_ci[metric] = ci_qd
                    mc["quality_delta"][metric] = ci_qd["mean"]
                    qd_sig[metric] = self._is_significant(ci_qd["lower"], ci_qd["upper"], 0)

                if q_ci:
                    mc["quality_ci"] = q_ci
                if qd_ci:
                    mc["quality_delta_ci"] = qd_ci
                mc["quality_delta_significant"] = qd_sig

        # Cluster scores
        for ci, cluster in enumerate(cluster_names):
            cs = cluster_scores[cluster]

            ci_prop = _ci_dict(arrays["c_prop"][:, ci])
            cs["proportion_ci"] = ci_prop
            cs["proportion"] = ci_prop["mean"]

            q_ci: Dict[str, Dict[str, float]] = {}
            qd_ci: Dict[str, Dict[str, float]] = {}
            qd_sig: Dict[str, bool] = {}
            for mj, metric in enumerate(metric_keys):
                ci_q = _ci_dict(arrays["c_quality"][:, ci, mj])
                q_ci[metric] = ci_q
                cs["quality"][metric] = ci_q["mean"]

                ci_qd = _ci_dict(arrays["c_quality_delta"][:, ci, mj])
                qd_ci[metric] = ci_qd
                cs["quality_delta"][metric] = ci_qd["mean"]
                qd_sig[metric] = self._is_significant(ci_qd["lower"], ci_qd["upper"], 0)

            if q_ci:
                cs["quality_ci"] = q_ci
            if qd_ci:
                cs["quality_delta_ci"] = qd_ci
            cs["quality_delta_significant"] = qd_sig

        # Model scores
        for mi, model in enumerate(model_names):
            ms = model_scores[model]

            ci_prop = _ci_dict(arrays["m_prop"][:, mi])
            ms["proportion_ci"] = ci_prop
            ms["proportion"] = ci_prop["mean"]

            q_ci: Dict[str, Dict[str, float]] = {}
            qd_ci: Dict[str, Dict[str, float]] = {}
            qd_sig: Dict[str, bool] = {}
            for mj, metric in enumerate(metric_keys):
                ci_q = _ci_dict(arrays["m_quality"][:, mi, mj])
                q_ci[metric] = ci_q
                ms["quality"][metric] = ci_q["mean"]

                ci_qd = _ci_dict(arrays["m_quality_delta"][:, mi, mj])
                qd_ci[metric] = ci_qd
                ms["quality_delta"][metric] = ci_qd["mean"]
                qd_sig[metric] = self._is_significant(ci_qd["lower"], ci_qd["upper"], 0)

            if q_ci:
                ms["quality_ci"] = q_ci
            if qd_ci:
                ms["quality_delta_ci"] = qd_ci
            ms["quality_delta_significant"] = qd_sig

    def _add_bootstrap_analysis(
        self,
        df: pd.DataFrame,
        model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]],
        cluster_scores: Dict[str, Dict[str, Any]],
        model_scores: Dict[str, Dict[str, Any]],
        *,
        cluster_names: List[str],
        model_names: List[str],
        progress_callback=None,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Add bootstrap confidence intervals and significance testing.

        This implementation is a **true with-replacement bootstrap** over `conversation_id`s.
        To make it fast, it uses per-conversation **weights** (draw counts) instead of
        materializing a duplicated DataFrame for each replicate.

        Inputs:
            df:
                Long dataframe with columns:
                - conversation_id: hashable conversation id
                - model: model name (str)
                - cluster: cluster label (str)
                - scores: dict[str, number] of metric values (may be missing keys)
            cluster_names, model_names:
                The cluster/model order to use for bootstrap arrays and output attachment.

        Behavior:
            - Always uses exactly `bootstrap_samples` replicates (no skipping).
            - Empty subsets (e.g., a model gets 0 draws) yield empty metrics (zeros), matching
              `empty_metrics()` behavior.
            - Point estimates are set to bootstrap means (same behavior as prior implementation).
        """
        import numpy as np

        self.log(f"Computing bootstrap confidence intervals with {self.bootstrap_samples} samples...")

        # ---- Setup deterministic RNG (optional) ----
        rng = np.random.default_rng(self.bootstrap_seed)

        prep = self._bootstrap_prepare(df=df, cluster_names=cluster_names, model_names=model_names)
        if prep["n_conv"] == 0:
            return model_cluster_scores, cluster_scores, model_scores

        S = int(self.bootstrap_samples)
        arrays = self._bootstrap_allocate_arrays(
            S=S, n_models=prep["n_models"], n_clusters=prep["n_clusters"], n_metrics=prep["n_metrics"]
        )

        # Bootstrap sampling distribution over conversation ids (uniform)
        p = np.full(prep["n_conv"], 1.0 / float(prep["n_conv"]), dtype=float)

        for i in range(S):
            if i % 20 == 0:
                self.log(f"Bootstrap progress: {i}/{S} ({i/S*100:.1f}%)")
            if progress_callback and i % 5 == 0:
                try:
                    progress_callback(i / S)
                except Exception:
                    pass

            # True with-replacement bootstrap counts for each conversation_id
            conv_weights = rng.multinomial(prep["n_conv"], p).astype(float)
            self._bootstrap_compute_one_replicate(prep=prep, arrays=arrays, i=i, conv_weights=conv_weights)

        self._bootstrap_attach_results(
            prep=prep,
            arrays=arrays,
            model_cluster_scores=model_cluster_scores,
            cluster_scores=cluster_scores,
            model_scores=model_scores,
        )

        self.log(f"✅ Bootstrap analysis completed with {S} samples")
        return model_cluster_scores, cluster_scores, model_scores
    
    def _resample_conversations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample conversations with replacement for bootstrap (materializing duplicates).

        Notes:
            This is a **true with-replacement** resample over `conversation_id`. Duplicate draws
            are preserved by concatenating conversation slices multiple times.

            The main implementation uses a faster weight-based approach; this helper remains
            useful for debugging and small equivalence checks.
        """
        import numpy as np
        
        # Get unique conversation IDs
        unique_conversations = df["conversation_id"].unique()
        
        # Sample with replacement
        sample_conversations = np.random.choice(
            unique_conversations, 
            size=len(unique_conversations), 
            replace=True
        )
        
        # Preserve multiplicity: each drawn conversation appears as many times as it was drawn.
        counts: Dict[Any, int] = {}
        for cid in sample_conversations:
            counts[cid] = counts.get(cid, 0) + 1

        parts = []
        for cid, k in counts.items():
            if k <= 0:
                continue
            block = df[df["conversation_id"] == cid]
            if block.empty:
                continue
            parts.extend([block] * k)

        if not parts:
            return df.iloc[0:0].copy()

        return pd.concat(parts, ignore_index=True).copy()
    
    def _compute_ci(self, values, lower_percentile=2.5, upper_percentile=97.5):
        """Compute confidence interval for a list of values."""
        import numpy as np
        
        if not values:
            return None
            
        return {
            'lower': float(np.percentile(values, lower_percentile)),
            'upper': float(np.percentile(values, upper_percentile)),
            'mean': float(np.mean(values))
        }
    
    def _is_significant(self, lower, upper, contains=0):
        """Check for significant difference. 
        If the interval range contains 0, the difference is not significant.
        If the interval range does not contain 0, the difference is significant.
        """
        return not (lower <= contains <= upper)

    def _add_confidence_intervals(self, model_cluster_scores, cluster_scores, model_scores, bootstrap_samples):
        """Add bootstrap confidence intervals to all score dictionaries."""
        
        # Add CIs to model_cluster_scores
        for model in model_cluster_scores:
            for cluster in model_cluster_scores[model]:
                proportions = []
                proportion_deltas = []  # NEW
                quality_scores = {key: [] for key in model_cluster_scores[model][cluster].get('quality', {})}
                quality_deltas = {key: [] for key in model_cluster_scores[model][cluster].get('quality_delta', {})}  # NEW
                
                for sample in bootstrap_samples:
                    if model in sample['model_cluster'] and cluster in sample['model_cluster'][model]:
                        sample_metrics = sample['model_cluster'][model][cluster]
                        proportions.append(sample_metrics.get('proportion', 0))
                        proportion_deltas.append(sample_metrics.get('proportion_delta', 0))  # NEW
                        
                        for key in quality_scores:
                            if key in sample_metrics.get('quality', {}):
                                quality_scores[key].append(sample_metrics['quality'][key])
                        
                        for key in quality_deltas:  # NEW
                            if key in sample_metrics.get('quality_delta', {}):
                                quality_deltas[key].append(sample_metrics['quality_delta'][key])
                
                # Add proportion CI
                proportion_ci = self._compute_ci(proportions)
                if proportion_ci:
                    model_cluster_scores[model][cluster]['proportion_ci'] = proportion_ci
                    # Use bootstrap mean as the point estimate
                    model_cluster_scores[model][cluster]['proportion'] = proportion_ci['mean']
                
                # Add proportion_delta CI (salience)  # NEW
                proportion_delta_ci = self._compute_ci(proportion_deltas)
                if proportion_delta_ci:
                    model_cluster_scores[model][cluster]['proportion_delta_ci'] = proportion_delta_ci
                    # Use bootstrap mean as the point estimate
                    model_cluster_scores[model][cluster]['proportion_delta'] = proportion_delta_ci['mean']
                    # Add significance testing for proportion_delta
                    model_cluster_scores[model][cluster]['proportion_delta_significant'] = self._is_significant(
                        proportion_delta_ci['lower'], proportion_delta_ci['upper'], 0
                    )
                else:
                    # If no CI could be computed (0 samples), significance is False
                    model_cluster_scores[model][cluster]['proportion_delta_significant'] = False
                
                # Add quality score CIs
                quality_ci = {}
                for key, values in quality_scores.items():
                    ci = self._compute_ci(values)
                    if ci:
                        quality_ci[key] = ci
                        # Use bootstrap mean as the point estimate
                        model_cluster_scores[model][cluster]['quality'][key] = ci['mean']
                
                if quality_ci:
                    model_cluster_scores[model][cluster]['quality_ci'] = quality_ci
                
                # Add quality_delta CIs  # NEW
                quality_delta_ci = {}
                quality_delta_significant = {}
                for key, values in quality_deltas.items():
                    ci = self._compute_ci(values)
                    if ci:
                        quality_delta_ci[key] = ci
                        # Use bootstrap mean as the point estimate
                        model_cluster_scores[model][cluster]['quality_delta'][key] = ci['mean']
                        # Add significance testing for quality_delta
                        quality_delta_significant[key] = self._is_significant(ci['lower'], ci['upper'], 0)
                    else:
                        # If no CI could be computed (0 samples), significance is False
                        quality_delta_significant[key] = False
                
                if quality_delta_ci:
                    model_cluster_scores[model][cluster]['quality_delta_ci'] = quality_delta_ci
                # Always add significance, even if some CIs couldn't be computed
                model_cluster_scores[model][cluster]['quality_delta_significant'] = quality_delta_significant
        
        # Add CIs to cluster_scores (across all models)
        for cluster in cluster_scores:
            proportions = []
            quality_scores = {key: [] for key in cluster_scores[cluster].get('quality', {})}
            quality_deltas = {key: [] for key in cluster_scores[cluster].get('quality_delta', {})}  # NEW
            
            for sample in bootstrap_samples:
                if cluster in sample['cluster']:
                    sample_metrics = sample['cluster'][cluster]
                    proportions.append(sample_metrics.get('proportion', 0))
                    
                    for key in quality_scores:
                        if key in sample_metrics.get('quality', {}):
                            quality_scores[key].append(sample_metrics['quality'][key])
                    
                    for key in quality_deltas:  # NEW
                        if key in sample_metrics.get('quality_delta', {}):
                            quality_deltas[key].append(sample_metrics['quality_delta'][key])
            
            # Add proportion CI
            proportion_ci = self._compute_ci(proportions)
            if proportion_ci:
                cluster_scores[cluster]['proportion_ci'] = proportion_ci
                # Use bootstrap mean as the point estimate
                cluster_scores[cluster]['proportion'] = proportion_ci['mean']
            
            # Add quality score CIs
            quality_ci = {}
            for key, values in quality_scores.items():
                ci = self._compute_ci(values)
                if ci:
                    quality_ci[key] = ci
                    # Use bootstrap mean as the point estimate
                    cluster_scores[cluster]['quality'][key] = ci['mean']
            
            if quality_ci:
                cluster_scores[cluster]['quality_ci'] = quality_ci
            
            # Add quality_delta CIs  # NEW
            quality_delta_ci = {}
            quality_delta_significant = {}
            for key, values in quality_deltas.items():
                ci = self._compute_ci(values)
                if ci:
                    quality_delta_ci[key] = ci
                    # Use bootstrap mean as the point estimate
                    cluster_scores[cluster]['quality_delta'][key] = ci['mean']
                    # Add significance testing for quality_delta
                    quality_delta_significant[key] = self._is_significant(ci['lower'], ci['upper'], 0)
                else:
                    # If no CI could be computed (0 samples), significance is False
                    quality_delta_significant[key] = False
            
            if quality_delta_ci:
                cluster_scores[cluster]['quality_delta_ci'] = quality_delta_ci
            # Always add significance, even if some CIs couldn't be computed
            cluster_scores[cluster]['quality_delta_significant'] = quality_delta_significant
        
        # Add CIs to model_scores (across all clusters)  
        for model in model_scores:
            proportions = []
            quality_scores = {key: [] for key in model_scores[model].get('quality', {})}
            quality_deltas = {key: [] for key in model_scores[model].get('quality_delta', {})}  # NEW
            
            for sample in bootstrap_samples:
                if model in sample['model']:
                    sample_metrics = sample['model'][model]
                    proportions.append(sample_metrics.get('proportion', 0))
                    
                    for key in quality_scores:
                        if key in sample_metrics.get('quality', {}):
                            quality_scores[key].append(sample_metrics['quality'][key])
                    
                    for key in quality_deltas:  # NEW
                        if key in sample_metrics.get('quality_delta', {}):
                            quality_deltas[key].append(sample_metrics['quality_delta'][key])
            
            # Add CIs
            proportion_ci = self._compute_ci(proportions)
            if proportion_ci:
                model_scores[model]['proportion_ci'] = proportion_ci
                # Use bootstrap mean as the point estimate
                model_scores[model]['proportion'] = proportion_ci['mean']
            
            # Add quality score CIs
            quality_ci = {}
            for key, values in quality_scores.items():
                ci = self._compute_ci(values)
                if ci:
                    quality_ci[key] = ci
                    # Use bootstrap mean as the point estimate
                    model_scores[model]['quality'][key] = ci['mean']
            
            if quality_ci:
                model_scores[model]['quality_ci'] = quality_ci
            
            # Add quality_delta CIs  # NEW
            quality_delta_ci = {}
            quality_delta_significant = {}
            for key, values in quality_deltas.items():
                ci = self._compute_ci(values)
                if ci:
                    quality_delta_ci[key] = ci
                    # Use bootstrap mean as the point estimate
                    model_scores[model]['quality_delta'][key] = ci['mean']
                    # Add significance testing for quality_delta
                    quality_delta_significant[key] = self._is_significant(ci['lower'], ci['upper'], 0)
                else:
                    # If no CI could be computed (0 samples), significance is False
                    quality_delta_significant[key] = False
            
            if quality_delta_ci:
                model_scores[model]['quality_delta_ci'] = quality_delta_ci
            # Always add significance, even if some CIs couldn't be computed
            model_scores[model]['quality_delta_significant'] = quality_delta_significant 

    def process_wandb_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe for wandb logging by handling NA values and converting complex types to strings."""
        import json
        
        df = df.copy()
        
        # Fill NA values with "None" string
        df = df.fillna("None")

        # only include up to 5 examples
        if "examples" in df.columns:
            cut_examples = df["examples"].apply(lambda x: x[:5] if isinstance(x, list) else x)
            df["examples"] = cut_examples
        
        # Convert dict and list columns to pretty-printed strings
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
        return df
    
    def _save_dataframe_files(self, model_cluster_scores, cluster_scores, model_scores):
        """Save dataframe versions as JSONL files for easier data analysis.
        
        Uses the data_transformers module for consistent, well-tested transformations.
        
        Returns:
            Tuple of (model_cluster_df, cluster_df, model_df) for use in model_stats
        """
        from .data_transformers import (
            flatten_model_cluster_scores,
            flatten_cluster_scores, 
            flatten_model_scores,
            save_flattened_jsonl
        )
        
        # Transform using the utility functions
        model_cluster_df = flatten_model_cluster_scores(model_cluster_scores)
        cluster_df = flatten_cluster_scores(cluster_scores)
        model_df = flatten_model_scores(model_scores)
        
        # Save all three JSONL files
        model_cluster_path = self.output_dir / "model_cluster_scores_df.jsonl"
        save_flattened_jsonl(model_cluster_df, model_cluster_path)
        self.log(f"📄 Saved model-cluster dataframe to {model_cluster_path} ({len(model_cluster_df)} rows)")
        
        cluster_path = self.output_dir / "cluster_scores_df.jsonl"
        save_flattened_jsonl(cluster_df, cluster_path)
        self.log(f"📄 Saved cluster dataframe to {cluster_path} ({len(cluster_df)} rows)")
        
        model_path = self.output_dir / "model_scores_df.jsonl"
        save_flattened_jsonl(model_df, model_path)
        self.log(f"📄 Saved model scores dataframe to {model_path} ({len(model_df)} rows)")
        
        return model_cluster_df, cluster_df, model_df

    def _log_to_wandb(self, model_cluster_scores, cluster_scores, model_scores):
        """Log the three score dataframes to wandb as tables."""
        if importlib.util.find_spec("wandb") is None:
            raise ModuleNotFoundError(
                "wandb is not installed, but log_to_wandb=True. "
                "Install it with: pip install 'stringsight[wandb]' (or: pip install wandb), "
                "or set log_to_wandb=False."
            )
        import wandb
        self.log("📊 Logging metrics to wandb...")
        
        # Create dataframes for wandb (reusing the logic from _save_dataframe_files)
        df = pd.DataFrame(model_cluster_scores).T
        tidy_rows = []
        for model, row in df.iterrows():
            for property_name, metrics in row.items():
                if isinstance(metrics, dict):
                    tidy_row = {"model": model, "property": property_name}
                    tidy_row.update(metrics)
                    tidy_rows.append(tidy_row)
        
        model_cluster_df = pd.DataFrame(tidy_rows)
        # Ensure model and property are first two columns
        cols = ['model', 'property'] + [col for col in model_cluster_df.columns if col not in ['model', 'property']]
        model_cluster_df = model_cluster_df[cols]
        model_cluster_df = self.process_wandb_dataframe(model_cluster_df)
        
        # Create cluster_df
        cluster_df = pd.DataFrame(cluster_scores).T
        cluster_df["property"] = cluster_df.index
        cluster_df["model"] = "all"
        # Ensure model and property are first two columns
        cols = ['model', 'property'] + [col for col in cluster_df.columns if col not in ['model', 'property']]
        cluster_df = cluster_df[cols]
        cluster_df = self.process_wandb_dataframe(cluster_df)
        
        # Create model_scores_df
        model_scores_df = pd.DataFrame(model_scores).T
        model_scores_df["model"] = model_scores_df.index
        model_scores_df["property"] = "all_clusters"
        # Ensure model and property are first two columns
        cols = ['model', 'property'] + [col for col in model_scores_df.columns if col not in ['model', 'property']]
        model_scores_df = model_scores_df[cols]
        model_scores_df = self.process_wandb_dataframe(model_scores_df)
        
        # Log to wandb
        wandb.log({
            "Metrics/model_cluster_scores": wandb.Table(dataframe=model_cluster_df),
            "Metrics/cluster_scores": wandb.Table(dataframe=cluster_df),
            "Metrics/model_scores": wandb.Table(dataframe=model_scores_df)
        })
        
        self.log(f"✅ Successfully logged {len(model_cluster_df)} model-cluster, {len(cluster_df)} cluster, and {len(model_scores_df)} model records to wandb")

    def _save_results(self, model_cluster_scores, cluster_scores, model_scores):
        """Save the three result files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure JSON-serializable structures (replace Ellipsis and unknown types)
        def _json_safe(obj):
            from numpy import ndarray, generic
            if obj is Ellipsis:
                return None
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, ndarray):
                return obj.tolist()
            if isinstance(obj, generic):
                return obj.item()
            if isinstance(obj, (list, tuple, set)):
                return [_json_safe(x) for x in obj]
            if isinstance(obj, dict):
                safe = {}
                for k, v in obj.items():
                    # Convert non-JSON-safe keys to strings
                    if isinstance(k, (str, int, float, bool)) or k is None:
                        sk = k
                    else:
                        sk = str(k)
                    safe[sk] = _json_safe(v)
                return safe
            # Fallback: stringify unknown types
            return str(obj)
        
        model_cluster_scores = _json_safe(model_cluster_scores)
        cluster_scores = _json_safe(cluster_scores)
        model_scores = _json_safe(model_scores)

        # Save model-cluster scores
        model_cluster_path = str(self.output_dir / "model_cluster_scores.json")
        self.storage.write_json(model_cluster_path, model_cluster_scores)
        self.log(f"📄 Saved model-cluster scores to {model_cluster_path}")

        # Save cluster scores
        cluster_scores_path = str(self.output_dir / "cluster_scores.json")
        self.storage.write_json(cluster_scores_path, cluster_scores)
        self.log(f"📄 Saved cluster scores to {cluster_scores_path}")

        # Save model scores
        model_scores_path = str(self.output_dir / "model_scores.json")
        self.storage.write_json(model_scores_path, model_scores)
        self.log(f"📄 Saved model scores to {model_scores_path}")
        
        # Save dataframe versions as JSONL files (previously only saved when wandb was enabled)
        self._save_dataframe_files(model_cluster_scores, cluster_scores, model_scores)


    def _generate_plots(self, model_cluster_scores, cluster_scores, model_scores):
        """Generate comprehensive plots using the plotting module."""
        self.log("📊 Generating comprehensive metric plots...")

        if importlib.util.find_spec("wandb") is None:
            log_to_wandb = False
        else:
            import wandb
            log_to_wandb = self.log_to_wandb and wandb.run is not None

        # Use the plotting module to generate all plots
        num_quality_metrics = plotting.generate_all_plots(
            model_cluster_scores=model_cluster_scores,
            cluster_scores=cluster_scores,
            model_scores=model_scores,
            output_dir=self.output_dir / "plots" if self.output_dir else Path("plots"),
            log_to_wandb=log_to_wandb
        )
        
        # Informational logging (no local files are saved)
        self.log(f"✅ Generated interactive figures for {num_quality_metrics} quality metrics (no local files saved)")
        if log_to_wandb:
            self.log("📊 Figures logged to wandb under the 'Plots/' namespace")

    def _convert_to_legacy_format(self, model_cluster_scores, cluster_scores, model_scores) -> Dict[str, Any]:
        """Convert new format back to legacy format for backward compatibility."""
        # TODO: Implement conversion to legacy ModelStats format
        # For now, return a simple structure
        return {
            "functional_metrics": {
                "model_cluster_scores": model_cluster_scores,
                "cluster_scores": cluster_scores,
                "model_scores": model_scores
            }
        } 