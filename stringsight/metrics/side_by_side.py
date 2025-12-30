"""stringsight.metrics.side_by_side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Side-by-side metrics implemented on top of the functional metrics pipeline.

This adapts the Arena-style pairwise inputs by expanding each conversation into
per-model rows and converting the 'winner' field into a numeric score per model
(+1 winner, -1 loser, 0 tie). Other numeric quality metrics in the score dict
are preserved as-is if present.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .functional_metrics import FunctionalMetrics


class SideBySideMetrics(FunctionalMetrics):
    """Metrics stage for side-by-side data using functional metrics.

    The output artifacts and wandb logging are identical to `FunctionalMetrics`.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        compute_bootstrap: bool = True,
        bootstrap_samples: int = 100,
        bootstrap_seed: int | None = None,
        log_to_wandb: bool = True,
        generate_plots: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            compute_bootstrap=compute_bootstrap,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            log_to_wandb=log_to_wandb,
            generate_plots=generate_plots,
            **kwargs,
        )

    def _prepare_data(self, data) -> pd.DataFrame:
        """Prepare SxS data as per-model, per-property rows.

        This version respects the *model* associated with each property:
        clusters are linked to individual properties via ``property_ids``,
        and properties carry the ``model`` field.  Conversation-level
        metadata and scores are then joined on ``(conversation_id, model)``.
        """
        # Require both clusters and properties. If either is missing, we cannot compute metrics.
        if not data.clusters or not data.properties:
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # 1) Cluster information at the property level
        # ------------------------------------------------------------------
        clusters_df = pd.DataFrame([cluster.to_dict() for cluster in data.clusters])

        # Explode aligned list columns so each row corresponds to a single property id
        list_cols = ["property_ids", "property_descriptions", "question_ids"]
        existing_list_cols = [c for c in list_cols if c in clusters_df.columns]
        if existing_list_cols:
            clusters_df = clusters_df.explode(existing_list_cols, ignore_index=True)

        clusters_df = clusters_df.rename(
            {
                "property_ids": "property_id",
                "property_descriptions": "property_description",
                "question_ids": "conversation_id",
                "label": "cluster",
            },
            axis=1,
        )

        # Keep only the columns needed downstream for metrics
        cluster_cols = ["property_id", "cluster", "property_description"]
        if "meta" in clusters_df.columns:
            clusters_df["cluster_metadata"] = clusters_df["meta"]
            cluster_cols.append("cluster_metadata")
        clusters_df = clusters_df[cluster_cols]

        # ------------------------------------------------------------------
        # 2) Property information (includes the model that owns the property)
        # ------------------------------------------------------------------
        properties_df = pd.DataFrame([prop.to_dict() for prop in data.properties])
        properties_df = properties_df.rename(
            {"id": "property_id", "question_id": "conversation_id"}, axis=1
        )

        # ------------------------------------------------------------------
        # 3) Conversation-level scores and metadata, expanded per model
        # ------------------------------------------------------------------
        expanded_rows: List[Dict[str, Any]] = []
        for conv in data.conversations:
            conversation_id = conv.question_id
            conversation_metadata = conv.meta

            # Side-by-side: conv.model is a pair of models
            if isinstance(conv.model, (list, tuple)) and len(conv.model) == 2:
                model_a, model_b = conv.model[0], conv.model[1]

                expanded_rows.append(
                    {
                        "conversation_id": conversation_id,
                        "model": model_a,
                        "scores": self._transform_scores_for_model(
                            conv.scores, model_a, model_b, conv
                        ),
                        "conversation_metadata": conversation_metadata,
                    }
                )
                expanded_rows.append(
                    {
                        "conversation_id": conversation_id,
                        "model": model_b,
                        "scores": self._transform_scores_for_model(
                            conv.scores, model_b, model_a, conv
                        ),
                        "conversation_metadata": conversation_metadata,
                    }
                )

        conversations_df = pd.DataFrame(expanded_rows)
        if conversations_df.empty:
            # Ensure required columns exist even if empty to prevent merge errors
            conversations_df = pd.DataFrame(columns=["conversation_id", "model", "scores", "conversation_metadata"])

        # ------------------------------------------------------------------
        # 4) Join: properties ↔ conversations ↔ clusters
        # ------------------------------------------------------------------
        # First, attach per-model scores/metadata to properties via (conversation_id, model)
        properties_with_conv = properties_df.merge(
            conversations_df,
            on=["conversation_id", "model"],
            how="left",
        )

        # Then attach cluster labels/metadata via property_id. This may produce
        # suffixed property_description columns (e.g. _x / _y); we'll reconcile
        # those immediately afterwards.
        full_df = properties_with_conv.merge(
            clusters_df,
            on="property_id",
            how="left",
        )

        # Normalise property_description column name after merge
        if "property_description" not in full_df.columns:
            prop_x = full_df.get("property_description_x")
            prop_y = full_df.get("property_description_y")
            if prop_x is not None and prop_y is not None:
                full_df["property_description"] = prop_x.combine_first(prop_y)
                full_df = full_df.drop(
                    columns=[c for c in ["property_description_x", "property_description_y"] if c in full_df.columns]
                )
            elif prop_x is not None:
                full_df["property_description"] = prop_x
                full_df = full_df.drop(columns=["property_description_x"])
            elif prop_y is not None:
                full_df["property_description"] = prop_y
                full_df = full_df.drop(columns=["property_description_y"])

        # Derive property_metadata from the property description if not provided
        if "property_metadata" not in full_df.columns:
            full_df["property_metadata"] = full_df["property_description"].apply(
                lambda x: {"property_description": x}
            )

        # Ensure conversation_metadata and cluster_metadata columns exist
        if "conversation_metadata" not in full_df.columns:
            full_df["conversation_metadata"] = {}
        if "cluster_metadata" not in full_df.columns:
            full_df["cluster_metadata"] = {}

        # ------------------------------------------------------------------
        # 5) Match the schema expected by FunctionalMetrics
        # ------------------------------------------------------------------
        important_columns = [
            "conversation_id",
            "conversation_metadata",
            "property_metadata",
            "model",
            "cluster",
            "property_description",
            "scores",
            "cluster_metadata",
        ]

        # Ensure all required columns exist before filtering
        for col in important_columns:
            if col not in full_df.columns:
                if col == "scores":
                    full_df[col] = {}
                elif col == "model":
                    full_df[col] = "unknown"
                elif col in ["cluster_metadata", "conversation_metadata"]:
                    full_df[col] = {}
                else:
                    full_df[col] = ""

        return full_df[important_columns]

    @staticmethod
    def _transform_scores_for_model(all_scores: List[Dict[str, Any]], this_model: str, other_model: str, conversation=None) -> Dict[str, float]:
        """Convert the side-by-side score list into per-model numeric scores.

        Expects scores in list format [scores_a, scores_b].
        
        - "winner": +1 if this_model won, -1 if lost, 0 if tie
        - Preserve other numeric keys as floats when possible
        """
        result: Dict[str, float] = {}
        
        # Handle list format [scores_a, scores_b]
        if isinstance(all_scores, list) and len(all_scores) == 2:
            scores_a, scores_b = all_scores[0], all_scores[1]

            # Ensure scores_a and scores_b are dicts
            if not isinstance(scores_a, dict):
                scores_a = {}
            if not isinstance(scores_b, dict):
                scores_b = {}

            # Match this_model to the appropriate scores based on conversation order
            if conversation and isinstance(conversation.model, (list, tuple)) and len(conversation.model) == 2:
                model_a, model_b = conversation.model[0], conversation.model[1]
                if this_model == model_a:
                    model_scores = scores_a
                elif this_model == model_b:
                    model_scores = scores_b
                else:
                    # Fallback: use scores_a for first model, scores_b for second
                    model_scores = scores_a if this_model == model_a else scores_b
            else:
                # Fallback: use scores_a for first model, scores_b for second
                model_scores = scores_a if this_model < other_model else scores_b

            # Copy all numeric metrics from the model's scores
            if isinstance(model_scores, dict):
                for k, v in model_scores.items():
                    if isinstance(v, (int, float)):
                        result[k] = float(v)
            
        return result

    # --- Robust metrics computation for SxS to handle empty bootstrap subsets ---
    def _infer_metric_keys(self, df: pd.DataFrame) -> List[str]:
        """Infer score metric keys from any available non-empty scores dict in df."""
        if df is None or df.empty or "scores" not in df.columns:
            return []
        for val in df["scores"]:
            if isinstance(val, dict) and val:
                return list(val.keys())
        return []

    def _compute_salience(self, model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute salience for side-by-side as difference between the two models.

        For SxS, proportion_delta = this_model_proportion - other_model_proportion
        (instead of deviation from average across all models).
        """
        df = pd.DataFrame(model_cluster_scores).reset_index().rename({"index": "cluster"}, axis=1)

        model_names = [col for col in df.columns if col not in ['cluster']]

        # Extract proportion values
        for model in model_names:
            df[f'{model}_proportion'] = df[model].apply(lambda x: x.get('proportion', 0) if isinstance(x, dict) else 0)

        # For side-by-side with exactly 2 models, compute pairwise difference
        if len(model_names) == 2:
            model_a, model_b = model_names[0], model_names[1]
            df[f'{model_a}_deviation'] = df[f'{model_a}_proportion'] - df[f'{model_b}_proportion']
            df[f'{model_b}_deviation'] = df[f'{model_b}_proportion'] - df[f'{model_a}_proportion']
        else:
            # Fallback to average-based deviation if not exactly 2 models
            proportion_cols = [f'{model}_proportion' for model in model_names]
            df['avg_proportion'] = df[proportion_cols].mean(axis=1)
            for model in model_names:
                df[f'{model}_deviation'] = df[f'{model}_proportion'] - df['avg_proportion']

        # Add deviation into model_cluster_scores
        for i, row in df.iterrows():
            cluster = row['cluster']
            for model in model_names:
                deviation_value = row[f'{model}_deviation']
                if model in model_cluster_scores and cluster in model_cluster_scores[model]:
                    model_cluster_scores[model][cluster]['proportion_delta'] = deviation_value

        return model_cluster_scores

    def _bootstrap_salience_from_proportions(self, proportions: Any, *, n_models: int) -> Any:
        """Compute bootstrap salience for side-by-side from proportions.

        For SxS with exactly 2 models, `proportion_delta` is the **pairwise difference**:
            - model_a: p_a - p_b
            - model_b: p_b - p_a

        If there are not exactly 2 models, we fall back to the base definition
        (deviation from the average of other models).
        """
        import numpy as np

        if n_models == 2:
            out = np.zeros_like(proportions, dtype=float)
            out[0, :] = proportions[0, :] - proportions[1, :]
            out[1, :] = proportions[1, :] - proportions[0, :]
            return out
        return super()._bootstrap_salience_from_proportions(proportions, n_models=n_models)

    def compute_cluster_metrics(self, df: pd.DataFrame, clusters: List[str] | str, models: List[str] | str, *, include_metadata: bool = True) -> Dict[str, Any]:
        """Override to avoid indexing into empty DataFrames during bootstrap.

        Mirrors FunctionalMetrics.compute_cluster_metrics but with guards for
        empty model subsets and key alignment without assertions.
        """
        if isinstance(clusters, str):
            clusters = [clusters]
        if isinstance(models, str):
            models = [models]

        model_df = df[df["model"].isin(models)]
        if model_df.empty:
            metric_keys = self._infer_metric_keys(df)
            return self.empty_metrics(metric_keys)

        cluster_model_df = model_df[model_df["cluster"].isin(clusters)]

        # Determine metric keys from available rows
        metric_keys = self._infer_metric_keys(model_df)
        if not metric_keys:
            metric_keys = self._infer_metric_keys(df)

        if len(cluster_model_df) == 0:
            return self.empty_metrics(metric_keys)

        # Compute sizes and raw quality scores
        model_size, model_scores = self.compute_size_and_score(model_df)
        cluster_model_size, cluster_model_scores = self.compute_size_and_score(cluster_model_df)

        # Align keys without asserting strict equality
        all_keys = set(metric_keys) | set(model_scores.keys()) | set(cluster_model_scores.keys())
        for k in all_keys:
            if k not in model_scores:
                model_scores[k] = 0.0
            if k not in cluster_model_scores:
                cluster_model_scores[k] = 0.0

        quality_raw_delta = self.compute_relative_quality(cluster_model_scores, model_scores)
        proportion = cluster_model_size / model_size if model_size != 0 else 0

        # Quality delta is just the raw difference in scores (no proportion weighting)
        quality_delta = quality_raw_delta

        # Extract cluster metadata (take the first non-empty metadata from the cluster)
        cluster_metadata = {}
        if include_metadata:
            if "cluster_metadata" in cluster_model_df.columns:
                non_empty_metadata = cluster_model_df["cluster_metadata"].dropna()
                if not non_empty_metadata.empty:
                    cluster_metadata = non_empty_metadata.iloc[0]

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