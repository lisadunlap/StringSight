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
        log_to_wandb: bool = True,
        generate_plots: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            compute_bootstrap=compute_bootstrap,
            bootstrap_samples=bootstrap_samples,
            log_to_wandb=log_to_wandb,
            generate_plots=generate_plots,
            **kwargs,
        )

    def _prepare_data(self, data) -> pd.DataFrame:
        """Prepare SxS data: expand each pair into two rows (one per model).

        Produces the same schema expected by FunctionalMetrics:
        columns: [conversation_id, conversation_metadata, property_metadata, model, cluster, property_description, scores]
        """
        # Extract clusters and properties data
        if not data.clusters:
            return pd.DataFrame()

        clusters = pd.DataFrame([cluster.to_dict() for cluster in data.clusters])

        # FIXED: Use the same approach as functional_metrics - explode only property_descriptions and question_ids
        # This ensures the join works correctly and doesn't depend on complex property-model mappings
        clusters = clusters.explode(["property_descriptions", "question_ids", "property_ids"]).drop_duplicates(
            subset=["property_descriptions", "question_ids", "property_ids"]
        )
        clusters = clusters.dropna(subset=["property_descriptions", "question_ids", "property_ids"])
        clusters = clusters.rename(
            {"question_ids": "question_id", "property_descriptions": "property_description", "property_ids": "property_id"}, axis=1
        )
        
        properties = pd.DataFrame([property.to_dict() for property in data.properties])

        properties_df = []
        for index, row in properties.iterrows():
            properties_df.append({
                "property_id": row["id"],
                "model": row["model"],
            })
        properties_df = pd.DataFrame(properties_df)
        print(f"Number of properties: {len(properties_df)}")

        properties = clusters.merge(properties_df, on="property_id", how="left").rename(
            {"label": "cluster"},
            axis=1,
        )

        # Expand conversations: one row per model with per-model scores
        expanded_rows: List[Dict[str, Any]] = []
        for conv in data.conversations:
            qid = conv.question_id
            meta = conv.meta

            # Side-by-side: conv.model is a list/tuple of two models
            model_a, model_b = conv.model[0], conv.model[1]
            expanded_rows.append(
                {
                    "question_id": qid,
                    "scores": self._transform_scores_for_model(conv.scores, model_a, model_b, conv),
                    "conversation_metadata": meta,
                    "model_name": model_a,
                }
            )
            expanded_rows.append(
                {
                    "question_id": qid,
                    "scores": self._transform_scores_for_model(conv.scores, model_b, model_a, conv),
                    "conversation_metadata": meta,
                    "model_name": model_b,
                }
            )

        conversations = pd.DataFrame(expanded_rows)

        properties = properties.merge(conversations, on="question_id", how="left").rename(
            {"label": "cluster", "question_id": "conversation_id"},
            axis=1,
        )

        # remove any rows where model_name != model
        print(f"Length before: {len(properties)}")
        properties = properties[properties["model_name"] == properties["model"]]
        print(f"Length after: {len(properties)}")
        properties = properties.drop("model_name", axis=1)
        
        # Ensure conversation_metadata exists - fill missing values with empty dict
        if "conversation_metadata" not in properties.columns:
            properties["conversation_metadata"] = {}
        else:
            properties["conversation_metadata"] = properties["conversation_metadata"].fillna({})

        # print(properties['cluster_metadata'].head())
        
        # Handle cluster_metadata from the cluster's meta field
        if "meta" in properties.columns:
            properties["cluster_metadata"] = properties["meta"]
            properties = properties.drop("meta", axis=1)
        else:
            properties["cluster_metadata"] = {}
        
        properties["property_metadata"] = properties["property_description"].apply(
            lambda x: {"property_description": x}
        )

        # Match the column selection from functional_metrics exactly
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
        return properties

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
            for k, v in model_scores.items():
                if isinstance(v, (int, float)):
                    result[k] = float(v)
            
            # Handle winner if present in meta field
            if conversation and hasattr(conversation, 'meta'):
                winner = conversation.meta.get("winner")
                if isinstance(winner, str):
                    if winner == this_model:
                        result["winner"] = 1.0
                    elif "tie" in winner.lower():
                        result["winner"] = 0.0
                    else:
                        result["winner"] = -1.0
        
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

        quality_delta = self.compute_relative_quality(cluster_model_scores, model_scores)
        proportion = cluster_model_size / model_size if model_size != 0 else 0

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