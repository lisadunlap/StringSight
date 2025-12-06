from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd

from .base import BaseClusterer
from ..core.data_objects import PropertyDataset, Cluster

# Unified config
try:
    from .config import ClusterConfig
except ImportError:
    from config import ClusterConfig


class DummyClusterer(BaseClusterer):
    """A no-op clustering stage used for fixed-taxonomy pipelines.

    Every unique `category` becomes its own fine cluster. No
    embeddings or distance computations are performed.

    Parameters
    ----------
    allowed_labels:
        List of labels that are present in the user-supplied taxonomy.
    unknown_label:
        Name assigned to properties whose description is not in
        `allowed_labels` (default: "Other").
    output_dir:
        Directory to save clustering results (optional)
    include_embeddings:
        Whether to include embeddings in output (default: False)
    """

    def __init__(
        self,
        allowed_labels: List[str],
        unknown_label: str = "Other",
        output_dir: str | None = None,
        include_embeddings: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            output_dir=output_dir,
            include_embeddings=include_embeddings,
            **kwargs,
        )
        self.allowed_labels = set(allowed_labels)
        self.unknown_label = unknown_label
        # Minimal config for saving/logging and consistency
        self.config = ClusterConfig(
            min_cluster_size=1,
            embedding_model="dummy",
            assign_outliers=False,
            use_wandb=False,
            wandb_project=None,
            disable_dim_reduction=True,
            include_embeddings=include_embeddings,
        )

    async def run(self, data: PropertyDataset, column_name: str = "property_description", progress_callback=None) -> PropertyDataset:
        """Execute clustering using `property_description` as the key for fixed-axes.

        We intentionally ignore the incoming `column_name` and cluster by
        the `property_description` field emitted by the fixed-axes extractor.
        """
        return await super().run(data, column_name="property_description", progress_callback=progress_callback)

    def cluster(self, data: PropertyDataset, column_name: str) -> pd.DataFrame:
        """Map properties to a fixed taxonomy and return a standardized DataFrame."""
        # 1) Sanitize property descriptions in-memory for clustering
        for prop in data.properties:
            desc = (getattr(prop, column_name, "") or "").strip()
            if desc not in self.allowed_labels:
                setattr(prop, column_name, self.unknown_label)

        # 2) Build a properties DataFrame
        df = data.to_dataframe(type="properties")
        if df.empty:
            return df

        # 3) Compute deterministic ordering: allowed labels first, then 'Other' if not included
        ordered_labels = list(self.allowed_labels)
        if self.unknown_label not in self.allowed_labels:
            ordered_labels.append(self.unknown_label)
        label_to_id: Dict[str, int] = {label: idx for idx, label in enumerate(ordered_labels)}

        # 4) Add standardized cluster columns
        id_col = f"{column_name}_cluster_id"
        label_col = f"{column_name}_cluster_label"
        df[label_col] = df[column_name].map(lambda x: x if x in label_to_id else self.unknown_label)
        df[id_col] = df[label_col].map(label_to_id)
        # Add base convenience aliases used elsewhere
        df["cluster_id"] = df[id_col]
        df["cluster_label"] = df[label_col]

        return df 

    def _build_clusters_from_df(self, df: pd.DataFrame, column_name: str):
        """Construct clusters while preserving human-readable descriptions.

        Although we cluster by `category`, we keep the cluster's
        `property_descriptions` populated from the `property_description`
        column when available so downstream merges and displays remain
        intuitive.
        """
        label_col = f"{column_name}_cluster_label"
        id_col = f"{column_name}_cluster_id"

        clusters: List[Cluster] = []
        for cid, group in df.groupby(id_col):
            cid_group = group[group[id_col] == cid]
            label = str(cid_group[label_col].iloc[0])

            property_ids = cid_group["id"].tolist() if "id" in cid_group.columns else []
            question_ids = cid_group["question_id"].tolist() if "question_id" in cid_group.columns else []
            if "property_description" in cid_group.columns:
                property_descriptions = cid_group["property_description"].tolist()
            else:
                property_descriptions = cid_group[column_name].tolist()

            clusters.append(
                Cluster(
                    id=int(cid),
                    label=label,
                    size=len(cid_group),
                    property_descriptions=property_descriptions,
                    property_ids=property_ids,
                    question_ids=question_ids,
                    meta={},
                )
            )

        self.log(f"Created {len(clusters)} clusters")

        return clusters

    def save(self, df: pd.DataFrame, clusters: List[Cluster]) -> Dict[str, str]:
        """Ensure summary utilities receive expected columns.

        The shared saving helpers expect `property_description_*` columns.
        For the fixed-axes path we cluster by `category`, so we add
        on-the-fly alias columns before delegating to the base saver.
        """
        df_to_save = df.copy()
        if "category_cluster_label" in df_to_save.columns and "property_description_cluster_label" not in df_to_save.columns:
            df_to_save["property_description_cluster_label"] = df_to_save["category_cluster_label"]
        if "category_cluster_id" in df_to_save.columns and "property_description_cluster_id" not in df_to_save.columns:
            df_to_save["property_description_cluster_id"] = df_to_save["category_cluster_id"]
        return super().save(df_to_save, clusters)