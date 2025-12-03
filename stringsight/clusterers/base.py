from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import asyncio
import pandas as pd
import litellm
from ..core.llm_utils import parallel_completions_async, LLMConfig

from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Cluster
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin

# Import unified config
try:
    from .config import ClusterConfig
except ImportError:
    from config import ClusterConfig


class BaseClusterer(LoggingMixin, TimingMixin, WandbMixin, PipelineStage, ABC):
    """Abstract base class for clustering stages.

    This class defines a minimal, unified contract for clustering steps
    in the pipeline. Subclasses implement the clustering strategy while
    reusing shared orchestration, saving, and metadata-handling provided
    by the base class.

    Responsibilities
    ----------------
    - Define a single entry point (`run`) for the clustering pipeline
    - Define an abstract `cluster` method that returns a standardized
      DataFrame schema
    - Provide hooks for post-processing, configuration, saving, and
      converting DataFrames into `Cluster` objects

    Standardized DataFrame Contract
    --------------------------------
    Implementations of `cluster` must return a DataFrame containing:
    - `question_id`
    - `{column_name}` (by default `property_description`)
    - `{column_name}_cluster_id` (int)
    - `{column_name}_cluster_label` (str)
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        include_embeddings: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        prettify_labels: bool = False,
        config: Optional[ClusterConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the clusterer with common options.

        Parameters
        ----------
        output_dir:
            Directory where clustering artifacts should be saved.
        include_embeddings:
            Whether embedding columns should be included in saved artifacts.
        use_wandb:
            Enable Weights & Biases logging for clustering outputs.
        wandb_project:
            W&B project name to log under when enabled.
        prettify_labels:
            Whether to use an LLM to highlight key parts of cluster names in bold for improved readability.
            Defaults to False to avoid 30-60s overhead from 100+ LLM calls.
        config:
            Optional pre-constructed ClusterConfig to use for this clusterer.
        kwargs:
            Additional implementation-specific options for derived classes.
        """
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project, **kwargs)
        self.output_dir = output_dir
        self.include_embeddings = include_embeddings
        self.prettify_labels = prettify_labels
        self.config: Optional[ClusterConfig] = config

    @abstractmethod
    def cluster(self, data: PropertyDataset, column_name: str, progress_callback=None) -> pd.DataFrame:
        """Produce a standardized clustered DataFrame from the dataset.

        Implementations may compute embeddings or use heuristic rules, but
        should not mutate `data`. The returned DataFrame must follow the
        standardized column naming contract described in the class docstring.

        Parameters
        ----------
        data:
            The input `PropertyDataset` containing conversations and properties.
        column_name:
            The name of the textual feature column to cluster (default
            expected value is "property_description").
        progress_callback:
            Optional callback(completed, total) for progress updates.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the standardized cluster columns.
        """
        ...

    @staticmethod
    async def prettify_labels(df: pd.DataFrame, column_name: str, config: ClusterConfig) -> pd.DataFrame:
        """Use an LLM to highlight key parts of cluster names in bold for improved readability."""
        # Filter out outliers and collect labels to process
        labels_to_process = []
        indices_to_update = []
        
        for i, row in df.iterrows():
            label = row[column_name]
            if not (label == "Outliers" or label.startswith("Outliers - ")):
                labels_to_process.append(label)
                indices_to_update.append(i)
        
        if not labels_to_process:
            return df
            
        system_prompt = """The following is a property of a model response. To improve readability of the property for a user, please provide a 2-5 word summary of the property, which you will put in bold before the property (i.e. "**[property summary]**: [property]"). 
            
Make sure to include the minimal set of words that can be used to understand the property. Users at a glance should be able to understand the property and whether it's positive, negative, or neutral given the summary.

Respond with the summary along with the original property in the following format: **[property summary]**: [property]

Do not include any other text in your response."""

        # Process labels in parallel with caching
        prettified_labels = await parallel_completions_async(
            labels_to_process,
            model=config.summary_model,
            system_prompt=system_prompt,
            max_workers=min(getattr(config, "llm_max_workers", 64), len(labels_to_process)),
            show_progress=True,
            progress_desc="Prettifying cluster labels"
        )
        
        # Update the dataframe with prettified labels
        for idx, prettified_label in zip(indices_to_update, prettified_labels):
            df.at[idx, column_name] = prettified_label
            
        return df

    async def postprocess_clustered_df(self, df: pd.DataFrame, column_name: str, prettify_labels: bool = False) -> pd.DataFrame:
        """Optional hook to modify the clustered DataFrame.

        Called after `cluster` and before converting to `Cluster` objects.
        Typical use cases include:
        - Reassigning small clusters to an "Outliers" label
        - Thresholding cluster assignments or cleaning labels

        Parameters
        ----------
        df:
            The clustered DataFrame produced by `cluster`.
        column_name:
            The feature column used for clustering.
        prettify_labels:
            Whether to use an LLM to highlight key parts of cluster names in bold for improved readability.

        Returns
        -------
        pd.DataFrame
            The potentially modified DataFrame. Default: return `df` unchanged.
        """
        if prettify_labels:
            config = self.get_config()
            df = await self.prettify_labels(df, column_name, config)
        return df

    def get_config(self) -> ClusterConfig:
        """Return a configuration object for saving/logging.

        Returns
        -------
        ClusterConfig
            The configuration object used by this clusterer. If one was not
            provided, construct a default `ClusterConfig` aligned with the
            clusterer's base options.
        """
        if isinstance(self.config, ClusterConfig):
            return self.config
        # Construct a default config aligned with existing defaults
        self.config = ClusterConfig(
            include_embeddings=bool(self.include_embeddings),
            use_wandb=bool(self.use_wandb),
            wandb_project=getattr(self, "wandb_project", None),
        )
        return self.config

    async def run(self, data: PropertyDataset, column_name: str = "property_description", progress_callback=None) -> PropertyDataset:
        """Execute the clustering pipeline and return an updated dataset.

        Expected orchestration steps:
        1. Create a standardized clustered DataFrame via `cluster(...)`.
        2. Optionally post-process via `postprocess_clustered_df(...)`.
        3. Convert groups to `Cluster` domain objects.
        4. Add a synthetic "No properties" cluster via `add_no_properties_cluster(...)`
           to cover conversations without properties (when desired).
        5. Attach `cluster_id` and `cluster_label` to each
           `Property` in the input dataset when possible.
        6. Persist artifacts via `save(...)` if an `output_dir` is provided.
        7. Return a new `PropertyDataset` that includes the clusters and any
           property annotations.
        """
        self.log(f"Clustering {len(data.properties)} properties using {self.__class__.__name__}")

        if not data.properties:
            raise ValueError("No properties to cluster")

        # Handle both sync and async cluster() methods
        import inspect
        if inspect.iscoroutinefunction(self.cluster):
            # Check if cluster accepts progress_callback
            sig = inspect.signature(self.cluster)
            if 'progress_callback' in sig.parameters:
                clustered_df = await self.cluster(data, column_name, progress_callback=progress_callback)
            else:
                clustered_df = await self.cluster(data, column_name)
        else:
            # Check if cluster accepts progress_callback
            sig = inspect.signature(self.cluster)
            if 'progress_callback' in sig.parameters:
                clustered_df = self.cluster(data, column_name, progress_callback=progress_callback)
            else:
                clustered_df = self.cluster(data, column_name)
        if "meta" not in clustered_df.columns:
            clustered_df["meta"] = [{} for _ in range(len(clustered_df))]
        clustered_df = await self.postprocess_clustered_df(clustered_df, column_name, prettify_labels=self.prettify_labels)

        clusters = self._build_clusters_from_df(clustered_df, column_name)
        self.add_no_properties_cluster(data, clusters)
        self._attach_cluster_attrs_to_properties(data, clustered_df, column_name)

        self.save(clustered_df, clusters)

        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats,
        )

    def save(self, df: pd.DataFrame, clusters: List[Cluster]) -> Dict[str, str]:
        """Persist clustering artifacts to disk (and optionally external loggers).

        Implementations should leverage a common saving utility to ensure
        consistent artifact formats across clusterers.
        """
        if not self.output_dir:
            return {}

        from .clustering_utils import save_clustered_results

        base_filename = os.path.basename(self.output_dir.rstrip("/"))
        config = self.get_config()

        paths = save_clustered_results(
            df=df,
            base_filename=base_filename,
            include_embeddings=bool(self.include_embeddings),
            config=config,
            output_dir=self.output_dir,
        )

        self.log(f"✅ Auto-saved clustering results to: {self.output_dir}")
        for key, path in paths.items():
            if path:
                self.log(f"  • {key}: {path}")

        if self.use_wandb:
            self.init_wandb(project=self.wandb_project)
            import wandb
            # import weave
            log_df = pd.DataFrame([c.to_dict() for c in clusters]).astype(str)
            self.log_wandb({
                f"Clustering/{self.__class__.__name__}_clustered_table": wandb.Table(dataframe=log_df)
            })

        return paths

    def _build_clusters_from_df(self, df: pd.DataFrame, column_name: str) -> List[Cluster]:
        """Construct `Cluster` objects from a standardized DataFrame.

        Group rows by cluster id, extract labels and collect
        `question_id`, `{column_name}` and `id` values for each cluster.
        """
        label_col = f"{column_name}_cluster_label"
        id_col = f"{column_name}_cluster_id"

        clusters: List[Cluster] = []
        for cid, group in df.groupby(id_col):
            cid_group = group[group[id_col] == cid]
            label = str(cid_group[label_col].iloc[0])

            property_ids = cid_group["id"].tolist() if "id" in cid_group.columns else []
            question_ids = cid_group["question_id"].tolist() if "question_id" in cid_group.columns else []
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

    def _attach_cluster_attrs_to_properties(self, data: PropertyDataset, df: pd.DataFrame, column_name: str) -> None:
        """Attach cluster annotations to properties in the dataset.

        For each `Property` whose `{column_name}` value appears in the
        standardized DataFrame, set `cluster_id` and `cluster_label`
        on the property instance.
        """
        id_map = dict(zip(df[column_name], df[f"{column_name}_cluster_id"]))
        label_map = dict(zip(df[column_name], df[f"{column_name}_cluster_label"]))

        for prop in data.properties:
            value = getattr(prop, column_name, None)
            if value in id_map:
                setattr(prop, "cluster_id", int(id_map[value]))
                setattr(prop, "cluster_label", label_map[value])

    def add_no_properties_cluster(self, data: PropertyDataset, clusters: List[Cluster]) -> None:
        """Append a synthetic "No properties" cluster when applicable.

        Detect conversations that lack any associated properties and create a
        dedicated cluster entry so they are represented in downstream metrics
        and visualizations. Subclasses may override to disable or customize
        this behavior.
        """
        # Only add the synthetic cluster when *no* properties were extracted for the
        # entire dataset.  If at least one property exists, we leave conversations
        # without properties unclustered so that partial extraction failures do not
        # pollute the clustering results with a global "No properties" bucket.

        if data.properties:
            # There is at least one property in the dataset – skip creating the
            # special cluster entirely.
            self.log("Dataset contains properties; skipping creation of 'No properties' cluster")
            return

        conversations_with_properties = set()
        for prop in data.properties:
            conversations_with_properties.add((prop.question_id, prop.model))

        conversations_without_properties: List[tuple] = []
        for conv in data.conversations:
            if isinstance(conv.model, str):
                key = (conv.question_id, conv.model)
                if key not in conversations_with_properties:
                    conversations_without_properties.append(key)
            elif isinstance(conv.model, list):
                for model in conv.model:
                    key = (conv.question_id, model)
                    if key not in conversations_with_properties:
                        conversations_without_properties.append(key)

        if not conversations_without_properties:
            self.log("All conversations have properties - no 'No properties' cluster needed")
            return

        self.log(
            f"Found {len(conversations_without_properties)} conversations without properties - creating 'No properties' cluster"
        )

        # For the "No properties" cluster, we need all lists to have matching lengths for explode() to work.
        # Use sentinel values to represent the lack of actual properties.
        no_props_cluster = Cluster(
            id=-2,
            label="No properties",
            size=len(conversations_without_properties),
            property_descriptions=["No properties"] * len(conversations_without_properties),  # One per conversation
            property_ids=["-2"] * len(conversations_without_properties),  # Sentinel property ID
            question_ids=[qid for qid, _ in conversations_without_properties],
            meta={},
        )
        clusters.append(no_props_cluster)
        self.log(
            f"Created 'No properties' cluster with {len(conversations_without_properties)} conversations"
        )


__all__ = ["BaseClusterer"] 