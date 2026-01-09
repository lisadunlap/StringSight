"""
Pipeline builder functions for constructing analysis pipelines.
"""

from typing import Dict, Callable, Union, Any
import pandas as pd
from ..pipeline import Pipeline, PipelineBuilder
from ..constants import DEFAULT_MAX_WORKERS


def build_default_pipeline(
    method: str,
    system_prompt: str,
    prompt_builder: Callable[[pd.Series, str], str] | None,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
    include_scores_in_prompt: bool,
    clusterer: Union[str, Any],
    min_cluster_size: int | None,
    embedding_model: str,
    assign_outliers: bool,
    prettify_labels: bool,
    summary_model: str,
    cluster_assignment_model: str,
    metrics_kwargs: Dict[str, Any | None],
    use_wandb: bool,
    wandb_project: str | None,
    include_embeddings: bool,
    verbose: bool,
    extraction_cache_dir: str | None = None,
    clustering_cache_dir: str | None = None,
    metrics_cache_dir: str | None = None,
    output_dir: str | None = "./results",
    custom_clustering_prompts: Dict[str, str] | None = None,
    **kwargs
) -> Pipeline:
    """
    Build the default pipeline based on configuration.

    This function constructs the standard pipeline stages based on the user's
    configuration. It handles the complexity of importing and configuring
    the appropriate stages.
    """

    # Import stages (lazy imports to avoid circular dependencies)
    from ..extractors import get_extractor
    from ..postprocess import LLMJsonParser, PropertyValidator
    from ..clusterers import get_clusterer
    from ..metrics import get_metrics

    # Build pipeline using PipelineBuilder
    builder = PipelineBuilder(name=f"StringSight-{method}")

    # Configure common options
    common_config = {
        'verbose': verbose,
        'use_wandb': use_wandb,
        'wandb_project': wandb_project or "StringSight"
    }

    # Create stage-specific output directories if output_dir is provided
    if output_dir:
        extraction_output: str | None = output_dir
        parsing_output: str | None = output_dir
        validation_output: str | None = output_dir
        clustering_output: str | None = output_dir
        metrics_output: str | None = output_dir
    else:
        extraction_output = parsing_output = validation_output = clustering_output = metrics_output = None

    # 1. Property extraction stage
    extractor_kwargs = {
        'model_name': model_name,
        'system_prompt': system_prompt,
        'prompt_builder': prompt_builder,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'max_workers': max_workers,
        'include_scores_in_prompt': include_scores_in_prompt,
        'output_dir': extraction_output,
        **common_config
    }

    # Add cache directory for extraction if provided
    if extraction_cache_dir:
        extractor_kwargs['cache_dir'] = extraction_cache_dir

    extractor = get_extractor(**extractor_kwargs)  # type: ignore[arg-type]
    builder.extract_properties(extractor)

    # 2. JSON parsing stage
    parser_kwargs = {
        'output_dir': parsing_output,
        **common_config
    }
    parser = LLMJsonParser(**parser_kwargs)  # type: ignore[arg-type]
    builder.parse_properties(parser)

    # 3. Property validation stage
    validator_kwargs = {
        'output_dir': validation_output,
        **common_config
    }
    validator = PropertyValidator(**validator_kwargs)  # type: ignore[arg-type]
    builder.add_stage(validator)

    # 4. Clustering stage
    clusterer_kwargs = {
        'min_cluster_size': min_cluster_size,
        'embedding_model': embedding_model,
        'assign_outliers': assign_outliers,
        'include_embeddings': include_embeddings,
        'prettify_labels': prettify_labels,
        'summary_model': summary_model,
        'cluster_assignment_model': cluster_assignment_model,
        'output_dir': clustering_output,
        **common_config
    }
    # Default to stratified clustering by behavior_type unless overridden by caller
    if not kwargs or 'groupby_column' not in kwargs:
        clusterer_kwargs['groupby_column'] = 'behavior_type'
    # Forward any additional clusterer-specific kwargs (e.g., groupby_column)
    if kwargs:
        clusterer_kwargs.update(kwargs)

    # Ensure LLM concurrency for clustering calls follows extraction max_workers by default
    # unless explicitly overridden by caller via kwargs
    clusterer_kwargs.setdefault('llm_max_workers', max_workers)

    # Add custom clustering prompts if provided (from dynamic prompt generation)
    if custom_clustering_prompts:
        clusterer_kwargs['custom_prompts'] = custom_clustering_prompts

    # Add cache directory for clustering if provided
    if clustering_cache_dir:
        clusterer_kwargs['cache_dir'] = clustering_cache_dir

    if isinstance(clusterer, str):
        clusterer_stage = get_clusterer(clusterer, **clusterer_kwargs)  # type: ignore[arg-type]
    else:
        clusterer_stage = clusterer

    builder.cluster_properties(clusterer_stage)

    # 5. Metrics computation stage
    metrics_kwargs_dict = {
        'method': method,
        'output_dir': metrics_output,
        'compute_bootstrap': metrics_kwargs.get('compute_confidence_intervals', True) if metrics_kwargs else True,
        'bootstrap_samples': metrics_kwargs.get('bootstrap_samples', 100) if metrics_kwargs else 100,
        'log_to_wandb': use_wandb,
        'generate_plots': True,
        **(metrics_kwargs or {}),
        **common_config
    }

    # Add cache directory for metrics if provided
    if metrics_cache_dir:
        metrics_kwargs_dict['cache_dir'] = metrics_cache_dir

    metrics_stage = get_metrics(method, **{k: v for k, v in metrics_kwargs_dict.items() if k != 'method'})
    builder.compute_metrics(metrics_stage)

    # Build and return the pipeline
    pipeline = builder.configure(output_dir=output_dir, **common_config).build()

    # If wandb is already initialized globally, mark the pipeline as having wandb available
    if use_wandb:
        import wandb
        if wandb.run is not None and hasattr(pipeline, '_wandb_ok'):
            pipeline._wandb_ok = True

    return pipeline


def build_fixed_axes_pipeline(
    *,
    extractor: Any,
    taxonomy: Dict[str, str],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
    metrics_kwargs: Dict[str, Any | None] | None = None,
    use_wandb: bool,
    wandb_project: str | None,
    include_embeddings: bool,
    verbose: bool,
    output_dir: str | None,
    extraction_cache_dir: str | None = None,
    metrics_cache_dir: str | None = None,
    **kwargs: Any,
) -> Pipeline:
    """
    Internal helper that constructs a pipeline for *label()* calls.

    Args:
        extractor: Extractor instance used to label properties.
        taxonomy: Mapping of allowed label -> human-readable description.
        model_name: Model identifier used in labeling.
        temperature: Sampling temperature for the model.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum tokens for model outputs.
        max_workers: Parallelism for extraction/processing stages.
        metrics_kwargs: Optional keyword arguments forwarded into `get_metrics(...)`.
            Expected to be a dict of metric-stage configuration keys to values (or None).
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name (if enabled).
        include_embeddings: Whether to compute / include embeddings.
        verbose: Whether to emit verbose logs.
        output_dir: Optional output directory for artifacts.
        extraction_cache_dir: Optional cache directory for extraction stage.
        metrics_cache_dir: Optional cache directory for metrics stage.
        **kwargs: Additional configuration forwarded to downstream components.

    Returns:
        Pipeline configured for fixed-axis labeling.
    """

    from ..postprocess import LLMJsonParser, PropertyValidator
    from ..clusterers.dummy_clusterer import DummyClusterer
    from ..metrics import get_metrics

    builder = PipelineBuilder(name="StringSight-fixed-axes")

    common_cfg = {"verbose": verbose, "use_wandb": use_wandb, "wandb_project": wandb_project or "StringSight"}

    # 1️⃣  Extraction / labeling (use pre-created extractor)
    builder.extract_properties(extractor)

    # 2️⃣  JSON parsing
    parser = LLMJsonParser(output_dir=output_dir, fail_fast=True, **common_cfg)  # type: ignore[arg-type]
    builder.parse_properties(parser)

    # 3️⃣  Validation
    validator = PropertyValidator(output_dir=output_dir, **common_cfg)  # type: ignore[arg-type]
    builder.add_stage(validator)

    # 4️⃣  Dummy clustering
    dummy_clusterer = DummyClusterer(allowed_labels=list(taxonomy.keys()), output_dir=output_dir, **common_cfg)  # type: ignore[arg-type]
    builder.cluster_properties(dummy_clusterer)

    # 5️⃣  Metrics (single-model only)
    metrics_stage = get_metrics(method="single_model", output_dir=output_dir, **(metrics_kwargs or {}), **({"cache_dir": metrics_cache_dir} if metrics_cache_dir else {}), **common_cfg)
    builder.compute_metrics(metrics_stage)

    return builder.configure(output_dir=output_dir, **common_cfg).build()
