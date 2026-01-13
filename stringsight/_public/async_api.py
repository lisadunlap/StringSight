"""
Async versions of the public API functions for use in async contexts (e.g., FastAPI).
"""

from typing import Dict, List, Any, Callable, Union, Tuple
import pandas as pd
from ..core.data_objects import PropertyDataset
from ..pipeline import Pipeline, PipelineBuilder
from ..prompts import get_system_prompt
from ..utils.validation import validate_openai_api_key
from ..logging_config import get_logger
from ..constants import DEFAULT_MAX_WORKERS

logger = get_logger(__name__)


async def extract_properties_only_async(
    df: pd.DataFrame,
    *,
    method: str = "single_model",
    system_prompt: str | None = None,
    task_description: str | None = None,
    fail_on_empty_properties: bool = True,
    prompt_builder: Callable[[pd.Series, str], str] | None = None,
    model_name: str = "gpt-4.1",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    include_scores_in_prompt: bool = False,
    score_columns: List[str] | None = None,
    sample_size: int | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    prompt_column: str = "prompt",
    model_column: str | None = None,
    model_response_column: str | None = None,
    question_id_column: str | None = None,
    model_a_column: str | None = None,
    model_b_column: str | None = None,
    model_a_response_column: str | None = None,
    model_b_response_column: str | None = None,
    output_dir: str | None = None,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    verbose: bool = False,
    extraction_cache_dir: str | None = None,
    return_debug: bool = False,
    **kwargs: Any
) -> PropertyDataset | Tuple[PropertyDataset, List[Dict[str, Any]]]:
    """Async version of extract_properties_only for use in async contexts (e.g., FastAPI).

    See extract_properties_only() for full documentation.
    """
    # Just call the sync version's implementation but await the pipeline
    from ..extractors import get_extractor
    from ..postprocess import LLMJsonParser, PropertyValidator
    from ..core.preprocessing import validate_and_prepare_dataframe

    validate_openai_api_key(model_name=model_name)

    system_prompt = get_system_prompt(method, system_prompt, task_description)

    if verbose:
        logger.info("\n" + "="*80)
        logger.info("SYSTEM PROMPT")
        logger.info("="*80)
        logger.info(system_prompt)
        logger.info("="*80 + "\n")
    if len(system_prompt) < 50:
        raise ValueError("System prompt is too short. Please provide a longer system prompt.")

    df = validate_and_prepare_dataframe(
        df,
        method=method,
        score_columns=score_columns,
        sample_size=sample_size,
        model_a=model_a,
        model_b=model_b,
        prompt_column=prompt_column,
        model_column=model_column,
        model_response_column=model_response_column,
        question_id_column=question_id_column,
        model_a_column=model_a_column,
        model_b_column=model_b_column,
        model_a_response_column=model_a_response_column,
        model_b_response_column=model_b_response_column,
        verbose=verbose,
    )

    dataset = PropertyDataset.from_dataframe(df, method=method)

    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)

    common_cfg = {"verbose": verbose, "use_wandb": use_wandb, "wandb_project": wandb_project or "StringSight"}

    extractor_kwargs = {
        "model_name": model_name,
        "system_prompt": system_prompt,
        "prompt_builder": None,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_workers": max_workers,
        "include_scores_in_prompt": include_scores_in_prompt,
        **common_cfg,
    }

    extractor = get_extractor(**extractor_kwargs)  # type: ignore[arg-type]
    parser = LLMJsonParser(fail_fast=False, **common_cfg)  # type: ignore[arg-type]
    validator = PropertyValidator(fail_on_empty=fail_on_empty_properties, **common_cfg)  # type: ignore[arg-type]

    if output_dir:
        extractor.output_dir = output_dir  # type: ignore[attr-defined]
        parser.output_dir = output_dir
        validator.output_dir = output_dir

    pipeline = Pipeline(
        name=f"extract-{method}",
        stages=[extractor, parser, validator],
        **common_cfg,  # type: ignore[arg-type]
    )

    result_dataset = await pipeline.run(dataset)

    if return_debug:
        return result_dataset, []
    return result_dataset


async def explain_async(
    df: pd.DataFrame,
    method: str = "single_model",
    system_prompt: str | None = None,
    prompt_builder: Callable[[pd.Series, str], str] | None = None,
    task_description: str | None = None,
    *,
    sample_size: int | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    score_columns: List[str] | None = None,
    prompt_column: str = "prompt",
    model_column: str | None = None,
    model_response_column: str | None = None,
    question_id_column: str | None = None,
    model_a_column: str | None = None,
    model_b_column: str | None = None,
    model_a_response_column: str | None = None,
    model_b_response_column: str | None = None,
    model_name: str = "gpt-4.1",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    include_scores_in_prompt: bool = False,
    clusterer: Union[str, Any] = "hdbscan",
    min_cluster_size: int | None = 5,
    embedding_model: str = "text-embedding-3-large",
    prettify_labels: bool = False,
    assign_outliers: bool = False,
    summary_model: str = "gpt-4.1",
    cluster_assignment_model: str = "gpt-4.1-mini",
    metrics_kwargs: Dict[str, Any | None] | None = None,
    use_wandb: bool = True,
    wandb_project: str | None = None,
    include_embeddings: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
    custom_pipeline: Pipeline | None = None,
    extraction_cache_dir: str | None = None,
    clustering_cache_dir: str | None = None,
    metrics_cache_dir: str | None = None,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Async version of explain() for use in async contexts (e.g., FastAPI).

    This is identical to explain() but can be awaited from async code.
    See explain() for full documentation of parameters.
    """
    from ..prompts import get_system_prompt
    from ..pipeline import PipelineBuilder

    if custom_pipeline is not None:
        pipeline = custom_pipeline
        dataset = PropertyDataset.from_dataframe(df, method=method)
        result_dataset = await pipeline.run(dataset)
        return result_dataset.to_dataframe(), result_dataset.model_stats

    system_prompt = get_system_prompt(method, system_prompt, task_description)
    dataset = PropertyDataset.from_dataframe(df, method=method)

    from ..extractors import get_extractor
    from ..postprocess import LLMJsonParser, PropertyValidator
    from ..clusterers import get_clusterer
    from ..metrics import get_metrics

    common_cfg = {
        'verbose': verbose,
        'use_wandb': use_wandb,
        'wandb_project': wandb_project,
    }

    # Create extractor
    extractor = get_extractor(
        model_name=model_name,
        system_prompt=system_prompt,
        prompt_builder=prompt_builder,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        include_scores_in_prompt=include_scores_in_prompt,
        **common_cfg
    )

    # Create parser
    parser = LLMJsonParser(**common_cfg)  # type: ignore[arg-type]

    # Create validator
    validator = PropertyValidator(**common_cfg)  # type: ignore[arg-type]

    # Create clusterer
    clusterer_kwargs = {
        'min_cluster_size': min_cluster_size,
        'embedding_model': embedding_model,
        'prettify_labels': prettify_labels,
        'assign_outliers': assign_outliers,
        'summary_model': summary_model,
        'cluster_assignment_model': cluster_assignment_model,
        'include_embeddings': include_embeddings,
        **common_cfg
    }
    if isinstance(clusterer, str):
        clusterer_stage = get_clusterer(clusterer, **clusterer_kwargs)  # type: ignore[arg-type]
    else:
        clusterer_stage = clusterer

    # Create metrics
    metrics_stage = get_metrics(method, **(metrics_kwargs or {}), **common_cfg)

    # Build pipeline
    pipeline = PipelineBuilder(name=f"StringSight-{method}") \
        .extract_properties(extractor) \
        .parse_properties(parser) \
        .add_stage(validator) \
        .cluster_properties(clusterer_stage) \
        .compute_metrics(metrics_stage) \
        .configure(output_dir=output_dir, **common_cfg) \
        .build()

    result_dataset = await pipeline.run(dataset)
    return result_dataset.to_dataframe(), result_dataset.model_stats
