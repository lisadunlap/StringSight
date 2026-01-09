"""
Fixed-taxonomy labeling API for StringSight.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import time
from ..core.data_objects import PropertyDataset
from ..pipeline import Pipeline
from ..logging_config import get_logger
from ..constants import DEFAULT_MAX_WORKERS
from .helpers import run_pipeline_smart, print_analysis_summary, save_final_summary
from .pipeline_builders import build_fixed_axes_pipeline

logger = get_logger(__name__)


def label(
    df: pd.DataFrame,
    *,
    taxonomy: Dict[str, str],
    sample_size: int | None = None,
    # Column mapping parameters
    score_columns: List[str] | None = None,
    prompt_column: str = "prompt",
    model_column: str | None = None,
    model_response_column: str | None = None,
    question_id_column: str | None = None,
    model_name: str = "gpt-4.1",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    max_workers: int = DEFAULT_MAX_WORKERS,
    metrics_kwargs: Dict[str, Any | None] | None = None,
    use_wandb: bool = True,
    wandb_project: str | None = None,
    include_embeddings: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
    extraction_cache_dir: str | None = None,
    metrics_cache_dir: str | None = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run the *fixed-taxonomy* analysis pipeline. This is just you're run of the mill LLM-judge with a given rubric.

    The user provides a dataframe with a model and its responses alone with a taxonomy.

    Unlike :pyfunc:`explain`, this entry point does **not** perform clustering;
    each taxonomy label simply becomes its own cluster.  The input `df` **must**
    be in *single-model* format (columns `question_id`, `prompt`, `model`, `model_response`, …).

    Args:
        df: DataFrame with single-model conversation data
        taxonomy: Dictionary mapping label names to their descriptions
        sample_size: Optional number of rows to sample from the dataset before processing.
                    If None, uses the entire dataset. For balanced datasets (each prompt answered
                    by all models), automatically samples prompts evenly across models.
        score_columns: Optional list of column names containing score metrics. Instead of
                    providing scores as a dictionary in a 'score' column, you can specify
                    separate columns for each metric (e.g., ['accuracy', 'helpfulness']).
                    If provided, these columns will be converted to the expected score dict format.
        prompt_column: Name of the prompt column in your dataframe (default: "prompt")
        model_column: Name of the model column (default: "model")
        model_response_column: Name of the model response column (default: "model_response")
        question_id_column: Name of the question_id column (default: "question_id" if column exists)
        model_name: LLM model for property extraction (default: "gpt-4.1")
        temperature: Temperature for LLM (default: 0.0)
        top_p: Top-p for LLM (default: 1.0)
        max_tokens: Max tokens for LLM (default: 2048)
        max_workers: Max parallel workers for API calls (default: 16)
        metrics_kwargs: Additional metrics configuration
        use_wandb: Whether to log to Weights & Biases (default: True)
        wandb_project: W&B project name
        include_embeddings: Whether to include embeddings in output (default: True)
        verbose: Whether to print progress (default: True)
        output_dir: Directory to save results (optional)
        extraction_cache_dir: Cache directory for extraction results
        metrics_cache_dir: Cache directory for metrics results
        **kwargs: Additional configuration options

    Returns:
        Tuple of (clustered_df, model_stats)
        - clustered_df: Original DataFrame with added property and cluster columns
        - model_stats: Dictionary containing three DataFrames:
            - "model_cluster_scores": Per model-cluster metrics (size, proportion, quality, etc.)
            - "cluster_scores": Per cluster aggregated metrics (across all models)
            - "model_scores": Per model aggregated metrics (across all clusters)
    """
    t0 = time.perf_counter()
    timings = {}

    method = "single_model"  # hard-coded, we only support single-model here

    # Align environment with wandb toggle early to avoid accidental logging on import
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)
    if "model_b" in df.columns:
        raise ValueError("label() currently supports only single-model data.  Use explain() for side-by-side analyses.")

    # Preprocess data: handle score_columns, sampling, and column mapping
    # For label() mode, use row-level sampling to get exact sample_size
    from ..core.preprocessing import validate_and_prepare_dataframe
    df = validate_and_prepare_dataframe(
        df,
        method=method,
        score_columns=score_columns,
        sample_size=sample_size,
        prompt_column=prompt_column,
        model_column=model_column,
        model_response_column=model_response_column,
        question_id_column=question_id_column,
        verbose=verbose,
        use_row_sampling=True,  # Use row-level sampling for label() to get exact count
    )

    timings['preprocessing'] = time.perf_counter() - t0
    logger.info(f"[TIMING] Preprocessing completed in {timings['preprocessing']:.3f}s")

    # ------------------------------------------------------------------
    # Create extractor first to get the system prompt
    # ------------------------------------------------------------------
    from ..extractors.fixed_axes_labeler import FixedAxesLabeler

    # Create the extractor to generate the system prompt from taxonomy
    extractor = FixedAxesLabeler(
        taxonomy=taxonomy,
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        cache_dir=extraction_cache_dir or ".cache/stringsight",
        output_dir=output_dir,
        verbose=verbose,
        use_wandb=use_wandb,
        wandb_project=wandb_project or "StringSight"
    )

    timings['extractor_init'] = time.perf_counter() - t0
    logger.info(f"[TIMING] Extractor initialization completed in {timings['extractor_init'] - timings['preprocessing']:.3f}s")

    # Print the system prompt for verification
    if verbose:
        logger.info("\n" + "="*80)
        logger.info("SYSTEM PROMPT")
        logger.info("="*80)
        logger.info(extractor.system_prompt)
        logger.info("="*80 + "\n")

    # ------------------------------------------------------------------
    # Build dataset & pipeline
    # ------------------------------------------------------------------
    dataset = PropertyDataset.from_dataframe(df, method=method)

    timings['dataset_creation'] = time.perf_counter() - t0
    logger.info(f"[TIMING] Dataset creation completed in {timings['dataset_creation'] - timings['extractor_init']:.3f}s")

    # Initialize wandb if enabled - short-circuit early to avoid expensive string operations
    if use_wandb:
        try:
            import wandb
            import os
            import re

            # Try to get input filename from the DataFrame or use a default
            input_filename = "unknown_dataset"
            if hasattr(df, 'name') and df.name:
                input_filename = df.name
            elif hasattr(df, '_metadata') and df._metadata and 'filename' in df._metadata:
                input_filename = df._metadata['filename']
            else:
                # Try to infer from the DataFrame source if it has a path attribute
                # This is a fallback for when we can't determine the filename
                input_filename = f"dataset_{len(df)}_rows"

            # Clean the filename for wandb (remove extension, replace spaces/special chars)
            if isinstance(input_filename, str):
                # Remove file extension and clean up the name
                input_filename = os.path.splitext(os.path.basename(input_filename))[0]
                # Replace spaces and special characters with underscores
                input_filename = input_filename.replace(' ', '_').replace('-', '_')
                # Remove any remaining special characters
                input_filename = re.sub(r'[^a-zA-Z0-9_]', '', input_filename)

            wandb_run_name = os.path.basename(os.path.normpath(output_dir)) if output_dir else f"{input_filename}_label"

            wandb.init(
                project=wandb_project or "StringSight",
                name=wandb_run_name,
                config={
                    "method": method,
                    "model_name": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "max_workers": max_workers,
                    "taxonomy_size": len(taxonomy),
                    "include_embeddings": include_embeddings,
                    "output_dir": output_dir,
                },
                reinit=False  # Don't reinitialize if already exists
            )
        except ImportError:
            # wandb not installed or not available
            use_wandb = False
    # If wandb is disabled, skip all the initialization overhead entirely

    timings['wandb_init'] = time.perf_counter() - t0
    logger.info(f"[TIMING] Wandb initialization completed in {timings['wandb_init'] - timings['dataset_creation']:.3f}s")

    pipeline = build_fixed_axes_pipeline(
        extractor=extractor,
        taxonomy=taxonomy,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        metrics_kwargs=metrics_kwargs,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        include_embeddings=include_embeddings,
        verbose=verbose,
        output_dir=output_dir,
        extraction_cache_dir=extraction_cache_dir,
        metrics_cache_dir=metrics_cache_dir,
        **kwargs,
    )

    timings['pipeline_build'] = time.perf_counter() - t0
    logger.info(f"[TIMING] Pipeline build completed in {timings['pipeline_build'] - timings['wandb_init']:.3f}s")

    timings['setup_total'] = time.perf_counter() - t0
    logger.info(f"[TIMING] Setup total (before pipeline execution): {timings['setup_total']:.3f}s")
    logger.info(f"[TIMING] Label() setup breakdown: {timings}")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    t_pipeline_start = time.perf_counter()
    result_dataset = run_pipeline_smart(pipeline, dataset)
    timings['pipeline_execution'] = time.perf_counter() - t_pipeline_start
    logger.info(f"[TIMING] Pipeline execution completed in {timings['pipeline_execution']:.3f}s")

    # Check for 0 properties before attempting to save
    if len([p for p in result_dataset.properties if p.property_description is not None]) == 0:
        raise RuntimeError("Label pipeline completed with 0 valid properties. Check logs for parsing errors or API issues.")

    clustered_df = result_dataset.to_dataframe(type="clusters", method=method)

    # Save final summary and full dataset if output_dir is provided (same as explain() function)
    if output_dir is not None:
        save_final_summary(result_dataset, clustered_df, result_dataset.model_stats, output_dir, verbose)

        # Also save the full dataset for backward compatibility with compute_metrics_only and other tools
        import pathlib

        output_path = pathlib.Path(output_dir)

        # Save full dataset as JSON
        full_dataset_json_path = output_path / "full_dataset.json"
        result_dataset.save(str(full_dataset_json_path))
        if verbose:
            logger.info(f"  ✓ Saved full dataset: {full_dataset_json_path}")

    # Print analysis summary if verbose
    print_analysis_summary(result_dataset.model_stats, max_behaviors=5)

    return clustered_df, result_dataset.model_stats
