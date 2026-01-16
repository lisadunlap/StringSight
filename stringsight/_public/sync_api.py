"""
Synchronous public API functions - the main entry points for StringSight.
"""

from typing import Dict, List, Any, Callable, Union, Tuple
import pandas as pd
from ..core.data_objects import PropertyDataset
from ..pipeline import Pipeline, PipelineBuilder
from ..prompts import get_system_prompt
from ..utils.validation import validate_openai_api_key
from ..logging_config import get_logger
from ..constants import DEFAULT_MAX_WORKERS
from .helpers import run_pipeline_smart, print_analysis_summary, log_final_results_to_wandb, save_final_summary
from .pipeline_builders import build_default_pipeline

logger = get_logger(__name__)


def extract_properties_only(
    df: pd.DataFrame,
    *,
    method: str = "single_model",
    system_prompt: str | None = None,
    task_description: str | None = None,
    fail_on_empty_properties: bool = True,
    # Data preparation
    score_columns: List[str] | None = None,
    sample_size: int | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    # Column mapping parameters
    prompt_column: str = "prompt",
    model_column: str | None = None,
    model_response_column: str | None = None,
    question_id_column: str | None = None,
    model_a_column: str | None = None,
    model_b_column: str | None = None,
    model_a_response_column: str | None = None,
    model_b_response_column: str | None = None,
    # Extraction parameters
    model_name: str = "gpt-4.1",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    include_scores_in_prompt: bool = False,
    # Logging & output
    use_wandb: bool = True,
    wandb_project: str | None = None,
    verbose: bool = False,
    output_dir: str | None = None,
    # Caching
    extraction_cache_dir: str | None = None,
    return_debug: bool = False,
) -> PropertyDataset | tuple[PropertyDataset, list[dict[str, Any]]]:
    """Run only the extraction → parsing → validation stages and return a PropertyDataset.

    Args:
        df: Input conversations dataframe (single_model or side_by_side format)
        method: "single_model" | "side_by_side"
        system_prompt: Explicit system prompt text or a short prompt name from stringsight.prompts
        task_description: Optional task-aware description (used only if the chosen prompt has {task_description})
        fail_on_empty_properties: If True, raise a RuntimeError when 0 valid properties remain after validation.
            If False, return an empty PropertyDataset.properties list.
        score_columns: Optional list of column names containing score metrics to convert to dict format
        sample_size: Optional number of rows to sample from the dataset before processing
        model_a: For side_by_side method with tidy data, specifies first model to select
        model_b: For side_by_side method with tidy data, specifies second model to select
        prompt_column: Name of the prompt column in your dataframe (default: "prompt")
        model_column: Name of the model column for single_model (default: "model")
        model_response_column: Name of the model response column for single_model (default: "model_response")
        question_id_column: Name of the question_id column (default: "question_id" if column exists)
        model_a_column: Name of the model_a column for side_by_side (default: "model_a")
        model_b_column: Name of the model_b column for side_by_side (default: "model_b")
        model_a_response_column: Name of the model_a_response column for side_by_side (default: "model_a_response")
        model_b_response_column: Name of the model_b_response column for side_by_side (default: "model_b_response")
        model_name, temperature, top_p, max_tokens, max_workers: LLM config for extraction
        include_scores_in_prompt: Whether to include any provided score fields in the prompt context
        use_wandb, wandb_project, verbose: Logging configuration
        output_dir: If provided, stages will auto-save their artefacts to this directory
        extraction_cache_dir: Optional cache directory for extractor

    Returns:
        PropertyDataset containing parsed Property objects (no clustering or metrics).
    """
    # Validate OpenAI API key is set if using GPT models
    validate_openai_api_key(
        model_name=model_name
    )

    # Resolve system prompt using centralized resolver
    system_prompt = get_system_prompt(method, system_prompt, task_description)

    if verbose:
        logger.info("\n" + "="*80)
        logger.info("SYSTEM PROMPT")
        logger.info("="*80)
        logger.info(system_prompt)
        logger.info("="*80 + "\n")
    if len(system_prompt) < 50:
        raise ValueError("System prompt is too short. Please provide a longer system prompt.")

    # Preprocess data: handle score_columns, sampling, tidy→side_by_side conversion, column mapping
    from ..core.preprocessing import validate_and_prepare_dataframe
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

    # Prepare dataset
    dataset = PropertyDataset.from_dataframe(df, method=method)

    # Align env with wandb toggle early
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)

    # Build a minimal pipeline: extractor → parser → validator
    from ..extractors import get_extractor
    from ..postprocess import LLMJsonParser, PropertyValidator

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
        "output_dir": output_dir,
        **({"cache_dir": extraction_cache_dir} if extraction_cache_dir else {}),
        **common_cfg,
    }

    extractor = get_extractor(**extractor_kwargs)  # type: ignore[arg-type]
    # Do not fail the whole run on parsing errors – collect failures and drop those rows
    parser = LLMJsonParser(fail_fast=False, output_dir=output_dir, **common_cfg)  # type: ignore[arg-type]
    validator = PropertyValidator(output_dir=output_dir, fail_on_empty=fail_on_empty_properties, **common_cfg)  # type: ignore[arg-type]

    pipeline = PipelineBuilder(name=f"StringSight-extract-{method}") \
        .extract_properties(extractor) \
        .parse_properties(parser) \
        .add_stage(validator) \
        .configure(output_dir=output_dir, **common_cfg) \
        .build()

    result_dataset = run_pipeline_smart(pipeline, dataset)
    if return_debug:
        try:
            failures = parser.get_parsing_failures()
        except Exception:
            failures = []
        return result_dataset, failures
    return result_dataset


def explain(
    df: pd.DataFrame,
    method: str = "single_model",
    system_prompt: str | None = None,
    prompt_builder: Callable[[pd.Series, str], str] | None = None,
    task_description: str | None = None,
    *,
    # Data preparation
    sample_size: int | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    score_columns: List[str] | None = None,
    # Column mapping parameters
    prompt_column: str = "prompt",
    model_column: str | None = None,
    model_response_column: str | None = None,
    question_id_column: str | None = None,
    model_a_column: str | None = None,
    model_b_column: str | None = None,
    model_a_response_column: str | None = None,
    model_b_response_column: str | None = None,
    # Extraction parameters
    model_name: str = "gpt-4.1",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = DEFAULT_MAX_WORKERS,
    include_scores_in_prompt: bool = False,
    # Prompt expansion parameters
    prompt_expansion: bool = False,
    expansion_num_traces: int = 5,
    expansion_model: str = "gpt-4.1",
    # Dynamic prompt generation parameters
    use_dynamic_prompts: bool = True,
    dynamic_prompt_samples: int = 5,
    dynamic_prompt_model: str | None = None,
    # Clustering parameters
    clusterer: Union[str, Any] = "hdbscan",
    min_cluster_size: int | None = 5,
    embedding_model: str = "text-embedding-3-large",
    prettify_labels: bool = False,
    assign_outliers: bool = False,
    summary_model: str = "gpt-4.1",
    cluster_assignment_model: str = "gpt-4.1-mini",
    # Metrics parameters
    metrics_kwargs: Dict[str, Any | None] | None = None,
    # Caching & logging
    use_wandb: bool = True,
    wandb_project: str | None = None,
    include_embeddings: bool = False,
    verbose: bool = False,
    # Output parameters
    output_dir: str | None = None,
    # Pipeline configuration
    custom_pipeline: Pipeline | None = None,
    # Cache configuration
    extraction_cache_dir: str | None = None,
    clustering_cache_dir: str | None = None,
    metrics_cache_dir: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Explain model behavior patterns from conversation data.

    This is the main entry point for StringSight. It takes a DataFrame of conversations
    and returns the same data with extracted properties and clusters.

    Args:
        df: DataFrame with conversation data
        method: "side_by_side" or "single_model"
        system_prompt: System prompt for property extraction (if None, will be auto-determined)
        prompt_builder: Optional custom prompt builder function
        task_description: Optional description of the task; when provided with
            method="single_model" and no explicit system_prompt, a task-aware
            system prompt is constructed from single_model_system_prompt_custom.
            If prompt_expansion=True, this description will be expanded using
            example traces before being used in prompts.

        # Data preparation
        sample_size: Optional number of rows to sample from the dataset before processing.
                    If None, uses the entire dataset. For single_model method with balanced
                    datasets (each prompt answered by all models), automatically samples prompts
                    evenly across models. Otherwise falls back to row-level sampling.
        model_a: For side_by_side method with tidy data, specifies first model to select
        model_b: For side_by_side method with tidy data, specifies second model to select
        score_columns: Optional list of column names containing score metrics. Instead of
                    providing scores as a dictionary in a 'score' column, you can specify
                    separate columns for each metric. For single_model: columns should be
                    named like 'accuracy', 'helpfulness'. For side_by_side: columns should
                    be named like 'accuracy_a', 'accuracy_b', 'helpfulness_a', 'helpfulness_b'.
                    If provided, these columns will be converted to the expected score dict format.

        # Column mapping parameters
        prompt_column: Name of the prompt column in your dataframe (default: "prompt")
        model_column: Name of the model column for single_model (default: "model")
        model_response_column: Name of the model response column for single_model (default: "model_response")
        question_id_column: Name of the question_id column (default: "question_id" if column exists)
        model_a_column: Name of the model_a column for side_by_side (default: "model_a")
        model_b_column: Name of the model_b column for side_by_side (default: "model_b")
        model_a_response_column: Name of the model_a_response column for side_by_side (default: "model_a_response")
        model_b_response_column: Name of the model_b_response column for side_by_side (default: "model_b_response")

        # Extraction parameters
        model_name: LLM model for property extraction
        temperature: Temperature for LLM
        top_p: Top-p for LLM
        max_tokens: Max tokens for LLM
        max_workers: Max parallel workers for API calls

        # Prompt expansion parameters
        prompt_expansion: If True, expand task_description using example traces
            before extraction (default: False)
        expansion_num_traces: Number of traces to sample for expansion (default: 5)
        expansion_model: LLM model to use for expansion (default: "gpt-4.1")

        # Clustering parameters
        clusterer: Clustering method ("hdbscan", "hdbscan_native") or PipelineStage
        min_cluster_size: Minimum cluster size
        embedding_model: Embedding model ("openai" or sentence-transformer model)
        assign_outliers: Whether to assign outliers to nearest clusters
        summary_model: LLM model for generating cluster summaries (default: "gpt-4.1")
        cluster_assignment_model: LLM model for assigning outliers to clusters (default: "gpt-4.1-mini")

        # Metrics parameters
        metrics_kwargs: Additional metrics configuration

        # Caching & logging
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        include_embeddings: Whether to include embeddings in output
        verbose: Whether to print progress

        # Output parameters
        output_dir: Directory to save results (optional). If provided, saves:
                   - clustered_results.parquet: DataFrame with all results
                   - full_dataset.json: Complete PropertyDataset (JSON format)
                   - full_dataset.parquet: Complete PropertyDataset (parquet format)
                   - model_stats.json: Model statistics and rankings
                   - summary.txt: Human-readable summary

        # Pipeline configuration
        custom_pipeline: Custom pipeline to use instead of default
        **kwargs: Additional configuration options

    Returns:
        Tuple of (clustered_df, model_stats)
        - clustered_df: Original DataFrame with added property and cluster columns
        - model_stats: Dictionary containing three DataFrames:
            - "model_cluster_scores": Per model-cluster metrics (size, proportion, quality, etc.)
            - "cluster_scores": Per cluster aggregated metrics (across all models)
            - "model_scores": Per model aggregated metrics (across all clusters)

    Notes on input format:
        - For method="single_model": expect columns [question_id, prompt, model, model_response, (optional) score]
        - For method="side_by_side": expect columns [question_id, prompt, model_a, model_b, model_a_response, model_b_response]
        - Alternatively, for method="side_by_side" you may pass tidy single-model-like data
          (columns [prompt, model, model_response] and optionally question_id) and specify
          `model_a` and `model_b` parameters. The function will select these two
          models and convert the input to the expected side-by-side schema.

    Example:
        >>> import pandas as pd
        >>> from stringsight import explain
        >>>
        >>> # Load your conversation data
        >>> df = pd.read_csv("conversations.csv")
        >>>
        >>> # Explain model behavior and save results
        >>> clustered_df, model_stats = explain(
        ...     df,
        ...     method="side_by_side",
        ...     min_cluster_size=5,
        ...     output_dir="results/"  # Automatically saves results
        ... )
        >>>
        >>> # Explore the results
        >>> print(clustered_df.columns)
        >>> print(model_stats.keys())
    """

    # Validate OpenAI API key is set if using GPT models
    validate_openai_api_key(
        model_name=model_name,
        embedding_model=embedding_model,
        **kwargs
    )

    # Preprocess data: handle score_columns, sampling, tidy→side_by_side conversion, column mapping
    from ..core.preprocessing import validate_and_prepare_dataframe
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

    # Create PropertyDataset from input DataFrame (needed for prompt generation)
    dataset = PropertyDataset.from_dataframe(df, method=method)

    # Prompt generation: dynamic or static
    # Dynamic prompts use task description + sampled conversations to generate custom prompts
    # Old prompt_expansion parameter is deprecated but still supported for backward compatibility
    if prompt_expansion and not use_dynamic_prompts:
        # Legacy behavior: only expand task description (deprecated)
        from ..prompts.expansion.trace_based import expand_task_description
        from ..formatters.traces import format_single_trace_from_row, format_side_by_side_trace_from_row

        if task_description is None:
            raise ValueError(
                "task_description must be provided when prompt_expansion=True and use_dynamic_prompts=False."
            )

        if verbose:
            logger.info("[DEPRECATED] Using old prompt_expansion. Consider use_dynamic_prompts instead.")
            logger.info("Expanding task description using example traces...")

        # Convert dataframe rows to traces
        traces = []
        for idx, row in df.iterrows():
            if method == "single_model":
                trace = format_single_trace_from_row(row)
            else:  # side_by_side
                trace = format_side_by_side_trace_from_row(row)
            traces.append(trace)

        # Expand task description
        expanded_description = expand_task_description(
            task_description=task_description,
            traces=traces,
            model=expansion_model,
            num_traces=expansion_num_traces,
        )

        if verbose:
            logger.info(f"Original task description length: {len(task_description)}")
            logger.info(f"Expanded task description length: {len(expanded_description)}")

        # Use expanded description
        task_description = expanded_description
        system_prompt = get_system_prompt(method, system_prompt, task_description)
        custom_clustering_prompts = None
    else:
        # New behavior: dynamic prompt generation (or static if disabled)
        from ..prompt_generation import generate_prompts

        system_prompt, custom_clustering_prompts, _ = generate_prompts(
            task_description=task_description,
            dataset=dataset,
            method=method,
            use_dynamic_prompts=use_dynamic_prompts,
            dynamic_prompt_samples=dynamic_prompt_samples,
            model=dynamic_prompt_model or model_name,
            system_prompt_override=system_prompt,
            output_dir=output_dir,
        )

    # Print the system prompt for verification
    if verbose:
        logger.info("\n" + "="*80)
        logger.info("SYSTEM PROMPT")
        logger.info("="*80)
        logger.info(system_prompt)
        logger.info("="*80 + "\n")
    if len(system_prompt) < 50:
        raise ValueError("System prompt is too short. Please provide a longer system prompt.")

    print(f"df length: {len(df)}")

    # Dataset already created above for prompt generation
    # Print initial dataset information
    if verbose:
        logger.info(f"\n📋 Initial dataset summary:")
        logger.info(f"   • Conversations: {len(dataset.conversations)}")
        logger.info(f"   • Models: {len(dataset.all_models)}")
        if len(dataset.all_models) <= 20:
            logger.info(f"   • Model names: {', '.join(sorted(dataset.all_models))}")
        logger.info("")

    # 2️⃣  Initialize wandb if enabled (and explicitly disable via env when off)
    # Ensure environment flag aligns with the provided setting to prevent
    # accidental logging by submodules that import wandb directly.
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)

    # 2️⃣  Initialize wandb if enabled
    # Create run name based on input filename if available
    if use_wandb:
        try:
            import wandb
            import os

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
                import re
                input_filename = re.sub(r'[^a-zA-Z0-9_]', '', input_filename)

            wandb_run_name = os.path.basename(os.path.normpath(output_dir)) if output_dir else f"{input_filename}_{method}"

            wandb.init(
                project=wandb_project or "StringSight",
                name=wandb_run_name,
                config={
                    "method": method,
                    "system_prompt": system_prompt,
                    "model_name": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "max_workers": max_workers,
                    "clusterer": clusterer,
                    "min_cluster_size": min_cluster_size,
                    "embedding_model": embedding_model,
                    "assign_outliers": assign_outliers,
                    "include_embeddings": include_embeddings,
                    "output_dir": output_dir,
                },
                reinit=False  # Don't reinitialize if already exists
            )
        except (ImportError, TypeError, Exception) as e:
            # wandb not installed, has corrupted package metadata, or initialization failed
            logger.warning(f"Wandb initialization failed: {e}. Disabling wandb tracking.")
            use_wandb = False
            _os.environ["WANDB_DISABLED"] = "true"

    # Use custom pipeline if provided, otherwise build default pipeline
    if custom_pipeline is not None:
        pipeline = custom_pipeline
        # Ensure the custom pipeline uses the same wandb configuration
        if hasattr(pipeline, 'use_wandb'):
            pipeline.use_wandb = use_wandb
            pipeline.wandb_project = wandb_project or "StringSight"
            if use_wandb:
                pipeline._wandb_ok = True  # Mark that wandb is already initialized
    else:
        pipeline = build_default_pipeline(
            method=method,
            system_prompt=system_prompt,
            prompt_builder=prompt_builder,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_workers=max_workers,
            include_scores_in_prompt=include_scores_in_prompt,
            clusterer=clusterer,
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            assign_outliers=assign_outliers,
            prettify_labels=prettify_labels,
            summary_model=summary_model,
            cluster_assignment_model=cluster_assignment_model,
            metrics_kwargs=metrics_kwargs or {},  # type: ignore[arg-type]
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            include_embeddings=include_embeddings,
            verbose=verbose,
            extraction_cache_dir=extraction_cache_dir,
            clustering_cache_dir=clustering_cache_dir,
            metrics_cache_dir=metrics_cache_dir,
            output_dir=output_dir,
            custom_clustering_prompts=custom_clustering_prompts,
            **kwargs
        )

    # 4️⃣  Execute pipeline
    result_dataset = run_pipeline_smart(pipeline, dataset, progress_callback=progress_callback)

       # Check for 0 properties before attempting to save
    if len([p for p in result_dataset.properties if p.property_description is not None]) == 0:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "ERROR: Pipeline completed with 0 valid properties!\n"
            "="*60 + "\n"
            "This indicates that all property extraction attempts failed.\n"
            "Common causes:\n\n"
            "1. JSON PARSING FAILURES:\n"
            "   - LLM returning natural language instead of JSON\n"
            "   - Check logs above for 'Failed to parse JSON' errors\n\n"
            "2. SYSTEM PROMPT MISMATCH:\n"
            "   - Current system_prompt may not suit your data format\n"
            "   - Try a different system_prompt parameter\n\n"
            "3. API/MODEL ISSUES:\n"
            "   - OpenAI API key invalid or quota exceeded\n"
            "   - Model configuration problems\n\n"
            "Cannot save results with 0 properties.\n"
            "="*60
        )

    # Convert back to DataFrame format
    clustered_df = result_dataset.to_dataframe(type="all", method=method)
    model_stats = result_dataset.model_stats

    # Save final summary if output_dir is provided
    if output_dir is not None:
        save_final_summary(result_dataset, clustered_df, model_stats, output_dir, verbose)

        # Also save the full dataset for backward compatibility with compute_metrics_only and other tools
        import pathlib

        output_path = pathlib.Path(output_dir)

        # Save full dataset as JSON
        full_dataset_json_path = output_path / "full_dataset.json"
        result_dataset.save(str(full_dataset_json_path))
        if verbose:
            logger.info(f"  ✓ Saved full dataset: {full_dataset_json_path}")

    # Log accumulated summary metrics from pipeline stages
    if use_wandb and hasattr(pipeline, 'log_final_summary'):
        pipeline.log_final_summary()

    # Log final results to wandb if enabled
    if use_wandb:
        try:
            import wandb
            log_final_results_to_wandb(clustered_df, model_stats)
        except ImportError:
            # wandb not installed or not available
            use_wandb = False

    # Print analysis summary if verbose
    print_analysis_summary(model_stats, max_behaviors=5)

    return clustered_df, model_stats
