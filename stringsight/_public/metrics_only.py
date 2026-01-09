"""
Utility function for recomputing metrics on existing pipeline results.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from ..core.data_objects import PropertyDataset
from ..pipeline import Pipeline
from ..logging_config import get_logger
from .helpers import run_pipeline_smart, save_final_summary

logger = get_logger(__name__)


def compute_metrics_only(
    input_path: str,
    method: str = "single_model",
    output_dir: str | None = None,
    metrics_kwargs: Dict[str, Any | None] | None = None,
    use_wandb: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run only the metrics computation stage on existing pipeline results.

    This function loads existing pipeline results (from extraction and clustering stages)
    and runs only the metrics computation stage. Useful for:
    - Recomputing metrics with different parameters
    - Running metrics on results from previous pipeline runs
    - Debugging metrics computation without re-running the full pipeline

    Args:
        input_path: Path to existing pipeline results (file or directory)
        method: "single_model" or "side_by_side"
        output_dir: Directory to save metrics results (optional)
        metrics_kwargs: Additional arguments for metrics computation
        use_wandb: Whether to enable wandb logging
        verbose: Whether to print verbose output

    Returns:
        Tuple of (clustered_df, model_stats)

    Example:
        >>> from stringsight import compute_metrics_only
        >>>
        >>> # Run metrics on existing pipeline results
        >>> clustered_df, model_stats = compute_metrics_only(
        ...     input_path="results/previous_run/full_dataset.json",
        ...     method="single_model",
        ...     output_dir="results/metrics_only"
        ... )
        >>>
        >>> # Or run on a directory containing pipeline outputs
        >>> clustered_df, model_stats = compute_metrics_only(
        ...     input_path="results/previous_run/",
        ...     method="side_by_side"
        ... )
    """
    from ..metrics import get_metrics

    # Align environment with wandb toggle early to avoid accidental logging on import
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)

    path = Path(input_path)

    # Load existing dataset
    if path.is_dir():
        # Try to load from a directory containing pipeline outputs
        possible_files = [
            path / "full_dataset.json",
            path / "full_dataset.parquet",
            path / "clustered_results.parquet",
            path / "dataset.json",
            path / "dataset.parquet"
        ]

        for file_path in possible_files:
            if file_path.exists():
                if verbose:
                    logger.info(f"Loading from: {file_path}")
                dataset = PropertyDataset.load(str(file_path))
                break
        else:
            raise FileNotFoundError(f"No recognizable dataset file found in {path}")

    elif path.is_file():
        # Load from a specific file
        if verbose:
            logger.info(f"Loading from: {path}")
        dataset = PropertyDataset.load(str(path))

    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    # Verify we have the required data for metrics
    if not dataset.clusters:
        raise ValueError("No clusters found in the dataset. Metrics computation requires clustered data.")

    if not dataset.properties:
        raise ValueError("No properties found in the dataset. Metrics computation requires extracted properties.")

    if verbose:
        logger.info(f"Loaded dataset with:")
        logger.info(f"  - {len(dataset.conversations)} conversations")
        logger.info(f"  - {len(dataset.properties)} properties")
        logger.info(f"  - {len(dataset.clusters)} clusters")
        logger.info(f"  - Models: {dataset.all_models}")

        # Count unique models from conversations for verification
        unique_models = set()
        for conv in dataset.conversations:
            if isinstance(conv.model, list):
                unique_models.update(conv.model)
            else:
                unique_models.add(conv.model)

        logger.info(f"  - Total unique models: {len(unique_models)}")
        if len(unique_models) <= 20:
            model_list = sorted(list(unique_models))
            logger.info(f"  - Model names: {', '.join(model_list)}")
        logger.info("")

    # Create metrics stage
    metrics_config = {
        'method': method,
        'use_wandb': use_wandb,
        'verbose': verbose,
        **(metrics_kwargs or {})
    }

    # Add output directory if provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_config['output_dir'] = str(output_path)

    # Initialize wandb if enabled
    if use_wandb:
        try:
            import wandb
            import os

            # Try to get input filename from the input path
            input_filename = "unknown_dataset"
            if path.is_file():
                input_filename = path.name
            elif path.is_dir():
                # Try to find a recognizable dataset file in the directory
                possible_files = [
                    path / "full_dataset.json",
                    path / "full_dataset.parquet",
                    path / "clustered_results.parquet",
                    path / "dataset.json",
                    path / "dataset.parquet"
                ]

                for file_path in possible_files:
                    if file_path.exists():
                        input_filename = file_path.name
                        break
                else:
                    # If no recognizable file found, use the directory name
                    input_filename = path.name

            # Clean the filename for wandb (remove extension, replace spaces/special chars)
            if isinstance(input_filename, str):
                # Remove file extension and clean up the name
                input_filename = os.path.splitext(os.path.basename(input_filename))[0]
                # Replace spaces and special characters with underscores
                input_filename = input_filename.replace(' ', '_').replace('-', '_')
                # Remove any remaining special characters
                import re
                input_filename = re.sub(r'[^a-zA-Z0-9_]', '', input_filename)

            wandb_run_name = os.path.basename(os.path.normpath(output_dir)) if output_dir else f"{input_filename}_metrics_only"

            wandb.init(
                project="StringSight",
                name=wandb_run_name,
                config={
                    "method": method,
                    "input_path": str(path),
                    "output_dir": output_dir,
                    "metrics_kwargs": metrics_kwargs,
                },
                reinit=False  # Don't reinitialize if already exists
            )
        except ImportError:
            # wandb not installed or not available
            use_wandb = False

    metrics_stage = get_metrics(method, **{k: v for k, v in metrics_config.items() if k != 'method'})

    # Create a minimal pipeline with just the metrics stage
    pipeline = Pipeline("Metrics-Only", [metrics_stage])

    # Run metrics computation
    if verbose:
        logger.info("\n" + "="*60)
        logger.info("COMPUTING METRICS")
        logger.info("="*60)

    result_dataset = run_pipeline_smart(pipeline, dataset)

    # Convert back to DataFrame format
    clustered_df = result_dataset.to_dataframe()
    model_stats = result_dataset.model_stats

    # Save results if output_dir is provided
    if output_dir:
        if verbose:
            logger.info(f"\nSaving results to: {output_dir}")

        # Use the same saving mechanism as the full pipeline
        save_final_summary(
            result_dataset=result_dataset,
            clustered_df=clustered_df,
            model_stats=model_stats,
            output_dir=output_dir,
            verbose=verbose
        )

        # Print summary
        logger.info(f"\n📊 Metrics Summary:")
        logger.info(f"  - Models analyzed: {len(model_stats)}")

        # Handle new functional metrics format
        if model_stats and "functional_metrics" in model_stats:
            functional_metrics = model_stats["functional_metrics"]
            model_scores = functional_metrics.get("model_scores", {})
            cluster_scores = functional_metrics.get("cluster_scores", {})

            logger.info(f"  - Functional metrics computed:")
            logger.info(f"    • Model scores: {len(model_scores)} models")
            logger.info(f"    • Cluster scores: {len(cluster_scores)} clusters")

            # Print model-level summary
            for model_name, model_data in model_scores.items():
                if isinstance(model_data, dict):
                    size = model_data.get("size", 0)
                    quality = model_data.get("quality", {})
                    logger.info(f"    • {model_name}: {size} conversations")
                    if quality:
                        for metric_name, metric_value in quality.items():
                            if isinstance(metric_value, (int, float)):
                                logger.info(f"      - {metric_name}: {metric_value:.3f}")

        # Handle legacy format for backward compatibility
        else:
            for model_name, stats in model_stats.items():
                if "fine" in stats:
                    logger.info(f"  - {model_name}: {len(stats['fine'])} fine clusters")
                if "coarse" in stats:
                    logger.info(f"    {len(stats['coarse'])} coarse clusters")

    return clustered_df, model_stats
