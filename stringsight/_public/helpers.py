"""
Helper functions for the public API.
"""

from typing import Dict, Callable
import asyncio
import pandas as pd
from ..core.data_objects import PropertyDataset
from ..pipeline import Pipeline
from ..logging_config import get_logger

logger = get_logger(__name__)


# ==================== Helper for Event Loop Management ====================

def run_pipeline_smart(pipeline: Pipeline, dataset: PropertyDataset, progress_callback: Callable[[float], None] | None = None) -> PropertyDataset:
    """Run pipeline, handling both sync and async contexts automatically."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # We're in a Jupyter notebook or similar - use nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(pipeline.run(dataset, progress_callback=progress_callback))
        except ImportError:
            raise RuntimeError(
                "Running in a Jupyter notebook or async context requires nest_asyncio. "
                "Install it with: pip install nest_asyncio"
            )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # No event loop - safe to use asyncio.run()
            return asyncio.run(pipeline.run(dataset, progress_callback=progress_callback))
        else:
            raise


# ==================== Utility Functions ====================

def print_analysis_summary(model_stats: Dict[str, pd.DataFrame], max_behaviors: int = 3):
    """Print a quick analysis summary of model behaviors and performance patterns."""
    if not model_stats or "model_cluster_scores" not in model_stats:
        return

    model_cluster_scores = model_stats['model_cluster_scores']

    if model_cluster_scores.empty:
        return

    logger.info("\n" + "="*80)
    logger.info("📊 ANALYSIS SUMMARY")
    logger.info("="*80)

    for model in model_cluster_scores.model.unique():
        model_data = model_cluster_scores[model_cluster_scores.model == model]

        logger.info(f"\n🤖 {model}")
        logger.info("-" * 80)

        # Most common behaviors
        logger.info(f"\n  Most common behaviors:")
        top_behaviors = model_data.sort_values(by='proportion', ascending=False).head(max_behaviors)
        for _, row in top_behaviors.iterrows():
            cluster = row['cluster']
            proportion = row['proportion']
            logger.info(f"    • {cluster} ({proportion:.1%})")

        # Find quality delta columns
        score_delta_columns = [c for c in model_cluster_scores.columns
                             if c.startswith("quality_delta_")
                             and not c.endswith("_ci_lower")
                             and not c.endswith("_ci_upper")
                             and not c.endswith("_ci_mean")
                             and not c.endswith("_significant")]

        if score_delta_columns:
            for col in score_delta_columns:
                metric_name = col.replace("quality_delta_", "")

                # Behaviors leading to worse performance
                logger.info(f"\n  Behaviors leading to worse {metric_name}:")
                worst = model_data.sort_values(by=col, ascending=True).head(max_behaviors)
                for _, row in worst.iterrows():
                    cluster = row['cluster']
                    delta = row[col]
                    if pd.notna(delta):
                        logger.info(f"    • {cluster} ({delta:+.3f})")

                # Behaviors leading to better performance
                logger.info(f"\n  Behaviors leading to better {metric_name}:")
                best = model_data.sort_values(by=col, ascending=False).head(max_behaviors)
                for _, row in best.iterrows():
                    cluster = row['cluster']
                    delta = row[col]
                    if pd.notna(delta):
                        logger.info(f"    • {cluster} ({delta:+.3f})")

    logger.info("\n" + "="*80)


def log_final_results_to_wandb(df: pd.DataFrame, model_stats: Dict[str, pd.DataFrame]):
    """Log final results to wandb."""
    try:
        import wandb

        # Log dataset summary as summary metrics (not regular metrics)
        if wandb.run is not None:
            wandb.run.summary["final_dataset_shape"] = str(df.shape)
            wandb.run.summary["final_total_conversations"] = len(df['question_id'].unique()) if 'question_id' in df.columns else len(df)
            wandb.run.summary["final_total_properties"] = len(df)
            wandb.run.summary["final_unique_models"] = len(df['model'].unique()) if 'model' in df.columns else 0

        # Log clustering results if present
        cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
        if cluster_cols:
            for col in cluster_cols:
                if col.endswith('_id'):
                    cluster_ids = df[col].unique()

                    # Safe conversion to int/float for counting
                    def _safe_to_num(x):
                        try:
                            return float(x)
                        except (ValueError, TypeError):
                            return None

                    valid_ids = [_safe_to_num(c) for c in cluster_ids if pd.notna(c)]
                    valid_ids = [c for c in valid_ids if c is not None]

                    n_clusters = len([c for c in valid_ids if c >= 0])
                    n_outliers = sum(1 for c in valid_ids if c < 0)

                    level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
                    # Log these as summary metrics
                    if wandb.run is not None:
                        wandb.run.summary[f"final_{level}_clusters"] = n_clusters
                        wandb.run.summary[f"final_{level}_outliers"] = n_outliers
                        wandb.run.summary[f"final_{level}_outlier_rate"] = n_outliers / len(df) if len(df) > 0 else 0

        # Handle new dataframe format
        if model_stats and isinstance(model_stats, dict):
            model_scores_df = model_stats.get("model_scores")
            cluster_scores_df = model_stats.get("cluster_scores")
            model_cluster_scores_df = model_stats.get("model_cluster_scores")

            # Log summary statistics
            if wandb.run is not None and model_scores_df is not None:
                wandb.run.summary["final_models_analyzed"] = len(model_scores_df)

                # Log model-level summary statistics
                for _, row in model_scores_df.iterrows():
                    model_name = row.get("model", "unknown")
                    size = row.get("size", 0)

                    wandb.run.summary[f"model_{model_name}_total_size"] = size

                    # Log quality metrics (columns starting with quality_)
                    quality_cols = [col for col in model_scores_df.columns if col.startswith("quality_") and not col.endswith("_ci_lower") and not col.endswith("_ci_upper") and not col.endswith("_ci_mean") and not col.endswith("_significant")]
                    for col in quality_cols:
                        metric_name = col.replace("quality_", "").replace("quality_delta_", "")
                        value = row.get(col)
                        if pd.notna(value):
                            wandb.run.summary[f"model_{model_name}_avg_{metric_name}"] = value

            if wandb.run is not None and cluster_scores_df is not None:
                wandb.run.summary["final_clusters_analyzed"] = len(cluster_scores_df)

            logger.info("✅ Successfully logged metrics to wandb")
            logger.info(f"   • Dataset summary metrics")
            logger.info(f"   • Clustering results")
            logger.info(f"   • Metrics: {len(model_scores_df) if model_scores_df is not None else 0} models, {len(cluster_scores_df) if cluster_scores_df is not None else 0} clusters")
            logger.info(f"   • Summary metrics logged to run summary")
    except ImportError:
        # wandb not installed or not available
        return


def save_final_summary(
    result_dataset: PropertyDataset,
    clustered_df: pd.DataFrame,
    model_stats: Dict[str, pd.DataFrame],
    output_dir: str,
    verbose: bool = False
):
    """Save a final summary of the explain run to a text file."""
    import pathlib

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(f"\nSaving final summary to: {output_path / 'summary.txt'}")

    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("StringSight Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total conversations: {len(clustered_df['question_id'].unique()) if 'question_id' in clustered_df.columns else len(clustered_df)}\n")
        f.write(f"Total properties: {len(clustered_df)}\n")

        # Count models from dataframe
        model_scores_df = model_stats.get("model_scores") if model_stats else None
        num_models = len(model_scores_df) if model_scores_df is not None else 0
        f.write(f"Models analyzed: {num_models}\n")

        # Clustering info
        if 'property_description_cluster_id' in clustered_df.columns:
            n_clusters = len(clustered_df['property_description_cluster_id'].unique())
            f.write(f"Clusters: {n_clusters}\n")

        f.write(f"\nOutput files:\n")
        f.write(f"  - raw_properties.jsonl: Raw LLM responses\n")
        f.write(f"  - extraction_stats.json: Extraction statistics\n")
        f.write(f"  - extraction_samples.jsonl: Sample inputs/outputs\n")
        f.write(f"  - parsed_properties.jsonl: Parsed property objects\n")
        f.write(f"  - parsing_stats.json: Parsing statistics\n")
        f.write(f"  - parsing_failures.jsonl: Failed parsing attempts\n")
        f.write(f"  - validated_properties.jsonl: Validated properties\n")
        f.write(f"  - validation_stats.json: Validation statistics\n")
        f.write(f"  - clustered_results.jsonl: Complete clustered data\n")
        f.write(f"  - embeddings.parquet: Embeddings data\n")
        f.write(f"  - clustered_results_lightweight.jsonl: Data without embeddings\n")
        f.write(f"  - summary_table.jsonl: Clustering summary\n")
        f.write(f"  - model_cluster_scores.json: Per model-cluster combination metrics\n")
        f.write(f"  - cluster_scores.json: Per cluster metrics (aggregated across models)\n")
        f.write(f"  - model_scores.json: Per model metrics (aggregated across clusters)\n")
        f.write(f"  - full_dataset.json: Complete PropertyDataset (JSON format)\n")
        f.write(f"  - full_dataset.parquet: Complete PropertyDataset (parquet format, or .jsonl if mixed data types)\n")

        # Model rankings - extract from dataframes
        f.write(f"\nModel Rankings (by average quality score):\n")
        model_avg_scores = {}

        if model_scores_df is not None and not model_scores_df.empty:
            # Find the first quality column to use for ranking
            quality_cols = [col for col in model_scores_df.columns
                          if col.startswith("quality_")
                          and not col.endswith("_ci_lower")
                          and not col.endswith("_ci_upper")
                          and not col.endswith("_ci_mean")
                          and not col.endswith("_significant")
                          and not col.startswith("quality_delta_")]

            if quality_cols:
                ranking_col = quality_cols[0]  # Use first quality metric for ranking
                for _, row in model_scores_df.iterrows():
                    model_name = row.get("model", "unknown")
                    score = row.get(ranking_col)
                    if pd.notna(score):
                        model_avg_scores[model_name] = score

        if model_avg_scores:
            for i, (model_name, avg_score) in enumerate(sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)):
                f.write(f"  {i+1}. {model_name}: {avg_score:.3f}\n")
        else:
            f.write(f"  (No quality scores available)\n")

    if verbose:
        logger.info(f"  ✓ Saved final summary: {summary_path}")
