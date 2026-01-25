"""
persistence.py

Save/load utilities for clustered results and WandB logging.
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from stringsight.logging_config import get_logger

logger = get_logger(__name__)


def load_clustered_results(parquet_path):
    """Load previously clustered results from parquet file."""
    df = pd.read_parquet(parquet_path)

    logger.info(f"Loaded {len(df)} rows from {parquet_path}")
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower() or 'topic' in col.lower()]
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]

    if cluster_cols:
        logger.info(f"Cluster columns: {cluster_cols}")
    if embedding_cols:
        logger.info(f"Embedding columns: {embedding_cols}")

    return df


def save_clustered_results(df, base_filename, include_embeddings=True, config=None, output_dir=None):
    """Save clustered results in multiple formats and optionally log to wandb.

    Args:
        df: DataFrame with clustered results
        base_filename: Base name for output files
        include_embeddings: Whether to include embeddings in output
        config: ClusterConfig object for wandb logging
        output_dir: Output directory (if None, uses cluster_results/{base_filename})
    """

    # Determine output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_dir = output_dir
    else:
        os.makedirs(f"cluster_results/{base_filename}", exist_ok=True)
        save_dir = f"cluster_results/{base_filename}"

    # Handle JSON serialization - preserve dictionaries for score columns
    df = df.copy()
    score_columns = [col for col in df.columns if 'score' in col.lower()]

    # Debug print to see what score columns we're finding
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Available columns: {list(df.columns)}")
        logger.debug(f"Detected score columns: {score_columns}")

    for col in df.columns:
        if df[col].dtype == 'object':
            # Preserve dictionary columns (like scores) - don't convert to string
            if col in score_columns:
                # Keep as-is for JSON serialization - pandas.to_json can handle dicts
                continue
            else:
                # Convert other object columns to strings for safety
                df[col] = df[col].astype(str)

    # 1. Save clustered results as JSON (preserves all data structures)
    df.to_json(f"{save_dir}/clustered_results.jsonl", orient='records', lines=True)
    logger.info(f"Saved clustered results (JSON): {save_dir}/clustered_results.jsonl")

    # 2. Save embeddings separately if they exist
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if embedding_cols and include_embeddings:
        # Create embeddings-only DataFrame
        embedding_df = df[embedding_cols].copy()

        # Add key columns for reference
        key_cols = ['property_description', 'question_id', 'model', 'property_description_cluster_label']
        for col in key_cols:
            if col in df.columns:
                embedding_df[col] = df[col]

        # Save embeddings as parquet (more efficient for large arrays)
        embeddings_path = os.path.join(save_dir, "embeddings.parquet")
        embedding_df.to_parquet(embeddings_path, compression='snappy')
        logger.info(f"Saved embeddings: {embeddings_path}")

        # Also save as JSON for compatibility
        embedding_df.to_json(f"{save_dir}/embeddings.jsonl", orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved embeddings (JSON): {save_dir}/embeddings.jsonl")

    # 3. Save lightweight version without embeddings
    df_light = df.drop(columns=embedding_cols) if embedding_cols else df

    # Save lightweight as json
    df_light.to_json(f"{save_dir}/clustered_results_lightweight.jsonl", orient='records', lines=True)
    logger.info(f"Saved lightweight results (JSON): {save_dir}/clustered_results_lightweight.jsonl")

    # 4. Create and save summary table
    summary_table = create_summary_table(df_light, config)
    summary_table.to_json(f"{save_dir}/summary_table.jsonl", orient='records', lines=True)
    logger.info(f"Saved summary table: {save_dir}/summary_table.jsonl")

    # 6. Log to wandb if enabled
    if config and config.use_wandb:
        log_results_to_wandb(df_light, f"{save_dir}/clustered_results_lightweight.jsonl", base_filename, config)
        logger.info(f"Logged results to wandb")

    return {
        'clustered_json': f"{save_dir}/clustered_results.jsonl",
        'embeddings_parquet': f"{save_dir}/embeddings.parquet" if embedding_cols and include_embeddings else None,
        'summary_table': f"{save_dir}/summary_table.jsonl"
    }


def log_results_to_wandb(df_light, light_json_path, base_filename, config):
    """Log clustering results to wandb."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb is required for logging but is not installed. Install with: pip install wandb")
        return

    if not wandb.run:
        logger.warning("⚠️ wandb not initialized, skipping logging")
        return

    logger.info("📊 Logging results to wandb...")

    # Log the lightweight CSV file
    artifact = wandb.Artifact(
        name=f"{base_filename}_clustered_data",
        type="clustered_dataset",
        description=f"Clustered dataset without embeddings - {base_filename}"
    )
    artifact.add_file(light_json_path)
    wandb.log_artifact(artifact)

    # Log the actual clustering results as a table
    # Find the original column that was clustered
    original_col = None
    for col in df_light.columns:
        if not any(suffix in col for suffix in ['_cluster', '_embedding']):
            # This is likely the original column
            original_col = col
            break

    if original_col:
        # Create a table with the key clustering results
        cluster_cols = [col for col in df_light.columns if 'cluster' in col.lower()]
        table_cols = [original_col] + cluster_cols

        # Sample the data if it's too large (wandb has limits)
        sample_size = min(100, len(df_light))
        if len(df_light) > sample_size:
            df_sample = df_light[table_cols].sample(n=sample_size, random_state=42)
            logger.info(f"📋 Logging sample of {sample_size} rows (out of {len(df_light)} total)")
        else:
            df_sample = df_light[table_cols]
            logger.info(f"📋 Logging all {len(df_sample)} rows")

        # Convert to string to handle any non-serializable data
        df_sample_str = df_sample.astype(str)
        wandb.log({f"{base_filename}_clustering_results": wandb.Table(dataframe=df_sample_str)})

    # Calculate clustering metrics
    cluster_cols = [col for col in df_light.columns if 'cluster_id' in col.lower()]
    metrics = {"clustering_dataset_size": len(df_light)}

    for col in cluster_cols:
        cluster_ids = df_light[col].values
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_outliers = list(cluster_ids).count(-1)

        level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
        metrics[f"clustering_{level}_clusters"] = n_clusters
        metrics[f"clustering_{level}_outliers"] = n_outliers
        metrics[f"clustering_{level}_outlier_rate"] = n_outliers / len(cluster_ids) if len(cluster_ids) > 0 else 0

        # Calculate cluster size distribution
        cluster_sizes = [list(cluster_ids).count(cid) for cid in set(cluster_ids) if cid >= 0]
        if cluster_sizes:
            metrics[f"clustering_{level}_avg_cluster_size"] = np.mean(cluster_sizes)
            metrics[f"clustering_{level}_min_cluster_size"] = min(cluster_sizes)
            metrics[f"clustering_{level}_max_cluster_size"] = max(cluster_sizes)

    # Log clustering configuration
    config_dict = {
        "clustering_min_cluster_size": config.min_cluster_size,
        "clustering_embedding_model": config.embedding_model,
        "clustering_assign_outliers": config.assign_outliers,
        "clustering_disable_dim_reduction": config.disable_dim_reduction,
        "clustering_min_samples": config.min_samples,
        "clustering_cluster_selection_epsilon": config.cluster_selection_epsilon
    }

    # Log all metrics as summary metrics (not regular metrics)
    # Note: This function doesn't have access to WandbMixin, so we'll log directly to wandb.run.summary
    all_metrics = {**metrics, **config_dict}
    for key, value in all_metrics.items():
        wandb.run.summary[key] = value

    logger.info(f"✅ Logged clustering results to wandb")
    logger.info(f"   - Dataset artifact: {base_filename}_clustered_data")
    logger.info(f"   - Clustering results table: {base_filename}_clustering_results")
    logger.info(f"   - Summary metrics: {list(all_metrics.keys())}")


def initialize_wandb(config, method_name, input_file):
    """Initialize wandb logging if enabled."""
    if not config.use_wandb:
        return

    try:
        import wandb
    except ImportError:
        logger.error("wandb is required for logging but is not installed. Install with: pip install wandb")
        return

    logger.info("🔧 Initializing wandb...")

    # Create run name if not provided
    run_name = config.wandb_run_name
    if not run_name:
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        run_name = f"{input_basename}_{method_name}_clustering"

    # Initialize wandb
    wandb.init(
        project=config.wandb_project or "hierarchical_clustering",
        entity=config.wandb_entity,
        name=run_name,
        config={
            "method": method_name,
            "input_file": input_file,
            "min_cluster_size": config.min_cluster_size,
            "embedding_model": config.embedding_model,
            "assign_outliers": config.assign_outliers,
            "disable_dim_reduction": config.disable_dim_reduction,
            "min_samples": config.min_samples,
            "cluster_selection_epsilon": config.cluster_selection_epsilon
        }
    )

    logger.info(f"✅ Initialized wandb run: {run_name}")


def load_precomputed_embeddings(embeddings_path, verbose=True):
    """Load precomputed embeddings from various file formats."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if verbose:
        logger.info(f"Loading precomputed embeddings from {embeddings_path}...")

    if embeddings_path.endswith('.pkl'):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                # Check if it's a cache file with 'embeddings' key
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                    if verbose:
                        logger.info(f"Loaded {len(embeddings)} embeddings from cache file")
                else:
                    # Assume it's a direct mapping of values to embeddings
                    embeddings = data
                    if verbose:
                        logger.info(f"Loaded {len(embeddings)} embeddings from mapping file")
            else:
                # Assume it's a direct array/list of embeddings
                embeddings = data
                if verbose:
                    logger.info(f"Loaded {len(embeddings)} embeddings from array file")

    elif embeddings_path.endswith('.npy'):
        embeddings = np.load(embeddings_path)
        if verbose:
            logger.info(f"Loaded {len(embeddings)} embeddings from numpy file")

    elif embeddings_path.endswith('.parquet'):
        # Load from parquet file with embedding column
        if verbose:
            logger.info("Loading parquet file...")
        df = pd.read_parquet(embeddings_path)

        embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
        if not embedding_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")

        embedding_col = embedding_cols[0]  # Use first embedding column

        # Find the column that was clustered (should be the base name of embedding column)
        base_col = embedding_col.replace('_embedding', '')
        if base_col not in df.columns:
            # Try to find any text column that might be the source
            text_cols = [col for col in df.columns if col not in embedding_cols and
                        df[col].dtype == 'object']
            if text_cols:
                base_col = text_cols[0]
                if verbose:
                    logger.info(f"Using column '{base_col}' as source column")
            else:
                raise ValueError(f"Cannot find source text column in {embeddings_path}")

        if verbose:
            logger.info(f"Creating value-to-embedding mapping from column '{base_col}'...")

        # Create mapping from values to embedding
        embeddings = {}
        for _, row in df.iterrows():
            value = str(row[base_col])
            embedding = row[embedding_col]
            embeddings[value] = embedding

        if verbose:
            logger.info(f"Loaded {len(embeddings)} embeddings from parquet file (column: {embedding_col})")

    else:
        raise ValueError(f"Unsupported file format: {embeddings_path}. Supported: .pkl, .npy, .parquet")

    return embeddings


def create_summary_table(df, config=None, **kwargs):
    """Create summary table of cluster labels with model counts and examples."""
    labels = df.property_description_cluster_label.value_counts()
    cols = [
        'property_description',
    ]
    existing_cols = [c for c in cols if c in df.columns]
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results
