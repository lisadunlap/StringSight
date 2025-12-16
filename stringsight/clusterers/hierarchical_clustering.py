#!/usr/bin/env python3
"""
Hierarchical Text Clustering Module

Provides scalable hierarchical clustering for text data using semantic embeddings.
Supports multiple clustering algorithms including HDBSCAN, and traditional methods.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import asyncio
import time
from collections import defaultdict
from ..core.llm_utils import parallel_completions
import random
import os
import pickle
import argparse
from tqdm import tqdm
from ..logging_config import get_logger

logger = get_logger(__name__)
# from dataclasses import dataclass  # removed: ClusterConfig now imported from config
from typing import Optional, Dict, Union, List, Any

# Core ML libraries
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

# Try relative import first, fall back to absolute import
from .clustering_utils import llm_coarse_cluster_with_centers, _get_embeddings, _setup_embeddings, save_clustered_results, initialize_wandb, load_precomputed_embeddings

# Import the new modular functions
from .clustering_utils import generate_coarse_labels, assign_fine_to_coarse

# Import the unified config
from .config import ClusterConfig

# Prompts for LLM clustering
from stringsight.prompts.clustering.prompts import coarse_clustering_systems_prompt, deduplication_clustering_systems_prompt, outlier_clustering_systems_prompt

# Optional imports (will be checked when needed)
# sentence-transformers is optional - imported lazily when needed
import hdbscan
import litellm
# import weave

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

# ClusterConfig is now provided by stringsight.clusterers.config

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_embeddings(unique_values: List[Any], config: ClusterConfig) -> np.ndarray:
    """
    Prepare embeddings for clustering with caching and optional dimensionality reduction.
    
    Args:
        unique_values: List of unique values to embed
        config: ClusterConfig containing embedding parameters
        
    Returns:
        np.ndarray: Processed embeddings ready for clustering
    """
    unique_strings = [str(value) for value in unique_values]
    
    if config.verbose:
        logger.info(f"Preparing embeddings for {len(unique_values)} unique values...")
    
    # Get embeddings (either precomputed or compute fresh)
    if config.precomputed_embeddings is not None:
        if config.verbose:
            logger.info("Using precomputed embeddings...")
        embeddings = config.precomputed_embeddings
        if isinstance(embeddings, dict):
            if config.verbose:
                logger.info(f"Mapping {len(unique_values)} values to embeddings from dict with {len(embeddings)} entries...")
            try:
                embeddings = np.array([embeddings[str(val)] for val in unique_values])
                if config.verbose:
                    logger.info(f"Successfully mapped to {len(embeddings)} embeddings")
            except KeyError as e:
                logger.error(f"Error: Some values not found in precomputed embeddings: {e}")
                logger.error(f"Available keys (first 5): {list(embeddings.keys())[:5]}")
                logger.error(f"Missing values (first 5): {[str(val) for val in unique_values if str(val) not in embeddings][:5]}")
                raise
        else:
            if config.verbose:
                logger.info(f"Using precomputed embeddings array with {len(embeddings)} entries...")
            embeddings = np.array(embeddings)
        
        if config.verbose:
            logger.info(f"Embeddings shape: {embeddings.shape}")
    else:
       
        embeddings, _ = _setup_embeddings(unique_strings, config.embedding_model, config.verbose)
        embeddings = np.array(embeddings)
    
    # Normalize embeddings
    if config.verbose:
        logger.info("Normalizing embeddings...")
    if len(embeddings) > 1:
        embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)
    
    # Keep original embeddings for output (before any dimensionality reduction)
    original_embeddings = embeddings.copy()
    
    # Improved dimension reduction that preserves semantic coherence
    if not config.disable_dim_reduction:
        n_points, n_dims = embeddings.shape
        
        # Determine method (improved adaptive logic)
        if config.dim_reduction_method == "adaptive":
            # More conservative adaptive logic that considers dataset size
            if n_points < 100:
                # For very small datasets, skip dimension reduction entirely
                method = "none"
            elif n_points > 5000 or n_dims > 200:
                # For large datasets, use PCA (portable and fast; avoids optional compiled stacks)
                method = "pca"
            else:
                # For medium datasets, skip dimension reduction
                method = "none"
        else:
            method = config.dim_reduction_method
        
        if method == "pca":
            if config.verbose:
                logger.info(f"Applying PCA dimensionality reduction...")
            
            from sklearn.decomposition import PCA
            n_components = min(100, n_dims - 1, n_points - 1)
            reducer = PCA(n_components=n_components, random_state=42)
            embeddings = reducer.fit_transform(embeddings)
            
            if config.verbose:
                logger.info(f"Reduced to shape: {embeddings.shape}")
                
        elif method == "none" and config.verbose:
            logger.info("Skipping dimension reduction")
    
    return embeddings, original_embeddings


def generate_cluster_summaries(cluster_values: Dict[int, List], config: ClusterConfig, 
                             column_name: str, cluster_type: str = "cluster") -> Dict[int, str]:
    """
    Generate cluster summaries using LLM or generic labels.
    
    Args:
        cluster_values: Dict mapping cluster IDs to lists of values
        config: ClusterConfig containing summary parameters
        column_name: Name of the column being clustered
        
    Returns:
        Dict mapping cluster IDs to summary labels
    """
    if config.verbose:
        logger.info(f"Generating LLM-based cluster summaries for {cluster_type} clusters...")
    
    # Prepare data for parallel processing
    cluster_ids = []
    messages = []
    
    for cluster_id, values in cluster_values.items():
        if cluster_id < 0:
            continue  # Handle outliers separately
            
        cluster_ids.append(cluster_id)
        
        # Sample values and create prompt (same logic as _get_llm_cluster_summary)
        sampled_vals = values if len(values) <= 50 else random.sample(values, 50)
        values_text = "\n".join(map(str, sampled_vals))
        messages.append(values_text)
    
    cluster_label_map = {-1: "Outliers"}  # Handle outliers
    
    if not messages:
        return cluster_label_map
    
    # Get the system prompt
    from stringsight.prompts.clustering.prompts import clustering_systems_prompt
    
    # Parallel LLM calls!
    summaries = parallel_completions(
        messages,
        model=config.summary_model,
        system_prompt=clustering_systems_prompt,
        max_workers=getattr(config, 'llm_max_workers', 64),
        show_progress=config.verbose,
        progress_desc=f"Generating {cluster_type} summaries"
    )
    
    # Build result map
    for cluster_id, summary in zip(cluster_ids, summaries):
        # Clean up summary (same logic as _get_llm_cluster_summary)
        content = summary.strip()
        if content.startswith(("'", '"')):
            content = content[1:]
        if content.endswith(("'", '"')):
            content = content[:-1]
        cluster_label_map[cluster_id] = content
        
        if config.verbose:
            logger.info(f"    Cluster {cluster_id}: {content} ({len(cluster_values[cluster_id])} items)")
    
    return cluster_label_map


def format_clustering_results(df: pd.DataFrame, column_name: str, 
                            unique_values: List, original_embeddings: np.ndarray,
                            cluster_labels: np.ndarray, cluster_label_map: Dict[int, str],
                            config: ClusterConfig) -> pd.DataFrame:
    """
    Format clustering results into output DataFrame.
    
    Args:
        df: Original DataFrame
        column_name: Name of the column that was clustered
        unique_values: List of unique values that were clustered
        original_embeddings: Original embeddings before dimensionality reduction
        cluster_labels: Cluster assignment for each unique value
        cluster_label_map: Mapping from cluster ID to label
        config: ClusterConfig containing formatting parameters
        
    Returns:
        pd.DataFrame: Formatted results with cluster assignments
    """
    df_copy = df.copy()
    
    # Create basic mappings
    value_to_cluster = dict(zip(unique_values, cluster_labels))
    
    # Ensure cluster_label_map contains -1 key for outliers if needed
    if -1 in cluster_labels and -1 not in cluster_label_map:
        # Find any existing outlier labels or use default
        outlier_labels = [lbl for lbl in cluster_label_map.values() if lbl == "Outliers" or lbl.startswith("Outliers - ")]
        if outlier_labels:
            cluster_label_map[-1] = outlier_labels[0]
        else:
            cluster_label_map[-1] = "Outliers"
    
    value_to_label = {v: cluster_label_map[c] for v, c in value_to_cluster.items()}
    
    # Add cluster columns
    df_copy[f'{column_name}_cluster_label'] = df_copy[column_name].map(value_to_label)
    df_copy[f'{column_name}_cluster_id'] = df_copy[column_name].map(value_to_cluster)
    
    # Add embeddings if requested
    if config.include_embeddings:
        value_to_embedding = dict(zip(unique_values, original_embeddings.tolist()))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)
        
        # Get embeddings for cluster names
        unique_cluster_names = list(set(value_to_label.values()))
        cluster_name_embeddings = _get_embeddings(unique_cluster_names, config.embedding_model, config.verbose)
        cluster_name_to_embedding = dict(zip(unique_cluster_names, cluster_name_embeddings))
        df_copy[f'{column_name}_cluster_label_embedding'] = df_copy[f'{column_name}_cluster_label'].map(cluster_name_to_embedding)
    
    return df_copy


# =============================================================================
# MAIN CLUSTERING FUNCTIONS
# =============================================================================

async def hdbscan_cluster_categories(df, column_name, config=None, **kwargs) -> pd.DataFrame:
    """
    Fast HDBSCAN clustering for medium to large datasets.
    Supports LLM-based outlier reassignment.
    """
    # Handle backward compatibility by creating config from kwargs
    if config is None:
        config_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(ClusterConfig, k):
                config_kwargs[k] = v
        config = ClusterConfig(**config_kwargs)

    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]

    if config.verbose:
        logger.info(f"HDBSCAN clustering for {len(unique_values)} unique values...")

    # Step 1: Prepare embeddings
    embeddings, original_embeddings = prepare_embeddings(unique_values, config)

    # Step 2: Run HDBSCAN clustering
    if config.verbose:
        logger.info("Starting HDBSCAN clustering...")

    # Determine effective min_cluster_size (autoset if None)
    n_points = embeddings.shape[0]
    if getattr(config, 'min_cluster_size', None) is None:
        # Less conservative: Fixed at 5 for most datasets
        # This reduces outliers significantly compared to old 1%/0.5% rule
        effective_min_cluster_size = 5
    else:
        effective_min_cluster_size = int(config.min_cluster_size)
    
    # Ensure min_cluster_size is at least 2 (HDBSCAN requirement)
    if effective_min_cluster_size < 2:
        if config.verbose:
            logger.warning(
                f"min_cluster_size ({effective_min_cluster_size}) is less than 2. "
                f"HDBSCAN requires min_cluster_size >= 2. Setting to 2."
            )
        effective_min_cluster_size = 2

    if config.verbose:
        logger.info(f"Parameters: min_cluster_size={effective_min_cluster_size}, data_shape={embeddings.shape}")
        
    # 🛡️ Guard against small datasets ------------------------------------
    if n_points <= effective_min_cluster_size:
        if config.verbose:
            logger.warning(
                f"Number of points ({n_points}) is less than or equal to min_cluster_size "
                f"({effective_min_cluster_size}). Assigning all items to outliers (-1)."
            )
        cluster_labels = np.full(n_points, -1, dtype=int)
        clusterer = None
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=config.min_samples if config.min_samples else 1,
            metric='euclidean',  # Euclidean is fine since embeddings are normalized
            cluster_selection_method='eom',
            prediction_data=True,
            algorithm='best',
            core_dist_n_jobs=-1,
            cluster_selection_epsilon=config.cluster_selection_epsilon
        )
        # Run CPU-intensive HDBSCAN in thread pool to avoid blocking event loop
        if config.verbose:
            logger.info(f"HDBSCAN clustering (CPU) - {len(embeddings)} items...")
        
        def _run_hdbscan():
            try:
                return clusterer.fit_predict(embeddings), False
            except ValueError as e:
                if "Min cluster size must be greater than one" in str(e):
                    # Return all outliers if clustering fails, along with error flag
                    return np.full(n_points, -1, dtype=int), True
                else:
                    raise
        
        cluster_labels, error_occurred = await asyncio.to_thread(_run_hdbscan)
        
        # If an error occurred, set clusterer to None
        if error_occurred:
            if config.verbose:
                logger.warning(
                    f"HDBSCAN error: Min cluster size must be greater than one. "
                    f"Dataset too small for clustering. Assigning all items to outliers (-1)."
                )
            clusterer = None
        
        if config.verbose:
            logger.info("HDBSCAN clustering complete")

    if config.verbose:
        n_initial_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"HDBSCAN clustering completed! Found {n_initial_clusters} clusters and {n_noise} outliers")

    # Step 3: Generate cluster summaries
    from collections import defaultdict
    cluster_values = defaultdict(list)
    for value, cluster_id in zip(unique_values, cluster_labels):
        cluster_values[cluster_id].append(value)

    cluster_label_map = generate_cluster_summaries(cluster_values, config, column_name)

    # -------------------------------------------------------------
    # Step 3a: Optionally cluster outliers using LLM
    # -------------------------------------------------------------
    if config.verbose:
        logger.info("Clustering outliers using LLM...")

    # Get outlier items
    outlier_items = [unique_values[i] for i, label in enumerate(cluster_labels) if label < 0]

    # Skip LLM-based outlier clustering if fewer than min_cluster_size * 2 outliers (not worth the overhead)
    if len(outlier_items) >= effective_min_cluster_size * 2:
        # Generate outlier cluster summaries
        outlier_cluster_names = generate_coarse_labels(
            outlier_items,
            max_coarse_clusters=len(outlier_items) // max(1, effective_min_cluster_size),
            systems_prompt=outlier_clustering_systems_prompt,
            model=config.summary_model,
            verbose=config.verbose,
        )
        
        # Assign outlier items to outlier clusters
        outlier_items = outlier_items + [unique_values[i] for i, label in enumerate(cluster_labels) if label < 0]
        outlier_assignments = await assign_fine_to_coarse(
            outlier_items,
            outlier_cluster_names + list(cluster_label_map.values()),
            model=config.cluster_assignment_model,
            strategy="llm",
            verbose=config.verbose,
            max_workers=getattr(config, 'llm_max_workers', 64),
        )
        
        if config.verbose:
            logger.info(f"Created {len(outlier_cluster_names)} outlier clusters")
            logger.info(f"Outlier cluster names: {outlier_cluster_names}")
        
        # -------------------------------------------------------------
        # Merge outlier clusters with fine clusters
        # -------------------------------------------------------------
        
        # Get the next available cluster ID (after existing fine clusters)
        next_cluster_id = max(cluster_values.keys()) + 1 if cluster_values else 0
        
        # Create mapping from outlier cluster names to new cluster IDs
        outlier_name_to_id = {}
        for i, name in enumerate(outlier_cluster_names):
            if not (name == "Outliers" or name.startswith("Outliers - ")):  # Skip if LLM returned outlier names
                outlier_name_to_id[name] = next_cluster_id + i
        
        # Update cluster_labels to assign outlier items to their new clusters
        new_cluster_labels = cluster_labels.copy()
        for i, label in enumerate(cluster_labels):
            if label < 0:  # This was an outlier
                item = unique_values[i]
                assigned_cluster = outlier_assignments.get(item, "Outliers")
                if assigned_cluster in outlier_name_to_id:
                    new_cluster_labels[i] = outlier_name_to_id[assigned_cluster]
                # If assigned_cluster is "Outliers" or starts with "Outliers - " or not found, keep as -1
        
        cluster_labels = new_cluster_labels
        
        # Rebuild cluster_values and cluster_label_map to include outlier clusters
        cluster_values = defaultdict(list)
        for value, cluster_id in zip(unique_values, cluster_labels):
            cluster_values[cluster_id].append(value)
        
        # Update cluster_label_map to include outlier clusters
        for name, cluster_id in outlier_name_to_id.items():
            cluster_label_map[cluster_id] = name
        
        if config.verbose:
            n_outlier_clusters = len(outlier_name_to_id)
            n_remaining_outliers = list(cluster_labels).count(-1)
            logger.info(f"Assigned outliers to {n_outlier_clusters} clusters, {n_remaining_outliers} remain as outliers")
    else:
        if config.verbose:
            logger.info(f"No outliers/too few outliers ({len(outlier_items)}) to cluster")

    # -------------------------------------------------------------
    # Step 4b: Deduplicate cluster labels via LLM            
    # -------------------------------------------------------------
    
    # Only perform deduplication if there are more than 1 non-outlier cluster
    non_outlier_clusters = [cid for cid in cluster_values.keys() if cid >= 0]
    if len(non_outlier_clusters) > 1:
        if config.verbose:
            logger.info("Deduplicating cluster labels…")

        cluster_names = [cluster_label_map[cid] for cid in cluster_values.keys() if cid >= 0]

        # Generate deduplicated labels
        deduped_names = generate_coarse_labels(
            cluster_names,
            max_coarse_clusters=None,
            systems_prompt=deduplication_clustering_systems_prompt,
            model=config.summary_model,
            verbose=config.verbose,
        )

        # Assign fine labels to deduplicated labels
        fine_to_dedupe = await assign_fine_to_coarse(
            cluster_names,
            deduped_names,
            model=config.cluster_assignment_model,
            strategy="llm",
            verbose=config.verbose,
            max_workers=getattr(config, 'llm_max_workers', 64),
        )

        # -------------------------------------------------------------
        # Merge fine clusters that were deduplicated
        # -------------------------------------------------------------

        # 1. Update the label map so every original cluster id points to its deduped label
        for cid in cluster_values.keys():
            if cid < 0:
                continue
            original_label = cluster_label_map[cid]
            # Handle case where fine label maps to 'Outliers' in deduplication
            if original_label in fine_to_dedupe:
                deduped_label = fine_to_dedupe[original_label]
            else:
                deduped_label = original_label
            cluster_label_map[cid] = deduped_label

        # 2. Build mapping from deduped label → new sequential id
        unique_dedup_labels = [lbl for lbl in sorted(set(cluster_label_map.values())) if not (lbl == "Outliers" or lbl.startswith("Outliers - "))]
        label_to_new_id = {lbl: idx for idx, lbl in enumerate(unique_dedup_labels)}

        # 3. Re-assign cluster_labels array so duplicates share the same id
        new_cluster_labels = []
        for original_cid in cluster_labels:
            if original_cid < 0:
                new_cluster_labels.append(-1)
            else:
                deduped = cluster_label_map[original_cid]
                if deduped == "Outliers" or deduped.startswith("Outliers - "):
                    new_cluster_labels.append(-1)
                else:
                    new_cluster_labels.append(label_to_new_id[deduped])

        cluster_labels = np.array(new_cluster_labels)

        # 4. Rebuild cluster_values & cluster_label_map using new ids
        cluster_values = defaultdict(list)
        for val, cid in zip(unique_values, cluster_labels):
            cluster_values[cid].append(val)

        cluster_label_map = {new_id: lbl for lbl, new_id in label_to_new_id.items()}
        # Ensure outliers are properly mapped (handle both standard and group-specific outliers)
        # Find any outlier labels that might exist
        outlier_labels = [lbl for lbl in cluster_label_map.values() if lbl == "Outliers" or lbl.startswith("Outliers - ")]
        if outlier_labels:
            # Use the first outlier label found, or default to "Outliers"
            cluster_label_map[-1] = outlier_labels[0]
        else:
            # If no outlier labels found, add default mapping
            cluster_label_map[-1] = "Outliers"
    else:
        if config.verbose:
            logger.info("Skipping deduplication - only 1 non-outlier cluster found")

    # Step 5: Format results
    df_result = format_clustering_results(
        df, column_name, unique_values, original_embeddings,
        cluster_labels, cluster_label_map, config
    )

    if config.verbose:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        total_time = time.time() - start_time
        logger.info(f"Found {n_clusters} clusters and {n_noise} outliers in {total_time:.1f} seconds")

    return df_result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with command-line argument support."""
    parser = argparse.ArgumentParser(description='Hierarchical Text Clustering')
    parser.add_argument('--file', '-f', required=True, 
                       help='Path to input JSONL file')
    parser.add_argument('--column', '-c', default='property_description',
                       help='Column name to cluster on (default: property_description)')
    parser.add_argument('--method', '-m', choices=['hdbscan', 'hdbscan_native'], 
                       default='hdbscan',
                       help='Clustering method (default: hdbscan)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum cluster size (default: 15)')
    parser.add_argument('--min-samples', type=int,
                       help='min_samples for HDBSCAN. Lower values reduce outliers. (default: based on min_cluster_size)')
    parser.add_argument('--cluster-selection-epsilon', type=float, default=0.0,
                       help='Epsilon value for HDBSCAN cluster selection to merge clusters (default: 0.0, disabled, higher values merge more clusters)')
    parser.add_argument('--embedding-model', default='openai',
                       help='Embedding model: openai, all-MiniLM-L6-v2, etc. (default: openai)')
    parser.add_argument('--output', '-o', 
                       help='Output filename prefix (default: auto-generated)')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Exclude embeddings from output')
    parser.add_argument('--context', default='properties seen in AI responses',
                       help='Context for LLM summaries (default: "properties seen in AI responses")')
    parser.add_argument('--precomputed-embeddings', 
                       help='Path to precomputed embeddings file (.pkl or .npy)')
    parser.add_argument('--disable-dim-reduction', action='store_true',
                       help='Disable dimensionality reduction (default: False)')
    parser.add_argument('--dim-reduction-method', choices=['adaptive', 'pca', 'none'], default='adaptive',
                       help='Dimension reduction method: adaptive (auto-choose), pca, or none (default: adaptive)')
    parser.add_argument('--assign-outliers', action='store_true',
                       help='Assign HDBSCAN outliers to their nearest clusters (default: False)')
    parser.add_argument('--input-model-name', 
                       help='Name of the input model being analyzed (for cache differentiation)')
    parser.add_argument('--type', choices=['all', 'context-specific', 'general'], default='all',
                       help='Type of data to cluster (default: context-specific)')
    parser.add_argument('--remove-low-impact', action='store_true',
                       help='Remove low impact properties (default: False)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb-project', 
                       help='wandb project name')
    parser.add_argument('--wandb-entity', 
                       help='wandb entity name')
    parser.add_argument('--wandb-run-name', 
                       help='wandb run name')
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.file}...")
    df = pd.read_json(args.file, lines=True)

    if args.type == 'context-specific':
        def is_context_specific(row):
            return 'context' in row['type'].lower()
        df = df[df.apply(is_context_specific, axis=1)]
    elif args.type == 'general':
        def is_general(row):
            return 'general' in row['type'].lower()
        df = df[df.apply(is_general, axis=1)]
    elif args.type == 'all':
        pass
    
    if args.remove_low_impact:
        df = df[(df.impact.str.lower() == 'high') | (df.impact.str.lower() == 'medium')]
    
    if args.column not in df.columns:
        logger.error(f"Error: Column '{args.column}' not found in data. Available columns: {list(df.columns)}")
        return None
    
    logger.info(f"Loaded {len(df)} rows with {len(df[args.column].unique())} unique values in '{args.column}'")
    
    # Set up parameters
    include_embeddings = not args.no_embeddings
    
    # Load precomputed embeddings if provided
    precomputed_embeddings = None
    if args.precomputed_embeddings:
        precomputed_embeddings = load_precomputed_embeddings(args.precomputed_embeddings, verbose=True)
    
    # Create config from args
    config = ClusterConfig.from_args(args)
    if precomputed_embeddings:
        config.precomputed_embeddings = precomputed_embeddings
    
    # Initialize wandb if enabled
    method_name = args.method
    initialize_wandb(config, method_name, args.file)
    
    # Run clustering based on method
    if args.method == 'hdbscan':
        logger.info(f"Running HDBSCAN clustering...")
        df_clustered = hdbscan_cluster_categories(df, args.column, config=config)
        method_name = "hdbscan"
    
    # Generate output filename
    if args.output:
        output_prefix = args.output
    else:
        input_basename = os.path.splitext(os.path.basename(args.file))[0]
        output_prefix = f"{input_basename}_{method_name}_clustered"
    
    # Save results
    save_clustered_results(df_clustered, output_prefix, include_embeddings=include_embeddings, config=config)
    
    logger.info(f"\n✅ Clustering complete! Final dataset shape: {df_clustered.shape}")
    
    return df_clustered


if __name__ == "__main__":
    df_result = main() 