#!/usr/bin/env python3
"""
Run the complete StringSight pipeline on full datasets.

This script runs the full pipeline using the explain() function on complete datasets
with configurable parameters.
"""

import argparse
import os
import sys
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime

# Ensure local StringSightNew package is preferred over any globally installed version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stringsight import explain
from stringsight.core.preprocessing import sample_prompts_evenly
from stringsight.core.data_objects import PropertyDataset
from typing import Optional, Dict, Any, Tuple, List


def load_dataset(
    data_path: str,
    method: str = "single_model",
    tidy_side_by_side_models: Optional[Tuple[str, str]] = None,
):
    """Load dataset from json, jsonl, or csv file.

    Args:
        data_path: Path to input file (.json, .jsonl, or .csv)
        method: "single_model" or "side_by_side"
        tidy_side_by_side_models: When provided and method=="side_by_side",
            indicates the input is tidy single-model-like data to be converted.
            Expects columns [question_id, prompt, model, model_response].
    """
    print(f"Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Determine file type and load accordingly
    file_ext = os.path.splitext(data_path)[1].lower()

    if file_ext == '.jsonl':
        df = pd.read_json(data_path, lines=True)
    elif file_ext == '.json':
        df = pd.read_json(data_path)
    elif file_ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .json, .jsonl, .csv")
    
    # Attach the filename to the DataFrame for wandb naming
    df.name = os.path.basename(data_path)
    
    # Note: Column validation is deferred to validate_and_prepare_dataframe()
    # which handles column mapping. This allows users to specify custom column
    # names via prompt_column, model_column, etc. parameters.
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    return df


def run_pipeline(
    data_path,
    output_dir,
    method="single_model",
    system_prompt=None,
    task_description: Optional[str] = None,
    clusterer="hdbscan",
    min_cluster_size=15,
    embedding_model="text-embedding-3-large",
    max_workers=16,
    use_wandb=True,
    verbose=False,
    sample_size=None,
    extraction_cache_dir=None,
    clustering_cache_dir=None,
    metrics_cache_dir=None,
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    *,
    groupby_column: str | None = None,
    assign_outliers: bool | None = None,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
    filter_models: Optional[List[str]] = None,
    score_columns: Optional[List[str]] = None,
    extraction_model: Optional[str] = None,
    summary_model: Optional[str] = None,
    cluster_assignment_model: Optional[str] = None,
    # Column mapping parameters
    prompt_column: str = "prompt",
    model_column: Optional[str] = None,
    model_response_column: Optional[str] = None,
    question_id_column: Optional[str] = None,
    model_a_column: Optional[str] = None,
    model_b_column: Optional[str] = None,
    model_a_response_column: Optional[str] = None,
    model_b_response_column: Optional[str] = None,
):
    """Run the complete pipeline on a dataset.

    Args:
        filter_models: Optional list of model names. When provided and the
            input is in long/tidy format with a 'model' column, the dataset is
            filtered to only these models before further processing.
        score_columns: Optional list of column names containing score metrics.
            Instead of providing scores as a dictionary in a single column,
            you can specify separate columns for each metric.
        extraction_model: Optional model name for property extraction (e.g., 'gpt-4.1').
            If not provided, uses the default from explain().
        summary_model: Optional model name for cluster summarization (e.g., 'gpt-4.1').
            If not provided, uses the default from ClusterConfig.
        cluster_assignment_model: Optional model name for cluster matching (e.g., 'gpt-4.1-mini').
            If not provided, uses the default from ClusterConfig.
        prompt_column: Name of the prompt column in your dataframe (default: "prompt")
        model_column: Name of the model column for single_model (default: "model" if None)
        model_response_column: Name of the model response column for single_model (default: "model_response")
        question_id_column: Name of the question_id column (default: "question_id" if column exists)
        model_a_column: Name of the model_a column for side_by_side (default: "model_a")
        model_b_column: Name of the model_b column for side_by_side (default: "model_b")
        model_a_response_column: Name of the model_a_response column for side_by_side (default: "model_a_response")
        model_b_response_column: Name of the model_b_response column for side_by_side (default: "model_b_response")
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(data_path, method, tidy_side_by_side_models=(model_a, model_b) if (method == "side_by_side" and model_a and model_b) else None)
    
    # Optional generic model filtering for tidy (long) inputs
    if filter_models is not None and len(filter_models) > 0 and "model" in df.columns:
        print(f"Filtering to models list: {filter_models}")
        df = df[df["model"].isin(filter_models)].copy()
        print(f"After list model filtering: {len(df)} rows")
    
    # For side_by_side with model_a/model_b specified, filter to paired prompts first, then sample
    if method == "side_by_side" and model_a is not None and model_b is not None:
        # Filter to only the two requested models
        print(f"Filtering to models: {model_a} and {model_b}")
        df = df[df["model"].isin([model_a, model_b])].copy()
        print(f"After model filtering: {len(df)} rows")
        
        # Further filter to only prompts that have responses from BOTH models
        prompt_model_counts = df.groupby("prompt")["model"].nunique()
        prompts_with_both = prompt_model_counts[prompt_model_counts == 2].index
        df = df[df["prompt"].isin(prompts_with_both)].copy()
        print(f"After filtering to paired prompts: {len(df)} rows ({len(prompts_with_both)} prompts with both models)")
        
        # For tidy-to-side_by_side conversion, we need to sample PROMPTS not rows
        # because we need both model responses for each prompt
        if sample_size and sample_size < len(df):
            # Each prompt has 2 rows (one per model), so sample_size/2 prompts
            num_prompts_to_sample = max(1, int(sample_size // 2))
            print(f"Sampling {num_prompts_to_sample} prompts (= {num_prompts_to_sample * 2} rows) from {len(prompts_with_both)} available prompts")
            sampled_prompts = pd.Series(list(prompts_with_both)).sample(n=min(num_prompts_to_sample, len(prompts_with_both)), random_state=42)
            df = df[df["prompt"].isin(sampled_prompts)].copy()
            print(f"After prompt sampling: {len(df)} rows ({df['prompt'].nunique()} prompts)")
    elif sample_size and sample_size < len(df):
        # Standard sampling for other cases
        print(f"Sampling evenly by prompts for target size {sample_size} from {len(df)} total rows")
        df = sample_prompts_evenly(df, sample_size=int(sample_size), method=method, prompt_column="prompt", random_state=42)
    
    print(f"Starting pipeline with {len(df)} conversations")
    
    # Record start time
    start_time = time.time()
    
    # Run the full pipeline
    print("Running full pipeline with explain()...")

    # Build kwargs for explain() call
    explain_kwargs = {
        "method": method,
        "system_prompt": system_prompt,
        "task_description": task_description,
        "clusterer": clusterer,
        "min_cluster_size": min_cluster_size,
        "embedding_model": embedding_model,
        "assign_outliers": bool(assign_outliers) if assign_outliers is not None else False,
        "max_workers": max_workers,
        "use_wandb": use_wandb,
        "verbose": verbose,
        "output_dir": str(output_path),
        "extraction_cache_dir": extraction_cache_dir,
        "clustering_cache_dir": clustering_cache_dir,
        "metrics_cache_dir": metrics_cache_dir,
        "metrics_kwargs": metrics_kwargs,
        "groupby_column": groupby_column,
        "track_costs": True,  # Enable cost tracking
        "model_a": model_a,
        "model_b": model_b,
        "score_columns": score_columns,
        # Column mapping parameters
        "prompt_column": prompt_column,
        "model_column": model_column,
        "model_response_column": model_response_column,
        "question_id_column": question_id_column,
        "model_a_column": model_a_column,
        "model_b_column": model_b_column,
        "model_a_response_column": model_a_response_column,
        "model_b_response_column": model_b_response_column,
    }

    # Add model parameters if provided
    if extraction_model is not None:
        explain_kwargs["model_name"] = extraction_model
    if summary_model is not None:
        explain_kwargs["summary_model"] = summary_model
    if cluster_assignment_model is not None:
        explain_kwargs["cluster_assignment_model"] = cluster_assignment_model

    clustered_df, model_stats = explain(df, **explain_kwargs)
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Verify outputs
    print("\nVerifying pipeline outputs...")
    
    # Check basic structure
    assert len(clustered_df) > 0, "Should return clustered DataFrame"
    assert len(model_stats) > 0, "Should return model statistics"
    
    # Check required columns (match clusterer output)
    required_columns = ['cluster_id', 'cluster_label']
    print(clustered_df.columns)
    for col in required_columns:
        if col not in clustered_df.columns:
            print(f"Warning: Missing column: {col}")
    
    # Check for properties - FAIL if no properties were extracted
    if 'property_description' in clustered_df.columns:
        properties_with_desc = clustered_df['property_description'].notna().sum()
        print(f"Properties extracted: {properties_with_desc}")
        
        if properties_with_desc == 0:
            raise RuntimeError(
                "ERROR: No properties were successfully extracted from the conversations. "
                "This could be due to:\n"
                "1. OpenAI API errors (check your API key and quotas)\n"
                "2. Data format issues (verify your input data structure)\n"
                "3. Model or prompt configuration problems\n"
                "Check the logs above for specific error messages."
            )
    else:
        raise RuntimeError(
            "ERROR: 'property_description' column not found in results. "
            "This indicates a fundamental issue with the pipeline execution."
        )
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Output directory: {output_path}")
    print(f"Runtime: {runtime:.2f} seconds ({runtime/60:.1f} minutes)")
    print(f"Input conversations: {len(df)}")
    print(f"Clustered properties: {len(clustered_df)}")
    print(f"Models analyzed: {len(model_stats)}")
    print(f"Number of clusters: {len(clustered_df['cluster_id'].unique())}")
    
    if 'coarse_cluster_id' in clustered_df.columns:
        coarse_clusters = len(clustered_df['coarse_cluster_id'].unique())
        print(f"Coarse clusters: {coarse_clusters}")
    
    # Show sample clusters
    print(f"\nSample cluster labels:")
    unique_labels = clustered_df['cluster_label'].dropna().unique()
    for i, label in enumerate(unique_labels[:5]):
        cluster_size = (clustered_df['cluster_label'] == label).sum()
        print(f"  {i+1}. {label} (size: {cluster_size})")
    
    print("="*60)
    print("✅ Full pipeline completed successfully!")
    print("="*60)
    
    return clustered_df, model_stats


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run StringSight pipeline on full datasets")
    
    # Dataset and output
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input dataset (.json, .jsonl, or .csv file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    
    # Pipeline parameters
    parser.add_argument("--method", type=str, default="single_model",
                        choices=["single_model", "side_by_side"],
                        help="Analysis method (default: single_model)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help=(
                            "System prompt name (e.g., 'single_model_system_prompt', 'sbs_system_prompt', "
                            "'single_model_system_prompt_custom'), or omit to auto-select based on method"
                        ))
    parser.add_argument("--task_description", type=str, default=None,
                        help=(
                            "Optional task-specific description to guide property extraction "
                            "(e.g., 'Evaluate web development for code quality, security, and UX')"
                        ))
    parser.add_argument("--clusterer", type=str, default="hdbscan",
                        choices=["hdbscan", "dummy"],
                        help="Clustering algorithm (default: hdbscan)")
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (default: 15)")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large",
                        help="Embedding model to use (default: openai)")
    parser.add_argument("--max_workers", type=int, default=64,
                        help="Maximum number of workers (default: 64)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size to use (default: use full dataset)")
    
    # Flags
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable wandb logging (default: enabled)")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    parser.add_argument("--groupby_column", type=str, default="behavior_type",
                        help="Column to group by for stratified clustering (default: behavior_type, if None, no grouping will be done)")
    parser.add_argument("--assign_outliers", action="store_true",
                        help="Assign outliers to clusters when supported")
    parser.add_argument("--model_a", type=str, default=None,
                        help=(
                            "When method=side_by_side and the input is tidy single-model-like data, "
                            "select this model as model_a"
                        ))
    parser.add_argument("--model_b", type=str, default=None,
                        help=(
                            "When method=side_by_side and the input is tidy single-model-like data, "
                            "select this model as model_b"
                        ))
    parser.add_argument("--score_columns", nargs="+", type=str, default=None,
                        help="Optional list of column names containing score metrics (e.g., accuracy, helpfulness)")
    parser.add_argument("--extraction_model", type=str, default=None,
                        help="Model for property extraction (e.g., gpt-4.1)")
    parser.add_argument("--summary_model", type=str, default=None,
                        help="Model for cluster summarization (e.g., gpt-4.1)")
    parser.add_argument("--cluster_assignment_model", type=str, default=None,
                        help="Model for cluster matching/assignment (e.g., gpt-4.1-mini)")

    args = parser.parse_args()
    
    # Run pipeline
    clustered_df, model_stats = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        method=args.method,
        system_prompt=args.system_prompt,
        task_description=args.task_description,
        clusterer=args.clusterer,
        min_cluster_size=args.min_cluster_size,
        embedding_model=args.embedding_model,
        max_workers=args.max_workers,
        use_wandb=not args.disable_wandb,
        verbose=not args.quiet,
        sample_size=args.sample_size,
        groupby_column=args.groupby_column,
        assign_outliers=args.assign_outliers,
        model_a=args.model_a,
        model_b=args.model_b,
        score_columns=args.score_columns,
        extraction_model=args.extraction_model,
        summary_model=args.summary_model,
        cluster_assignment_model=args.cluster_assignment_model,
    )
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 