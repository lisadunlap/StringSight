#!/usr/bin/env python3
"""
Run the StringSight pipeline on the WebDev dataset.

This is a convenience script for running the full pipeline on the webdev dataset
with optimized parameters.
"""

import argparse
import os
from run_full_pipeline import run_pipeline
from stringsight import compute_metrics_only
import pandas as pd
import json

def main():
    """Main function for webdev dataset processing."""
    parser = argparse.ArgumentParser(description="Run StringSight pipeline on WebDev dataset")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, 
                        help="Output directory for results (default: results/webdev_full_pipeline)")
    parser.add_argument("--input_file", type=str,
                        help="Input file for results (default: data/arena_webdev_sbs.jsonl)")
    parser.add_argument("--system_prompt", type=str,
                        default="single_model_system_prompt",
                        help=(
                            "System prompt name (e.g., 'single_model_system_prompt', 'sbs_system_prompt', "
                            "'single_model_system_prompt_custom'), or omit to auto-select based on method"
                        ))
    parser.add_argument("--method", type=str,
                        default="single_model",
                        help="Method for the pipeline: 'single_model' or 'side_by_side' (default: single_model)")
    
    # Optional overrides
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size to use (default: use full dataset)")
    parser.add_argument("--min_cluster_size", type=int, default=8,
                        help="Minimum cluster size (default: 8)")
    parser.add_argument("--max_coarse_clusters", type=int, default=12,
                        help="Maximum number of coarse clusters (default: 12)")
    parser.add_argument("--max_workers", type=int, default=64,
                        help="Maximum number of workers (default: 16)")
    
    # Flags
    parser.add_argument("--hierarchical", action="store_true",
                        help="Enable hierarchical clustering")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    parser.add_argument("--clusterer", type=str, default="hdbscan",
                        choices=["hdbscan", "hierarchical", "dummy"],
                        help="Clustering method to use (default: hdbscan)")
    parser.add_argument("--groupby_column", type=str,  default="behavior_type",
                        help="Column to group by to enable stratified clustering (effective only for hdbscan)")
    parser.add_argument("--assign_outliers", action="store_true",
                        help="Assign outliers to clusters when supported")
    
    # run specific components (only metrics)
    parser.add_argument("--run_metrics", action="store_true",
                        help="Run only the metrics component")
    # metrics flags
    parser.add_argument("--bootstrap_samples", type=int, default=100,
                        help="Number of bootstrap samples to draw when computing CIs")
    
    args = parser.parse_args()
    
    # Set the data path
    data_path = args.input_file
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at {data_path}")
        print("Please make sure the dataset is available.")
        return
    
    print("="*60)
    print("StringSight PIPELINE")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Output: {args.output_dir}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    else:
        print("Using full dataset")
    print("="*60)
    
    # Handle metrics-only mode
    if args.run_metrics:
        print("\n🔧 Running metrics-only mode...")
        print("This will load existing pipeline results and compute metrics only.")
        data_path = args.output_dir
        
        # Check if the input path exists and looks like pipeline results
        if not os.path.exists(data_path):
            print(f"Error: Input path not found: {data_path}")
            print("For metrics-only mode, provide a path to existing pipeline results.")
            print("This can be:")
            print("  - A file: results/previous_run/full_dataset.json")
            print("  - A directory: results/previous_run/")
            return
        
        try:
            # Run metrics-only computation
            metrics_kwargs = {
                "compute_confidence_intervals": True,
                "bootstrap_samples": args.bootstrap_samples,
            }

            clustered_df, model_stats = compute_metrics_only(
                input_path=data_path,
                method=args.method,
                output_dir=args.output_dir,
                metrics_kwargs=metrics_kwargs,
                use_wandb=args.use_wandb,
                verbose=not args.quiet
            )
            # save_examples(args.output_dir, args.method)
            # # Convert ModelStats objects to dictionaries for JSON serialization
            # stats_for_json = {}
            # for model_name, stats in model_stats.items():
            #     print(stats)
            #     stats_for_json[str(model_name)] = {
            #         "fine": [stat.to_dict() for stat in stats["fine"]]
            #     }
            #     if "coarse" in stats:
            #         stats_for_json[str(model_name)]["coarse"] = [stat.to_dict() for stat in stats["coarse"]]
            #     assert "stats" in stats, "stats should be in the model stats"
            #     if "stats" in stats:
            #         print(stats["stats"])
            #         stats_for_json[str(model_name)]["stats"] = stats["stats"]
            
            # with open(f"{args.output_dir}/model_stats.json", 'w') as f:
            #     json.dump(stats_for_json, f, indent=2)
            print(f"  ✓ Saved model statistics (JSON): {args.output_dir}/model_stats.json")
            
            print(f"\n🎉 Metrics computation completed! Results saved to: {args.output_dir}")
            return
            
        except Exception as e:
            print(f"\n❌ Error during metrics computation: {e}")
            print("\nThis could be because:")
            print("1. The input path doesn't contain valid pipeline results")
            print("2. The results don't have the required clustering data")
            print("3. The data format is incompatible")
            print("\nTry running the full pipeline first to generate the required data.")
            raise
    
    # Prepare metrics configuration
    metrics_kwargs = {
        "compute_confidence_intervals": args.bootstrap_samples > 0,
        "bootstrap_samples": args.bootstrap_samples,
    }

    # Run pipeline with webdev-optimized parameters
    clustered_df, model_stats = run_pipeline(
        data_path=data_path,
        output_dir=args.output_dir,
        method=args.method,
        system_prompt=args.system_prompt,
        clusterer=args.clusterer,
        min_cluster_size=args.min_cluster_size,
        max_coarse_clusters=args.max_coarse_clusters,
        embedding_model="text-embedding-3-small",
        hierarchical=args.hierarchical,
        max_workers=args.max_workers,
        use_wandb=args.use_wandb,
        verbose=not args.quiet,
        sample_size=args.sample_size,
        groupby_column=args.groupby_column,
        assign_outliers=args.assign_outliers,
        metrics_kwargs=metrics_kwargs,
    )
    # save_examples(args.output_dir, args.method)
    
    print(f"\n🎉 Pipeline completed! Results saved to: {args.output_dir}")

def save_examples(output_dir, method):
    dataset = "koala"

    with open(f"{output_dir}/model_stats.json", "r") as f:
        data = json.load(f)
    all_examples = []
    for model in data:
        for example in data[model]['fine']:
            all_examples.extend(example['examples'][:1])
    print(len(set(all_examples)))
    if method == "single_model":
        selected_columns = ['question_id', 'prompt', 'model_response', 'score', 'property_description',
            'category', 'type', 'impact', 'reason', 'evidence',
            'behavior_type', 'raw_response', 'contains_errors',
            'unexpected_behavior', 'model', 'property_description_cluster_label','property_description_coarse_cluster_label']
    elif method == "side_by_side":
        selected_columns = ['question_id', 'prompt', 'model_a_response', 'model_b_response', 'score', 'property_description',
            'category', 'type', 'impact', 'reason', 'evidence',
            'behavior_type', 'raw_response', 'contains_errors',
            'unexpected_behavior', 'model_a', 'model_b', 'property_description_cluster_label','property_description_coarse_cluster_label']
    else:
        raise ValueError(f"Invalid method: {method}")
    
    with open(f"{output_dir}/full_dataset.json", "r") as f:
        data = json.load(f)
    pd.DataFrame(data['clusters']).to_json(f"{output_dir}/clusters.json", orient="records", lines=True)

    df = pd.read_json(f"{output_dir}/clustered_results.jsonl", lines=True)
    print(df.columns)
    df[df.id.isin(all_examples)][selected_columns].to_json(f"{output_dir}/clustered_results_examples.json", lines=True, orient="records")


if __name__ == "__main__":
    main() 