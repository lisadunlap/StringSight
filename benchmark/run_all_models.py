#!/usr/bin/env python3
"""
Script to run evaluate_stringsight.py on all individual model files
in aci_bench and instructeval, excluding baselines and all_behaviors files.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import argparse


def find_model_files(results_dir: Path, benchmarks: List[str]) -> List[Tuple[str, Path]]:
    """
    Find all individual model .jsonl files, excluding baselines and all_behaviors.
    
    Returns:
        List of (benchmark_name, model_file_path) tuples
    """
    model_files = []
    
    for benchmark in benchmarks:
        benchmark_dir = results_dir / benchmark
        
        if not benchmark_dir.exists():
            print(f"Warning: Directory {benchmark_dir} not found, skipping...")
            continue
        
        # Find all .jsonl files
        for model_file in benchmark_dir.glob("*.jsonl"):
            filename = model_file.name
            
            # Skip baseline and all_behaviors files
            if "baseline" in filename or "all_behaviors" in filename:
                print(f"  Skipping: {filename} (baseline or all_behaviors)")
                continue
            
            model_files.append((benchmark, model_file))
    
    return model_files


def run_evaluation(
    model_file: Path,
    benchmark_name: str,
    output_base_dir: Path,
    args: argparse.Namespace
) -> bool:
    """
    Run evaluate_stringsight.py on a single model file.
    
    Returns:
        True if successful, False otherwise
    """
    model_name = model_file.stem  # Remove .jsonl extension
    output_dir = output_base_dir / benchmark_name / model_name
    
    # Build command
    cmd = [
        "python", "benchmark/evaluate_stringsight.py",
        "--benchmark-results", str(model_file),
        "--output-dir", str(output_dir),
        "--min-cluster-size", str(args.min_cluster_size),
        "--embedding-model", args.embedding_model,
        "--extraction-model", args.extraction_model,
        "--judge-model", args.judge_model,
    ]
    
    # Add optional flags
    if args.subset_size:
        cmd.extend(["--subset-size", str(args.subset_size)])
    
    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    
    if args.log_to_wandb:
        cmd.append("--log-to-wandb")
    else:
        cmd.append("--no-wandb")
    
    print(f"\n{'='*80}")
    print(f"Processing: {benchmark_name}/{model_name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Successfully completed: {benchmark_name}/{model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {benchmark_name}/{model_name}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run StringSight evaluation on all individual model files"
    )
    
    # Directory configuration
    parser.add_argument("--results-dir", type=str, default="benchmark/results",
                        help="Base directory containing benchmark results")
    parser.add_argument("--output-dir", type=str, default="benchmark/evaluation_results",
                        help="Base output directory for evaluation results")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                        default=["aci_bench", "instructeval", "omni_math", "medication_qa", "harm_bench"],
                        help="List of benchmark directories to process")
    
    # StringSight parameters
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Number of prompts to sample (None = use all)")
    parser.add_argument("--min-cluster-size", type=int, default=4,
                        help="Minimum cluster size for StringSight")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large",
                        help="Embedding model for StringSight clustering")
    parser.add_argument("--extraction-model", type=str, default="gpt-4.1-mini",
                        help="Model for StringSight property extraction")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1",
                        help="Model for LLM-as-judge evaluation")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top behaviors to evaluate per model (None = all)")
    parser.add_argument("--log-to-wandb", dest="log_to_wandb", action="store_true", default=True,
                        help="Log results to wandb (default: True)")
    parser.add_argument("--no-wandb", dest="log_to_wandb", action="store_false",
                        help="Disable wandb logging")
    
    # Execution control
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("Running StringSight Evaluation on All Models")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Results directory: {results_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Benchmarks: {', '.join(args.benchmarks)}")
    print(f"  Subset size: {args.subset_size or 'All data'}")
    print(f"  Min cluster size: {args.min_cluster_size}")
    print(f"  Top K behaviors: {args.top_k or 'All'}")
    print(f"  Log to wandb: {args.log_to_wandb}")
    print()
    
    # Find all model files
    model_files = find_model_files(results_dir, args.benchmarks)
    
    if not model_files:
        print("No model files found to process!")
        return
    
    print(f"\nFound {len(model_files)} model files to process:")
    for benchmark, model_file in model_files:
        print(f"  - {benchmark}/{model_file.name}")
    print()
    
    if args.dry_run:
        print("Dry run mode - no evaluations will be executed")
        return
    
    # Run evaluations
    successful = 0
    failed = 0
    
    for benchmark, model_file in model_files:
        success = run_evaluation(model_file, benchmark, output_dir, args)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print("="*80)
    print(f"Total models processed: {len(model_files)}")
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
