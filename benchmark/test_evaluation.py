"""
Quick test script for the evaluation pipeline.

This runs the evaluation on a small subset to verify the pipeline works.
"""

from evaluate_stringsight import EvaluationConfig, evaluate_stringsight

# Test with a small subset
config = EvaluationConfig(
    benchmark_results_path="benchmark/results/instructeval/all_behaviors.jsonl",
    output_dir="benchmark/evaluation_results/test_run/",
    subset_size=5,  # Sample 5 prompts (if 12 behaviors, this = 5×12=60 total responses)
    min_cluster_size=3,  # Lower threshold for small test
    embedding_model="text-embedding-3-large",
    extraction_model="gpt-4.1-mini",  # Cheaper model for testing
    judge_model="gpt-4.1-mini",  # Cheaper model for testing
    hierarchical=True,
    top_k_behaviors=5,  # Evaluate top 5 behaviors per model
    log_to_wandb=False  # Disable wandb for testing
)

print("Running evaluation pipeline test...")
print(f"This will sample {config.subset_size} prompts (×all behaviors) from the benchmark")
print(f"Using {config.extraction_model} for extraction and {config.judge_model} for judging\n")

evaluate_stringsight(config)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("Check benchmark/evaluation_results/test_run/ for outputs")
print("If this works, run the full evaluation with:")
print("  --subset-size 100  (100 prompts × N behaviors)")
print("  or no --subset-size to use all data")

