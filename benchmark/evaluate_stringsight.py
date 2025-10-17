"""
Evaluation pipeline for StringSight on benchmark datasets.

This script:
1. Loads benchmark results with ground truth behaviors
2. Runs StringSight to discover behavioral clusters
3. Uses LLM-as-judge to match discovered clusters to ground truth
4. Calculates precision, recall, and F1 scores
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from stringsight import explain
from stringsight.core.llm_utils import single_completion

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Skipping wandb logging.")


@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""
    benchmark_results_path: str  # Path to all_behaviors.jsonl
    output_dir: str = "benchmark/evaluation_results/"
    subset_size: Optional[int] = None  # Number of prompts to sample (includes all behaviors per prompt)
    
    # StringSight parameters
    min_cluster_size: int = 3
    embedding_model: str = "text-embedding-3-small"
    extraction_model: str = "gpt-4.1-mini"
    hierarchical: bool = False
    
    # Evaluation parameters
    judge_model: str = "gpt-4.1"
    top_k_behaviors: Optional[int] = None  # Number of top behaviors to evaluate per model (None = all)
    random_seed: int = 42
    
    # Logging parameters
    log_to_wandb: bool = True


@dataclass
class GroundTruthBehavior:
    """Ground truth behavior from benchmark."""
    name: str
    description: str
    category: str
    count: int  # Number of examples in benchmark


@dataclass
class DiscoveredCluster:
    """Discovered cluster from StringSight."""
    cluster_id: int
    label: str
    description: str
    count: int
    example_properties: List[str]  # Sample property descriptions


@dataclass
class MatchResult:
    """Result of rating a discovered cluster against ground truth."""
    ground_truth_behavior: str
    discovered_cluster_id: int
    discovered_cluster_label: str
    score: int  # 1-10 rating
    reasoning: str


def load_benchmark_data(benchmark_path: str, subset_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[GroundTruthBehavior]]:
    """
    Load benchmark data and extract ground truth behaviors.
    
    Args:
        benchmark_path: Path to all_behaviors.jsonl
        subset_size: Optional number of PROMPTS to sample (not total responses)
                     If specified, samples N prompts and includes ALL behavior responses for those prompts
    
    Returns:
        Tuple of (dataframe, list of ground truth behaviors)
    """
    print(f"Loading benchmark data from {benchmark_path}")
    
    # Load the benchmark data
    df = pd.read_json(benchmark_path, lines=True)
    
    print(f"Loaded {len(df)} total responses")
    print(f"Behaviors in benchmark: {df['model'].nunique()}")
    print(f"Columns in benchmark data: {list(df.columns)}")
    num_behaviors = df['model'].nunique()
    
    # Sample if requested - sample PROMPTS, not individual responses
    if subset_size:
        # Get unique prompts
        unique_prompts = df['prompt'].unique()
        print(f"Total unique prompts: {len(unique_prompts)}")
        
        # Sample prompts
        n_prompts = min(subset_size, len(unique_prompts))
        sampled_prompts = np.random.RandomState(42).choice(unique_prompts, size=n_prompts, replace=False)
        
        # Filter to only responses for sampled prompts (includes ALL behaviors for those prompts)
        df = df[df['prompt'].isin(sampled_prompts)]
        
        print(f"Sampled {n_prompts} prompts × {num_behaviors} behaviors = {len(df)} total responses")
    
    # Extract ground truth behaviors (skip baselines as they have no meaningful pattern to evaluate)
    # Note: Baselines are kept in the data for StringSight (provides contrast/reference)
    # but excluded from ground truth evaluation
    ground_truth = []
    for behavior_name in df['model'].unique():
        behavior_df = df[df['model'] == behavior_name]
        category = behavior_df['category'].iloc[0]
        
        # Skip baseline behaviors
        if category == 'baseline':
            print(f"  Skipping baseline behavior: {behavior_name}")
            continue
            
        ground_truth.append(GroundTruthBehavior(
            name=behavior_name,
            description=behavior_df['behavior_description'].iloc[0],
            category=category,
            count=len(behavior_df)
        ))
    
    print(f"\nGround Truth Behaviors ({len(ground_truth)} excluding baselines):")
    for gt in ground_truth:
        print(f"  - {gt.name} ({gt.category}): {gt.count} examples")
    
    return df, ground_truth


def run_stringsight(df: pd.DataFrame, config: EvaluationConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Run StringSight on benchmark data to discover behavioral clusters.
    
    Args:
        df: Benchmark dataframe with ground truth labels
        config: Evaluation configuration
    
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    print(f"\n{'='*80}")
    print("RUNNING STRINGSIGHT")
    print(f"{'='*80}")
    
    # Check if system_prompt column exists
    if 'system_prompt' not in df.columns:
        print("WARNING: 'system_prompt' column not found in benchmark data!")
        print(f"Available columns: {list(df.columns)}")
        print("StringSight will analyze without system prompts, which may reduce property extraction quality.")
        has_system_prompt = False
    else:
        print(f"✓ Found system_prompt column - including in StringSight analysis")
        has_system_prompt = True
    
    # Prepare StringSight input format
    stringsight_input_dict = {
        'prompt': df['prompt'],
        'model': df['model'],
        'model_response': df['model_response'],
    }
    
    # Add system_prompt if available
    if has_system_prompt:
        stringsight_input_dict['system_prompt'] = df['system_prompt']
    
    stringsight_input = pd.DataFrame(stringsight_input_dict)
    
    # Add question_id if not present
    if 'question_id' not in stringsight_input.columns:
        stringsight_input['question_id'] = range(len(stringsight_input))
    
    # Run StringSight
    # Use benchmark filename for output directory (e.g., "instructeval" from "benchmark/results/instructeval/all_behaviors.jsonl")
    benchmark_path = Path(config.benchmark_results_path)
    benchmark_name = benchmark_path.parent.name  # Get parent directory name (e.g., "instructeval")
    output_dir = Path(config.output_dir) / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clustered_df, model_stats = explain(
        stringsight_input,
        method="single_model",
        model_name=config.extraction_model,
        embedding_model=config.embedding_model,
        min_cluster_size=config.min_cluster_size,
        hierarchical=config.hierarchical,
        output_dir=str(output_dir),
        use_wandb=config.log_to_wandb,
        wandb_project="stringsight-benchmark",
        wandb_group=benchmark_name
    )
    
    # Check what cluster columns exist
    cluster_columns = [col for col in clustered_df.columns if 'cluster' in col.lower()]
    print(f"\nAvailable cluster columns: {cluster_columns}")
    
    # Use the fine cluster label column if it exists
    if 'property_description_cluster_label' in clustered_df.columns:
        n_clusters = clustered_df['property_description_cluster_label'].nunique()
    elif 'cluster_label' in clustered_df.columns:
        n_clusters = clustered_df['cluster_label'].nunique()
    elif 'cluster_label' in clustered_df.columns:
        n_clusters = clustered_df['cluster_label'].nunique()
    else:
        n_clusters = "Unknown"
    
    print(f"\nStringSight discovered {n_clusters} clusters")
    
    return clustered_df, model_stats


def extract_top_behaviors_per_model(
    stringsight_output_dir: str,
    top_k: Optional[int] = None
) -> Dict[str, List[DiscoveredCluster]]:
    """
    Extract top K behaviors per model from StringSight output files.
    
    Args:
        stringsight_output_dir: Directory containing StringSight output files
        top_k: Number of top behaviors to extract per model (None = all)
    
    Returns:
        Dict mapping model name to list of its top discovered behaviors
    
    The model_cluster_scores.json file has structure:
    {
      "model_name": {
        "cluster_name": {
          "size": int,
          "proportion": float,
          "proportion_delta": float,
          ...
        }
      }
    }
    """
    print(f"\n{'='*80}")
    print("EXTRACTING TOP BEHAVIORS PER MODEL")
    print(f"{'='*80}")
    
    # Load model_cluster_scores.json which has per-model, per-cluster statistics
    model_cluster_path = Path(stringsight_output_dir) / "model_cluster_scores.json"
    
    if not model_cluster_path.exists():
        print(f"ERROR: Could not find {model_cluster_path}")
        print(f"Available files in {stringsight_output_dir}:")
        for f in Path(stringsight_output_dir).iterdir():
            print(f"  - {f.name}")
        return {}
    
    print(f"Loading model-cluster results from: {model_cluster_path}")
    
    with open(model_cluster_path, 'r') as f:
        model_cluster_scores = json.load(f)
    
    print(f"\nLoaded model_cluster_scores for {len(model_cluster_scores)} models")
    
    # Group by model and extract top K clusters per model
    model_behaviors = {}
    
    for model_name, clusters in model_cluster_scores.items():
        print(f"\n  Model: {model_name}")
        print(f"  Found {len(clusters)} clusters (before filtering)")
        
        # Convert clusters dict to list of (cluster_name, metrics) tuples
        cluster_list = []
        outliers_filtered = 0
        for cluster_name, metrics in clusters.items():
            # Skip clusters containing "outliers" in the name
            if 'outlier' in cluster_name.lower():
                outliers_filtered += 1
                continue
                
            cluster_list.append({
                'cluster': cluster_name,
                'size': metrics.get('size', 0),
                'proportion': metrics.get('proportion', 0),
                'proportion_delta': metrics.get('proportion_delta', 0),
                'quality': metrics.get('quality', {}),
                'quality_delta': metrics.get('quality_delta', {})
            })
        
        if outliers_filtered > 0:
            print(f"  Filtered out {outliers_filtered} outlier cluster(s)")
        print(f"  {len(cluster_list)} clusters remaining after filtering")
        
        # Sort by proportion if only 1 model, otherwise by proportion_delta
        num_models = len(model_cluster_scores)
        if num_models == 1:
            # Single model: sort by frequency (proportion)
            cluster_list.sort(key=lambda x: x['proportion'], reverse=True)
            print(f"  Sorting by proportion (single model analysis)")
        else:
            # Multiple models: sort by frequency delta (proportion_delta)
            cluster_list.sort(key=lambda x: x['proportion_delta'], reverse=True)
            print(f"  Sorting by proportion_delta (multi-model analysis)")
        
        # Limit to top K
        if top_k is not None:
            cluster_list = cluster_list[:top_k]
        
        # Extract discovered behaviors
        discovered_behaviors = []
        for idx, cluster_info in enumerate(cluster_list):
            cluster_label = cluster_info['cluster']
            
            discovered_behaviors.append(DiscoveredCluster(
                cluster_id=idx,
                label=cluster_label,
                description=f"Cluster: {cluster_label}",
                count=int(cluster_info['size']),
                example_properties=[cluster_label]
            ))
        
        model_behaviors[model_name] = discovered_behaviors
        print(f"  Extracted {len(discovered_behaviors)} behaviors (top_k={top_k or 'all'})")
        
        # Print the top behaviors
        print(f"\n  Top {len(discovered_behaviors)} behaviors for {model_name}:")
        for idx, cluster_info in enumerate(cluster_list, 1):
            cluster_label = cluster_info['cluster']
            proportion = cluster_info['proportion']
            proportion_delta = cluster_info['proportion_delta']
            
            print(f"    {idx}. {cluster_label}")
            if num_models == 1:
                print(f"       prop={proportion:.3f}")
            else:
                print(f"       prop={proportion:.3f} Δ={proportion_delta:+.3f}")
    
    return model_behaviors


def match_cluster_to_ground_truth(
    ground_truth: GroundTruthBehavior,
    discovered: DiscoveredCluster,
    model: str
) -> MatchResult:
    """
    Use LLM-as-judge to rate how well a discovered cluster matches ground truth.
    
    Args:
        ground_truth: Ground truth behavior from benchmark
        discovered: Discovered cluster from StringSight
        model: LLM model for judging
    
    Returns:
        Match result with 1-10 score and reasoning
    """
    judge_prompt = f"""You are a machine learning export tasks with comparing the similarity of two behaviors seen in an LLM. Behavior 1 is in the format of a system prompt, and Behavior 2 is a behavior seen in a different model's response. 

Behavior 1: {ground_truth.description}

Behavior 2: {discovered.label}

TASK: Rate how well the behaviors are similar on a scale of 1-10.

RATING SCALE:
- 10: Perfect match - a model response exhibiting the behavior 1 is very likely to exhibit the behavior 2
- 7-9: Strong match - the behaviors highly overlap, a model response exhibiting the behavior 1 is likely to exhibit the behavior 2
- 4-6: Partial match - the behaviors some overlap but there are differences in scope or interpretation
- 1-3: Weak match - behaviors are mostly unrelated or contradictory, a model response exhibiting the behavior 1 is not likely to exhibit the behavior 2

Respond ONLY with valid JSON in this exact format:
{{
  "score": 1-10,
  "reasoning": "brief explanation of your rating"
}}"""
    response_text = single_completion(
        judge_prompt,
        model=model,
        temperature=0.0  # Deterministic for evaluation
    )
    
    # Parse response
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    try:
        result = json.loads(response_text)
        return MatchResult(
            ground_truth_behavior=ground_truth.name,
            discovered_cluster_id=discovered.cluster_id,
            discovered_cluster_label=discovered.label,
            score=result['score'],
            reasoning=result['reasoning']
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse judge response: {e}")
        print(f"Response: {response_text[:200]}")
        return MatchResult(
            ground_truth_behavior=ground_truth.name,
            discovered_cluster_id=discovered.cluster_id,
            discovered_cluster_label=discovered.label,
            score=1,
            reasoning="Failed to parse judge response"
        )


def evaluate_behavior_recovery(
    ground_truth_behaviors: List[GroundTruthBehavior],
    model_behaviors: Dict[str, List[DiscoveredCluster]],
    config: EvaluationConfig
) -> Tuple[List[MatchResult], Dict]:
    """
    Evaluate how well StringSight recovered ground truth behaviors.
    
    For each ground truth behavior, rates the top K discovered behaviors for that model.
    
    Args:
        ground_truth_behaviors: List of ground truth behaviors
        model_behaviors: Dict mapping model name to its top discovered behaviors
        config: Evaluation configuration
    
    Returns:
        Tuple of (all_match_results, summary_metrics)
    """
    print(f"\n{'='*80}")
    print("EVALUATING BEHAVIOR RECOVERY")
    print(f"{'='*80}")
    
    all_match_results = []
    total_comparisons = sum(len(model_behaviors.get(gt.name, [])) for gt in ground_truth_behaviors)
    
    print(f"\nRunning LLM-as-judge for {len(ground_truth_behaviors)} ground truth behaviors")
    print(f"Total comparisons: {total_comparisons}")
    
    # For each ground truth behavior, rate its model's top discovered behaviors
    for gt in tqdm(ground_truth_behaviors, desc="Ground truth behaviors"):
        # Get discovered behaviors for this model
        discovered = model_behaviors.get(gt.name, [])
        
        if not discovered:
            print(f"  Warning: No discovered behaviors for model '{gt.name}'")
            continue
        
        print(f"\n  {gt.name}: Rating {len(discovered)} discovered behaviors")
        
        # Rate each discovered behavior against ground truth
        for disc in discovered:
            match_result = match_cluster_to_ground_truth(gt, disc, config.judge_model)
            all_match_results.append(match_result)
    
    # For each ground truth, find the best matching discovered behavior
    best_matches_per_behavior = {}
    for gt in ground_truth_behaviors:
        gt_matches = [m for m in all_match_results if m.ground_truth_behavior == gt.name]
        if gt_matches:
            best_match = max(gt_matches, key=lambda x: x.score)
            best_matches_per_behavior[gt.name] = {
                'best_score': best_match.score,
                'best_cluster_id': best_match.discovered_cluster_id,
                'best_cluster_label': best_match.discovered_cluster_label,
                'reasoning': best_match.reasoning,
                'all_scores': [m.score for m in gt_matches]  # All scores for this behavior
            }
        else:
            best_matches_per_behavior[gt.name] = {
                'best_score': 0,
                'best_cluster_id': -1,
                'best_cluster_label': 'None',
                'reasoning': 'No discovered behaviors for this model',
                'all_scores': []
            }
    
    # Calculate aggregate metrics
    all_best_scores = [m['best_score'] for m in best_matches_per_behavior.values() if m['best_score'] > 0]
    avg_best_score = np.mean(all_best_scores) if all_best_scores else 0
    
    # Count behaviors with strong matches (score >= 7)
    strong_matches = sum(1 for score in all_best_scores if score >= 7)
    moderate_matches = sum(1 for score in all_best_scores if 4 <= score < 7)
    weak_matches = sum(1 for score in all_best_scores if score < 4)
    
    total_discovered = sum(len(behaviors) for behaviors in model_behaviors.values())
    
    # Calculate top-1, top-5, and top-k scores
    # Note: all_scores are already in order by cluster ranking (proportion_delta descending)
    top1_scores = []
    top5_scores = []
    topk_scores = []
    
    for gt in ground_truth_behaviors:
        if gt.name in best_matches_per_behavior:
            scores = best_matches_per_behavior[gt.name]['all_scores']
            if scores:
                # Top-1: score of the #1 ranked cluster
                top1_scores.append(scores[0])
                
                # Top-5: best score among top 5 clusters
                top5_scores.append(max(scores[:5]) if len(scores) >= 5 else max(scores))
                
                # Top-K: best score among top K clusters (what we already have)
                topk_scores.append(max(scores))
    
    avg_top1_score = np.mean(top1_scores) if top1_scores else 0
    avg_top5_score = np.mean(top5_scores) if top5_scores else 0
    avg_topk_score = np.mean(topk_scores) if topk_scores else 0
    
    metrics = {
        'ground_truth_count': len(ground_truth_behaviors),
        'total_discovered_behaviors': total_discovered,
        'avg_best_score': round(avg_best_score, 2),
        'avg_top1_score': round(avg_top1_score, 2),
        'avg_top5_score': round(avg_top5_score, 2),
        'avg_topk_score': round(avg_topk_score, 2),
        'strong_matches': strong_matches,  # score >= 7
        'moderate_matches': moderate_matches,  # 4 <= score < 7
        'weak_matches': weak_matches,  # score < 4
        'behaviors_with_no_match': len(ground_truth_behaviors) - len(all_best_scores),
        'best_matches_per_behavior': best_matches_per_behavior
    }
    
    return all_match_results, metrics


def save_evaluation_results(
    match_results: List[MatchResult],
    metrics: Dict,
    config: EvaluationConfig
):
    """Save evaluation results to output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ALL match results (ground truth x discovered cluster scores)
    all_scores_dicts = [
        {
            'ground_truth_behavior': m.ground_truth_behavior,
            'discovered_cluster_id': m.discovered_cluster_id,
            'discovered_cluster_label': m.discovered_cluster_label,
            'score': m.score,
            'reasoning': m.reasoning
        }
        for m in match_results
    ]
    
    with open(output_dir / "all_scores.json", 'w') as f:
        json.dump(all_scores_dicts, f, indent=2)
    
    # Save summary metrics (with best matches per behavior)
    with open(output_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save summary text
    summary = f"""StringSight Evaluation Results
{'='*80}

CONFIGURATION:
- Benchmark: {config.benchmark_results_path}
- Subset Size: {config.subset_size or 'All data'} prompts
- Min Cluster Size: {config.min_cluster_size}
- Embedding Model: {config.embedding_model}
- Extraction Model: {config.extraction_model}
- Judge Model: {config.judge_model}

SUMMARY METRICS:
- Ground Truth Behaviors: {metrics['ground_truth_count']}
- Total Discovered Behaviors: {metrics['total_discovered_behaviors']}
- Top-1 Score: {metrics['avg_top1_score']:.2f}/10 (score of #1 ranked cluster)
- Top-5 Score: {metrics['avg_top5_score']:.2f}/10 (best score in top 5 clusters)
- Top-K Score: {metrics['avg_topk_score']:.2f}/10 (best score in top K clusters)
- Average Best Score: {metrics['avg_best_score']:.2f}/10 (overall best match)

MATCH QUALITY:
- Strong Matches (7-10): {metrics['strong_matches']}/{metrics['ground_truth_count']} behaviors
- Moderate Matches (4-6): {metrics['moderate_matches']}/{metrics['ground_truth_count']} behaviors
- Weak Matches (1-3): {metrics['weak_matches']}/{metrics['ground_truth_count']} behaviors
- No Match: {metrics['behaviors_with_no_match']}/{metrics['ground_truth_count']} behaviors

BEST MATCHES PER BEHAVIOR:
"""
    
    for behavior_name, match_info in metrics['best_matches_per_behavior'].items():
        summary += f"\n{behavior_name}:\n"
        summary += f"  Score: {match_info['best_score']}/10\n"
        summary += f"  Matched to: {match_info['best_cluster_label']}\n"
        summary += f"  Reasoning: {match_info['reasoning']}\n"
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - all_scores.json (all GT x Cluster scores)")
    print(f"  - evaluation_metrics.json (summary metrics)")
    print(f"  - summary.txt (human-readable report)")


def log_to_wandb(
    ground_truth_behaviors: List[GroundTruthBehavior],
    model_behaviors: Dict[str, List[DiscoveredCluster]],
    match_results: List[MatchResult],
    metrics: Dict,
    config: EvaluationConfig
):
    """Log evaluation results table to wandb (using StringSight's existing run)."""
    if not config.log_to_wandb or not WANDB_AVAILABLE:
        print("\nSkipping wandb logging (disabled or wandb not available)")
        return
    
    if not wandb.run:
        print("\nWarning: No active wandb run found. Skipping evaluation table logging.")
        return
    
    print(f"\n{'='*80}")
    print("LOGGING EVALUATION TABLE TO WANDB")
    print(f"{'='*80}")
    
    # Create a single table with 3 columns: gt behavior, generated behaviors, scores
    # Group match results by ground truth behavior
    table_data = []
    
    for gt in ground_truth_behaviors:
        # Get all matches for this ground truth behavior
        gt_matches = [m for m in match_results if m.ground_truth_behavior == gt.name]
        
        if gt_matches:
            # Get the best match
            best_match = max(gt_matches, key=lambda x: x.score)
            
            # Create a comma-separated list of discovered behaviors with scores
            discovered_list = ", ".join([
                f"{m.discovered_cluster_label} ({m.score}/10)" 
                for m in sorted(gt_matches, key=lambda x: x.score, reverse=True)
            ])
            
            table_data.append([
                gt.name,
                discovered_list,
                best_match.score
            ])
        else:
            table_data.append([
                gt.name,
                "No matches found",
                0
            ])
    
    evaluation_table = wandb.Table(
        columns=["gt behavior", "generated behaviors", "scores"],
        data=table_data
    )
    
    # Log the table and summary metrics to the existing StringSight run
    wandb.log({
        "Evaluation/benchmark_scores": evaluation_table
    })
    
    # Log summary metrics (these will appear in the wandb summary)
    wandb.summary["eval/top1_score"] = metrics["avg_top1_score"]
    wandb.summary["eval/top5_score"] = metrics["avg_top5_score"]
    wandb.summary["eval/topk_score"] = metrics["avg_topk_score"]
    wandb.summary["eval/avg_best_score"] = metrics["avg_best_score"]
    wandb.summary["eval/strong_matches"] = metrics["strong_matches"]
    wandb.summary["eval/moderate_matches"] = metrics["moderate_matches"]
    wandb.summary["eval/weak_matches"] = metrics["weak_matches"]
    
    print(f"\n✅ Logged to wandb:")
    print(f"  - Evaluation table ({len(table_data)} ground truth behaviors)")
    print(f"  - Summary metrics (top-1: {metrics['avg_top1_score']:.2f}, top-5: {metrics['avg_top5_score']:.2f}, top-k: {metrics['avg_topk_score']:.2f})")


def evaluate_stringsight(config: EvaluationConfig):
    """Main evaluation pipeline."""
    print(f"{'='*80}")
    print("STRINGSIGHT BENCHMARK EVALUATION")
    print(f"{'='*80}\n")
    
    # 1. Load benchmark data
    df, ground_truth_behaviors = load_benchmark_data(
        config.benchmark_results_path,
        config.subset_size
    )
    
    # 2. Run StringSight
    clustered_df, model_stats = run_stringsight(df, config)
    
    # 3. Extract top behaviors per model from StringSight output files
    benchmark_name = Path(config.benchmark_results_path).parent.name
    stringsight_output_dir = str(Path(config.output_dir) / benchmark_name)
    model_behaviors = extract_top_behaviors_per_model(
        stringsight_output_dir,
        top_k=config.top_k_behaviors
    )
    wandb.config["benchmark_name"] = benchmark_name
    
    # 4. Evaluate behavior recovery
    match_results, metrics = evaluate_behavior_recovery(
        ground_truth_behaviors,
        model_behaviors,
        config
    )
    
    # 5. Save results
    save_evaluation_results(match_results, metrics, config)
    
    # 6. Log to wandb
    log_to_wandb(ground_truth_behaviors, model_behaviors, match_results, metrics, config)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nScore Metrics:")
    print(f"  Top-1 Score: {metrics['avg_top1_score']:.2f}/10 (score of #1 ranked cluster)")
    print(f"  Top-5 Score: {metrics['avg_top5_score']:.2f}/10 (best in top 5 clusters)")
    print(f"  Top-K Score: {metrics['avg_topk_score']:.2f}/10 (best in top K clusters)")
    print(f"  Average Best: {metrics['avg_best_score']:.2f}/10 (overall best match)")
    print(f"\nMatch Quality:")
    print(f"  Strong (7-10): {metrics['strong_matches']}/{metrics['ground_truth_count']} behaviors")
    print(f"  Moderate (4-6): {metrics['moderate_matches']}/{metrics['ground_truth_count']} behaviors")
    print(f"  Weak (1-3): {metrics['weak_matches']}/{metrics['ground_truth_count']} behaviors")
    print(f"  No Match: {metrics['behaviors_with_no_match']}/{metrics['ground_truth_count']} behaviors")
    print(f"\nGround Truth Behaviors: {metrics['ground_truth_count']}")
    print(f"Total Discovered Behaviors: {metrics['total_discovered_behaviors']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate StringSight on benchmark dataset")
    parser.add_argument("--benchmark-results", type=str, required=True,
                        help="Path to all_behaviors.jsonl from benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark/evaluation_results/",
                        help="Output directory for evaluation results")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Number of prompts to sample (None = use all). "
                             "Samples N prompts and includes ALL behavior responses for those prompts. "
                             "E.g., --subset-size 10 with 12 behaviors = 10×12=120 total responses.")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="Minimum cluster size for StringSight")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small",
                        help="Embedding model for StringSight clustering")
    parser.add_argument("--extraction-model", type=str, default="gpt-4.1-mini",
                        help="Model for StringSight property extraction")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1",
                        help="Model for LLM-as-judge evaluation")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of top behaviors to evaluate per model (None = all)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Enable hierarchical clustering in StringSight")
    parser.add_argument("--log-to-wandb", dest="log_to_wandb", action="store_true", default=True,
                        help="Log results to wandb (default: True)")
    parser.add_argument("--no-wandb", dest="log_to_wandb", action="store_false",
                        help="Disable wandb logging")
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        benchmark_results_path=args.benchmark_results,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        min_cluster_size=args.min_cluster_size,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        hierarchical=args.hierarchical,
        judge_model=args.judge_model,
        top_k_behaviors=args.top_k,
        log_to_wandb=args.log_to_wandb
    )
    
    evaluate_stringsight(config)

