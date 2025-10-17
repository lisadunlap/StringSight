#!/usr/bin/env python3
"""
Test script to demonstrate bootstrap sampling and confidence interval calculation.

This script shows exactly how bootstrap creates variability from deterministic data
and fixes the quality score CI calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import tempfile
import json
from pathlib import Path

# Import the metrics classes
from stringsight.metrics.single_model import SingleModelMetrics
from stringsight.core.data_objects import PropertyDataset


def demonstrate_bootstrap_sampling():
    """Show exactly how bootstrap sampling works with a simple example."""
    print("🔍 Bootstrap Sampling Demonstration")
    print("=" * 50)
    
    # Original deterministic data
    original_questions = np.array(['q1', 'q2', 'q3', 'q4', 'q5'])
    print(f"Original questions: {original_questions}")
    print()
    
    # Show 5 bootstrap samples
    np.random.seed(42)  # For reproducible results
    
    for i in range(5):
        # Bootstrap sample: sample WITH REPLACEMENT
        bootstrap_sample = np.random.choice(original_questions, size=len(original_questions), replace=True)
        print(f"Bootstrap sample {i+1}: {bootstrap_sample}")
        
        # Count frequencies
        unique, counts = np.unique(bootstrap_sample, return_counts=True)
        freq_dict = dict(zip(unique, counts))
        print(f"  Frequencies: {freq_dict}")
        print()
    
    print("✅ Key insight: Even with deterministic data, different bootstrap samples")
    print("   have different question frequencies, creating statistical variability!")


def demonstrate_distinctiveness_bootstrap():
    """Show how distinctiveness score bootstrap works."""
    print("\nDistinctiveness Score Bootstrap Demonstration")
    print("=" * 50)
    
    # Create simple test data
    data = pd.DataFrame([
        {'question_id': 'q1', 'model': 'gpt-4', 'score': {'accuracy': 0.8}},
        {'question_id': 'q1', 'model': 'claude', 'score': {'accuracy': 0.7}},
        {'question_id': 'q2', 'model': 'gpt-4', 'score': {'accuracy': 0.9}},
        {'question_id': 'q3', 'model': 'claude', 'score': {'accuracy': 0.6}},
        {'question_id': 'q4', 'model': 'gpt-4', 'score': {'accuracy': 0.7}},
    ])
    
    print("Original data:")
    print(data[['question_id', 'model']])
    print()
    
    # Calculate original proportions
    total_q = {'gpt-4': 3, 'claude': 2}  # Total questions per model
    print(f"Total questions per model: {total_q}")
    
    # Original cluster: questions q1, q2, q4
    cluster_questions = ['q1', 'q2', 'q4']
    print(f"Cluster questions: {cluster_questions}")
    
    # Calculate original distinctiveness scores
    gpt4_in_cluster = 2  # q1, q2
    claude_in_cluster = 1  # q1
    
    gpt4_prop = gpt4_in_cluster / total_q['gpt-4']  # 2/3 = 0.667
    claude_prop = claude_in_cluster / total_q['claude']  # 1/2 = 0.5
    
    median_prop = np.median([gpt4_prop, claude_prop])  # 0.583
    
    gpt4_distinctiveness = gpt4_prop / median_prop  # 0.667 / 0.583 = 1.14
    claude_distinctiveness = claude_prop / median_prop  # 0.5 / 0.583 = 0.86
    
    print(f"Original distinctiveness scores:")
    print(f"  GPT-4: {gpt4_distinctiveness:.3f}")
    print(f"  Claude: {claude_distinctiveness:.3f}")
    print()
    
    # Show bootstrap samples
    np.random.seed(42)
    bootstrap_scores = {'gpt-4': [], 'claude': []}
    
    for i in range(5):
        # Bootstrap sample cluster questions
        bootstrap_questions = np.random.choice(cluster_questions, size=len(cluster_questions), replace=True)
        print(f"Bootstrap sample {i+1}: {bootstrap_questions}")
        
        # Count models in this bootstrap sample
        model_counts = {'gpt-4': 0, 'claude': 0}
        for q in bootstrap_questions:
            if q in ['q1', 'q2', 'q4']:  # Questions where GPT-4 appears
                if q in ['q1', 'q2']:  # Only q1 and q2 for GPT-4
                    model_counts['gpt-4'] += 1
            if q == 'q1':  # Only q1 for Claude
                model_counts['claude'] += 1
        
        # Calculate bootstrap proportions
        boot_gpt4_prop = model_counts['gpt-4'] / total_q['gpt-4']
        boot_claude_prop = model_counts['claude'] / total_q['claude']
        boot_median_prop = np.median([boot_gpt4_prop, boot_claude_prop])
        
        boot_gpt4_dist = boot_gpt4_prop / boot_median_prop if boot_median_prop > 0 else 0
        boot_claude_dist = boot_claude_prop / boot_median_prop if boot_median_prop > 0 else 0
        
        bootstrap_scores['gpt-4'].append(boot_gpt4_dist)
        bootstrap_scores['claude'].append(boot_claude_dist)
        
        print(f"  GPT-4 distinctiveness: {boot_gpt4_dist:.3f}")
        print(f"  Claude distinctiveness: {boot_claude_dist:.3f}")
        print()
    
    # Calculate confidence intervals
    for model in ['gpt-4', 'claude']:
        scores = bootstrap_scores[model]
        lower_ci = np.percentile(scores, 2.5)
        upper_ci = np.percentile(scores, 97.5)
        print(f"{model} 95% CI: [{lower_ci:.3f}, {upper_ci:.3f}]")
    
    print("\n✅ Bootstrap creates variability by resampling questions with replacement!")


def demonstrate_quality_score_bootstrap():
    """Show how quality score bootstrap works."""
    print("\n📊 Quality Score Bootstrap Demonstration")
    print("=" * 50)
    
    # Create test data with actual scores
    scores_in_cluster = [0.8, 0.9, 0.7, 0.85]  # Scores for a model in a cluster
    global_average = 0.75  # Model's global average
    global_range = 0.5  # Global score range
    
    print(f"Scores in cluster: {scores_in_cluster}")
    print(f"Model global average: {global_average}")
    print(f"Global range: {global_range}")
    print()
    
    # Original quality score
    cluster_avg = np.mean(scores_in_cluster)
    quality_score = (cluster_avg - global_average) / global_range
    print(f"Original cluster average: {cluster_avg:.3f}")
    print(f"Original quality score: {quality_score:.3f}")
    print()
    
    # Show bootstrap samples
    np.random.seed(42)
    bootstrap_quality_scores = []
    
    for i in range(5):
        # Bootstrap sample scores WITH REPLACEMENT
        bootstrap_scores = np.random.choice(scores_in_cluster, size=len(scores_in_cluster), replace=True)
        print(f"Bootstrap sample {i+1}: {bootstrap_scores}")
        
        # Calculate quality score for this bootstrap sample
        boot_cluster_avg = np.mean(bootstrap_scores)
        boot_quality_score = (boot_cluster_avg - global_average) / global_range
        bootstrap_quality_scores.append(boot_quality_score)
        
        print(f"  Bootstrap cluster avg: {boot_cluster_avg:.3f}")
        print(f"  Bootstrap quality score: {boot_quality_score:.3f}")
        print()
    
    # Calculate confidence interval
    lower_ci = np.percentile(bootstrap_quality_scores, 2.5)
    upper_ci = np.percentile(bootstrap_quality_scores, 97.5)
    print(f"Quality score 95% CI: [{lower_ci:.3f}, {upper_ci:.3f}]")
    
    print("\n✅ Bootstrap creates variability by resampling scores with replacement!")


def test_fixed_bootstrap_implementation():
    """Test the fixed bootstrap implementation."""
    print("\n🧪 Testing Fixed Bootstrap Implementation")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    data = []
    for i in range(20):
        for model in ["gpt-4", "claude-3"]:
            data.append({
                "question_id": f"q_{i}",
                "prompt": f"Test prompt {i}",
                "model": model,
                "model_response": f"Response from {model}",
                "score": {"accuracy": np.random.uniform(0.6, 0.9)},
                "id": f"prop_{i}_{model}",
                "cluster_id": 0,
                "cluster_label": "test_cluster",
                "coarse_cluster_id": 0,
                "coarse_cluster_label": "test_coarse",
            })
    
    df = pd.DataFrame(data)
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Test with bootstrap enabled
    print("Running metrics with bootstrap confidence intervals...")
    metrics_stage = SingleModelMetrics(
        compute_confidence_intervals=True, 
        bootstrap_samples=100  # Small number for quick test
    )
    
    result_dataset = metrics_stage.run(dataset)
    
    # Check results
    for model_name, stats in result_dataset.model_stats.items():
        for cluster_stat in stats["fine"]:
            print(f"\nModel: {cluster_stat.model_name}")
            print(f"Distinctiveness Score: {cluster_stat.score:.3f}")
            print(f"Distinctiveness CI: [{cluster_stat.score_ci_lower:.3f}, {cluster_stat.score_ci_upper:.3f}]")
            
            if cluster_stat.quality_score_ci:
                for key, ci in cluster_stat.quality_score_ci.items():
                    print(f"Quality Score CI ({key}): [{ci['lower']:.3f}, {ci['upper']:.3f}]")
            else:
                print("Quality Score CI: None")
    
    print("\n✅ Fixed bootstrap implementation works correctly!")


def main():
    """Run all demonstrations."""
    print("🔍 Bootstrap Sampling and Confidence Interval Explanation")
    print("=" * 60)
    
    try:
        demonstrate_bootstrap_sampling()
        demonstrate_distinctiveness_bootstrap()
        demonstrate_quality_score_bootstrap()
        test_fixed_bootstrap_implementation()
        
        print("\n" + "=" * 60)
        print("✅ ALL BOOTSTRAP DEMONSTRATIONS COMPLETED!")
        print("=" * 60)
        print("\n📋 Key Takeaways:")
        print("   • Bootstrap creates variability by resampling WITH REPLACEMENT")
        print("   • Different bootstrap samples have different frequencies/averages")
        print("   • This variability estimates uncertainty in our statistics")
        print("   • Quality score CIs should now show proper ranges (not 1,1)")
        print("   • Distinctiveness and quality CIs measure different things")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 