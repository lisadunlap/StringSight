#!/usr/bin/env python3
"""
Test script to demonstrate confidence interval functionality in StringSight metrics.

This script creates a simple synthetic dataset and shows how confidence intervals
are computed for both distinctiveness scores and quality scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Import the metrics classes
from stringsight.metrics.single_model import SingleModelMetrics
from stringsight.metrics.side_by_side import SideBySideMetrics
from stringsight.core.data_objects import PropertyDataset


def create_synthetic_single_model_data():
    """Create synthetic single-model data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create synthetic data
    n_questions = 100
    n_models = 3
    models = ["gpt-4", "claude-3", "llama-2"]
    
    data = []
    for i in range(n_questions):
        for model in models:
            # Create some clusters
            cluster_id = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            cluster_label = f"cluster_{cluster_id}"
            
            # Create synthetic scores
            accuracy = np.random.normal(0.8, 0.1)
            helpfulness = np.random.normal(4.0, 0.5)
            
            data.append({
                "question_id": f"q_{i}",
                "prompt": f"Test prompt {i}",
                "model": model,
                "model_response": f"Response from {model}",
                "score": {
                    "accuracy": max(0, min(1, accuracy)),
                    "helpfulness": max(1, min(5, helpfulness))
                },
                "id": f"prop_{i}_{model}",
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "coarse_cluster_id": cluster_id // 2,
                "coarse_cluster_label": f"coarse_{cluster_id // 2}",
            })
    
    return pd.DataFrame(data)


def create_synthetic_side_by_side_data():
    """Create synthetic side-by-side data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create synthetic data
    n_questions = 100
    models = ["gpt-4", "claude-3", "llama-2"]
    
    data = []
    for i in range(n_questions):
        # Randomly select two models for comparison
        model_a, model_b = np.random.choice(models, size=2, replace=False)
        
        # Create some clusters
        cluster_id = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        cluster_label = f"cluster_{cluster_id}"
        
        # Create synthetic scores
        accuracy_a = np.random.normal(0.8, 0.1)
        accuracy_b = np.random.normal(0.8, 0.1)
        helpfulness_a = np.random.normal(4.0, 0.5)
        helpfulness_b = np.random.normal(4.0, 0.5)
        
        # Determine winner
        total_score_a = accuracy_a + helpfulness_a / 5
        total_score_b = accuracy_b + helpfulness_b / 5
        
        if total_score_a > total_score_b:
            winner = model_a
        elif total_score_b > total_score_a:
            winner = model_b
        else:
            winner = "tie"
        
        # Create two rows for the comparison
        for model, response, accuracy, helpfulness in [
            (model_a, f"Response from {model_a}", accuracy_a, helpfulness_a),
            (model_b, f"Response from {model_b}", accuracy_b, helpfulness_b)
        ]:
            data.append({
                "question_id": f"q_{i}",
                "prompt": f"Test prompt {i}",
                "model": model,
                "model_response": response,
                "score": {
                    "accuracy": max(0, min(1, accuracy)),
                    "helpfulness": max(1, min(5, helpfulness)),
                    "winner": winner
                },
                "id": f"prop_{i}_{model}",
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "coarse_cluster_id": cluster_id // 2,
                "coarse_cluster_label": f"coarse_{cluster_id // 2}",
            })
    
    return pd.DataFrame(data)


def test_single_model_confidence_intervals():
    """Test confidence intervals with single model data."""
    print("=" * 60)
    print("TESTING SINGLE MODEL CONFIDENCE INTERVALS")
    print("=" * 60)
    
    # Create synthetic data
    df = create_synthetic_single_model_data()
    print(f"Created synthetic dataset with {len(df)} rows")
    print(f"Models: {df['model'].unique()}")
    print(f"Clusters: {df['cluster_label'].unique()}")
    
    # Create PropertyDataset
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Test with confidence intervals enabled
    print("\n🔧 Testing with confidence intervals enabled...")
    metrics_stage = SingleModelMetrics(
        compute_confidence_intervals=True,
        bootstrap_samples=500  # Use fewer samples for faster testing
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        result_dataset = metrics_stage.run(dataset)
        
        # Save results
        stats_path = Path(temp_dir) / "model_stats.json"
        stats_for_json = {}
        for model_name, stats in result_dataset.model_stats.items():
            stats_for_json[str(model_name)] = {
                "fine": [stat.to_dict() for stat in stats["fine"]],
                "coarse": [stat.to_dict() for stat in stats["coarse"]],
                "stats": stats.get("stats", {})
            }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        
        # Display some results
        print(f"\n📊 Results saved to: {stats_path}")
        
        # Show confidence intervals for first model and cluster
        first_model = list(result_dataset.model_stats.keys())[0]
        first_cluster = result_dataset.model_stats[first_model]["fine"][0]
        
        print(f"\n📈 Example confidence intervals for {first_model}:")
        print(f"  Cluster: {first_cluster.property_description}")
        print(f"  Distinctiveness Score: {first_cluster.score:.3f}")
        print(f"  Score CI: [{first_cluster.score_ci_lower:.3f}, {first_cluster.score_ci_upper:.3f}]")
        
        if first_cluster.quality_score_ci:
            print(f"  Quality Score CIs:")
            for key, ci in first_cluster.quality_score_ci.items():
                print(f"    {key}: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    
    # Test with confidence intervals disabled
    print("\n🔧 Testing with confidence intervals disabled...")
    metrics_stage_no_ci = SingleModelMetrics(
        compute_confidence_intervals=False
    )
    
    result_dataset_no_ci = metrics_stage_no_ci.run(dataset)
    first_cluster_no_ci = result_dataset_no_ci.model_stats[first_model]["fine"][0]
    
    print(f"  Score CI (disabled): [{first_cluster_no_ci.score_ci_lower}, {first_cluster_no_ci.score_ci_upper}]")
    print(f"  Quality Score CI (disabled): {first_cluster_no_ci.quality_score_ci}")


def test_side_by_side_confidence_intervals():
    """Test confidence intervals with side-by-side data."""
    print("\n" + "=" * 60)
    print("TESTING SIDE-BY-SIDE CONFIDENCE INTERVALS")
    print("=" * 60)
    
    # Create synthetic data
    df = create_synthetic_side_by_side_data()
    print(f"Created synthetic dataset with {len(df)} rows")
    print(f"Models: {df['model'].unique()}")
    print(f"Clusters: {df['cluster_label'].unique()}")
    
    # Create PropertyDataset
    dataset = PropertyDataset.from_dataframe(df, method="side_by_side")
    
    # Test with confidence intervals enabled
    print("\n🔧 Testing with confidence intervals enabled...")
    metrics_stage = SideBySideMetrics(
        compute_confidence_intervals=True,
        bootstrap_samples=500  # Use fewer samples for faster testing
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        result_dataset = metrics_stage.run(dataset)
        
        # Save results
        stats_path = Path(temp_dir) / "model_stats.json"
        stats_for_json = {}
        for model_name, stats in result_dataset.model_stats.items():
            stats_for_json[str(model_name)] = {
                "fine": [stat.to_dict() for stat in stats["fine"]],
                "coarse": [stat.to_dict() for stat in stats["coarse"]],
                "stats": stats.get("stats", {})
            }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        
        # Display some results
        print(f"\n📊 Results saved to: {stats_path}")
        
        # Show confidence intervals for first model and cluster
        first_model = list(result_dataset.model_stats.keys())[0]
        first_cluster = result_dataset.model_stats[first_model]["fine"][0]
        
        print(f"\n📈 Example confidence intervals for {first_model}:")
        print(f"  Cluster: {first_cluster.property_description}")
        print(f"  Distinctiveness Score: {first_cluster.score:.3f}")
        print(f"  Score CI: [{first_cluster.score_ci_lower:.3f}, {first_cluster.score_ci_upper:.3f}]")
        
        if first_cluster.quality_score_ci:
            print(f"  Quality Score CIs:")
            for key, ci in first_cluster.quality_score_ci.items():
                print(f"    {key}: [{ci['lower']:.3f}, {ci['upper']:.3f}]")


def main():
    """Run all tests."""
    print("🧪 Testing StringSight Confidence Intervals")
    print("=" * 60)
    
    try:
        test_single_model_confidence_intervals()
        test_side_by_side_confidence_intervals()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 