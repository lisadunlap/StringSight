#!/usr/bin/env python3
"""
Test script to verify the new metrics functionality:
1. Global stats computation
2. Normalized quality scores
"""

import pandas as pd
import numpy as np
from stringsight import PropertyDataset
from stringsight.metrics.single_model import SingleModelMetrics
from stringsight.metrics.side_by_side import SideBySideMetrics

def create_test_data_single_model():
    """Create test data for single model metrics."""
    data = []
    models = ["model_a", "model_b", "model_c"]
    
    for i in range(30):
        model = models[i % 3]
        cluster_id = i // 10  # 3 clusters
        
        # Create score dict with multiple metrics
        score = {
            "accuracy": np.random.uniform(0.5, 1.0),
            "helpfulness": np.random.uniform(1, 5),
            "safety": np.random.uniform(0, 1)
        }
        
        data.append({
            "id": f"test_{i}",
            "model": model,
            "question_id": f"q_{i}",
            "score": score,
            "cluster_id": cluster_id,
            "cluster_label": f"cluster_{cluster_id}",
            "coarse_cluster_id": 0,
            "coarse_cluster_label": "coarse_cluster_0"
        })
    
    return pd.DataFrame(data)

def create_test_data_side_by_side():
    """Create test data for side-by-side metrics."""
    data = []
    models = ["model_a", "model_b", "model_c"]
    
    for i in range(30):
        model = models[i % 3]
        cluster_id = i // 10  # 3 clusters
        
        # Create score dict with winner and other metrics
        winner = model if np.random.random() > 0.3 else "model_b"  # model_a wins 70% of the time
        score = {
            "winner": winner,
            "helpfulness": np.random.uniform(1, 5),
            "safety": np.random.uniform(0, 1)
        }
        
        data.append({
            "id": f"test_{i}",
            "model": model,
            "question_id": f"q_{i}",
            "score": score,
            "cluster_id": cluster_id,
            "cluster_label": f"cluster_{cluster_id}",
            "coarse_cluster_id": 0,
            "coarse_cluster_label": "coarse_cluster_0"
        })
    
    return pd.DataFrame(data)

def test_single_model_metrics():
    """Test single model metrics with new functionality."""
    print("Testing Single Model Metrics...")
    
    # Create test data
    df = create_test_data_single_model()
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Run metrics
    metrics = SingleModelMetrics()
    result = metrics.run(dataset)
    
    # Check results
    model_stats = result.model_stats
    
    print(f"Number of models: {len(model_stats)}")
    
    for model, stats in model_stats.items():
        print(f"\nModel: {model}")
        print(f"  Global stats: {stats.get('stats', {})}")
        
        if "fine" in stats:
            print(f"  Fine clusters: {len(stats['fine'])}")
            for cluster in stats["fine"][:2]:  # Show first 2 clusters
                print(f"    Cluster: {cluster.property_description}")
                print(f"      Quality score: {cluster.quality_score}")
                print(f"      Score: {cluster.score}")

def test_side_by_side_metrics():
    """Test side-by-side metrics with new functionality."""
    print("\nTesting Side-by-Side Metrics...")
    
    # Create test data
    df = create_test_data_side_by_side()
    dataset = PropertyDataset.from_dataframe(df, method="side_by_side")
    
    # Run metrics
    metrics = SideBySideMetrics()
    result = metrics.run(dataset)
    
    # Check results
    model_stats = result.model_stats
    
    print(f"Number of models: {len(model_stats)}")
    
    for model, stats in model_stats.items():
        print(f"\nModel: {model}")
        print(f"  Global stats: {stats.get('stats', {})}")
        
        if "fine" in stats:
            print(f"  Fine clusters: {len(stats['fine'])}")
            for cluster in stats["fine"][:2]:  # Show first 2 clusters
                print(f"    Cluster: {cluster.property_description}")
                print(f"      Quality score: {cluster.quality_score}")
                print(f"      Score: {cluster.score}")

def test_quality_score_normalization():
    """Test that quality scores are properly normalized."""
    print("\nTesting Quality Score Normalization...")
    
    # Create test data with known patterns
    data = []
    models = ["model_a", "model_b"]
    
    # Create data where model_a has higher scores than model_b
    for i in range(20):
        model = models[i % 2]
        cluster_id = i // 10
        
        if model == "model_a":
            score = {"accuracy": 0.9, "helpfulness": 4.5}  # High scores
        else:
            score = {"accuracy": 0.6, "helpfulness": 2.5}  # Lower scores
        
        data.append({
            "id": f"test_{i}",
            "model": model,
            "question_id": f"q_{i}",
            "score": score,
            "cluster_id": cluster_id,
            "cluster_label": f"cluster_{cluster_id}",
            "coarse_cluster_id": 0,
            "coarse_cluster_label": "coarse_cluster_0"
        })
    
    df = pd.DataFrame(data)
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Run metrics
    metrics = SingleModelMetrics()
    result = metrics.run(dataset)
    
    # Check that quality scores are normalized
    model_stats = result.model_stats
    
    for model, stats in model_stats.items():
        print(f"\nModel: {model}")
        if "fine" in stats and stats["fine"]:
            cluster = stats["fine"][0]
            print(f"  Quality scores: {cluster.quality_score}")
            
            # Check that scores are normalized (should be positive for model_a, negative for model_b)
            for metric, score in cluster.quality_score.items():
                if model == "model_a":
                    assert score > 0, f"Model A should have positive normalized score for {metric}"
                else:
                    assert score < 0, f"Model B should have negative normalized score for {metric}"
    
    print("✅ Quality score normalization test passed!")

if __name__ == "__main__":
    test_single_model_metrics()
    test_side_by_side_metrics()
    test_quality_score_normalization()
    print("\n🎉 All tests completed successfully!") 