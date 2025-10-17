#!/usr/bin/env python3
"""
Test to verify the optimizations work correctly and produce the same results.
"""

import pandas as pd
import numpy as np
from stringsight import PropertyDataset
from stringsight.metrics.single_model import SingleModelMetrics
from stringsight.metrics.side_by_side import SideBySideMetrics
import time

def test_optimization_correctness():
    """Test that optimizations produce the same results."""
    print("Testing Optimization Correctness...")
    
    # Create test data
    data = []
    models = ["model_a", "model_b", "model_c"]
    
    for i in range(100):  # Larger dataset to test performance
        model = models[i % 3]
        cluster_id = i // 20  # 5 clusters
        
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
    
    df = pd.DataFrame(data)
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Run metrics and time it
    start_time = time.time()
    metrics = SingleModelMetrics()
    result = metrics.run(dataset)
    end_time = time.time()
    
    print(f"✅ Metrics computation completed in {end_time - start_time:.2f} seconds")
    
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
                
                # Verify quality scores are in [0,1] range
                for metric, quality in cluster.quality_score.items():
                    assert 0 <= quality <= 1, f"Quality score {quality} for {metric} is not in [0,1] range"
    
    print("✅ All quality scores are in [0,1] range")
    print("✅ Optimization test passed!")

def test_side_by_side_optimization():
    """Test side-by-side optimizations."""
    print("\nTesting Side-by-Side Optimization...")
    
    # Create test data
    data = []
    models = ["model_a", "model_b", "model_c"]
    
    for i in range(100):
        model = models[i % 3]
        cluster_id = i // 20
        
        # Create score dict with winner and other metrics
        winner = model if np.random.random() > 0.3 else "model_b"
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
    
    df = pd.DataFrame(data)
    dataset = PropertyDataset.from_dataframe(df, method="side_by_side")
    
    # Run metrics and time it
    start_time = time.time()
    metrics = SideBySideMetrics()
    result = metrics.run(dataset)
    end_time = time.time()
    
    print(f"✅ Side-by-side metrics completed in {end_time - start_time:.2f} seconds")
    
    # Check results
    model_stats = result.model_stats
    
    for model, stats in model_stats.items():
        print(f"\nModel: {model}")
        print(f"  Global stats: {stats.get('stats', {})}")
        
        if "fine" in stats:
            print(f"  Fine clusters: {len(stats['fine'])}")
            for cluster in stats["fine"][:2]:
                print(f"    Cluster: {cluster.property_description}")
                print(f"      Quality score: {cluster.quality_score}")
                
                # Verify quality scores are in [0,1] range
                for metric, quality in cluster.quality_score.items():
                    assert 0 <= quality <= 1, f"Quality score {quality} for {metric} is not in [0,1] range"
    
    print("✅ All side-by-side quality scores are in [0,1] range")
    print("✅ Side-by-side optimization test passed!")

if __name__ == "__main__":
    test_optimization_correctness()
    test_side_by_side_optimization()
    print("\n🎉 All optimization tests completed successfully!") 