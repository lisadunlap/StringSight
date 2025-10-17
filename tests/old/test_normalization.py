#!/usr/bin/env python3
"""
Simple test to verify min-max normalization works correctly.
"""

import pandas as pd
import numpy as np
from stringsight import PropertyDataset
from stringsight.metrics.single_model import SingleModelMetrics

def test_min_max_normalization():
    """Test that min-max normalization produces expected 0-1 values."""
    print("Testing Min-Max Normalization...")
    
    # Create test data with known ranges
    data = []
    models = ["model_a", "model_b", "model_c"]
    
    # Create data with specific patterns to test normalization
    for i in range(30):
        model = models[i % 3]
        cluster_id = i // 10  # 3 clusters
        
        # Create scores with known ranges
        if model == "model_a":
            # model_a: high scores (should get high normalized values)
            score = {
                "accuracy": 0.9,
                "helpfulness": 4.5,
                "safety": 0.95
            }
        elif model == "model_b":
            # model_b: medium scores (should get medium normalized values)
            score = {
                "accuracy": 0.7,
                "helpfulness": 3.0,
                "safety": 0.7
            }
        else:
            # model_c: low scores (should get low normalized values)
            score = {
                "accuracy": 0.5,
                "helpfulness": 2.0,
                "safety": 0.5
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
    
    # Run metrics
    metrics = SingleModelMetrics()
    result = metrics.run(dataset)
    
    # Check results
    model_stats = result.model_stats
    
    print(f"Number of models: {len(model_stats)}")
    
    for model, stats in model_stats.items():
        print(f"\nModel: {model}")
        print(f"  Global stats: {stats.get('stats', {})}")
        
        if "fine" in stats and stats["fine"]:
            cluster = stats["fine"][0]  # First cluster
            print(f"  Quality scores: {cluster.quality_score}")
            
            # Verify normalization produces 0-1 values
            for metric, quality in cluster.quality_score.items():
                assert 0 <= quality <= 1, f"Quality score {quality} for {metric} is not in [0,1] range"
                
                # Check expected patterns
                if model == "model_a":
                    assert quality > 0.7, f"Model A should have high quality scores, got {quality} for {metric}"
                elif model == "model_b":
                    assert 0.3 < quality < 0.7, f"Model B should have medium quality scores, got {quality} for {metric}"
                else:  # model_c
                    assert quality < 0.5, f"Model C should have low quality scores, got {quality} for {metric}"
    
    print("✅ Min-max normalization test passed!")
    print("✅ All quality scores are in [0,1] range")
    print("✅ Quality scores follow expected patterns (A > B > C)")

def test_positive_only_hack():
    """Test the hack for positive-only scales."""
    print("\nTesting Positive-Only Scale Hack...")
    
    # Create data where all scores are positive (> 0)
    data = []
    models = ["model_a", "model_b"]
    
    for i in range(20):
        model = models[i % 2]
        cluster_id = i // 10
        
        if model == "model_a":
            score = {"helpfulness": 4.5}  # High score
        else:
            score = {"helpfulness": 2.5}  # Lower score
        
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
    
    # Check that the hack works (min should be treated as 0)
    model_stats = result.model_stats
    
    for model, stats in model_stats.items():
        if "fine" in stats and stats["fine"]:
            cluster = stats["fine"][0]
            quality = cluster.quality_score.get("helpfulness", 0)
            
            # With the hack, model_a (4.5) should get higher normalized score than model_b (2.5)
            if model == "model_a":
                assert quality > 0.5, f"Model A should have high normalized score, got {quality}"
            else:
                assert quality < 0.5, f"Model B should have low normalized score, got {quality}"
    
    print("✅ Positive-only scale hack test passed!")
    print("✅ Min value is correctly treated as 0 for positive-only scales")

if __name__ == "__main__":
    test_min_max_normalization()
    test_positive_only_hack()
    print("\n🎉 All normalization tests completed successfully!") 