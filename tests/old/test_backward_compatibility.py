#!/usr/bin/env python3
"""
Test script to verify backward compatibility of confidence interval changes.
This ensures existing visualization code continues to work unchanged.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Import the metrics classes
from stringsight.metrics.single_model import SingleModelMetrics
from stringsight.core.data_objects import PropertyDataset, ModelStats


def create_minimal_test_data():
    """Create minimal test data for compatibility testing."""
    np.random.seed(42)
    
    data = []
    for i in range(20):
        for model in ["gpt-4", "claude-3"]:
            data.append({
                "question_id": f"q_{i}",
                "prompt": f"Test prompt {i}",
                "model": model,
                "model_response": f"Response from {model}",
                "score": {"accuracy": np.random.random()},
                "id": f"prop_{i}_{model}",
                "cluster_id": 0,
                "cluster_label": "test_cluster",
                "coarse_cluster_id": 0,
                "coarse_cluster_label": "test_coarse",
            })
    
    return pd.DataFrame(data)


def test_existing_field_access():
    """Test that all existing fields can be accessed exactly as before."""
    print("🧪 Testing existing field access...")
    
    # Create test data
    df = create_minimal_test_data()
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Test with confidence intervals disabled (simulates old behavior)
    metrics_stage = SingleModelMetrics(compute_confidence_intervals=False)
    result_dataset = metrics_stage.run(dataset)
    
    # Access all existing fields exactly as before
    for model_name, stats in result_dataset.model_stats.items():
        for cluster_stat in stats["fine"]:
            # All existing fields should work exactly as before
            assert hasattr(cluster_stat, 'property_description')
            assert hasattr(cluster_stat, 'model_name')
            assert hasattr(cluster_stat, 'score')
            assert hasattr(cluster_stat, 'quality_score')
            assert hasattr(cluster_stat, 'size')
            assert hasattr(cluster_stat, 'proportion')
            assert hasattr(cluster_stat, 'examples')
            assert hasattr(cluster_stat, 'metadata')
            
            # New fields should be None when disabled
            assert cluster_stat.score_ci_lower is None
            assert cluster_stat.score_ci_upper is None
            assert cluster_stat.quality_score_ci is None
            
            print(f"✅ Model: {cluster_stat.model_name}")
            print(f"   Score: {cluster_stat.score}")
            print(f"   Quality: {cluster_stat.quality_score}")
    
    print("✅ All existing fields accessible and working correctly")


def test_json_serialization():
    """Test that JSON serialization works exactly as before."""
    print("\n🧪 Testing JSON serialization...")
    
    # Create test data
    df = create_minimal_test_data()
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Test with confidence intervals disabled
    metrics_stage = SingleModelMetrics(compute_confidence_intervals=False)
    result_dataset = metrics_stage.run(dataset)
    
    # Serialize to JSON
    stats_for_json = {}
    for model_name, stats in result_dataset.model_stats.items():
        stats_for_json[str(model_name)] = {
            "fine": [stat.to_dict() for stat in stats["fine"]],
            "coarse": [stat.to_dict() for stat in stats["coarse"]],
            "stats": stats.get("stats", {})
        }
    
    # Save to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(stats_for_json, f, indent=2)
        temp_path = f.name
    
    # Load back and verify structure
    with open(temp_path, 'r') as f:
        loaded_stats = json.load(f)
    
    # Verify all existing fields are present
    first_model = list(loaded_stats.keys())[0]
    first_cluster = loaded_stats[first_model]["fine"][0]
    
    required_fields = [
        "property_description", "model_name", "score", "quality_score",
        "size", "proportion", "examples", "metadata"
    ]
    
    for field in required_fields:
        assert field in first_cluster, f"Missing required field: {field}"
    
    # Verify new fields are present but null
    new_fields = ["score_ci_lower", "score_ci_upper", "quality_score_ci"]
    for field in new_fields:
        assert field in first_cluster, f"Missing new field: {field}"
        assert first_cluster[field] is None, f"New field should be null when disabled: {field}"
    
    print("✅ JSON serialization works correctly")
    print(f"   Saved to: {temp_path}")
    
    # Clean up
    Path(temp_path).unlink()


def test_visualization_compatibility():
    """Test that typical visualization code patterns still work."""
    print("\n🧪 Testing visualization compatibility...")
    
    # Create test data
    df = create_minimal_test_data()
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # Test both with and without confidence intervals
    for compute_ci in [False, True]:
        print(f"\n   Testing with confidence intervals: {compute_ci}")
        
        metrics_stage = SingleModelMetrics(compute_confidence_intervals=compute_ci)
        result_dataset = metrics_stage.run(dataset)
        
        # Simulate typical visualization code patterns
        for model_name, stats in result_dataset.model_stats.items():
            for cluster_stat in stats["fine"]:
                # Pattern 1: Basic field access
                model = cluster_stat.model_name
                score = cluster_stat.score
                quality = cluster_stat.quality_score
                
                # Pattern 2: Dictionary-style access (if using to_dict())
                stat_dict = cluster_stat.to_dict()
                model_dict = stat_dict["model_name"]
                score_dict = stat_dict["score"]
                
                # Pattern 3: Conditional access to new fields
                if compute_ci and cluster_stat.score_ci_lower is not None:
                    ci_lower = cluster_stat.score_ci_lower
                    ci_upper = cluster_stat.score_ci_upper
                    print(f"     {model}: {score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
                else:
                    print(f"     {model}: {score:.3f}")
                
                # Verify all patterns work
                assert model == model_dict
                assert score == score_dict
                assert isinstance(quality, dict)
    
    print("✅ All visualization patterns work correctly")


def test_existing_scripts():
    """Test that existing pipeline scripts continue to work."""
    print("\n🧪 Testing existing pipeline scripts...")
    
    # Simulate existing script behavior
    df = create_minimal_test_data()
    dataset = PropertyDataset.from_dataframe(df, method="single_model")
    
    # This is how existing scripts would work
    metrics_stage = SingleModelMetrics()  # Uses default settings
    result_dataset = metrics_stage.run(dataset)
    
    # Existing code patterns should work unchanged
    model_stats = result_dataset.model_stats
    
    # Pattern: Iterate through models and clusters
    for model_name, stats in model_stats.items():
        print(f"   Model: {model_name}")
        for level in ["fine", "coarse"]:
            if level in stats:
                for cluster_stat in stats[level]:
                    # All existing access patterns work
                    print(f"     {cluster_stat.property_description}: {cluster_stat.score:.3f}")
    
    print("✅ Existing pipeline scripts work correctly")


def main():
    """Run all compatibility tests."""
    print("🔍 Testing StringSight Backward Compatibility")
    print("=" * 60)
    
    try:
        test_existing_field_access()
        test_json_serialization()
        test_visualization_compatibility()
        test_existing_scripts()
        
        print("\n" + "=" * 60)
        print("✅ ALL COMPATIBILITY TESTS PASSED!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("   • All existing fields remain unchanged")
        print("   • JSON serialization works exactly as before")
        print("   • Visualization code continues to work")
        print("   • Pipeline scripts remain compatible")
        print("   • New confidence interval fields are optional")
        print("\nYour existing visualizations will work without any changes!")
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 