#!/usr/bin/env python3
"""
Test script to verify confidence interval functionality in the interactive app.
"""

import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

# Import the app functions
from stringsight.viz.pipeline_results_app import (
    format_confidence_interval, 
    has_confidence_intervals, 
    get_confidence_interval_width
)


def create_test_model_stats_with_ci():
    """Create test model stats with confidence intervals."""
    return {
        "gpt-4": {
            "fine": [
                {
                    "property_description": "helpful responses",
                    "model_name": "gpt-4",
                    "score": 1.25,
                    "score_ci_lower": 1.18,
                    "score_ci_upper": 1.32,
                    "quality_score": {"accuracy": 0.15},
                    "quality_score_ci": {
                        "accuracy": {"lower": 0.12, "upper": 0.18}
                    },
                    "size": 10,
                    "proportion": 0.2,
                    "cluster_size_global": 50,
                    "examples": ["prop_1", "prop_2"],
                    "metadata": {"cluster_id": 1, "level": "fine"}
                },
                {
                    "property_description": "creative responses",
                    "model_name": "gpt-4",
                    "score": 0.8,
                    "score_ci_lower": 0.75,
                    "score_ci_upper": 0.85,
                    "quality_score": {"creativity": 0.25},
                    "quality_score_ci": {
                        "creativity": {"lower": 0.20, "upper": 0.30}
                    },
                    "size": 5,
                    "proportion": 0.1,
                    "cluster_size_global": 30,
                    "examples": ["prop_3"],
                    "metadata": {"cluster_id": 2, "level": "fine"}
                }
            ],
            "coarse": [],
            "stats": {"accuracy": 0.85, "creativity": 0.70}
        },
        "claude-3": {
            "fine": [
                {
                    "property_description": "helpful responses",
                    "model_name": "claude-3",
                    "score": 1.1,
                    "score_ci_lower": 1.05,
                    "score_ci_upper": 1.15,
                    "quality_score": {"accuracy": 0.12},
                    "quality_score_ci": {
                        "accuracy": {"lower": 0.10, "upper": 0.14}
                    },
                    "size": 8,
                    "proportion": 0.16,
                    "cluster_size_global": 50,
                    "examples": ["prop_4", "prop_5"],
                    "metadata": {"cluster_id": 1, "level": "fine"}
                }
            ],
            "coarse": [],
            "stats": {"accuracy": 0.82, "creativity": 0.65}
        }
    }


def create_test_model_stats_without_ci():
    """Create test model stats without confidence intervals."""
    return {
        "gpt-4": {
            "fine": [
                {
                    "property_description": "helpful responses",
                    "model_name": "gpt-4",
                    "score": 1.25,
                    "score_ci_lower": None,
                    "score_ci_upper": None,
                    "quality_score": {"accuracy": 0.15},
                    "quality_score_ci": None,
                    "size": 10,
                    "proportion": 0.2,
                    "cluster_size_global": 50,
                    "examples": ["prop_1", "prop_2"],
                    "metadata": {"cluster_id": 1, "level": "fine"}
                }
            ],
            "coarse": [],
            "stats": {"accuracy": 0.85}
        }
    }


def test_confidence_interval_functions():
    """Test the confidence interval helper functions."""
    print("🧪 Testing confidence interval helper functions...")
    
    # Test format_confidence_interval
    assert format_confidence_interval(1.18, 1.32) == "[1.180, 1.320] (95% CI)"
    assert format_confidence_interval(None, 1.32) == "N/A"
    assert format_confidence_interval(1.18, None) == "N/A"
    print("✅ format_confidence_interval works correctly")
    
    # Test has_confidence_intervals
    cluster_with_ci = {
        "score_ci_lower": 1.18,
        "score_ci_upper": 1.32
    }
    cluster_without_ci = {
        "score_ci_lower": None,
        "score_ci_upper": None
    }
    cluster_missing_ci = {}
    
    assert has_confidence_intervals(cluster_with_ci) == True
    assert has_confidence_intervals(cluster_without_ci) == False
    assert has_confidence_intervals(cluster_missing_ci) == False
    print("✅ has_confidence_intervals works correctly")
    
    # Test get_confidence_interval_width
    assert get_confidence_interval_width(1.18, 1.32) == 0.14
    assert get_confidence_interval_width(None, 1.32) == 0.0
    assert get_confidence_interval_width(1.18, None) == 0.0
    print("✅ get_confidence_interval_width works correctly")


def test_model_stats_with_ci():
    """Test model stats with confidence intervals."""
    print("\n🧪 Testing model stats with confidence intervals...")
    
    model_stats = create_test_model_stats_with_ci()
    
    # Check that confidence intervals are present
    has_any_ci = False
    for model_name, model_data in model_stats.items():
        clusters = model_data.get("fine", [])
        for cluster in clusters:
            if has_confidence_intervals(cluster):
                has_any_ci = True
                break
        if has_any_ci:
            break
    
    assert has_any_ci == True
    print("✅ Model stats with CI detected correctly")
    
    # Test CI formatting for each cluster
    for model_name, model_data in model_stats.items():
        clusters = model_data.get("fine", [])
        for cluster in clusters:
            if has_confidence_intervals(cluster):
                ci_formatted = format_confidence_interval(
                    cluster["score_ci_lower"], 
                    cluster["score_ci_upper"]
                )
                assert "CI" in ci_formatted
                assert "95%" in ci_formatted
                print(f"✅ CI formatting works for {model_name}: {ci_formatted}")


def test_model_stats_without_ci():
    """Test model stats without confidence intervals."""
    print("\n🧪 Testing model stats without confidence intervals...")
    
    model_stats = create_test_model_stats_without_ci()
    
    # Check that confidence intervals are not present
    has_any_ci = False
    for model_name, model_data in model_stats.items():
        clusters = model_data.get("fine", [])
        for cluster in clusters:
            if has_confidence_intervals(cluster):
                has_any_ci = True
                break
        if has_any_ci:
            break
    
    assert has_any_ci == False
    print("✅ Model stats without CI detected correctly")


def test_json_serialization():
    """Test that model stats with CI can be serialized to JSON."""
    print("\n🧪 Testing JSON serialization...")
    
    model_stats = create_test_model_stats_with_ci()
    
    # Serialize to JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(model_stats, f, indent=2)
        temp_path = f.name
    
    # Load back and verify
    with open(temp_path, 'r') as f:
        loaded_stats = json.load(f)
    
    # Check that CI fields are preserved
    first_model = list(loaded_stats.keys())[0]
    first_cluster = loaded_stats[first_model]["fine"][0]
    
    assert "score_ci_lower" in first_cluster
    assert "score_ci_upper" in first_cluster
    assert "quality_score_ci" in first_cluster
    assert first_cluster["score_ci_lower"] == 1.18
    assert first_cluster["score_ci_upper"] == 1.32
    
    print("✅ JSON serialization works correctly")
    
    # Clean up
    Path(temp_path).unlink()


def main():
    """Run all tests."""
    print("🔍 Testing Confidence Interval App Functionality")
    print("=" * 60)
    
    try:
        test_confidence_interval_functions()
        test_model_stats_with_ci()
        test_model_stats_without_ci()
        test_json_serialization()
        
        print("\n" + "=" * 60)
        print("✅ ALL APP TESTS PASSED!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("   • Confidence interval helper functions work correctly")
        print("   • Model stats with CI are detected properly")
        print("   • Model stats without CI are handled gracefully")
        print("   • JSON serialization preserves CI data")
        print("\nThe interactive app is ready to display confidence intervals!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 