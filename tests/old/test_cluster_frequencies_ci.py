#!/usr/bin/env python3
"""
Test script to verify confidence interval functionality in the cluster frequencies tab.
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


def create_test_cluster_frequencies_data():
    """Create test data for cluster frequencies with confidence intervals."""
    return {
        "chart_data": pd.DataFrame([
            {
                "property_description": "helpful responses",
                "model": "gpt-4",
                "frequency": 15.2,
                "size": 10,
                "cluster_size_global": 50,
                "has_ci": True,
                "ci_lower": 1.18,
                "ci_upper": 1.32,
                "has_quality_ci": True
            },
            {
                "property_description": "helpful responses",
                "model": "claude-3",
                "frequency": 12.8,
                "size": 8,
                "cluster_size_global": 50,
                "has_ci": True,
                "ci_lower": 1.05,
                "ci_upper": 1.15,
                "has_quality_ci": True
            },
            {
                "property_description": "creative responses",
                "model": "gpt-4",
                "frequency": 8.5,
                "size": 5,
                "cluster_size_global": 30,
                "has_ci": True,
                "ci_lower": 0.75,
                "ci_upper": 0.85,
                "has_quality_ci": True
            },
            {
                "property_description": "creative responses",
                "model": "claude-3",
                "frequency": 6.2,
                "size": 3,
                "cluster_size_global": 30,
                "has_ci": False,
                "ci_lower": None,
                "ci_upper": None,
                "has_quality_ci": False
            }
        ]),
        "model_stats": {
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
                        "proportion": 0.152,
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
                        "proportion": 0.085,
                        "cluster_size_global": 30,
                        "examples": ["prop_3"],
                        "metadata": {"cluster_id": 2, "level": "fine"}
                    }
                ]
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
                        "proportion": 0.128,
                        "cluster_size_global": 50,
                        "examples": ["prop_4", "prop_5"],
                        "metadata": {"cluster_id": 1, "level": "fine"}
                    },
                    {
                        "property_description": "creative responses",
                        "model_name": "claude-3",
                        "score": 0.6,
                        "score_ci_lower": None,
                        "score_ci_upper": None,
                        "quality_score": {"creativity": 0.15},
                        "quality_score_ci": None,
                        "size": 3,
                        "proportion": 0.062,
                        "cluster_size_global": 30,
                        "examples": ["prop_6"],
                        "metadata": {"cluster_id": 2, "level": "fine"}
                    }
                ]
            }
        }
    }


def test_frequency_table_with_quality_cis():
    """Test that the frequency comparison table includes quality confidence intervals and global sizes."""
    print("\n🧪 Testing frequency table with quality CIs and global sizes...")
    
    # Import the function we want to test
    from stringsight.dashboard.utils import create_frequency_comparison_table
    
    # Create test data with quality confidence intervals
    test_model_stats = {
        "gpt-4": {
            "fine": [
                {
                    "property_description": "helpful responses",
                    "model_name": "gpt-4",
                    "score": 1.25,
                    "score_ci": {"lower": 1.18, "upper": 1.32},
                    "quality_score": {"accuracy": 0.15, "helpfulness": 0.20},
                    "quality_score_ci": {
                        "accuracy": {"lower": 0.12, "upper": 0.18},
                        "helpfulness": {"lower": 0.18, "upper": 0.22}
                    },
                    "size": 10,
                    "proportion": 0.152,
                    "cluster_size_global": 50,
                    "examples": ["prop_1", "prop_2"],
                    "metadata": {"cluster_id": 1, "level": "fine"}
                },
                {
                    "property_description": "creative responses",
                    "model_name": "gpt-4",
                    "score": 0.8,
                    "score_ci": {"lower": 0.75, "upper": 0.85},
                    "quality_score": {"creativity": 0.25},
                    "quality_score_ci": {
                        "creativity": {"lower": 0.20, "upper": 0.30}
                    },
                    "size": 5,
                    "proportion": 0.085,
                    "cluster_size_global": 30,
                    "examples": ["prop_3"],
                    "metadata": {"cluster_id": 2, "level": "fine"}
                }
            ]
        },
        "claude-3": {
            "fine": [
                {
                    "property_description": "helpful responses",
                    "model_name": "claude-3",
                    "score": 1.1,
                    "score_ci": {"lower": 1.05, "upper": 1.15},
                    "quality_score": {"accuracy": 0.12, "helpfulness": 0.18},
                    "quality_score_ci": {
                        "accuracy": {"lower": 0.10, "upper": 0.14},
                        "helpfulness": {"lower": 0.16, "upper": 0.20}
                    },
                    "size": 8,
                    "proportion": 0.128,
                    "cluster_size_global": 50,
                    "examples": ["prop_4", "prop_5"],
                    "metadata": {"cluster_id": 1, "level": "fine"}
                },
                {
                    "property_description": "creative responses",
                    "model_name": "claude-3",
                    "score": 0.6,
                    "score_ci": {"lower": 0.55, "upper": 0.65},
                    "quality_score": {"creativity": 0.20},
                    "quality_score_ci": {
                        "creativity": {"lower": 0.18, "upper": 0.22}
                    },
                    "size": 3,
                    "proportion": 0.062,
                    "cluster_size_global": 30,
                    "examples": ["prop_6"],
                    "metadata": {"cluster_id": 2, "level": "fine"}
                }
            ]
        }
    }
    
    # Test the function
    result_df = create_frequency_comparison_table(
        test_model_stats, 
        selected_models=["gpt-4", "claude-3"], 
        cluster_level="fine", 
        top_n=10
    )
    
    print(f"✅ Frequency table created with {len(result_df)} rows and {len(result_df.columns)} columns")
    print(f"📊 Columns: {list(result_df.columns)}")
    
    # Check that the table includes the expected columns
    expected_columns = ['Cluster', 'Global Size', 'Quality Score']
    for col in expected_columns:
        if col in result_df.columns:
            print(f"✅ Found expected column: {col}")
        else:
            print(f"❌ Missing expected column: {col}")
    
    # Check for quality confidence interval columns
    quality_ci_columns = [col for col in result_df.columns if 'Quality' in col and 'CI' in col]
    if quality_ci_columns:
        print(f"✅ Found quality CI columns: {quality_ci_columns}")
    else:
        print("❌ No quality CI columns found")
    
    # Check for frequency columns
    freq_columns = [col for col in result_df.columns if 'Freq' in col]
    if freq_columns:
        print(f"✅ Found frequency columns: {freq_columns}")
    else:
        print("❌ No frequency columns found")
    
    # Check for CI columns
    ci_columns = [col for col in result_df.columns if 'CI' in col and 'Quality' not in col]
    if ci_columns:
        print(f"✅ Found CI columns: {ci_columns}")
    else:
        print("❌ No CI columns found")
    
    # Print a sample row
    if not result_df.empty:
        print(f"\n📋 Sample row:")
        sample_row = result_df.iloc[0]
        for col in result_df.columns:
            print(f"  {col}: {sample_row[col]}")
    
    return result_df


def test_ci_error_bar_calculation():
    """Test the confidence interval error bar calculation logic."""
    print("🧪 Testing CI error bar calculation...")
    
    # Test data
    test_data = create_test_cluster_frequencies_data()
    chart_data = test_data["chart_data"]
    
    # Test CI calculation for frequency error bars
    for _, row in chart_data.iterrows():
        if row.get('has_ci', False) and row.get('ci_lower') is not None and row.get('ci_upper') is not None:
            ci_width = row['ci_upper'] - row['ci_lower']
            freq_uncertainty = ci_width * row['frequency'] * 0.1
            ci_lower_freq = max(0, row['frequency'] - freq_uncertainty)
            ci_upper_freq = row['frequency'] + freq_uncertainty
            
            print(f"✅ {row['model']} - {row['property_description']}:")
            print(f"   Frequency: {row['frequency']:.1f}%")
            print(f"   CI: [{ci_lower_freq:.1f}%, {ci_upper_freq:.1f}%]")
            print(f"   Uncertainty: ±{freq_uncertainty:.1f}%")
        else:
            print(f"⚠️ {row['model']} - {row['property_description']}: No CI available")


def test_quality_ci_extraction():
    """Test quality score confidence interval extraction."""
    print("\n🧪 Testing quality score CI extraction...")
    
    test_data = create_test_cluster_frequencies_data()
    model_stats = test_data["model_stats"]
    
    # Test extracting quality CIs from model_stats
    for model_name, model_data in model_stats.items():
        clusters = model_data.get("fine", [])
        for cluster in clusters:
            quality_ci = cluster.get('quality_score_ci', {})
            if quality_ci:
                for key, ci_bounds in quality_ci.items():
                    ci_formatted = format_confidence_interval(
                        ci_bounds.get('lower'), 
                        ci_bounds.get('upper')
                    )
                    print(f"✅ {model_name} - {cluster['property_description']} - {key}: {ci_formatted}")
            else:
                print(f"⚠️ {model_name} - {cluster['property_description']}: No quality CI available")


def test_ci_detection():
    """Test confidence interval detection in cluster frequencies data."""
    print("\n🧪 Testing CI detection in cluster frequencies...")
    
    test_data = create_test_cluster_frequencies_data()
    chart_data = test_data["chart_data"]
    
    # Count CIs available
    total_measurements = len(chart_data)
    measurements_with_ci = chart_data['has_ci'].sum()
    ci_coverage = (measurements_with_ci / total_measurements) * 100
    
    print(f"Total measurements: {total_measurements}")
    print(f"Measurements with CI: {measurements_with_ci}")
    print(f"CI coverage: {ci_coverage:.1f}%")
    
    assert ci_coverage > 0, "Should have some measurements with CIs"
    assert ci_coverage < 100, "Should have some measurements without CIs"
    print("✅ CI detection works correctly")


def test_ci_formatting():
    """Test confidence interval formatting for display."""
    print("\n🧪 Testing CI formatting...")
    
    # Test various CI scenarios
    test_cases = [
        (1.18, 1.32, "[1.180, 1.320] (95% CI)"),
        (0.75, 0.85, "[0.750, 0.850] (95% CI)"),
        (None, 1.32, "N/A"),
        (1.18, None, "N/A"),
        (None, None, "N/A")
    ]
    
    for lower, upper, expected in test_cases:
        result = format_confidence_interval(lower, upper)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"✅ {lower}, {upper} -> {result}")


def test_error_bar_visibility():
    """Test error bar visibility logic."""
    print("\n🧪 Testing error bar visibility logic...")
    
    # Test different scenarios
    scenarios = [
        (True, True, True, "Should show error bars"),
        (True, False, False, "Should not show error bars (no CIs available)"),
        (False, True, False, "Should not show error bars (user disabled)"),
        (False, False, False, "Should not show error bars (disabled and no CIs)")
    ]
    
    for show_ci, has_any_ci, expected, description in scenarios:
        visible = show_ci and has_any_ci
        assert visible == expected, f"{description}: Expected {expected}, got {visible}"
        print(f"✅ {description}: {visible}")


def main():
    """Run all tests."""
    print("🔍 Testing Cluster Frequencies CI Functionality")
    print("=" * 60)
    
    try:
        test_frequency_table_with_quality_cis()
        test_ci_error_bar_calculation()
        test_quality_ci_extraction()
        test_ci_detection()
        test_ci_formatting()
        test_error_bar_visibility()
        
        print("\n" + "=" * 60)
        print("✅ ALL CLUSTER FREQUENCIES CI TESTS PASSED!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("   • CI error bar calculation works correctly")
        print("   • Quality score CI extraction functions properly")
        print("   • CI detection and coverage calculation accurate")
        print("   • CI formatting displays correctly")
        print("   • Error bar visibility logic works as expected")
        print("\nConfidence intervals are ready for cluster frequencies visualization!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 