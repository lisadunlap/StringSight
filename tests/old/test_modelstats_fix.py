#!/usr/bin/env python3
"""
Test script to verify that ModelStats now works with confidence interval fields.
"""

from stringsight.core.data_objects import ModelStats
import json


def test_modelstats_creation():
    """Test creating ModelStats with confidence intervals."""
    print("🧪 Testing ModelStats creation with confidence intervals...")
    
    # Create a ModelStats object with confidence intervals
    ms = ModelStats(
        property_description="helpful responses",
        model_name="gpt-4",
        cluster_size_global=50,
        score=1.25,
        quality_score={"accuracy": 0.15},
        size=10,
        proportion=0.2,
        examples=["prop_1", "prop_2"],
        metadata={"cluster_id": 1, "level": "fine"},
        score_ci={"lower": 1.18, "upper": 1.32},
        quality_score_ci={
            "accuracy": {"lower": 0.12, "upper": 0.18}
        }
    )
    
    print(f"✅ Created ModelStats: {ms.model_name}")
    print(f"   Score: {ms.score}")
    print(f"   Score CI: {ms.score_ci}")
    print(f"   Quality CI: {ms.quality_score_ci}")
    
    # Test to_dict method
    ms_dict = ms.to_dict()
    print(f"✅ to_dict() works correctly")
    
    # Test JSON serialization
    json_str = json.dumps(ms_dict, indent=2)
    print(f"✅ JSON serialization works")
    print(f"   JSON length: {len(json_str)} characters")
    
    return ms


def test_modelstats_without_ci():
    """Test creating ModelStats without confidence intervals (backward compatibility)."""
    print("\n🧪 Testing ModelStats creation without confidence intervals...")
    
    # Create a ModelStats object without confidence intervals (old style)
    ms = ModelStats(
        property_description="creative responses",
        model_name="claude-3",
        cluster_size_global=30,
        score=0.8,
        quality_score={"creativity": 0.25},
        size=5,
        proportion=0.1,
        examples=["prop_3"],
        metadata={"cluster_id": 2, "level": "fine"}
        # Note: No CI fields provided - should default to None
    )
    
    print(f"✅ Created ModelStats: {ms.model_name}")
    print(f"   Score: {ms.score}")
    print(f"   Score CI: {ms.score_ci}")
    print(f"   Quality CI: {ms.quality_score_ci}")
    
    # Test to_dict method
    ms_dict = ms.to_dict()
    print(f"✅ to_dict() works correctly")
    
    # Verify CI fields are None
    assert ms.score_ci is None
    assert ms.quality_score_ci is None
    print(f"✅ CI fields default to None correctly")
    
    return ms


def test_json_serialization():
    """Test that model stats can be saved and loaded from JSON."""
    print("\n🧪 Testing JSON serialization...")
    
    # Create some test model stats
    stats = {
        "gpt-4": {
            "fine": [
                ModelStats(
                    property_description="helpful responses",
                    model_name="gpt-4",
                    cluster_size_global=50,
                    score=1.25,
                    quality_score={"accuracy": 0.15},
                    size=10,
                    proportion=0.2,
                    examples=["prop_1", "prop_2"],
                    metadata={"cluster_id": 1, "level": "fine"},
                    score_ci={"lower": 1.18, "upper": 1.32},
                    quality_score_ci={
                        "accuracy": {"lower": 0.12, "upper": 0.18}
                    }
                )
            ],
            "coarse": [],
            "stats": {"accuracy": 0.85}
        }
    }
    
    # Convert to JSON format (like the save function does)
    stats_for_json = {}
    for model_name, model_stats in stats.items():
        stats_for_json[str(model_name)] = {
            "fine": [stat.to_dict() for stat in model_stats["fine"]]
        }
        if "coarse" in model_stats:
            stats_for_json[str(model_name)]["coarse"] = [stat.to_dict() for stat in model_stats["coarse"]]
        if "stats" in model_stats:
            stats_for_json[str(model_name)]["stats"] = model_stats["stats"]
    
    # Serialize to JSON
    json_str = json.dumps(stats_for_json, indent=2)
    print(f"✅ JSON serialization successful")
    print(f"   JSON length: {len(json_str)} characters")
    
    # Load back from JSON
    loaded_stats = json.loads(json_str)
    print(f"✅ JSON deserialization successful")
    
    # Verify confidence intervals are preserved
    first_stat = loaded_stats["gpt-4"]["fine"][0]
    assert "score_ci" in first_stat
    assert "quality_score_ci" in first_stat
    assert first_stat["score_ci"]["lower"] == 1.18
    assert first_stat["score_ci"]["upper"] == 1.32
    print(f"✅ Confidence intervals preserved in JSON")
    
    return loaded_stats


def main():
    """Run all tests."""
    print("🔍 Testing ModelStats with Confidence Intervals")
    print("=" * 60)
    
    try:
        test_modelstats_creation()
        test_modelstats_without_ci()
        test_json_serialization()
        
        print("\n" + "=" * 60)
        print("✅ ALL MODELSTATS TESTS PASSED!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("   • ModelStats can be created with confidence intervals")
        print("   • ModelStats works without confidence intervals (backward compatible)")
        print("   • JSON serialization preserves all fields including CIs")
        print("   • The pipeline save function should now work correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 