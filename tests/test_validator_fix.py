#!/usr/bin/env python3
"""
Test that PropertyValidator accepts properties without behavior_type.

This verifies the fix for the validation error:
  "ERROR: 0 valid properties after validation"
  
The fix updates the validator to accept properties where behavior_type
is None or empty, which is the case for FixedAxesLabeler outputs.
"""

from stringsight.core.data_objects import Property, PropertyDataset, ConversationRecord
from stringsight.postprocess.validator import PropertyValidator

def test_validator_accepts_no_behavior_type():
    """Test that validator accepts properties without behavior_type."""
    
    # Scenario 1: Property with behavior_type=None (from FixedAxesLabeler)
    prop1 = Property(
        id="1",
        question_id="q1",
        model="gpt-4",
        property_description="Uses clear language",
        reason="The response is well-structured",
        evidence="Quote from response",
        behavior_type=None  # FixedAxesLabeler doesn't set this
    )
    
    validator = PropertyValidator()
    assert validator._is_valid_property(prop1), "Should accept property with behavior_type=None"
    print("✓ Test 1 passed: property with behavior_type=None is valid")


def test_validator_accepts_empty_behavior_type():
    """Test that validator accepts properties with empty behavior_type."""
    
    # Scenario 2: Property with behavior_type="" (empty string)
    prop2 = Property(
        id="2",
        question_id="q2",
        model="gpt-4",
        property_description="Provides detailed explanation",
        reason="The response includes examples",
        evidence="Quote from response",
        behavior_type=""  # Empty string should also be accepted
    )
    
    validator = PropertyValidator()
    assert validator._is_valid_property(prop2), "Should accept property with behavior_type=''"
    print("✓ Test 2 passed: property with behavior_type='' is valid")


def test_validator_accepts_valid_behavior_type():
    """Test that validator still validates allowed behavior_type values."""
    
    # Scenario 3: Property with valid behavior_type
    prop3 = Property(
        id="3",
        question_id="q3",
        model="gpt-4",
        property_description="Makes an error",
        reason="Incorrect calculation",
        evidence="Quote showing error",
        behavior_type="Negative (critical)"  # Valid value
    )
    
    validator = PropertyValidator()
    assert validator._is_valid_property(prop3), "Should accept property with valid behavior_type"
    print("✓ Test 3 passed: property with valid behavior_type is valid")


def test_validator_rejects_invalid_behavior_type():
    """Test that validator rejects invalid behavior_type values."""
    
    # Scenario 4: Property with invalid behavior_type
    prop4 = Property(
        id="4",
        question_id="q4",
        model="gpt-4",
        property_description="Does something",
        reason="Some reason",
        evidence="Some evidence",
        behavior_type="InvalidType"  # Invalid value
    )
    
    validator = PropertyValidator()
    assert not validator._is_valid_property(prop4), "Should reject property with invalid behavior_type"
    print("✓ Test 4 passed: property with invalid behavior_type is rejected")


def test_validator_run_with_mixed_properties():
    """Test that validator.run() works with mixed property types."""
    
    # Create a dataset with mixed properties (some with behavior_type, some without)
    conversations = [
        ConversationRecord(
            question_id="q1",
            model="gpt-4",
            prompt="What is AI?",
            responses="AI is...",
            scores={},
            meta={}
        )
    ]
    
    properties = [
        Property(
            id="1",
            question_id="q1",
            model="gpt-4",
            property_description="Clear explanation",
            reason="Well structured",
            evidence="Quote",
            behavior_type=None  # No behavior_type (from FixedAxesLabeler)
        ),
        Property(
            id="2",
            question_id="q1",
            model="gpt-4",
            property_description="Detailed response",
            reason="Includes examples",
            evidence="Quote",
            behavior_type=""  # Empty behavior_type
        ),
        Property(
            id="3",
            question_id="q1",
            model="gpt-4",
            property_description="Has error",
            reason="Mistake",
            evidence="Quote",
            behavior_type="Negative (critical)"  # Valid behavior_type
        ),
    ]
    
    dataset = PropertyDataset(
        conversations=conversations,
        all_models=["gpt-4"],
        properties=properties,
        clusters=[],
        model_stats={}
    )
    
    validator = PropertyValidator()
    result = validator.run(dataset)
    
    assert len(result.properties) == 3, f"Should keep all 3 valid properties, got {len(result.properties)}"
    print("✓ Test 5 passed: validator.run() accepts mixed property types")


if __name__ == "__main__":
    print("Testing PropertyValidator fix for behavior_type validation...\n")
    
    test_validator_accepts_no_behavior_type()
    test_validator_accepts_empty_behavior_type()
    test_validator_accepts_valid_behavior_type()
    test_validator_rejects_invalid_behavior_type()
    test_validator_run_with_mixed_properties()
    
    print("\n✅ All tests passed!")
    print("\nThe fix allows:")
    print("  1. Properties without behavior_type (None) - for FixedAxesLabeler")
    print("  2. Properties with empty behavior_type ('') - for edge cases")
    print("  3. Properties with valid behavior_type - for regular extraction")
    print("  4. Still rejects properties with invalid behavior_type values")

