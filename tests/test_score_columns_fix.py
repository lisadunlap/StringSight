#!/usr/bin/env python3
"""
Test that score_columns parameter is ignored when score dict already exists.

This verifies the fix for the issue:
  "Score columns not found in dataframe: ['reward']"
  
The fix adds smart detection: if the 'score' column already contains
dict values, skip the score_columns conversion step.
"""

import pandas as pd
from stringsight.core.preprocessing import validate_and_prepare_dataframe

def test_score_dict_already_exists():
    """Test that score_columns is ignored when score dict already exists."""
    
    # Scenario 1: score column already contains dicts (like from frontend)
    df = pd.DataFrame([
        {
            "question_id": "1",
            "prompt": "What is AI?",
            "model": "gpt-4",
            "model_response": "AI is...",
            "score": {"reward": 0.8, "helpfulness": 0.9}  # Already a dict!
        },
        {
            "question_id": "2",
            "prompt": "What is ML?",
            "model": "gpt-4",
            "model_response": "ML is...",
            "score": {"reward": 0.7, "helpfulness": 0.85}
        }
    ])
    
    # User mistakenly passes score_columns=["reward"] even though
    # scores are already in dict format
    result = validate_and_prepare_dataframe(
        df,
        method="single_model",
        score_columns=["reward"],  # This should be ignored
        verbose=True
    )
    
    # Verify the score column is unchanged (still contains dicts)
    assert "score" in result.columns
    assert isinstance(result["score"].iloc[0], dict)
    assert "reward" in result["score"].iloc[0]
    assert result["score"].iloc[0]["reward"] == 0.8
    
    print("✓ Test 1 passed: score_columns ignored when score dict exists")


def test_score_columns_conversion_when_needed():
    """Test that score_columns still works when conversion is needed."""
    
    # Scenario 2: separate score columns that need conversion
    df = pd.DataFrame([
        {
            "question_id": "1",
            "prompt": "What is AI?",
            "model": "gpt-4",
            "model_response": "AI is...",
            "reward": 0.8,  # Separate columns
            "helpfulness": 0.9
        },
        {
            "question_id": "2",
            "prompt": "What is ML?",
            "model": "gpt-4",
            "model_response": "ML is...",
            "reward": 0.7,
            "helpfulness": 0.85
        }
    ])
    
    # User correctly passes score_columns to convert them
    result = validate_and_prepare_dataframe(
        df,
        method="single_model",
        score_columns=["reward", "helpfulness"],
        verbose=True
    )
    
    # Verify the score dict was created
    assert "score" in result.columns
    assert isinstance(result["score"].iloc[0], dict)
    assert "reward" in result["score"].iloc[0]
    assert "helpfulness" in result["score"].iloc[0]
    assert result["score"].iloc[0]["reward"] == 0.8
    assert result["score"].iloc[0]["helpfulness"] == 0.9
    
    print("✓ Test 2 passed: score_columns conversion works when needed")


def test_no_score_columns_no_conversion():
    """Test that nothing breaks when no score_columns provided."""
    
    df = pd.DataFrame([
        {
            "question_id": "1",
            "prompt": "What is AI?",
            "model": "gpt-4",
            "model_response": "AI is...",
            "score": {"reward": 0.8}
        }
    ])
    
    # No score_columns provided - should work fine
    result = validate_and_prepare_dataframe(
        df,
        method="single_model",
        score_columns=None,
        verbose=True
    )
    
    assert "score" in result.columns
    assert isinstance(result["score"].iloc[0], dict)
    
    print("✓ Test 3 passed: no conversion when score_columns=None")


if __name__ == "__main__":
    print("Testing score_columns fix...\n")
    
    test_score_dict_already_exists()
    test_score_columns_conversion_when_needed()
    test_no_score_columns_no_conversion()
    
    print("\n✅ All tests passed!")
    print("\nThe fix allows:")
    print("  1. score_columns to be ignored when score dict already exists")
    print("  2. score_columns to work normally when conversion is needed")
    print("  3. No score_columns to work as before")

