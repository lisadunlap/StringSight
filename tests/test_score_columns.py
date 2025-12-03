"""
Tests for score_columns parameter functionality in explain() and label().

Tests use data/medhelm_gemini_flash_gpt4o_mini.jsonl to verify that:
1. score_columns parameter correctly converts separate columns to score dicts
2. Works with both single_model and side_by_side methods
3. Works with tidy data conversion (model_a/model_b parameters)
4. Works with both explain() and label() functions
"""

import pandas as pd
import pytest
from pathlib import Path
from stringsight import explain, label


# Define test data path
DATA_PATH = Path(__file__).parent.parent / "data" / "medhelm_gemini_flash_gpt4o_mini.jsonl"


def load_test_data(max_rows: int = 50) -> pd.DataFrame:
    """Load test data from JSONL file."""
    if not DATA_PATH.exists():
        pytest.skip(f"Test data not found at {DATA_PATH}")
    
    df = pd.read_json(DATA_PATH, lines=True, nrows=max_rows)
    return df


def test_explain_single_model_without_score_columns():
    """Test explain() with single model, using existing score dict (no score_columns)."""
    print("\n" + "="*80)
    print("TEST 1: explain() - single model without score_columns")
    print("="*80)
    
    df = load_test_data(max_rows=30)
    
    # Filter to just one model
    df_single = df[df["model"] == "openai/gpt-4.1-mini-2024-07-18"].copy()
    
    print(f"Loaded {len(df_single)} rows for single model")
    print(f"Columns: {df_single.columns.tolist()}")
    print(f"Sample score: {df_single['score'].iloc[0]}")
    
    # Run explain without score_columns
    clustered_df, model_stats = explain(
        df_single,
        method="single_model",
        sample_size=20,
        min_cluster_size=3,
        verbose=True,
        use_wandb=False,
        output_dir=None
    )
    
    print(f"\n✓ Success! Processed {len(clustered_df)} conversations")
    if 'property_description' in clustered_df.columns:
        print(f"  - Properties extracted: {clustered_df['property_description'].notna().sum()}")
    if 'cluster' in clustered_df.columns:
        print(f"  - Unique clusters: {clustered_df['cluster'].nunique()}")
    
    # Verify score column still exists as dict
    assert 'score' in clustered_df.columns, "Score column should be present"
    if len(clustered_df) > 0 and clustered_df['score'].notna().any():
        assert isinstance(clustered_df['score'].iloc[0], dict), "Score should be a dictionary"
    
    # Verify model_stats is dict of dataframes
    assert isinstance(model_stats, dict), "model_stats should be a dictionary"
    assert "model_cluster_scores" in model_stats, "model_stats should have model_cluster_scores"
    assert "cluster_scores" in model_stats, "model_stats should have cluster_scores"
    assert "model_scores" in model_stats, "model_stats should have model_scores"
    assert isinstance(model_stats["model_scores"], pd.DataFrame), "model_scores should be a DataFrame"
    
    return clustered_df, model_stats


def test_explain_single_model_with_score_columns():
    """Test explain() with single model, using score_columns parameter."""
    print("\n" + "="*80)
    print("TEST 2: explain() - single model with score_columns")
    print("="*80)
    
    df = load_test_data(max_rows=30)
    
    # Filter to just one model
    df_single = df[df["model"] == "openai/gpt-4.1-mini-2024-07-18"].copy()
    
    # Extract score metrics to separate columns
    score_cols = ['medication_qa_accuracy', 'gpt_accuracy', 'llama_accuracy', 'claude_accuracy']
    for col in score_cols:
        df_single[col] = df_single['score'].apply(lambda x: x.get(col, None))
    
    # Remove original score column to test score_columns conversion
    df_single = df_single.drop(columns=['score', 'annotations'])
    
    print(f"Loaded {len(df_single)} rows for single model")
    print(f"Score columns: {score_cols}")
    print(f"Sample values: {df_single[score_cols].iloc[0].to_dict()}")
    
    # Run explain with score_columns
    clustered_df, model_stats = explain(
        df_single,
        method="single_model",
        score_columns=score_cols,
        sample_size=20,
        min_cluster_size=3,
        verbose=True,
        use_wandb=False,
        output_dir=None
    )
    print(f"Cluster df columns: {clustered_df.columns}")
    exit()
    
    print(f"\n✓ Success! Processed {len(clustered_df)} conversations")
    print(f"  - Properties extracted: {clustered_df['property_description'].notna().sum()}")
    print(f"  - Unique clusters: {clustered_df['cluster'].nunique()}")
    
    # Verify score column was created as dict
    assert 'score' in clustered_df.columns, "Score column should be created"
    assert isinstance(clustered_df['score'].iloc[0], dict), "Score should be a dictionary"
    
    # Verify all metrics are in the score dict
    score_dict = clustered_df['score'].iloc[0]
    for col in score_cols:
        assert col in score_dict, f"Metric {col} should be in score dict"
    
    return clustered_df, model_stats


def test_explain_side_by_side_with_score_columns():
    """Test explain() with side-by-side, using model_a/model_b and score_columns."""
    print("\n" + "="*80)
    print("TEST 3: explain() - side-by-side with model_a/model_b and score_columns")
    print("="*80)
    
    df = load_test_data(max_rows=100)
    
    # Use tidy format with two models
    model_a = 'google/gemini-2.0-flash-001'
    model_b = 'openai/gpt-4.1-mini-2024-07-18'
    
    df_tidy = df[df["model"].isin([model_a, model_b])].copy()
    
    # Extract score metrics to separate columns
    score_cols = ['medication_qa_accuracy', 'gpt_accuracy', 'llama_accuracy', 'claude_accuracy']
    for col in score_cols:
        df_tidy[col] = df_tidy['score'].apply(lambda x: x.get(col, None))
    
    # Remove original score column to test score_columns conversion
    df_tidy = df_tidy.drop(columns=['score', 'annotations'])
    
    print(f"Loaded {len(df_tidy)} rows for side-by-side")
    print(f"Models: {model_a} vs {model_b}")
    print(f"Score columns: {score_cols}")
    
    # Run explain with side-by-side, model_a/model_b, and score_columns
    clustered_df, model_stats = explain(
        df_tidy,
        method="side_by_side",
        model_a=model_a,
        model_b=model_b,
        score_columns=score_cols,
        sample_size=30,
        min_cluster_size=3,
        verbose=True,
        use_wandb=False,
        output_dir=None
    )
    
    print(f"\n✓ Success! Processed {len(clustered_df)} conversations")
    print(f"  - Properties extracted: {clustered_df['property_description'].notna().sum()}")
    print(f"  - Unique clusters: {clustered_df['cluster'].nunique()}")
    
    # Verify score_a and score_b columns were created as dicts
    assert 'score_a' in clustered_df.columns, "score_a column should be created"
    assert 'score_b' in clustered_df.columns, "score_b column should be created"
    assert isinstance(clustered_df['score_a'].iloc[0], dict), "score_a should be a dictionary"
    assert isinstance(clustered_df['score_b'].iloc[0], dict), "score_b should be a dictionary"
    
    # Verify all metrics are in the score dicts
    score_a_dict = clustered_df['score_a'].iloc[0]
    score_b_dict = clustered_df['score_b'].iloc[0]
    for col in score_cols:
        assert col in score_a_dict, f"Metric {col} should be in score_a dict"
        assert col in score_b_dict, f"Metric {col} should be in score_b dict"
    
    return clustered_df, model_stats


def test_label_single_model_without_score_columns():
    """Test label() with single model, using existing score dict (no score_columns)."""
    print("\n" + "="*80)
    print("TEST 4: label() - single model without score_columns")
    print("="*80)
    
    df = load_test_data(max_rows=30)
    
    # Filter to just one model
    df_single = df[df["model"] == "openai/gpt-4.1-mini-2024-07-18"].copy()
    
    print(f"Loaded {len(df_single)} rows for single model")
    print(f"Sample score: {df_single['score'].iloc[0]}")
    
    # Define a simple taxonomy
    taxonomy = {
        "accurate": "Does the response provide accurate medical information?",
        "complete": "Does the response fully answer the question?",
        "clear": "Is the response clear and easy to understand?"
    }
    
    # Run label without score_columns
    clustered_df, model_stats = label(
        df_single,
        taxonomy=taxonomy,
        sample_size=15,
        verbose=True,
        use_wandb=False,
        output_dir=None
    )
    
    print(f"\n✓ Success! Processed {len(clustered_df)} conversations")
    print(f"  - Properties extracted: {clustered_df['property_description'].notna().sum()}")
    print(f"  - Labels assigned: {clustered_df['cluster'].value_counts().to_dict()}")
    
    # Verify score column still exists as dict
    assert 'score' in clustered_df.columns, "Score column should be present"
    
    return clustered_df, model_stats


def test_label_single_model_with_score_columns():
    """Test label() with single model, using score_columns parameter."""
    print("\n" + "="*80)
    print("TEST 5: label() - single model with score_columns")
    print("="*80)
    
    df = load_test_data(max_rows=30)
    
    # Filter to just one model
    df_single = df[df["model"] == "openai/gpt-4.1-mini-2024-07-18"].copy()
    
    # Extract score metrics to separate columns
    score_cols = ['medication_qa_accuracy', 'gpt_accuracy', 'llama_accuracy', 'claude_accuracy']
    for col in score_cols:
        df_single[col] = df_single['score'].apply(lambda x: x.get(col, None))
    
    # Remove original score column to test score_columns conversion
    df_single = df_single.drop(columns=['score', 'annotations'])
    
    print(f"Loaded {len(df_single)} rows for single model")
    print(f"Score columns: {score_cols}")
    print(f"Sample values: {df_single[score_cols].iloc[0].to_dict()}")
    
    # Define a simple taxonomy
    taxonomy = {
        "accurate": "Does the response provide accurate medical information?",
        "complete": "Does the response fully answer the question?",
        "clear": "Is the response clear and easy to understand?"
    }
    
    # Run label with score_columns
    clustered_df, model_stats = label(
        df_single,
        taxonomy=taxonomy,
        score_columns=score_cols,
        sample_size=15,
        verbose=True,
        use_wandb=False,
        output_dir=None
    )
    
    print(f"\n✓ Success! Processed {len(clustered_df)} conversations")
    print(f"  - Properties extracted: {clustered_df['property_description'].notna().sum()}")
    print(f"  - Labels assigned: {clustered_df['cluster'].value_counts().to_dict()}")
    
    # Verify score column was created as dict
    assert 'score' in clustered_df.columns, "Score column should be created"
    assert isinstance(clustered_df['score'].iloc[0], dict), "Score should be a dictionary"
    
    # Verify all metrics are in the score dict
    score_dict = clustered_df['score'].iloc[0]
    for col in score_cols:
        assert col in score_dict, f"Metric {col} should be in score dict"
    
    return clustered_df, model_stats


def test_all():
    """Run all tests in sequence."""
    print("\n" + "#"*80)
    print("# Running all score_columns tests")
    print("#"*80)
    
    try:
        test_explain_single_model_without_score_columns()
        print("\n✅ Test 1 passed")
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        raise
    
    try:
        test_explain_single_model_with_score_columns()
        print("\n✅ Test 2 passed")
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        raise
    
    try:
        test_explain_side_by_side_with_score_columns()
        print("\n✅ Test 3 passed")
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        raise
    
    try:
        test_label_single_model_without_score_columns()
        print("\n✅ Test 4 passed")
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        raise
    
    try:
        test_label_single_model_with_score_columns()
        print("\n✅ Test 5 passed")
    except Exception as e:
        print(f"\n❌ Test 5 failed: {e}")
        raise
    
    print("\n" + "#"*80)
    print("# ✅ ALL TESTS PASSED!")
    print("#"*80)


if __name__ == "__main__":
    # Run all tests when executed directly
    test_all()

