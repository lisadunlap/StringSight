"""
Test the complete StringSight pipeline end-to-end using the explain() function.

This test:
1. Loads arena test data (same as test_arena_subset.py)
2. Runs the full pipeline with explain()
3. Verifies outputs and saves results
"""

import pathlib
import pytest
import pandas as pd
import json

from stringsight import explain
from stringsight.core.data_objects import PropertyDataset


def test_full_pipeline_with_explain(tmp_path):
    """Test the complete pipeline using the explain() function."""
    # Load the pre-saved arena data

    first50 = pd.read_json("tests/outputs/single_model/arena_first50_single.jsonl", lines=True)
    print(first50.columns)
    required_cols = {"prompt", "model", "model_response"}
    assert required_cols.issubset(first50.columns)
    
    print(f"Loaded {len(first50)} rows")
    print(first50.columns)
    
    if len(first50) < 10:
        pytest.skip("Need at least 10 conversations for meaningful testing")
    
    print(f"Starting pipeline with {len(first50)} conversations")
    
    # Run the full pipeline with explain() - using real API calls
    print("Running full pipeline with explain()...")
    clustered_df, model_stats = explain(
        first50,
        method="single_model",
        system_prompt="single_model_system_prompt",
        clusterer="hdbscan",
        min_cluster_size=2,  # Small for test data
        max_coarse_clusters=5,
        embedding_model="text-embedding-3-small",
        hierarchical=True,
        max_workers=1,  # Sequential for predictable testing
        use_wandb=False,  # Disable wandb for testing
        verbose=True,
        output_dir=str(tmp_path)  # Test automatic saving
    )
    
    # =====================================================================
    # Verify Results
    # =====================================================================
    
    # Check that we got results back
    assert len(clustered_df) > 0, "Should return clustered DataFrame"
    assert len(model_stats) > 0, "Should return model statistics"
    
    # Check that clustering columns were added
    expected_columns = [
        'cluster_id',
        'cluster_label'
    ]
    print("clustered_df.columns ", clustered_df.columns)
    
    for col in expected_columns:
        assert col in clustered_df.columns, f"Missing column: {col}"
    
    # Check hierarchical clustering columns if enabled
    if any('coarse' in col for col in clustered_df.columns):
        hierarchical_columns = [
            'coarse_cluster_id',
            'coarse_cluster_label'
        ]
        for col in hierarchical_columns:
            assert col in clustered_df.columns, f"Missing hierarchical column: {col}"
    
    # Check that we have extracted properties
    assert 'property_description' in clustered_df.columns
    properties_with_desc = clustered_df['property_description'].notna().sum()
    assert properties_with_desc > 0, "Should have extracted some properties"
    
    # Check model statistics structure
    for model_name, stats in model_stats.items():
        assert "fine" in stats, f"Model {model_name} should have fine-grained stats"
        assert len(stats["fine"]) > 0, f"Model {model_name} should have at least one fine stat"
        
        # Check first stat object
        first_stat = stats["fine"][0]
        assert hasattr(first_stat, "property_description")
        assert hasattr(first_stat, "model_name")
        assert hasattr(first_stat, "score")
        assert hasattr(first_stat, "size")
        assert hasattr(first_stat, "proportion")
        assert hasattr(first_stat, "examples")
        
        print(f"Model {model_name}: {len(stats['fine'])} fine clusters")
        if "coarse" in stats:
            print(f"  + {len(stats['coarse'])} coarse clusters")
    
    # =====================================================================
    # Save Results for Inspection
    # =====================================================================
    
    # Verify that automatic saving worked
    expected_files = [
        "clustered_results.parquet",
        "full_dataset.json", 
        "full_dataset.parquet",
        "model_stats.json",
        "summary.txt"
    ]
    
    for filename in expected_files:
        file_path = f"{tmp_path}/{filename}"
        assert os.path.exists(file_path), f"Expected file {filename} was not created by automatic saving"
        print(f"✓ Found automatically saved file: {file_path}")
    
    # Load and verify the saved parquet matches returned DataFrame
    saved_df = pd.read_parquet(f"{tmp_path}/clustered_results.parquet")
    assert len(saved_df) == len(clustered_df), "Saved parquet should match returned DataFrame"
    
    # Verify the saved JSON can be loaded
    saved_dataset_json = PropertyDataset.load(f"{tmp_path}/full_dataset.json", format="json")
    assert len(saved_dataset_json.properties) > 0, "Saved JSON dataset should have properties"
    
    # # Verify the saved parquet can be loaded
    # saved_dataset_parquet = PropertyDataset.load(f"{tmp_path}/full_dataset.parquet", format="parquet")
    # assert len(saved_dataset_parquet.properties) > 0, "Saved parquet dataset should have properties"
    
    # # Verify both dataset formats have the same number of properties
    # assert len(saved_dataset_json.properties) == len(saved_dataset_parquet.properties), "JSON and parquet datasets should have same number of properties"
    
    # Verify the model stats JSON can be loaded
    with open(f"{tmp_path}/model_stats.json", 'r') as f:
        saved_stats = json.load(f)
    assert len(saved_stats) == len(model_stats), "Saved stats should match returned stats"
    
    # Save additional test results (keeping the old manual saving for compatibility)
    clustered_path = f"{tmp_path}/test_clustered.csv"
    clustered_df.to_csv(clustered_path, index=False)
    print(f"Also saved test DataFrame to: {clustered_path}")
    
    # Save model stats as JSON (manual backup)
    stats_path = f"{tmp_path}/test_model_stats.json"
    
    # Convert ModelStats objects to dicts for JSON serialization
    stats_for_json = {}
    for model_name, stats in model_stats.items():
        stats_for_json[model_name] = {
            "fine": [
                {
                    "property_description": stat.property_description,
                    "model_name": stat.model_name,
                    "score": stat.score,
                    "size": stat.size,
                    "proportion": stat.proportion,
                    "examples": stat.examples
                }
                for stat in stats["fine"]
            ]
        }
        if "coarse" in stats:
            stats_for_json[model_name]["coarse"] = [
                {
                    "property_description": stat.property_description,
                    "model_name": stat.model_name,
                    "score": stat.score,
                    "size": stat.size,
                    "proportion": stat.proportion,
                    "examples": stat.examples
                }
                for stat in stats["coarse"]
            ]
    
    with open(stats_path, 'w') as f:
        json.dump(stats_for_json, f, indent=2)
    print(f"Also saved test model stats to: {stats_path}")
    
    # =====================================================================
    # Print Summary
    # =====================================================================
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Input conversations: {len(first50)}")
    print(f"Properties extracted: {properties_with_desc}")
    print(f"Clustered properties: {len(clustered_df)}")
    print(f"Models analyzed: {len(model_stats)}")
    print(f"Fine clusters: {len(clustered_df['cluster_id'].unique())}")
    
    if 'coarse_cluster_id' in clustered_df.columns:
        coarse_clusters = len(clustered_df['coarse_cluster_id'].unique())
        print(f"Coarse clusters: {coarse_clusters}")
    
    # Show sample clusters
    print(f"\nSample cluster labels:")
    unique_labels = clustered_df['cluster_label'].dropna().unique()
    for i, label in enumerate(unique_labels[:3]):
        cluster_size = (clustered_df['cluster_label'] == label).sum()
        print(f"  {i+1}. {label} (size: {cluster_size})")
    
    # Show sample model stats
    print(f"\nSample model stats:")
    for model_name, stats in list(model_stats.items())[:2]:
        print(f"  {model_name}:")
        for stat in stats["fine"][:2]:
            print(f"    • {stat.property_description[:50]}... (score: {stat.score:.2f})")
    
    print("="*60)
    print("✅ Full pipeline test with explain() completed successfully!")
    print("="*60)


if __name__ == "__main__":
    import os
    # make dir if it doesn't exist
    if not os.path.exists("tests/outputs/full_pipeline_single_test"):
        os.makedirs("tests/outputs/full_pipeline_single_test")
    test_full_pipeline_with_explain("tests/outputs/full_pipeline_single_test") 