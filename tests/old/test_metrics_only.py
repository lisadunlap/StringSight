"""Test the metrics-only functionality."""

import pytest
import tempfile
import os
from pathlib import Path

from stringsight import compute_metrics_only
from stringsight.core.data_objects import PropertyDataset, ConversationRecord, Property, Cluster


def create_test_dataset():
    """Create a minimal test dataset with conversations, properties, and clusters."""
    # Create test conversations
    conversations = [
        ConversationRecord(
            question_id="test_1",
            prompt="What is 2+2?",
            model="gpt-4",
            responses="4",
            scores={"accuracy": 1.0},
            meta={}
        ),
        ConversationRecord(
            question_id="test_2", 
            prompt="What is 3+3?",
            model="gpt-4",
            responses="6",
            scores={"accuracy": 1.0},
            meta={}
        ),
        ConversationRecord(
            question_id="test_3",
            prompt="What is 2+2?",
            model="claude-3",
            responses="4",
            scores={"accuracy": 1.0},
            meta={}
        )
    ]
    
    # Create test properties
    properties = [
        Property(
            id="prop_1",
            question_id="test_1",
            model="gpt-4",
            property_description="Correct mathematical reasoning",
            category="reasoning",
            type="General",
            impact="High",
            reason="Model correctly solved basic arithmetic",
            evidence="Provided correct answer 4",
            contains_errors=False,
            unexpected_behavior=False
        ),
        Property(
            id="prop_2",
            question_id="test_2",
            model="gpt-4", 
            property_description="Correct mathematical reasoning",
            category="reasoning",
            type="General",
            impact="High",
            reason="Model correctly solved basic arithmetic",
            evidence="Provided correct answer 6",
            contains_errors=False,
            unexpected_behavior=False
        ),
        Property(
            id="prop_3",
            question_id="test_3",
            model="claude-3",
            property_description="Correct mathematical reasoning",
            category="reasoning", 
            type="General",
            impact="High",
            reason="Model correctly solved basic arithmetic",
            evidence="Provided correct answer 4",
            contains_errors=False,
            unexpected_behavior=False
        )
    ]
    
    # Create test clusters
    clusters = [
        Cluster(
            id=1,
            label="Mathematical reasoning",
            size=3,
            parent_id=None,
            parent_label=None,
            property_descriptions=["Correct mathematical reasoning"],
            question_ids=["test_1", "test_2", "test_3"],
            meta={"group": "math_problems", "difficulty": "easy"}
        )
    ]
    
    return PropertyDataset(
        conversations=conversations,
        properties=properties,
        clusters=clusters,
        all_models=["gpt-4", "claude-3"]
    )


def test_compute_metrics_only_basic():
    """Test basic metrics-only functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        dataset = create_test_dataset()
        
        # Save dataset to temp directory
        dataset_path = Path(temp_dir) / "test_dataset.json"
        dataset.save(str(dataset_path), format="json")
        
        # Run metrics-only computation
        clustered_df, model_stats = compute_metrics_only(
            input_path=str(dataset_path),
            method="single_model",
            output_dir=temp_dir,
            verbose=False
        )
        
        # Verify results
        assert len(clustered_df) > 0, "Should return clustered DataFrame"
        assert len(model_stats) > 0, "Should return model statistics"
        
        # Check that output files were created
        expected_files = [
            "metrics_results.parquet",
            "metrics_dataset.json", 
            "metrics_stats.json"
        ]
        
        for filename in expected_files:
            file_path = Path(temp_dir) / filename
            assert file_path.exists(), f"Expected file {filename} was not created"


def test_compute_metrics_only_directory():
    """Test metrics-only functionality with directory input."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        dataset = create_test_dataset()
        
        # Save dataset to temp directory with standard naming
        dataset_path = Path(temp_dir) / "full_dataset.json"
        dataset.save(str(dataset_path), format="json")
        
        # Run metrics-only computation on directory
        clustered_df, model_stats = compute_metrics_only(
            input_path=temp_dir,
            method="single_model",
            verbose=False
        )
        
        # Verify results
        assert len(clustered_df) > 0, "Should return clustered DataFrame"
        assert len(model_stats) > 0, "Should return model statistics"


def test_compute_metrics_only_no_clusters():
    """Test that metrics-only fails gracefully when no clusters are present."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dataset without clusters
        dataset = create_test_dataset()
        dataset.clusters = []  # Remove clusters
        
        # Save dataset
        dataset_path = Path(temp_dir) / "test_dataset.json"
        dataset.save(str(dataset_path), format="json")
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No clusters found"):
            compute_metrics_only(
                input_path=str(dataset_path),
                method="single_model",
                verbose=False
            )


def test_compute_metrics_only_no_properties():
    """Test that metrics-only fails gracefully when no properties are present."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dataset without properties
        dataset = create_test_dataset()
        dataset.properties = []  # Remove properties
        
        # Save dataset
        dataset_path = Path(temp_dir) / "test_dataset.json"
        dataset.save(str(dataset_path), format="json")
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No properties found"):
            compute_metrics_only(
                input_path=str(dataset_path),
                method="single_model",
                verbose=False
            )


def test_compute_metrics_only_invalid_path():
    """Test that invalid path raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        compute_metrics_only("nonexistent_path")


def test_metrics_include_cluster_metadata():
    """Test that cluster metadata is properly included in metrics output."""
    dataset = create_test_dataset()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save dataset to temp directory
        dataset.save(os.path.join(temp_dir, "test_dataset.json"))
        
        # Compute metrics
        result = compute_metrics_only(temp_dir)
        
        # Check that cluster metadata is included in the output
        assert "model_cluster_scores" in result
        assert "cluster_scores" in result
        
        # Check that the cluster has metadata in cluster_scores
        cluster_scores = result["cluster_scores"]
        assert "Mathematical reasoning" in cluster_scores
        cluster_metrics = cluster_scores["Mathematical reasoning"]
        assert "metadata" in cluster_metrics
        assert cluster_metrics["metadata"] == {"group": "math_problems", "difficulty": "easy"}
        
        # Check that the cluster has metadata in model_cluster_scores
        model_cluster_scores = result["model_cluster_scores"]
        for model in ["gpt-4", "claude-3"]:
            assert model in model_cluster_scores
            assert "Mathematical reasoning" in model_cluster_scores[model]
            cluster_metrics = model_cluster_scores[model]["Mathematical reasoning"]
            assert "metadata" in cluster_metrics
            assert cluster_metrics["metadata"] == {"group": "math_problems", "difficulty": "easy"} 