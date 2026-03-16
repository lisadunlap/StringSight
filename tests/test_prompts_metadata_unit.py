"""
Unit tests for PromptsMetadata generation.

Quick tests to verify the metadata structure without hitting the API.
"""

import pandas as pd

from stringsight.core.data_objects import PropertyDataset
from stringsight.prompt_generation import generate_prompts
from stringsight.schemas import PromptsMetadata


def test_metadata_structure() -> None:
    """Test that generate_prompts returns correct metadata structure."""
    print("\n" + "=" * 80)
    print("UNIT TEST: PromptsMetadata structure")
    print("=" * 80)

    # Create minimal test data
    data = pd.DataFrame([{
        "question_id": "test_001",
        "prompt": "Test prompt",
        "model": "gpt-4",
        "model_response": "Test response",
    }])

    dataset = PropertyDataset.from_dataframe(data, method="single_model")

    task_description = "This is a test task for airline booking."

    # Generate prompts with dynamic mode
    discovery_prompt, clustering_prompts, metadata = generate_prompts(
        task_description=task_description,
        dataset=dataset,
        method="single_model",
        use_dynamic_prompts=True,
        dynamic_prompt_samples=2,
        model="gpt-4.1-mini",
    )

    # Validate return types
    assert isinstance(discovery_prompt, str), "discovery_prompt should be a string"
    assert discovery_prompt is not None and len(discovery_prompt) > 0

    assert clustering_prompts is None or isinstance(clustering_prompts, dict)

    assert isinstance(metadata, PromptsMetadata), "metadata should be PromptsMetadata instance"

    # Validate metadata fields
    assert metadata.discovery_prompt == discovery_prompt
    assert metadata.task_description_original == task_description
    assert isinstance(metadata.dynamic_prompts_used, bool)

    print(f"✓ Discovery prompt: {len(discovery_prompt)} chars")
    print(f"✓ Dynamic prompts used: {metadata.dynamic_prompts_used}")
    print("✓ Metadata structure valid")

    # Validate .dict() method works for serialization
    metadata_dict = metadata.dict()
    assert isinstance(metadata_dict, dict)
    assert "discovery_prompt" in metadata_dict
    assert "dynamic_prompts_used" in metadata_dict

    print("✓ Metadata serialization works")
    print("\n✅ TEST PASSED\n")


def test_static_prompts_metadata() -> None:
    """Test metadata generation with static prompts."""
    print("\n" + "=" * 80)
    print("UNIT TEST: Static prompts metadata")
    print("=" * 80)

    data = pd.DataFrame([{
        "question_id": "test_001",
        "prompt": "Test prompt",
        "model": "gpt-4",
        "model_response": "Test response",
    }])

    dataset = PropertyDataset.from_dataframe(data, method="single_model")
    task_description = "Test task"

    # Generate with dynamic prompts disabled
    discovery_prompt, clustering_prompts, metadata = generate_prompts(
        task_description=task_description,
        dataset=dataset,
        method="single_model",
        use_dynamic_prompts=False,  # Static mode
        model="gpt-4.1-mini",
    )

    assert metadata.dynamic_prompts_used is False, "Should indicate static prompts"
    assert metadata.discovery_prompt is not None
    assert metadata.task_description_original == task_description
    assert clustering_prompts is None

    print(f"✓ Static prompts mode: dynamic_prompts_used = {metadata.dynamic_prompts_used}")
    print("✅ TEST PASSED\n")


def test_no_task_description() -> None:
    """Test metadata when no task description is provided."""
    print("\n" + "=" * 80)
    print("UNIT TEST: No task description")
    print("=" * 80)

    data = pd.DataFrame([{
        "question_id": "test_001",
        "prompt": "Test prompt",
        "model": "gpt-4",
        "model_response": "Test response",
    }])

    dataset = PropertyDataset.from_dataframe(data, method="single_model")

    # No task description
    discovery_prompt, clustering_prompts, metadata = generate_prompts(
        task_description=None,
        dataset=dataset,
        method="single_model",
        use_dynamic_prompts=True,
        model="gpt-4.1-mini",
    )

    # Should fall back to static prompts
    assert metadata.dynamic_prompts_used is False, "Should fall back to static when no task description"
    assert metadata.discovery_prompt is not None
    assert metadata.task_description_original is None, "Should store None when no task description is provided"
    assert isinstance(discovery_prompt, str) and discovery_prompt, "Discovery prompt should be a non-empty string"
    assert clustering_prompts is None

    print("✓ Correctly fell back to static prompts")
    print("✅ TEST PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING PROMPTS METADATA UNIT TESTS")
    print("=" * 80 + "\n")

    test_metadata_structure()
    test_static_prompts_metadata()
    test_no_task_description()

    print("=" * 80)
    print("✅ ALL UNIT TESTS PASSED!")
    print("=" * 80 + "\n")



