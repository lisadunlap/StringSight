"""
Integration test for dynamic prompt generation using airline demo data.

This test reads from `data/airline_data_demo.jsonl` and generates a custom
discovery prompt, then saves it to a text file for manual inspection.
"""

from pathlib import Path

import pandas as pd

from stringsight.core.data_objects import PropertyDataset
from stringsight.prompt_generation import generate_prompts


def test_generate_prompt_from_airline_data() -> None:
    """Test dynamic prompt generation with airline demo data."""
    print("\n" + "=" * 80)
    print("TESTING DYNAMIC PROMPT GENERATION WITH AIRLINE DATA")
    print("=" * 80 + "\n")

    repo_root = Path(__file__).resolve().parents[1]

    # Read airline data
    data_path = repo_root / "data" / "data/airline_data_demo.jsonl"
    print(f"Reading data from: {data_path}")

    df = pd.read_json(data_path, lines=True)
    print(f"Loaded {len(df)} conversations\n")

    # Convert to PropertyDataset
    dataset = PropertyDataset.from_dataframe(df, method="single_model")

    # Define task description
    task_description = """
This is a customer service chatbot for an airline. The chatbot helps customers
with booking flights, checking flight status, managing reservations, and
answering questions about airline policies. Analyze the model's ability to:
- Handle complex multi-turn conversations
- Use tools (search flights, book flights, cancel reservations)
- Follow airline policies correctly
- Provide helpful and accurate information
- Recover from errors gracefully
"""

    print("Task Description:")
    print("-" * 80)
    print(task_description)
    print("-" * 80 + "\n")

    # Generate prompts
    print("Generating prompts (this may take 15-25 seconds)...\n")

    output_dir = repo_root / "test_output" / "airline_prompts"
    discovery_prompt, clustering_prompts, prompts_metadata = generate_prompts(
        task_description=task_description,
        dataset=dataset,
        method="single_model",
        use_dynamic_prompts=True,
        dynamic_prompt_samples=5,
        model="gpt-4.1-mini",  # Use cheaper model for testing
        output_dir=str(output_dir),
    )

    # Assertions
    assert discovery_prompt is not None, "Discovery prompt should not be None"
    assert len(discovery_prompt) > 100, "Discovery prompt should be substantial"
    print(f"✓ Discovery prompt generated ({len(discovery_prompt)} chars)")

    # Check prompts metadata
    assert prompts_metadata is not None, "Prompts metadata should not be None"
    assert prompts_metadata.discovery_prompt == discovery_prompt, "Metadata discovery prompt should match"
    assert prompts_metadata.dynamic_prompts_used is True, "Should indicate dynamic prompts were used"
    assert prompts_metadata.task_description_original == task_description.strip(), "Should store original task description"
    print("✓ Prompts metadata captured correctly")

    # Check for task-specific keywords
    prompt_lower = discovery_prompt.lower()
    has_airline_keywords = any(
        keyword in prompt_lower
        for keyword in ["airline", "flight", "booking", "reservation", "customer service"]
    )
    if has_airline_keywords:
        print("✓ Discovery prompt contains airline-specific keywords")
    else:
        print("⚠ Discovery prompt may not be task-specific")

    # Check that prompt files were saved
    assert (output_dir / "discovery_prompt.txt").exists(), "Discovery prompt file should exist"
    assert (output_dir / "expanded_task_description.txt").exists(), "Expanded task description file should exist"
    print(f"✓ Prompt files saved to {output_dir}")

    if clustering_prompts:
        assert (output_dir / "clustering_prompts.txt").exists(), "Clustering prompts file should exist"
        print("✓ Clustering prompts generated and saved")

    # Print discovery prompt for manual inspection
    print("\n" + "=" * 80)
    print("GENERATED DISCOVERY PROMPT")
    print("=" * 80)
    print(discovery_prompt)
    print("\n" + "=" * 80)

    # Read and print expanded task description
    with open(output_dir / "expanded_task_description.txt") as f:
        expanded = f.read()
    print("\nEXPANDED TASK DESCRIPTION")
    print("=" * 80)
    print(expanded)
    print("=" * 80)

    print("\n✅ TEST PASSED!")
    print("\nGenerated files:")
    print(f"  - {output_dir / 'discovery_prompt.txt'}")
    print(f"  - {output_dir / 'expanded_task_description.txt'}")
    print(f"  - {output_dir / 'clustering_prompts.txt'}")
    print("\nPlease review these files to verify prompt quality.\n")


if __name__ == "__main__":
    # Allow running directly for manual testing
    test_generate_prompt_from_airline_data()



