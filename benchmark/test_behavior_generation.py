"""
Test script for validating behavior generation (MVP Stage 1).

This script tests only the behavior generation without generating responses,
allowing us to validate the quality of generated behaviors before running
the full expensive pipeline.
"""

import json
import pandas as pd
from create_benchmark import (
    BenchmarkConfig,
    load_dataset_description,
    load_dataset,
    generate_behaviors
)


def test_behavior_generation():
    """Test behavior generation and print results."""

    config = BenchmarkConfig(
        dataset_description_path="input_dataset_descriptions/instructeval.yaml",
        behavior_generation_model="gpt-4o",
        num_behaviors=10
    )

    print("=" * 80)
    print("TESTING BEHAVIOR GENERATION")
    print("=" * 80)

    # Load dataset description
    print(f"\nLoading dataset description: {config.dataset_description_path}")
    dataset_desc = load_dataset_description(config.dataset_description_path)
    print(f"Dataset: {dataset_desc['name']}")
    print(f"Description: {dataset_desc['description'][:200]}...")

    # Load dataset for few-shot examples
    print(f"\nLoading dataset from {dataset_desc['dataset_path']}")
    df = load_dataset(
        dataset_desc['dataset_path'],
        dataset_desc['prompt_column'],
        sample_size=None  # Load all for sampling
    )
    print(f"Loaded {len(df)} prompts")

    # Generate behaviors
    print(f"\nGenerating {config.num_behaviors} behaviors using {config.behavior_generation_model}...")

    behaviors = generate_behaviors(dataset_desc, df, dataset_desc['prompt_column'], config)

    # Display results
    print("\n" + "=" * 80)
    print("GENERATED BEHAVIORS")
    print("=" * 80)

    for i, behavior in enumerate(behaviors, 1):
        print(f"\n[{i}] {behavior.name.upper()}")
        print(f"Category: {behavior.category}")
        print(f"Description: {behavior.description}")
        print(f"\nFull System Prompt:")
        print(f"{behavior.full_system_prompt}")
        print("-" * 80)

    # Save behaviors to JSON for inspection
    behaviors_dict = [
        {
            "name": b.name,
            "description": b.description,
            "category": b.category,
            "full_system_prompt": b.full_system_prompt
        }
        for b in behaviors
    ]

    output_file = "benchmark/generated_behaviors.json"
    with open(output_file, 'w') as f:
        json.dump(behaviors_dict, f, indent=2)

    print(f"\nBehaviors saved to: {output_file}")
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the generated behaviors")
    print("2. If satisfied, run with --sample-size=10 to test response generation")
    print("3. Then run full benchmark generation")


if __name__ == "__main__":
    test_behavior_generation()
