#!/usr/bin/env python3
"""
Extract conversations, properties, and clusters from full_dataset.json to separate JSONL files

This script extracts data from a full_dataset.json file and saves each component
as a separate JSONL file in the same directory:
- conversation.jsonl
- properties.jsonl
- clusters.jsonl

Usage:
    python scripts/extract_conversations.py <results_directory>
    python scripts/extract_conversations.py results/koala
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def extract_conversations_from_dataset(dataset: Dict[str, Any]) -> list:
    """
    Extract conversations from a full dataset dictionary.

    Args:
        dataset: Dictionary containing the full dataset

    Returns:
        List of conversation dictionaries
    """
    return dataset.get("conversations", [])


def convert_conversation_format(conv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert conversation from internal format to input format.

    Converts:
    - scores -> score (single model)
    - scores -> score_a, score_b (side-by-side)
    - model/responses list -> model_a/model_b, model_a_response/model_b_response

    Args:
        conv: Conversation dictionary in internal format

    Returns:
        Conversation dictionary in input format
    """
    # Build base conversation dict
    conv_dict = {
        "question_id": conv.get("question_id"),
        "prompt": conv.get("prompt"),
    }

    model = conv.get("model")
    responses = conv.get("responses")
    scores = conv.get("scores")
    meta = conv.get("meta", {})

    # Handle side-by-side vs single model format
    if isinstance(model, list):
        # Side-by-side format
        conv_dict["model_a"] = model[0]
        conv_dict["model_b"] = model[1]
        conv_dict["model_a_response"] = responses[0] if isinstance(responses, list) else responses
        conv_dict["model_b_response"] = responses[1] if isinstance(responses, list) and len(responses) > 1 else responses

        # Convert scores list to score_a/score_b
        if isinstance(scores, list) and len(scores) == 2:
            conv_dict["score_a"] = scores[0]
            conv_dict["score_b"] = scores[1]
        else:
            conv_dict["score_a"] = {}
            conv_dict["score_b"] = {}

        # Add meta fields (includes winner)
        conv_dict.update(meta)
    else:
        # Single model format
        conv_dict["model"] = model
        conv_dict["model_response"] = responses
        conv_dict["score"] = scores

        # Add meta fields
        conv_dict.update(meta)

    return conv_dict


def save_conversations_jsonl(conversations: list, output_path: Path) -> None:
    """
    Save conversations to a JSONL file in input format.

    Args:
        conversations: List of conversation dictionaries
        output_path: Path to save the JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            # Convert to input format
            conv_dict = convert_conversation_format(conversation)
            json.dump(conv_dict, f, ensure_ascii=False)
            f.write('\n')


def save_properties_jsonl(properties: list, output_path: Path) -> None:
    """
    Save properties to a JSONL file.

    Args:
        properties: List of property dictionaries
        output_path: Path to save the JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for prop in properties:
            json.dump(prop, f, ensure_ascii=False)
            f.write('\n')


def save_clusters_jsonl(clusters: list, output_path: Path) -> None:
    """
    Save clusters to a JSONL file.

    Args:
        clusters: List of cluster dictionaries
        output_path: Path to save the JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for cluster in clusters:
            json.dump(cluster, f, ensure_ascii=False)
            f.write('\n')


def process_results_directory(results_dir: Path) -> None:
    """
    Process a results directory and extract conversations, properties, and clusters from full_dataset.json.

    Args:
        results_dir: Path to the results directory
    """
    full_dataset_path = results_dir / "full_dataset.json"

    if not full_dataset_path.exists():
        print(f"⚠️  No full_dataset.json found in {results_dir}")
        return

    print(f"📖 Reading: {full_dataset_path}")

    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Extract conversations
    conversations = extract_conversations_from_dataset(dataset)
    if conversations:
        output_path = results_dir / "conversation.jsonl"
        save_conversations_jsonl(conversations, output_path)
        print(f"✅ Extracted {len(conversations)} conversations")
        print(f"💾 Saved to: {output_path}")
    else:
        print(f"⚠️  No conversations found in {full_dataset_path}")

    # Extract properties
    properties = dataset.get("properties", [])
    if properties:
        properties_path = results_dir / "properties.jsonl"
        save_properties_jsonl(properties, properties_path)
        print(f"✅ Extracted {len(properties)} properties")
        print(f"💾 Saved to: {properties_path}")

    # Extract clusters
    clusters = dataset.get("clusters", [])
    if clusters:
        clusters_path = results_dir / "clusters.jsonl"
        save_clusters_jsonl(clusters, clusters_path)
        print(f"✅ Extracted {len(clusters)} clusters")
        print(f"💾 Saved to: {clusters_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/extract_conversations.py <results_directory>")
        print("\nExample:")
        print("  python scripts/extract_conversations.py results/koala")
        print("\nThis will extract and save:")
        print("  - conversation.jsonl (conversations in input format)")
        print("  - properties.jsonl (properties)")
        print("  - clusters.jsonl (clusters)")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"❌ Error: Directory not found: {results_dir}")
        sys.exit(1)

    if not results_dir.is_dir():
        print(f"❌ Error: Not a directory: {results_dir}")
        sys.exit(1)

    print(f"Processing results directory: {results_dir}\n")
    process_results_directory(results_dir)
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
