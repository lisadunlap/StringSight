#!/usr/bin/env python3
"""
Test script to verify the OpenAI extractor fix works with single model data.
"""

import pandas as pd
from stringsight.core.data_objects import PropertyDataset
from stringsight.extractors.openai import OpenAIExtractor

def test_openai_extractor():
    """Test that OpenAI extractor works with single model data."""
    
    # Load a small sample of the data
    print("Loading sample data...")
    df = pd.read_json("data/arena_single.jsonl", lines=True)
    
    # Attach the filename to the DataFrame for wandb naming
    df.name = "arena_single.jsonl"
    
    sample_df = df.head(5)  # Just test with 5 rows
    
    print(f"Sample data shape: {sample_df.shape}")
    print(f"Sample columns: {list(sample_df.columns)}")
    
    # Create PropertyDataset
    print("\nCreating PropertyDataset...")
    dataset = PropertyDataset.from_dataframe(sample_df, method="single_model")
    
    print(f"Created dataset with {len(dataset.conversations)} conversations")
    
    # Check the first conversation structure
    if dataset.conversations:
        conv = dataset.conversations[0]
        print(f"\nFirst conversation structure:")
        print(f"  question_id: {conv.question_id}")
        print(f"  model: {conv.model} (type: {type(conv.model)})")
        print(f"  responses: {type(conv.responses)}")
        print(f"  scores: {conv.scores}")
        
        # Test the prompt builder
        print("\nTesting prompt builder...")
        extractor = OpenAIExtractor(model="gpt-4o-mini")  # Use a cheaper model for testing
        
        try:
            prompt = extractor._default_prompt_builder(conv)
            print("✅ Prompt builder worked successfully!")
            print(f"Generated prompt preview: {prompt[:200]}...")
        except Exception as e:
            print(f"❌ Prompt builder failed: {e}")
            return False
    
    print("\n✅ OpenAI extractor test passed!")
    return True

if __name__ == "__main__":
    test_openai_extractor() 