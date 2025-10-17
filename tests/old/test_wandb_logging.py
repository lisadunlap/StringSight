#!/usr/bin/env python3
"""
Example script demonstrating wandb logging with StringSight.

This script shows how to use the explain function with wandb logging enabled
to track LLM inputs/outputs, parsing success rates, and final results.
"""

import pandas as pd
from stringsight import explain
from stringsight.datasets import load_data
from types import SimpleNamespace
import wandb
# import weave

def main():
    print("🚀 StringSight wandb logging example")
    print("=" * 50)
    
    # Load a small sample of arena data
    print("📊 Loading arena data...")
    args = SimpleNamespace(filter_english=True)
    df, _, _ = load_data("arena", args)
    
    # Take just the first 5 rows for quick demo
    sample_df = df.head(5)
    print(f"Using {len(sample_df)} conversations for demo")
    
    # Run explain with wandb logging enabled
    print("\n🔍 Running explain with wandb logging...")
    
    clustered_df, model_stats = explain(
        sample_df,
        method="side_by_side",
        system_prompt="one_sided_system_prompt",
        model_name="gpt-4o-mini",
        temperature=0.6,
        max_tokens=1000,
        max_workers=4,
        
        # Clustering parameters
        clusterer="hdbscan",
        min_cluster_size=2,  # Small for demo
        embedding_model="text-embedding-3-small",
        hierarchical=True,
        assign_outliers=False,
        
        # Wandb logging enabled
        use_wandb=True,
        wandb_project="StringSight-demo",
        include_embeddings=True,
    )
    
    print("\n✅ Processing complete!")
    print(f"Results shape: {clustered_df.shape}")
    print(f"Model stats: {list(model_stats.keys())}")
    
    # Show what was logged to wandb
    print("\n📈 Logged to wandb:")
    print("  - extraction_inputs_outputs: Table of LLM inputs/outputs")
    print("  - extraction_success_rate: API call success rate")
    print("  - parsing_success_rate: JSON parsing success rate")
    print("  - parsed_properties_sample: Sample of parsed properties")
    print("  - final_results_sample: Sample of final clustered results")
    print("  - final_dataset_shape: Overall dataset metrics")
    
    # Show a few sample properties
    if len(clustered_df) > 0:
        print("\n🔍 Sample properties extracted:")
        sample_properties = clustered_df[['model', 'property_description', 'category', 'impact']].head(3)
        for _, row in sample_properties.iterrows():
            print(f"  Model: {row['model']}")
            print(f"  Property: {row['property_description'][:100]}...")
            print(f"  Category: {row['category']} | Impact: {row['impact']}")
            print()
        
    # finally:
    #     # Clean up wandb run
    #     try:
    #         wandb.finish()
    #         print("🏁 Wandb run finished")
    #     except:
    #         pass
    wandb.finish()
    print("🏁 Wandb run finished")

if __name__ == "__main__":
    main() 