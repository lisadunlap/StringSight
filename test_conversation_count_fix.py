#!/usr/bin/env python
"""
Test script to verify the conversation count fix works correctly.
"""
import pandas as pd
from stringsight import explain
import json

# Create simple test data with multiple models
data = {
    "question_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
    "prompt": ["What is 2+2?"] * 3 + ["Explain gravity"] * 3,
    "model": ["gpt-4", "claude-3", "gemini-pro"] * 2,
    "model_response": [
        "4", "2+2 equals 4", "The answer is 4",
        "Gravity is a force", "Gravity pulls objects", "Mass attracts mass"
    ],
}
df = pd.DataFrame(data)

print("Test data:")
print(df)
print("\n" + "="*80 + "\n")

# Run analysis
print("Running analysis...")
result = explain(
    df,
    output_dir="/tmp/test_conversation_count",
    max_samples=10,  # Keep it small for testing
)

print("\n" + "="*80 + "\n")
print("Analysis complete! Results saved to: /tmp/test_conversation_count")
print("\n" + "="*80 + "\n")

# Check the cluster_scores output
import os
cluster_scores_path = "/tmp/test_conversation_count/cluster_scores_df.jsonl"
if os.path.exists(cluster_scores_path):
    print("✅ cluster_scores_df.jsonl exists")
    with open(cluster_scores_path) as f:
        cluster_scores = [json.loads(line) for line in f]

    print(f"\nFound {len(cluster_scores)} cluster(s)")
    for i, cluster in enumerate(cluster_scores[:3]):  # Show first 3
        print(f"\nCluster {i+1}:")
        print(f"  Label: {cluster.get('cluster', 'N/A')}")
        print(f"  Size (model responses): {cluster.get('size', 'N/A')}")
        print(f"  Total unique conversations: {cluster.get('total_unique_conversations', 'MISSING!')}")
        print(f"  Proportion: {cluster.get('proportion', 'N/A'):.2%}")
else:
    print("❌ cluster_scores_df.jsonl not found")

# Check model_cluster_scores
model_cluster_scores_path = "/tmp/test_conversation_count/model_cluster_scores_df.jsonl"
if os.path.exists(model_cluster_scores_path):
    print("\n✅ model_cluster_scores_df.jsonl exists")
    with open(model_cluster_scores_path) as f:
        model_cluster_scores = [json.loads(line) for line in f]

    print(f"\nFound {len(model_cluster_scores)} model-cluster pair(s)")

    # Group by cluster to show the comparison
    from collections import defaultdict
    by_cluster = defaultdict(list)
    for row in model_cluster_scores:
        by_cluster[row['cluster']].append(row)

    # Show first cluster
    if by_cluster:
        first_cluster = list(by_cluster.keys())[0]
        print(f"\nExample - First cluster '{first_cluster}':")
        total_responses = 0
        for row in by_cluster[first_cluster][:5]:  # Show first 5 models
            print(f"  {row['model']}: {row['size']} responses ({row['proportion']:.1%})")
            total_responses += row['size']
        print(f"  → Total model responses: {total_responses}")

        # Find corresponding cluster_scores entry
        cluster_entry = next((c for c in cluster_scores if c['cluster'] == first_cluster), None)
        if cluster_entry:
            print(f"  → Unique conversations: {cluster_entry.get('total_unique_conversations', 'N/A')}")
            print(f"\n  {'✅' if total_responses >= cluster_entry.get('total_unique_conversations', 0) else '❌'} Relationship correct: {total_responses} responses >= {cluster_entry.get('total_unique_conversations', 0)} unique conversations")
else:
    print("❌ model_cluster_scores_df.jsonl not found")

print("\n" + "="*80)
print("\nTo test the UI:")
print("  cd /tmp/test_conversation_count")
print("  stringsight launch")
print("  # Then check cluster cards - totals should match bar sums for single conversations")
print("="*80)
