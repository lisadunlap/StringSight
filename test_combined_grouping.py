#!/usr/bin/env python3
"""
Test combined role+behavior_type grouping.
"""

import pandas as pd
from stringsight import explain

# Create test data with multi-turn conversations
test_data = [
    {
        "question_id": "q1",
        "prompt": "Help me",
        "model": "gpt-4",
        "model_response": [
            {"role": "user", "content": "Great question!"},
            {"role": "assistant", "content": "Happy to help."}
        ]
    },
    {
        "question_id": "q2",
        "prompt": "Explain",
        "model": "gpt-4",
        "model_response": [
            {"role": "user", "content": "What is that?"},
            {"role": "assistant", "content": "Let me explain."}
        ]
    },
    {
        "question_id": "q3",
        "prompt": "Debug",
        "model": "gpt-4",
        "model_response": [
            {"role": "user", "content": "Fix this bug"},
            {"role": "assistant", "content": "I'll help debug."}
        ]
    },
]

df = pd.DataFrame(test_data)

print("=" * 80)
print("TESTING COMBINED ROLE+BEHAVIOR_TYPE GROUPING")
print("=" * 80)
print()

print("Running extraction with extract_by_role=['user', 'assistant']...")
print("-" * 80)

dataset, metrics = explain(
    df,
    method="single_model",
    extract_by_role=["user", "assistant"],
    sample_size=None,
    output_dir="test_combined_grouping_output",
    min_cluster_size=1,
    cluster_positive=True,
)

print(f"✓ Extraction completed!")
print()

# Check clusters
print("Checking cluster metadata...")
print("-" * 80)

with open("test_combined_grouping_output/clusters.jsonl") as f:
    import json
    clusters = [json.loads(line) for line in f]

print(f"Total clusters: {len(clusters)}")
print()

for i, cluster in enumerate(clusters[:10]):
    print(f"Cluster {i+1}:")
    print(f"  ID: {cluster.get('id')}")
    print(f"  Label: {cluster.get('label', '')[:80]}...")
    print(f"  Size: {cluster.get('size')}")
    meta = cluster.get('meta', {})
    print(f"  Meta:")
    print(f"    group: {meta.get('group')}")
    print(f"    role: {meta.get('role')}")
    print(f"    behavior_type: {meta.get('behavior_type')}")
    print(f"    role_behavior_type: {meta.get('role_behavior_type')}")
    print()

# Check if we have clusters for different role+behavior combinations
groups = set()
for cluster in clusters:
    meta = cluster.get('meta', {})
    if 'group' in meta:
        groups.add(meta['group'])

print(f"Unique groups found: {sorted(groups)}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

has_combined = any('_' in str(g) for g in groups if g)
has_role = any('role' in cluster.get('meta', {}) for cluster in clusters)
has_behavior = any('behavior_type' in cluster.get('meta', {}) for cluster in clusters)

if has_combined:
    print("✅ Clusters have combined grouping (e.g., 'user_Positive')")
else:
    print("⚠️  Clusters do NOT have combined grouping")

if has_role:
    print("✅ Clusters have 'role' metadata field")
else:
    print("⚠️  Clusters do NOT have 'role' metadata field")

if has_behavior:
    print("✅ Clusters have 'behavior_type' metadata field")
else:
    print("⚠️  Clusters do NOT have 'behavior_type' metadata field")
