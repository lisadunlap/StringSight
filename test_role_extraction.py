#!/usr/bin/env python3
"""
Test script to verify role extraction and clustering includes role metadata.

This script:
1. Creates a small test dataset with multi-turn conversations
2. Runs extraction with extract_by_role enabled
3. Runs clustering
4. Verifies that properties, clusters, and metrics contain role information
"""

import pandas as pd
import json
from stringsight import explain

# Create test data with multi-turn conversations in OpenAI format
test_data = [
    {
        "question_id": "q1",
        "prompt": "Help me with Python",
        "model": "gpt-4",
        "model_response": [
            {"role": "user", "content": "What is a list comprehension?"},
            {"role": "assistant", "content": "A list comprehension is a concise way to create lists in Python."}
        ]
    },
    {
        "question_id": "q2",
        "prompt": "Explain variables",
        "model": "gpt-4",
        "model_response": [
            {"role": "user", "content": "How do I define a variable?"},
            {"role": "assistant", "content": "You can define a variable with: name = value"}
        ]
    },
    {
        "question_id": "q3",
        "prompt": "Debug my code",
        "model": "gpt-4",
        "model_response": [
            {"role": "user", "content": "Why isn't my code working?"},
            {"role": "assistant", "content": "Let me help debug that for you."}
        ]
    },
]

df = pd.DataFrame(test_data)

print("=" * 80)
print("TESTING ROLE-BASED EXTRACTION AND CLUSTERING")
print("=" * 80)
print()

# Run extraction with role-based extraction enabled
print("Step 1: Running extraction with extract_by_role=['user', 'assistant']...")
print("-" * 80)

try:
    dataset, metrics = explain(
        df,
        method="single_model",
        extract_by_role=["user", "assistant"],
        sample_size=None,  # Use all data
        output_dir="test_role_extraction_output",
        min_cluster_size=1,  # Low threshold for small test data
        cluster_positive=True,
    )

    print(f"✓ Extraction completed!")
    print()

    # Check properties
    print("Step 2: Checking properties for role field...")
    print("-" * 80)

    properties = dataset.properties
    print(f"Total properties extracted: {len(properties)}")

    properties_with_role = [p for p in properties if p.role is not None]
    print(f"Properties with role field: {len(properties_with_role)}")

    if properties_with_role:
        print("\nSample properties with roles:")
        for i, prop in enumerate(properties_with_role[:3]):
            print(f"  {i+1}. Role: {prop.role}")
            print(f"     Description: {prop.property_description[:80]}...")
            print(f"     Model: {prop.model}")
            print()
    else:
        print("⚠️  WARNING: No properties have role field populated!")

    # Check role distribution
    role_counts = {}
    for prop in properties:
        role = prop.role or "None"
        role_counts[role] = role_counts.get(role, 0) + 1

    print("Role distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    print()

    # Check clusters
    print("Step 3: Checking clusters for role metadata...")
    print("-" * 80)

    clusters = dataset.clusters
    print(f"Total clusters: {len(clusters)}")

    if clusters:
        print("\nCluster details:")
        for i, cluster in enumerate(clusters[:5]):  # Show first 5
            print(f"  Cluster {i+1}:")
            print(f"    Label: {cluster.label}")
            print(f"    Size: {cluster.size}")
            print(f"    Meta keys: {list(cluster.meta.keys()) if cluster.meta else 'None'}")

            # Check if cluster has role info in meta
            if cluster.meta and 'group' in cluster.meta:
                print(f"    Group: {cluster.meta['group']}")

            # Check properties in this cluster for roles
            cluster_prop_ids = set(cluster.property_ids) if hasattr(cluster, 'property_ids') else set()
            cluster_props = [p for p in properties if p.id in cluster_prop_ids]
            if cluster_props:
                cluster_roles = set(p.role for p in cluster_props if p.role)
                if cluster_roles:
                    print(f"    Roles in cluster: {sorted(cluster_roles)}")
            print()

    # Check model_stats for role information
    print("Step 4: Checking model_stats (metrics) for role information...")
    print("-" * 80)

    if metrics:
        # Check model_cluster_scores
        if 'model_cluster_scores' in metrics:
            mcs = metrics['model_cluster_scores']
            if isinstance(mcs, pd.DataFrame):
                print(f"model_cluster_scores shape: {mcs.shape}")
                print(f"Columns: {list(mcs.columns)}")

                # Check if group column exists
                if 'group' in mcs.columns:
                    groups = mcs['group'].unique()
                    print(f"\nGroups in model_cluster_scores: {sorted(groups)}")
                else:
                    print("\n⚠️  WARNING: No 'group' column in model_cluster_scores")

                # Show sample rows
                print("\nSample model_cluster_scores rows:")
                print(mcs[['cluster', 'model', 'size', 'proportion']].head())
            else:
                print(f"model_cluster_scores type: {type(mcs)}")

        # Check cluster_scores
        if 'cluster_scores' in metrics:
            cs = metrics['cluster_scores']
            if isinstance(cs, pd.DataFrame):
                print(f"\ncluster_scores shape: {cs.shape}")
                print(f"Columns: {list(cs.columns)}")

                # Check if group column exists
                if 'group' in cs.columns:
                    groups = cs['group'].unique()
                    print(f"Groups in cluster_scores: {sorted(groups)}")
                else:
                    print("⚠️  WARNING: No 'group' column in cluster_scores")
    else:
        print("⚠️  WARNING: No metrics found")

    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    # Summary
    print("\nSUMMARY:")
    print(f"  ✓ Properties extracted: {len(properties)}")
    print(f"  ✓ Properties with role: {len(properties_with_role)}")
    print(f"  ✓ Clusters created: {len(clusters)}")

    if len(properties_with_role) == 0:
        print("\n⚠️  ISSUE: Role field is not being populated in properties!")
        print("    This means extract_by_role is not working as expected.")
    elif len(properties_with_role) < len(properties):
        print(f"\n⚠️  PARTIAL: Only {len(properties_with_role)}/{len(properties)} properties have roles")
    else:
        print("\n✓ SUCCESS: All properties have role information!")

    # Save results for inspection
    print("\nResults saved to: test_role_extraction_output/")
    print("  - Check full_dataset.json for complete data structure")
    print("  - Check parsed_properties.jsonl for property details")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
