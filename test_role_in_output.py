#!/usr/bin/env python3
"""
Simple test to verify role field appears in output files after running test_role_extraction.py.
This test doesn't run the pipeline - it just checks the output files.
"""

import json
import sys

print("=" * 80)
print("CHECKING ROLE IN OUTPUT FILES")
print("=" * 80)
print()

# Test 1: Check properties have role field
print("Test 1: Checking properties in clustered_results.jsonl")
print("-" * 80)

properties_with_role = 0
properties_without_role = 0
properties_by_role = {}
properties_by_behavior_type = {}

with open('test_role_extraction_output/clustered_results.jsonl') as f:
    for line in f:
        data = json.loads(line)
        role = data.get('role')
        behavior_type = data.get('behavior_type')

        if role:
            properties_with_role += 1
            properties_by_role[role] = properties_by_role.get(role, 0) + 1
        else:
            properties_without_role += 1

        if behavior_type:
            properties_by_behavior_type[behavior_type] = properties_by_behavior_type.get(behavior_type, 0) + 1

print(f"Properties with role field: {properties_with_role}")
print(f"Properties without role field: {properties_without_role}")
print(f"\nRole distribution:")
for role, count in sorted(properties_by_role.items()):
    print(f"  {role}: {count}")
print(f"\nBehavior type distribution:")
for bt, count in sorted(properties_by_behavior_type.items()):
    print(f"  {bt}: {count}")
print()

# Test 2: Check clusters have role metadata
print("Test 2: Checking clusters in full_dataset.json")
print("-" * 80)

with open('test_role_extraction_output/full_dataset.json') as f:
    full_data = json.load(f)

clusters_with_role = 0
clusters_without_role = 0
cluster_metadata = []

for cluster in full_data.get('clusters', []):
    meta = cluster.get('meta', {})
    cluster_id = cluster.get('id')
    cluster_label = cluster.get('label', '')[:60]

    if 'role' in meta or 'group' in meta:
        clusters_with_role += 1
        cluster_metadata.append({
            'id': cluster_id,
            'label': cluster_label,
            'group': meta.get('group'),
            'role': meta.get('role')
        })
    else:
        clusters_without_role += 1

print(f"Clusters with role metadata: {clusters_with_role}")
print(f"Clusters without role metadata: {clusters_without_role}")
print(f"\nCluster metadata:")
for cm in cluster_metadata:
    print(f"  Cluster {cm['id']}: {cm['label']}...")
    print(f"    group: {cm['group']}, role: {cm['role']}")
print()

# Test 3: Check model_cluster_scores have role metadata
print("Test 3: Checking model_cluster_scores")
print("-" * 80)

with open('test_role_extraction_output/model_cluster_scores_df.jsonl') as f:
    scores_with_role = 0
    scores_without_role = 0
    score_metadata = []

    for line in f:
        score = json.loads(line)
        metadata = score.get('metadata', {})

        if 'role' in metadata or 'group' in metadata:
            scores_with_role += 1
            score_metadata.append({
                'cluster': score.get('cluster', '')[:40],
                'model': score.get('model'),
                'group': metadata.get('group'),
                'role': metadata.get('role')
            })
        else:
            scores_without_role += 1

print(f"Model cluster scores with role metadata: {scores_with_role}")
print(f"Model cluster scores without role metadata: {scores_without_role}")
print(f"\nScore metadata:")
for sm in score_metadata:
    print(f"  {sm['model']} - {sm['cluster']}...")
    print(f"    group: {sm['group']}, role: {sm['role']}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

all_pass = True

if properties_with_role > 0 and properties_without_role == 0:
    print("✅ All properties have role field")
else:
    print(f"⚠️  Only {properties_with_role}/{properties_with_role + properties_without_role} properties have role field")
    all_pass = False

if clusters_with_role > 0 and clusters_without_role == 0:
    print("✅ All clusters have role metadata")
else:
    print(f"⚠️  Only {clusters_with_role}/{clusters_with_role + clusters_without_role} clusters have role metadata")
    all_pass = False

if scores_with_role > 0 and scores_without_role == 0:
    print("✅ All model_cluster_scores have role metadata")
else:
    print(f"⚠️  Only {scores_with_role}/{scores_with_role + scores_without_role} scores have role metadata")
    all_pass = False

print()
if all_pass:
    print("🎉 SUCCESS: Role information is present in all output files!")
    sys.exit(0)
else:
    print("❌ FAIL: Some outputs are missing role information")
    sys.exit(1)
