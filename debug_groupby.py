#!/usr/bin/env python3
"""
Debug script to show what groupby_column is being set.
"""

import pandas as pd
from stringsight._public.pipeline_builders import build_single_model_pipeline

# Simulate calling explain() with extract_by_role
extract_by_role = ["user", "assistant"]
kwargs = {}  # Add any kwargs you're passing to explain()

# Build pipeline to see what config is set
pipeline = build_single_model_pipeline(
    output_dir="debug_output",
    extract_by_role=extract_by_role,
    **kwargs
)

# Get the clusterer stage
clusterer = None
for stage in pipeline.stages:
    if 'Cluster' in stage.__class__.__name__:
        clusterer = stage
        break

if clusterer:
    config = clusterer.get_config()
    print("Clusterer config:")
    print(f"  groupby_column: {getattr(config, 'groupby_column', None)}")
    print(f"  cluster_positive: {getattr(config, 'cluster_positive', None)}")
    print()

    # Check if combined grouping would trigger
    test_df = pd.DataFrame({
        'role': ['user', 'assistant'],
        'behavior_type': ['Positive', 'Style']
    })

    group_col = getattr(config, 'groupby_column', None)
    print(f"Group column: {group_col}")
    print(f"Would combined grouping trigger? {group_col == 'role' and 'role' in test_df.columns and 'behavior_type' in test_df.columns}")
else:
    print("No clusterer found in pipeline")
