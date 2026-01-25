"""
salience.py

Salience (proportion deviation) computation for model-cluster analysis.
Computes how much each model over/under-represents in each cluster compared to other models.
"""

from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def compute_salience(
    model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute salience (proportion deviation from average of OTHER models) for each model-cluster combination.

    For each model-cluster combination, computes how much that model over/under-represents
    in that cluster compared to other models.

    Salience = model_proportion - avg_proportion_of_other_models

    Args:
        model_cluster_scores: Nested dictionary of model -> cluster -> metrics
                             Must contain 'proportion' field for each model-cluster pair

    Returns:
        Updated model_cluster_scores with 'proportion_delta' field added to each model-cluster pair

    Example:
        If Model A has 30% of its conversations in Cluster X, and all other models average 20%
        in Cluster X, then Model A's proportion_delta for Cluster X is +0.10 (10 percentage points).
    """
    df = pd.DataFrame(model_cluster_scores).reset_index().rename({"index": "cluster"}, axis=1)

    # Step 1: Extract proportion values
    model_names = [col for col in df.columns if col not in ['cluster']]

    # Parse the proportion field from the dictionary-like data
    for model in model_names:
        df[f'{model}_proportion'] = df[model].apply(lambda x: x.get('proportion', 0) if isinstance(x, dict) else 0)

    # Step 2 & 3: Compute deviation from average of OTHER models (excluding self)
    for model in model_names:
        # Get all other models' proportion columns
        other_model_cols = [f'{m}_proportion' for m in model_names if m != model]
        if other_model_cols:
            # Average proportion across all OTHER models
            df[f'{model}_avg_others'] = df[other_model_cols].mean(axis=1)
        else:
            # If only one model, deviation is 0
            df[f'{model}_avg_others'] = 0
        # Deviation = this model's proportion - average of others
        df[f'{model}_deviation'] = df[f'{model}_proportion'] - df[f'{model}_avg_others']

    # Step 4: Add deviation into model_cluster_scores
    for i, row in df.iterrows():
        cluster = row['cluster']
        for model in model_names:
            deviation_value = row[f'{model}_deviation']
            if model in model_cluster_scores and cluster in model_cluster_scores[model]:
                model_cluster_scores[model][cluster]['proportion_delta'] = deviation_value

    return model_cluster_scores
