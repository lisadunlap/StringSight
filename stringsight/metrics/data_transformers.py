"""
Data transformation utilities for metrics formatting.

This module provides functions to convert between different data formats
used by the metrics system:
- Nested JSON (legacy) ↔ Flattened DataFrames (frontend)
- Quality metric extraction and column naming
- Significance flag generation from confidence intervals

Design principles:
- Pure functions with no side effects
- Well-documented interfaces
- Backwards compatibility
- Easy to test and modify
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json


def sanitize_metric_name(metric_name: str) -> str:
    """Convert metric names to frontend-safe identifiers.
    
    Args:
        metric_name: Original metric name (e.g., "omni_math_accuracy (0/1)")
        
    Returns:
        Sanitized name (e.g., "omni_math_accuracy_0_1")
        
    Examples:
        >>> sanitize_metric_name("helpfulness (1-5)")
        "helpfulness_1_5"
        >>> sanitize_metric_name("accuracy (0/1)")
        "accuracy_0_1"
    """
    return (metric_name
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_"))


def flatten_model_cluster_scores(model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """Convert nested model-cluster scores to frontend-compatible DataFrame.
    
    This function transforms the nested JSON structure used by FunctionalMetrics
    into a flattened row-based format suitable for frontend consumption.
    
    Args:
        model_cluster_scores: Nested dict with structure:
            {model: {cluster: {metrics_dict}}}
            
    Returns:
        DataFrame with columns:
        - model: Model name
        - cluster: Cluster name
        - size: Cluster size
        - proportion: Cluster proportion
        - proportion_delta: Proportion delta vs average
        - quality_<metric>: Quality scores (flattened)
        - quality_delta_<metric>: Quality deltas (flattened)
        - *_significant: Boolean significance flags
        - *_ci_lower/upper/mean: Confidence intervals
        
    Example:
        >>> nested = {"gpt-4": {"cluster1": {"size": 10, "quality": {"acc": 0.9}}}}
        >>> df = flatten_model_cluster_scores(nested)
        >>> df.columns.tolist()
        ['model', 'cluster', 'size', 'proportion', 'quality_acc', ...]
    """
    rows = []
    
    for model_name, clusters in model_cluster_scores.items():
        for cluster_name, metrics in clusters.items():
            if not isinstance(metrics, dict):
                continue
                
            # Base row structure
            row = {
                "model": model_name,
                "cluster": cluster_name,
                "size": metrics.get("size", 0),
                "proportion": metrics.get("proportion", 0.0),
                "proportion_delta": metrics.get("proportion_delta", 0.0),
                "metadata": metrics.get("metadata", {}),
                "examples": metrics.get("examples", [])
            }
            
            # Flatten quality scores
            quality = metrics.get("quality", {})
            for metric_name, value in quality.items():
                safe_name = sanitize_metric_name(metric_name)
                row[f"quality_{safe_name}"] = value
                
            # Flatten quality_delta scores
            quality_delta = metrics.get("quality_delta", {})
            for metric_name, value in quality_delta.items():
                safe_name = sanitize_metric_name(metric_name)
                row[f"quality_delta_{safe_name}"] = value
            
            # Add confidence intervals
            _add_confidence_intervals(row, metrics, "proportion")
            _add_quality_confidence_intervals(row, metrics, "quality")
            _add_quality_confidence_intervals(row, metrics, "quality_delta")
            
            # Add significance flags
            row["proportion_delta_significant"] = metrics.get("proportion_delta_significant", False)
            
            quality_delta_significant = metrics.get("quality_delta_significant", {})
            for metric_name, is_significant in quality_delta_significant.items():
                safe_name = sanitize_metric_name(metric_name)
                row[f"quality_delta_{safe_name}_significant"] = is_significant
                
            rows.append(row)
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    
    # Ensure consistent column ordering
    base_cols = ["model", "cluster"]
    other_cols = [col for col in df.columns if col not in base_cols]
    return df[base_cols + other_cols]


def flatten_cluster_scores(cluster_scores: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert cluster scores (aggregated across models) to DataFrame.
    
    Args:
        cluster_scores: Dict with structure {cluster: {metrics_dict}}
        
    Returns:
        DataFrame with model="all" and flattened metrics
    """
    rows = []
    
    for cluster_name, metrics in cluster_scores.items():
        if not isinstance(metrics, dict):
            continue
            
        row = {
            "model": "all",  # Indicates aggregation across all models
            "cluster": cluster_name,
            "size": metrics.get("size", 0),
            "proportion": metrics.get("proportion", 0.0),
            "metadata": metrics.get("metadata", {}),
            "examples": metrics.get("examples", [])
        }
        
        # Flatten quality scores (same logic as model_cluster)
        quality = metrics.get("quality", {})
        for metric_name, value in quality.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"quality_{safe_name}"] = value
            
        quality_delta = metrics.get("quality_delta", {})
        for metric_name, value in quality_delta.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"quality_delta_{safe_name}"] = value
        
        # Add CIs and significance
        _add_confidence_intervals(row, metrics, "proportion")
        _add_quality_confidence_intervals(row, metrics, "quality")
        _add_quality_confidence_intervals(row, metrics, "quality_delta")
        
        quality_delta_significant = metrics.get("quality_delta_significant", {})
        for metric_name, is_significant in quality_delta_significant.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"quality_delta_{safe_name}_significant"] = is_significant
            
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    base_cols = ["model", "cluster"]
    other_cols = [col for col in df.columns if col not in base_cols]
    return df[base_cols + other_cols]


def flatten_model_scores(model_scores: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert model scores (aggregated across clusters) to DataFrame.
    
    Args:
        model_scores: Dict with structure {model: {metrics_dict}}
        
    Returns:
        DataFrame with cluster="all_clusters" and flattened metrics
    """
    rows = []
    
    for model_name, metrics in model_scores.items():
        if not isinstance(metrics, dict):
            continue
            
        row = {
            "model": model_name,
            "cluster": "all_clusters",  # Indicates aggregation across all clusters
            "size": metrics.get("size", 0),
            "proportion": metrics.get("proportion", 1.0),  # Always 1.0 for per-model aggregates
            "examples": metrics.get("examples", [])
        }
        
        # Flatten quality scores (same logic as above)
        quality = metrics.get("quality", {})
        for metric_name, value in quality.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"quality_{safe_name}"] = value
            
        quality_delta = metrics.get("quality_delta", {})
        for metric_name, value in quality_delta.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"quality_delta_{safe_name}"] = value
        
        # Add CIs and significance
        _add_confidence_intervals(row, metrics, "proportion")
        _add_quality_confidence_intervals(row, metrics, "quality")
        _add_quality_confidence_intervals(row, metrics, "quality_delta")
        
        quality_delta_significant = metrics.get("quality_delta_significant", {})
        for metric_name, is_significant in quality_delta_significant.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"quality_delta_{safe_name}_significant"] = is_significant
            
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    base_cols = ["model", "cluster"]
    other_cols = [col for col in df.columns if col not in base_cols]
    return df[base_cols + other_cols]


def extract_quality_metrics(df: pd.DataFrame) -> List[str]:
    """Extract available quality metrics from DataFrame columns.
    
    Args:
        df: DataFrame with flattened quality columns
        
    Returns:
        List of quality metric names (without prefixes)
        
    Example:
        >>> df = pd.DataFrame({"quality_accuracy_0_1": [0.9], "quality_delta_accuracy_0_1": [0.1]})
        >>> extract_quality_metrics(df)
        ["accuracy_0_1"]
    """
    quality_cols = [col for col in df.columns if col.startswith("quality_") and not col.startswith("quality_delta_")]
    
    # Remove "quality_" prefix to get metric names
    metric_names = []
    for col in quality_cols:
        # Skip CI columns
        if any(suffix in col for suffix in ["_ci_lower", "_ci_upper", "_ci_mean"]):
            continue
        metric_name = col.replace("quality_", "")
        metric_names.append(metric_name)
    
    return sorted(list(set(metric_names)))


def generate_significance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Generate boolean significance flags from confidence intervals.
    
    This function can be used to compute significance flags when they're missing
    from the original data, using the confidence interval columns.
    
    Args:
        df: DataFrame with CI columns
        
    Returns:
        DataFrame with additional *_significant columns
        
    Note:
        This modifies the input DataFrame in place and also returns it.
    """
    df = df.copy()
    
    # Check proportion_delta significance
    if all(col in df.columns for col in ["proportion_delta_ci_lower", "proportion_delta_ci_upper"]):
        df["proportion_delta_significant"] = ~(
            (df["proportion_delta_ci_lower"] <= 0) & (df["proportion_delta_ci_upper"] >= 0)
        )
    
    # Check quality_delta significance for each metric
    quality_metrics = extract_quality_metrics(df)
    for metric in quality_metrics:
        lower_col = f"quality_delta_{metric}_ci_lower"
        upper_col = f"quality_delta_{metric}_ci_upper"
        sig_col = f"quality_delta_{metric}_significant"
        
        if all(col in df.columns for col in [lower_col, upper_col]):
            df[sig_col] = ~(
                (df[lower_col] <= 0) & (df[upper_col] >= 0)
            )
    
    return df


def _add_confidence_intervals(row: dict, metrics: dict, key: str) -> None:
    """Add confidence interval columns to row for a given key."""
    ci_key = f"{key}_ci"
    if ci_key in metrics:
        ci = metrics[ci_key]
        row[f"{key}_ci_lower"] = ci.get("lower", None)
        row[f"{key}_ci_upper"] = ci.get("upper", None)
        row[f"{key}_ci_mean"] = ci.get("mean", None)


def _add_quality_confidence_intervals(row: dict, metrics: dict, quality_type: str) -> None:
    """Add quality confidence interval columns to row."""
    ci_key = f"{quality_type}_ci"
    if ci_key in metrics:
        quality_ci = metrics[ci_key]
        for metric_name, ci in quality_ci.items():
            safe_name = sanitize_metric_name(metric_name)
            row[f"{quality_type}_{safe_name}_ci_lower"] = ci.get("lower", None)
            row[f"{quality_type}_{safe_name}_ci_upper"] = ci.get("upper", None)
            row[f"{quality_type}_{safe_name}_ci_mean"] = ci.get("mean", None)


def load_and_transform_metrics(results_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load metrics from directory and transform to DataFrames.
    
    This is a convenience function that handles the full load-and-transform pipeline.
    
    Args:
        results_dir: Path to directory containing metrics JSON files
        
    Returns:
        Tuple of (model_cluster_df, cluster_df, model_df). Returns None for any
        files that cannot be loaded.
        
    Example:
        >>> model_cluster_df, cluster_df, model_df = load_and_transform_metrics(Path("results/"))
        >>> print(f"Loaded {len(model_cluster_df)} model-cluster combinations")
    """
    results_dir = Path(results_dir)
    
    # Load JSON files
    model_cluster_scores = None
    cluster_scores = None
    model_scores = None
    
    try:
        with open(results_dir / "model_cluster_scores.json") as f:
            model_cluster_scores = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
        
    try:
        with open(results_dir / "cluster_scores.json") as f:
            cluster_scores = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
        
    try:
        with open(results_dir / "model_scores.json") as f:
            model_scores = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Transform to DataFrames
    model_cluster_df = None
    if model_cluster_scores:
        model_cluster_df = flatten_model_cluster_scores(model_cluster_scores)
        
    cluster_df = None
    if cluster_scores:
        cluster_df = flatten_cluster_scores(cluster_scores)
        
    model_df = None
    if model_scores:
        model_df = flatten_model_scores(model_scores)
    
    return model_cluster_df, cluster_df, model_df


def save_flattened_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to JSONL format with proper handling of complex types.
    
    Args:
        df: DataFrame to save
        output_path: Output file path (should end in .jsonl)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)


# Backwards compatibility functions for legacy code

def convert_flattened_to_nested(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convert flattened DataFrame back to nested JSON structure.
    
    This function provides backwards compatibility for legacy code
    that expects the nested format.
    
    Args:
        df: Flattened DataFrame
        
    Returns:
        Nested dict structure matching original FunctionalMetrics format
        
    Note:
        This is primarily for migration/compatibility purposes. New code should
        use the flattened format directly.
    """
    nested = {}
    
    for _, row in df.iterrows():
        model = row["model"]
        cluster = row["cluster"]
        
        if model not in nested:
            nested[model] = {}
        if cluster not in nested[model]:
            nested[model][cluster] = {}
            
        metrics = nested[model][cluster]
        
        # Basic metrics
        metrics["size"] = row.get("size", 0)
        metrics["proportion"] = row.get("proportion", 0.0)
        metrics["proportion_delta"] = row.get("proportion_delta", 0.0)
        metrics["metadata"] = row.get("metadata", {})
        metrics["examples"] = row.get("examples", [])
        
        # Reconstruct nested quality structure
        quality = {}
        quality_delta = {}
        
        for col in df.columns:
            if col.startswith("quality_") and not any(x in col for x in ["delta_", "_ci_", "_significant"]):
                # Extract original metric name from sanitized column name
                metric_name = col.replace("quality_", "").replace("_", " ")
                quality[metric_name] = row[col]
            elif col.startswith("quality_delta_") and not any(x in col for x in ["_ci_", "_significant"]):
                metric_name = col.replace("quality_delta_", "").replace("_", " ")
                quality_delta[metric_name] = row[col]
        
        if quality:
            metrics["quality"] = quality
        if quality_delta:
            metrics["quality_delta"] = quality_delta
            
        # Add significance flags
        metrics["proportion_delta_significant"] = row.get("proportion_delta_significant", False)
        
        quality_delta_significant = {}
        for col in df.columns:
            if col.startswith("quality_delta_") and col.endswith("_significant"):
                metric_part = col.replace("quality_delta_", "").replace("_significant", "")
                metric_name = metric_part.replace("_", " ")
                quality_delta_significant[metric_name] = row[col]
        
        if quality_delta_significant:
            metrics["quality_delta_significant"] = quality_delta_significant
    
    return nested