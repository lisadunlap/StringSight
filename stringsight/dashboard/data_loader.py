"""
Data loading functionality for the StringSight Gradio app.

This module handles loading pipeline results and converting them to formats
suitable for the Gradio interface.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import os

from .state import app_state
from .plotting import create_model_cluster_dataframe


class DataCache:
    """Simple cache for loaded data to avoid re-loading."""
    _cache = {}
    
    @classmethod
    def get(cls, key: str):
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value: Any):
        cls._cache[key] = value
    
    @classmethod
    def clear(cls):
        cls._cache.clear()


def scan_for_result_subfolders(base_dir: str) -> List[str]:
    """Scan for subfolders that might contain pipeline results."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    # Look for subfolders that contain the required files
    subfolders = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if this subfolder contains pipeline results
            required_files = [
                "model_cluster_scores.json",
                "cluster_scores.json", 
                "model_scores.json",
                "clustered_results_lightweight.jsonl"
            ]
            if all((item / f).exists() for f in required_files):
                subfolders.append(item.name)
    
    return subfolders


def validate_results_directory(results_dir: str) -> Tuple[bool, str]:
    """Validate that the results directory contains the expected files."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return False, f"Directory does not exist: {results_dir}"
    
    if not results_path.is_dir():
        return False, f"Path is not a directory: {results_dir}"
    
    # Check for FunctionalMetrics format files
    required_files = [
        "model_cluster_scores.json",
        "cluster_scores.json",
        "model_scores.json",
    ]
    
    missing_files = []
    for filename in required_files:
        if not (results_path / filename).exists():
            missing_files.append(filename)
    
    # Check for clustered results
    if not (results_path / "clustered_results_lightweight.jsonl").exists():
        missing_files.append("clustered_results_lightweight.jsonl")
    
    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"
    
    return True, ""


def get_available_models(metrics: Dict[str, Any]) -> List[str]:
    """Extract available models from metrics data."""
    model_cluster_scores = metrics.get("model_cluster_scores", {})
    return list(model_cluster_scores.keys())


def get_all_models(metrics: Dict[str, Any]) -> List[str]:
    """Get all available models from metrics data."""
    return get_available_models(metrics)


def load_pipeline_results(results_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Path]:
    """Load pipeline outputs (FunctionalMetrics format only).
    Returns:
        clustered_df: DataFrame of per-conversation data loaded from clustered_results.jsonl
        metrics: Dict containing the three FunctionalMetrics score dictionaries
        model_cluster_df: DataFrame created from model_cluster_scores for plotting/analysis
        results_path: Path to the results directory
    """
    cache_key = f"pipeline_results_{results_dir}"
    cached = DataCache.get(cache_key)
    if cached:
        return cached
    
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    
    # ------------------------------------------------------------------
    # 1. Load FunctionalMetrics score files (must ALL be present)
    # ------------------------------------------------------------------
    required_files = [
        "model_cluster_scores.json",
        "cluster_scores.json",
        "model_scores.json",
    ]
    missing = [f for f in required_files if not (results_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required metrics files in {results_dir}: {', '.join(missing)}"
        )
    
    with open(results_path / "model_cluster_scores.json") as f:
        model_cluster_scores = json.load(f)
    with open(results_path / "cluster_scores.json") as f:
        cluster_scores = json.load(f)
    with open(results_path / "model_scores.json") as f:
        model_scores = json.load(f)
    
    metrics = {
        "model_cluster_scores": model_cluster_scores,
        "cluster_scores": cluster_scores,
        "model_scores": model_scores,
    }
    
    # ------------------------------------------------------------------
    # 2. Load clustered conversation data (JSON-Lines)
    # ------------------------------------------------------------------
    clustered_path = results_path / "clustered_results_lightweight.jsonl"
    if not clustered_path.exists():
        raise FileNotFoundError(f"clustered_results_lightweight.jsonl not found in {results_dir}")
    
    try:
        clustered_df = pd.read_json(clustered_path, lines=True)
    except Exception as e:
        raise ValueError(f"Could not load clustered results: {e}")
    
    # ------------------------------------------------------------------
    # 3. Create model_cluster_df from metrics for plotting/analysis
    # ------------------------------------------------------------------
    model_cluster_df = create_model_cluster_dataframe(model_cluster_scores)
    
    result = (clustered_df, metrics, model_cluster_df, results_path)
    DataCache.set(cache_key, result)
    return result


def load_property_examples(results_path: Path, property_ids: List[str]) -> pd.DataFrame:
    """Load specific property examples on-demand"""
    if not property_ids:
        return pd.DataFrame()
    
    cache_key = f"examples_{results_path}_{hash(tuple(sorted(property_ids)))}"
    cached = DataCache.get(cache_key)
    if cached is not None:
        return cached
        
    # Load full dataset to get prompt/response details
    clustered_path = results_path / "clustered_results_lightweight.jsonl"
    
    if not clustered_path.exists():
        raise FileNotFoundError("Could not load example data - clustered_results_lightweight.jsonl not found")
    
    try:
        full_df = pd.read_json(clustered_path, lines=True)
        result = full_df[full_df['id'].isin(property_ids)]
        DataCache.set(cache_key, result)
        return result
    except Exception as e:
        raise ValueError(f"Failed to load examples: {e}") 