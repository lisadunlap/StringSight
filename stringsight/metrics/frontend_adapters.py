"""
Frontend data adapters for metrics system.

This module provides a clean data access layer that abstracts away the
complexity of different data formats (JSONL vs JSON) and provides
consistent APIs for frontend consumption.

Key features:
- Automatic fallback from JSONL → JSON → computed from operational data
- Type-safe data structures with Pydantic models
- Caching for performance
- Error handling and graceful degradation
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback to dataclass if pydantic not available
    from dataclasses import dataclass as BaseModel
    def Field(**kwargs):
        return None


@dataclass
class MetricsLoadResult:
    """Result of loading metrics data with metadata about source."""
    model_cluster_df: Optional[pd.DataFrame]
    cluster_df: Optional[pd.DataFrame] 
    model_df: Optional[pd.DataFrame]
    source: str  # "jsonl", "json", "computed", or "none"
    quality_metrics: List[str]
    total_rows: int
    

class ModelClusterPayload(BaseModel):
    """Type-safe payload for model-cluster metrics data."""
    data: List[Dict[str, Any]]
    models: List[str]
    clusters: List[str] 
    quality_metrics: List[str]
    total_battles: int
    source: str
    
    if PYDANTIC_AVAILABLE:
        class Config:
            arbitrary_types_allowed = True


class ModelBenchmarkPayload(BaseModel):
    """Type-safe payload for model benchmark data."""
    data: List[Dict[str, Any]]
    models: List[str]
    quality_metrics: List[str]
    source: str
    
    if PYDANTIC_AVAILABLE:
        class Config:
            arbitrary_types_allowed = True


class MetricsDataAdapter:
    """
    Adapter to provide both legacy and frontend-compatible data formats.
    
    This class handles the complexity of loading from different sources
    and provides a consistent interface for frontend code.
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        """Initialize adapter with results directory."""
        self.results_dir = Path(results_dir)
        self._cached_result: Optional[MetricsLoadResult] = None
        
    def load_metrics(self, force_reload: bool = False) -> MetricsLoadResult:
        """
        Load metrics from best available source with fallback logic.
        
        Priority:
        1. JSONL files (preferred - frontend format)
        2. JSON files (legacy format)
        3. Computed from operational data (if available)
        4. None (graceful failure)
        
        Args:
            force_reload: Force reload even if cached data exists
            
        Returns:
            MetricsLoadResult with loaded data and metadata
        """
        if self._cached_result and not force_reload:
            return self._cached_result
            
        result = self._try_load_from_jsonl()
        if result.source != "none":
            self._cached_result = result
            return result
            
        result = self._try_load_from_json()
        if result.source != "none":
            self._cached_result = result
            return result
            
        # Could add fallback to compute from raw data here
        result = MetricsLoadResult(
            model_cluster_df=None,
            cluster_df=None,
            model_df=None,
            source="none",
            quality_metrics=[],
            total_rows=0
        )
        self._cached_result = result
        return result
    
    def get_frontend_payload(self) -> ModelClusterPayload:
        """Get data formatted for React frontend consumption."""
        result = self.load_metrics()
        
        if result.model_cluster_df is None or result.model_cluster_df.empty:
            return ModelClusterPayload(
                data=[],
                models=[],
                clusters=[],
                quality_metrics=[],
                total_battles=0,
                source=result.source
            )
        
        df = result.model_cluster_df
        
        # Calculate total battles (unique conversations) instead of model-cluster rows
        total_battles = self._calculate_total_battles(df)
        
        return ModelClusterPayload(
            data=df.to_dict('records'),
            models=sorted(df['model'].unique().tolist()),
            clusters=df['cluster'].unique().tolist(),
            quality_metrics=result.quality_metrics,
            total_battles=total_battles,
            source=result.source
        )
    
    def get_benchmark_payload(self) -> ModelBenchmarkPayload:
        """Get benchmark data (per-model aggregates) for frontend."""
        result = self.load_metrics()
        
        if result.model_df is None or result.model_df.empty:
            return ModelBenchmarkPayload(
                data=[],
                models=[],
                quality_metrics=[],
                source=result.source
            )
        
        df = result.model_df
        
        return ModelBenchmarkPayload(
            data=df.to_dict('records'),
            models=sorted(df['model'].unique().tolist()),
            quality_metrics=result.quality_metrics,
            source=result.source
        )
    
    def get_legacy_format(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return data in existing nested format for backwards compatibility."""
        result = self.load_metrics()
        
        if result.model_cluster_df is None or result.model_cluster_df.empty:
            return {}
            
        # Convert flattened back to nested for backwards compatibility
        from .data_transformers import convert_flattened_to_nested
        return convert_flattened_to_nested(result.model_cluster_df)
    
    def get_available_quality_metrics(self) -> List[str]:
        """Get list of available quality metrics."""
        result = self.load_metrics()
        return result.quality_metrics
    
    def _try_load_from_jsonl(self) -> MetricsLoadResult:
        """Try to load from JSONL files (preferred format)."""
        try:
            model_cluster_path = self.results_dir / "model_cluster_scores_df.jsonl"
            cluster_path = self.results_dir / "cluster_scores_df.jsonl"
            model_path = self.results_dir / "model_scores_df.jsonl"
            
            dfs = {}
            
            # Load model-cluster data (required)
            if model_cluster_path.exists():
                dfs['model_cluster'] = pd.read_json(model_cluster_path, lines=True)
            else:
                return self._empty_result()
                
            # Load cluster and model data (optional)
            if cluster_path.exists():
                dfs['cluster'] = pd.read_json(cluster_path, lines=True)
            if model_path.exists():
                dfs['model'] = pd.read_json(model_path, lines=True)
            
            # Extract quality metrics
            from .data_transformers import extract_quality_metrics
            quality_metrics = extract_quality_metrics(dfs['model_cluster'])
            
            return MetricsLoadResult(
                model_cluster_df=dfs.get('model_cluster'),
                cluster_df=dfs.get('cluster'),
                model_df=dfs.get('model'),
                source="jsonl",
                quality_metrics=quality_metrics,
                total_rows=len(dfs['model_cluster'])
            )
            
        except Exception:
            return self._empty_result()
    
    def _try_load_from_json(self) -> MetricsLoadResult:
        """Try to load from JSON files and transform to DataFrame."""
        try:
            model_cluster_path = self.results_dir / "model_cluster_scores.json"
            cluster_path = self.results_dir / "cluster_scores.json"
            model_path = self.results_dir / "model_scores.json"
            
            if not model_cluster_path.exists():
                return self._empty_result()
            
            # Load JSON files
            data = {}
            with open(model_cluster_path) as f:
                data['model_cluster_scores'] = json.load(f)
                
            if cluster_path.exists():
                with open(cluster_path) as f:
                    data['cluster_scores'] = json.load(f)
            else:
                data['cluster_scores'] = {}
                
            if model_path.exists():
                with open(model_path) as f:
                    data['model_scores'] = json.load(f)  
            else:
                data['model_scores'] = {}
            
            # Transform to DataFrames
            from .data_transformers import (
                flatten_model_cluster_scores,
                flatten_cluster_scores,
                flatten_model_scores,
                extract_quality_metrics
            )
            
            model_cluster_df = flatten_model_cluster_scores(data['model_cluster_scores'])
            cluster_df = flatten_cluster_scores(data['cluster_scores']) if data['cluster_scores'] else pd.DataFrame()
            model_df = flatten_model_scores(data['model_scores']) if data['model_scores'] else pd.DataFrame()
            
            quality_metrics = extract_quality_metrics(model_cluster_df)
            
            return MetricsLoadResult(
                model_cluster_df=model_cluster_df,
                cluster_df=cluster_df if not cluster_df.empty else None,
                model_df=model_df if not model_df.empty else None,
                source="json",
                quality_metrics=quality_metrics,
                total_rows=len(model_cluster_df)
            )
            
        except Exception:
            return self._empty_result()
    
    def _empty_result(self) -> MetricsLoadResult:
        """Return empty result."""
        return MetricsLoadResult(
            model_cluster_df=None,
            cluster_df=None,
            model_df=None,
            source="none",
            quality_metrics=[],
            total_rows=0
        )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded data."""
        result = self.load_metrics()
        
        if result.model_cluster_df is None:
            return {
                "source": result.source,
                "models": 0,
                "clusters": 0,
                "total_battles": 0,
                "quality_metrics": 0,
                "has_confidence_intervals": False,
                "significant_differences": 0
            }
        
        df = result.model_cluster_df
        
        # Check for confidence intervals
        ci_cols = [col for col in df.columns if "_ci_lower" in col]
        has_ci = len(ci_cols) > 0
        
        # Count significant differences
        sig_cols = [col for col in df.columns if col.endswith("_significant")]
        sig_count = sum(df[col].sum() if col in df.columns else 0 for col in sig_cols)
        
        # Calculate total battles (unique conversations) instead of model-cluster rows
        total_battles = self._calculate_total_battles(df)
        
        return {
            "source": result.source,
            "models": df['model'].nunique(),
            "clusters": df['cluster'].nunique(),
            "total_battles": total_battles,
            "quality_metrics": len(result.quality_metrics),
            "has_confidence_intervals": has_ci,
            "significant_differences": int(sig_count),
            "quality_metric_names": result.quality_metrics
        }
    
    def _calculate_total_battles(self, df: pd.DataFrame) -> int:
        """Calculate total number of unique conversations (battles) from model-cluster data.
        
        Args:
            df: Model-cluster DataFrame with 'examples' column containing conversation IDs
            
        Returns:
            Count of unique conversation IDs across all models and clusters
        """
        if df is None or df.empty:
            return 0
            
        # Extract unique conversation IDs from examples field
        unique_conversations = set()
        
        for _, row in df.iterrows():
            examples = row.get('examples', [])
            if examples:
                # examples format: [[conversation_id, conversation_metadata, property_metadata], ...]
                for example in examples:
                    if isinstance(example, (list, tuple)) and len(example) > 0:
                        conversation_id = example[0]  # First element is conversation_id
                        if conversation_id:
                            unique_conversations.add(str(conversation_id))
        
        return len(unique_conversations)


def create_adapter(results_dir: Union[str, Path]) -> MetricsDataAdapter:
    """Convenience function to create a MetricsDataAdapter.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Configured MetricsDataAdapter instance
        
    Example:
        >>> adapter = create_adapter("results/my_experiment/")
        >>> payload = adapter.get_frontend_payload()
        >>> print(f"Loaded {len(payload.data)} battles from {len(payload.models)} models")
    """
    return MetricsDataAdapter(results_dir)