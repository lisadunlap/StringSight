"""
Minimal FastAPI app exposing validation and conversation formatting.

Endpoints:
- GET /health
- POST /detect-and-validate   → parse, auto-detect, validate, preview
- POST /conversations         → parse, auto-detect, validate, return traces

This module is isolated from the Gradio app. It can be run independently:
    uvicorn stringsight.api:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import asyncio
import io
import os
import time

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path

from stringsight.formatters import (
    Method,
    detect_method,
    validate_required_columns,
    format_conversations,
)
from stringsight.utils.df_utils import explode_score_columns
from stringsight import public as public_api
from stringsight.clusterers import get_clusterer
from stringsight.metrics.cluster_subset import enrich_clusters_with_metrics, compute_total_conversations_by_model
from stringsight.logging_config import get_logger
from stringsight.schemas import ExtractBatchRequest, ExtractJobStartRequest
import threading, uuid
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timedelta
from datetime import datetime, timedelta
import hashlib

logger = get_logger(__name__)

# -------------------------------------------------------------------------
# Render persistent disk configuration
# -------------------------------------------------------------------------
from stringsight.utils.paths import _get_persistent_data_dir, _get_results_dir, _get_cache_dir

# -------------------------------------------------------------------------
# Simple in-memory cache for parsed JSONL data with TTL
# -------------------------------------------------------------------------
_JSONL_CACHE: Dict[str, tuple[List[Dict[str, Any]], datetime]] = {}
_CACHE_TTL = timedelta(minutes=15)  # Cache for 15 minutes
_CACHE_LOCK = threading.Lock()

def _get_file_hash(path: Path) -> str:
    """Get a hash of file path and modification time for cache key."""
    stat = path.stat()
    key_str = f"{path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_cached_jsonl(path: Path, nrows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read JSONL file with caching. Cache key includes file mtime to auto-invalidate on changes.

    Only caches full file reads (nrows=None) to avoid cache bloat. For partial reads,
    reads directly from disk.
    """
    # Only cache full file reads to avoid memory bloat
    if nrows is not None:
        logger.debug(f"Partial read requested for {path.name} (nrows={nrows}), skipping cache")
        return _read_jsonl_as_list(path, nrows)

    cache_key = _get_file_hash(path)

    with _CACHE_LOCK:
        if cache_key in _JSONL_CACHE:
            cached_data, cached_time = _JSONL_CACHE[cache_key]
            # Check if cache is still valid
            if datetime.now() - cached_time < _CACHE_TTL:
                logger.debug(f"Cache hit for {path.name}")
                return cached_data
            else:
                # Remove expired entry
                del _JSONL_CACHE[cache_key]
                logger.debug(f"Cache expired for {path.name}")

    # Cache miss - read from disk
    logger.debug(f"Cache miss for {path.name}, reading from disk")
    data = _read_jsonl_as_list(path, nrows)

    # Store in cache (only if full file read)
    if nrows is None:
        with _CACHE_LOCK:
            _JSONL_CACHE[cache_key] = (data, datetime.now())

    return data


def _get_base_browse_dir() -> Path:
    """Return the base directory allowed for server-side browsing.

    Defaults to the current working directory. You can override by setting
    environment variable `BASE_BROWSE_DIR` to an absolute path.
    """
    env = os.environ.get("BASE_BROWSE_DIR")
    base = Path(env).expanduser().resolve() if env else Path.cwd()
    return base


def _resolve_within_base(user_path: str) -> Path:
    """Resolve a user-supplied path and ensure it is within the allowed base.

    Args:
        user_path: Path provided by the client (file or directory)

    Returns:
        Absolute `Path` guaranteed to be within the base directory

    Raises:
        HTTPException: if the path is invalid or escapes the base directory
    """
    base = _get_base_browse_dir()
    target = Path(user_path).expanduser()
    # Treat relative paths as relative to base
    target = (base / target).resolve() if not target.is_absolute() else target.resolve()
    try:
        target.relative_to(base)
    except Exception:
        raise HTTPException(status_code=400, detail="Path is outside the allowed base directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    return target


def _read_json_safe(path: Path) -> Any:
    """Read a JSON file from disk into a Python object."""
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_as_list(path: Path, nrows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Optional row cap."""
    import json
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if nrows is not None and (i + 1) >= nrows:
                break
    return rows

class RowsPayload(BaseModel):
    rows: List[Dict[str, Any]]
    method: Optional[Literal["single_model", "side_by_side"]] = None


class ReadRequest(BaseModel):
    """Request body for reading a dataset from the server filesystem.

    Use with caution – this assumes the server has access to the path.
    """
    path: str
    method: Optional[Literal["single_model", "side_by_side"]] = None
    limit: Optional[int] = None  # return all rows if None


class ListRequest(BaseModel):
    path: str  # directory to list (server-side)
    exts: Optional[List[str]] = None  # e.g., [".jsonl", ".json", ".csv"]


class ResultsLoadRequest(BaseModel):
    """Request to load a results directory from the server filesystem.

    Attributes:
        path: Absolute or base-relative path to the results directory, which must
              be within BASE_BROWSE_DIR (defaults to current working directory).
        max_conversations: Maximum number of conversations to load (default: all).
                          Use this to limit memory usage for large datasets.
        max_properties: Maximum number of properties to load (default: all).
    """
    path: str
    max_conversations: Optional[int] = None
    max_properties: Optional[int] = None


class FlexibleColumnMapping(BaseModel):
    """Column mapping specification for flexible data processing."""
    prompt_col: str
    response_cols: List[str]
    model_cols: Optional[List[str]] = None
    score_cols: Optional[List[str]] = None
    method: Literal["single_model", "side_by_side"] = "single_model"


class FlexibleDataRequest(BaseModel):
    """Request for flexible data processing with user-specified column mapping."""
    rows: List[Dict[str, Any]]
    mapping: FlexibleColumnMapping


class AutoDetectRequest(BaseModel):
    """Request for auto-detecting column mappings."""
    rows: List[Dict[str, Any]]  # Sample of data for detection




# -----------------------------
# Extraction endpoints schemas
# -----------------------------

class ExtractSingleRequest(BaseModel):
    row: Dict[str, Any]
    method: Optional[Literal["single_model", "side_by_side"]] = None
    system_prompt: Optional[str] = None
    task_description: Optional[str] = None
    model_name: Optional[str] = "gpt-4.1"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 16000
    max_workers: Optional[int] = 128
    include_scores_in_prompt: Optional[bool] = False
    use_wandb: Optional[bool] = False
    output_dir: Optional[str] = None
    return_debug: Optional[bool] = False


# ExtractBatchRequest moved to schemas.py


# -----------------------------
# DataFrame operation schemas
# -----------------------------

class DFRows(BaseModel):
    rows: List[Dict[str, Any]]


class DFSelectRequest(DFRows):
    include: Dict[str, List[Any]] = {}
    exclude: Dict[str, List[Any]] = {}


class DFGroupPreviewRequest(DFRows):
    by: str
    numeric_cols: Optional[List[str]] = None


class DFGroupRowsRequest(DFRows):
    by: str
    value: Any
    page: int = 1
    page_size: int = 10


class DFCustomRequest(DFRows):
    code: str  # pandas expression using df


def _load_dataframe_from_upload(upload: UploadFile) -> pd.DataFrame:
    filename = (upload.filename or "").lower()
    raw = upload.file.read()
    # Decode text formats
    if filename.endswith(".jsonl"):
        text = raw.decode("utf-8")
        return pd.read_json(io.StringIO(text), lines=True)
    if filename.endswith(".json"):
        text = raw.decode("utf-8")
        return pd.read_json(io.StringIO(text))
    if filename.endswith(".csv"):
        text = raw.decode("utf-8")
        return pd.read_csv(io.StringIO(text))
    raise HTTPException(status_code=400, detail="Unsupported file format. Use JSONL, JSON, or CSV.")


def _load_dataframe_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _load_dataframe_from_path(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if p.endswith(".json"):
        return pd.read_json(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise HTTPException(status_code=400, detail="Unsupported file format. Use JSONL, JSON, or CSV.")


def _resolve_df_and_method(
    file: UploadFile | None,
    payload: RowsPayload | None,
) -> tuple[pd.DataFrame, Method]:
    if not file and not payload:
        raise HTTPException(status_code=400, detail="Provide either a file upload or a rows payload.")

    if file:
        df = _load_dataframe_from_upload(file)
        detected = detect_method(list(df.columns))
        method = detected or (payload.method if payload else None)  # type: ignore[assignment]
    else:
        assert payload is not None
        df = _load_dataframe_from_rows(payload.rows)
        method = payload.method or detect_method(list(df.columns))

    if method is None:
        raise HTTPException(status_code=422, detail="Unable to detect dataset method from columns.")

    # Validate required columns strictly (no defaults)
    missing = validate_required_columns(df, method)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Missing required columns for {method}",
                "missing": missing,
                "available": list(df.columns),
            },
        )

    return df, method


app = FastAPI(title="StringSight API", version="0.1.0")

# Ensure local installs work without external services.
# When using the default SQLite DB, create tables on startup.
from stringsight.database import init_db


@app.on_event("startup")
def _startup_init_db() -> None:
    """Initialize local SQLite database tables on application startup."""
    init_db()

# Initialize persistent disk configuration on startup
# This sets up environment variables for cache and results directories
_get_cache_dir()  # Call this to auto-configure cache if RENDER_DISK_PATH is set

# GZIP compression disabled - can add significant CPU overhead
# Uncomment below if network transfer is the bottleneck:
# from fastapi.middleware.gzip import GZipMiddleware
# app.add_middleware(GZipMiddleware, minimum_size=10000, compresslevel=1)

# CORS configuration - allow all origins for development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (cannot use with allow_credentials=True)
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicitly allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to frontend
)

from stringsight.routers.jobs import router as jobs_router

app.include_router(jobs_router)

# Include metrics endpoints (basic file serving)
@app.get("/metrics/summary/{results_dir}")
def get_metrics_summary(results_dir: str) -> Dict[str, Any]:
    """Get basic summary of available metrics files."""
    try:
        from pathlib import Path
        import pandas as pd
        
        base_path = Path("results") / results_dir
        model_cluster_file = base_path / "model_cluster_scores_df.jsonl"
        
        if not model_cluster_file.exists():
            raise HTTPException(status_code=404, detail=f"Metrics data not found for {results_dir}")
        
        # Read a small sample to get basic info
        df = pd.read_json(model_cluster_file, lines=True, nrows=100)
        models = sorted(df['model'].unique().tolist()) if 'model' in df.columns else []
        clusters = df['cluster'].unique().tolist() if 'cluster' in df.columns else []
        
        # Extract quality metrics from column names
        quality_metrics = []
        for col in df.columns:
            if col.startswith('quality_') and not col.endswith(('_delta', '_significant')):
                metric = col.replace('quality_', '')
                if metric not in quality_metrics:
                    quality_metrics.append(metric)
        
        return {
            "source": "jsonl",
            "models": len(models),
            "clusters": len(clusters),
            "total_battles": len(df),
            "has_confidence_intervals": any("_ci_" in col for col in df.columns),
            "quality_metric_names": quality_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")


@app.get("/metrics/model-cluster/{results_dir}")  
def get_model_cluster_metrics(results_dir: str) -> Dict[str, Any]:
    """Get model-cluster metrics data."""
    try:
        from pathlib import Path
        import pandas as pd
        
        base_path = Path("results") / results_dir
        model_cluster_file = base_path / "model_cluster_scores_df.jsonl"
        
        if not model_cluster_file.exists():
            raise HTTPException(status_code=404, detail=f"Model-cluster data not found for {results_dir}")
        
        df = pd.read_json(model_cluster_file, lines=True)
        
        models = sorted(df['model'].unique().tolist()) if 'model' in df.columns else []
        clusters = df['cluster'].unique().tolist() if 'cluster' in df.columns else []
        
        return {
            "source": "jsonl",
            "models": models,
            "data": df.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model-cluster data: {str(e)}")


@app.get("/metrics/benchmark/{results_dir}")
def get_benchmark_metrics(results_dir: str) -> Dict[str, Any]:
    """Get benchmark metrics data."""
    try:
        from pathlib import Path
        import pandas as pd
        
        base_path = Path("results") / results_dir
        model_scores_file = base_path / "model_scores_df.jsonl"
        
        if not model_scores_file.exists():
            raise HTTPException(status_code=404, detail=f"Benchmark data not found for {results_dir}")
        
        df = pd.read_json(model_scores_file, lines=True)
        
        models = sorted(df['model'].unique().tolist()) if 'model' in df.columns else []
        
        return {
            "source": "jsonl",
            "models": models,
            "data": df.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading benchmark data: {str(e)}")


@app.get("/metrics/quality-metrics/{results_dir}")
def get_quality_metrics(results_dir: str) -> Dict[str, Any]:
    """Get available quality metrics."""
    try:
        from pathlib import Path
        import pandas as pd
        
        base_path = Path("results") / results_dir
        model_cluster_file = base_path / "model_cluster_scores_df.jsonl"
        
        if not model_cluster_file.exists():
            raise HTTPException(status_code=404, detail=f"Metrics data not found for {results_dir}")
        
        # Read just the first row to get column names
        df = pd.read_json(model_cluster_file, lines=True, nrows=1)
        
        # Extract quality metrics from column names
        quality_metrics = []
        for col in df.columns:
            if col.startswith('quality_') and not col.endswith(('_delta', '_significant')):
                metric = col.replace('quality_', '')
                if metric not in quality_metrics:
                    quality_metrics.append(metric)
        
        return {"quality_metrics": quality_metrics}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading quality metrics: {str(e)}")


@app.get("/health")
def health() -> Dict[str, bool]:
    logger.debug("BACKEND: Health check called")
    return {"ok": True}


# Alias with /api prefix for clients expecting /api/health
@app.get("/api/health")
def api_health() -> Dict[str, bool]:
    """Health check alias at /api/health to match frontend expectations."""
    logger.debug("BACKEND: API Health check called")
    return {"ok": True}


# -----------------------------
# Clustering/metrics – embedding models
# -----------------------------

@app.get("/embedding-models")
def get_embedding_models() -> Dict[str, Any]:
    """Return a curated list of embedding model identifiers.

    Later we can make this dynamic via config/env. Keep it simple for now.
    """
    models = [
        "openai/text-embedding-3-large",
        "openai/text-embedding-3-large",
        "bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    return {"models": models}

@app.get("/debug")
def debug() -> Dict[str, Any]:
    import os
    if os.environ.get("STRINGSIGHT_DEBUG") in ("1", "true", "True"):
        logger.debug("BACKEND: Debug endpoint called")
    return {"status": "server_running", "message": "Backend is alive!"}

@app.post("/debug/post")
def debug_post(body: Dict[str, Any]) -> Dict[str, Any]:
    import os
    if os.environ.get("STRINGSIGHT_DEBUG") in ("1", "true", "True"):
        logger.debug(f"BACKEND: Debug POST called with keys: {list(body.keys())}")
    return {"status": "post_working", "received_keys": list(body.keys())}


# -----------------------------
# Clustering + Metrics Orchestration (simple contracts)
# -----------------------------

class ClusterRunParams(BaseModel):
    minClusterSize: int | None = None
    embeddingModel: str = "openai/text-embedding-3-large"
    groupBy: Optional[str] = "none"  # none | category | behavior_type


class ClusterRunRequest(BaseModel):
    operationalRows: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    params: ClusterRunParams
    output_dir: Optional[str] = None
    score_columns: Optional[List[str]] = None  # NEW: List of score column names to convert to dict format
    method: Optional[str] = "single_model"  # NEW: Method for score column conversion


@app.post("/cluster/run")
async def cluster_run(
    req: ClusterRunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Run clustering directly on existing properties without re-running extraction.
    
    This is much more efficient than the full explain() pipeline since it skips
    the expensive LLM property extraction step and works with already-extracted properties.
    
    Note: Cache is disk-backed (LMDB-based) and thread-safe.
    """
    from stringsight.core.data_objects import PropertyDataset, Property, ConversationRecord
    from stringsight.clusterers import get_clusterer
    import os
    
    # Preserve original cache setting; DiskCache does not use LMDB toggles
    original_cache_setting = os.environ.get("STRINGSIGHT_DISABLE_CACHE", "0")
    os.environ["STRINGSIGHT_DISABLE_CACHE"] = original_cache_setting

    # Force-drop any pre-initialized global LMDB caches so this request runs cacheless
    from stringsight.core import llm_utils as _llm_utils
    from stringsight.clusterers import clustering_utils as _cu
    _orig_default_cache = getattr(_llm_utils, "_default_cache", None)
    _orig_default_llm_utils = getattr(_llm_utils, "_default_llm_utils", None)
    _orig_embed_cache = getattr(_cu, "_cache", None)
    try:
        _llm_utils._default_cache = None
        _llm_utils._default_llm_utils = None
    except Exception:
        pass
    try:
        if hasattr(_cu, "_cache"):
            _cu._cache = None
    except Exception:
        pass
    except Exception:
        pass
    
    try:
        # NEW: Preprocess operationalRows to handle score_columns conversion
        # This ensures scores are in the expected nested dict format before creating ConversationRecords
        score_columns_to_use = req.score_columns
        
        # Auto-detect score columns if not provided
        if not score_columns_to_use and req.operationalRows:
            import pandas as pd
            operational_df = pd.DataFrame(req.operationalRows)

            # Check if 'score' or 'scores' column already exists (nested dict format)
            # Frontend may send either 'score' (singular) or 'scores' (plural)
            score_column_name = None
            if 'scores' in operational_df.columns:
                score_column_name = 'scores'
            elif 'score' in operational_df.columns:
                score_column_name = 'score'

            if score_column_name:
                # Check if it's actually a dict (not a string or number)
                sample_score = operational_df[score_column_name].iloc[0] if len(operational_df) > 0 else None
                if not isinstance(sample_score, dict):
                    logger.info(f"'{score_column_name}' column exists but is not a dict - will attempt to detect score columns")
                else:
                    logger.info(f"'{score_column_name}' column already in nested dict format - no conversion needed")
                    score_columns_to_use = None
                    # Normalize to 'score' for consistency
                    if score_column_name == 'scores':
                        operational_df.rename(columns={'scores': 'score'}, inplace=True)
            else:
                # Try to detect score columns based on naming patterns
                # Look for columns like: score_X, X_score, helpfulness, accuracy, etc.
                potential_score_cols = []
                score_related_keywords = ['score', 'rating', 'quality', 'helpfulness', 'accuracy', 'correctness', 'fluency', 'coherence', 'relevance']
                
                for col in operational_df.columns:
                    # Skip non-numeric columns
                    if not pd.api.types.is_numeric_dtype(operational_df[col]):
                        continue
                    
                    # Skip ID and size columns
                    if col in ['question_id', 'id', 'size', 'cluster_id'] or col.endswith('_id'):
                        continue
                    
                    # Check if column name contains score-related keywords
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in score_related_keywords):
                        potential_score_cols.append(col)
                
                if potential_score_cols:
                    logger.info(f"Auto-detected potential score columns: {potential_score_cols}")
                    score_columns_to_use = potential_score_cols
                else:
                    logger.info("No score columns detected")

            # If we normalized 'scores' to 'score', update req.operationalRows
            if score_column_name == 'scores':
                logger.info("🔄 Normalizing 'scores' column to 'score' for backend compatibility")
                req.operationalRows = operational_df.to_dict('records')
                # Log sample after normalization
                if req.operationalRows:
                    sample = req.operationalRows[0]
                    logger.info(f"  ✓ Sample after normalization:")
                    logger.info(f"    - Has 'score' key: {'score' in sample}")
                    logger.info(f"    - Score value: {sample.get('score')}")
                    logger.info(f"    - Score type: {type(sample.get('score'))}")

        # Convert score columns if needed
        if score_columns_to_use:
            logger.info(f"Converting score columns to dict format: {score_columns_to_use}")
            import pandas as pd
            from stringsight.core.preprocessing import convert_score_columns_to_dict
            
            # Convert to DataFrame for processing
            operational_df = pd.DataFrame(req.operationalRows)
            
            # Convert score columns to dict format
            operational_df = convert_score_columns_to_dict(
                operational_df,
                score_columns=score_columns_to_use,
                method=req.method
            )
            
            # Convert back to dict list
            req.operationalRows = operational_df.to_dict('records')
            
            logger.info(f"✓ Score columns converted successfully")
            if req.operationalRows:
                sample = req.operationalRows[0]
                logger.info(f"  - Sample operationalRow after conversion:")
                logger.info(f"    - Has 'score' key: {'score' in sample}")
                logger.info(f"    - Score value: {sample.get('score')}")
        
        # Convert properties data to Property objects
        properties: List[Property] = []
        for p in req.properties:
            try:
                # Strip property index suffix from question_id to get base conversation ID
                # Frontend sends compound IDs like "48-0", "48-1" but we need base ID "48" for matching
                raw_question_id = str(p.get("question_id", ""))
                base_question_id = raw_question_id.split('-')[0] if '-' in raw_question_id else raw_question_id

                prop = Property(
                    id=str(p.get("id", "")),
                    question_id=base_question_id,
                    model=str(p.get("model", "")),
                    property_description=p.get("property_description"),
                    category=p.get("category"),
                    reason=p.get("reason"),
                    evidence=p.get("evidence"),
                    behavior_type=p.get("behavior_type"),
                    raw_response=p.get("raw_response"),
                    contains_errors=p.get("contains_errors"),
                    unexpected_behavior=p.get("unexpected_behavior"),
                    meta=p.get("meta", {})
                )
                properties.append(prop)
            except Exception as e:
                logger.warning(f"Skipping invalid property: {e}")
                continue
        
        if not properties:
            return {"clusters": []}
        
        # Create minimal conversations that match the properties for to_dataframe() to work
        # We need conversations with matching (question_id, model) pairs for the merge to work
        conversations: List[ConversationRecord] = []
        all_models = set()
        
        # Create a set of unique (question_id, model) pairs from properties
        property_keys = {(prop.question_id, prop.model) for prop in properties}
        
        logger.info(f"Found {len(property_keys)} unique (question_id, model) pairs from {len(properties)} properties")
        logger.info(f"Sample property keys: {list(property_keys)[:3]}")
        
        # Debug: Check operationalRows structure
        if req.operationalRows:
            logger.info(f"OperationalRows count: {len(req.operationalRows)}")
            sample_op = req.operationalRows[0]
            logger.info(f"Sample operationalRow keys: {list(sample_op.keys())}")
            logger.info(f"Sample operationalRow: question_id={sample_op.get('question_id')}, model={sample_op.get('model')}, score={sample_op.get('score')}")
        
        # Create exactly one conversation per unique (question_id, model) pair
        matches_found = 0
        for question_id, model in property_keys:
            all_models.add(model)
            
            # Find matching operational row for this conversation
            matching_row = None
            for row in req.operationalRows:
                row_qid = str(row.get("question_id", ""))
                row_model = str(row.get("model", ""))
                
                # Try exact match first
                if row_qid == question_id and row_model == model:
                    matching_row = row
                    matches_found += 1
                    break
                
                # If no exact match, try matching on base question_id (strip suffix after '-')
                # This handles formats like "48-0" vs "48" or "0-0" vs "0"
                # Try stripping from both sides
                row_qid_base = row_qid.split('-')[0] if '-' in row_qid else row_qid
                question_id_base = question_id.split('-')[0] if '-' in question_id else question_id

                if (row_qid_base == question_id or row_qid == question_id_base) and row_model == model:
                    matching_row = row
                    matches_found += 1
                    break
            
            if not matching_row and matches_found == 0:
                # Log first failed match for debugging
                logger.warning(f"⚠️ No matching operationalRow for question_id={question_id}, model={model}")
                logger.warning(f"  Looking for: question_id='{question_id}' (type: {type(question_id)}), model='{model}' (type: {type(model)})")
                if req.operationalRows:
                    logger.warning(f"  Sample from operationalRows: question_id='{req.operationalRows[0].get('question_id')}' (type: {type(req.operationalRows[0].get('question_id'))}), model='{req.operationalRows[0].get('model')}' (type: {type(req.operationalRows[0].get('model'))})")
            
            # Create minimal conversation (use empty data if no matching row found)
            # Try both 'score' and 'scores' fields for compatibility
            if matching_row:
                scores = matching_row.get("score") or matching_row.get("scores") or {}
                # Debug logging for first match
                if matches_found == 1:
                    logger.info(f"🔍 First matching_row debug:")
                    logger.info(f"  - Keys in matching_row: {list(matching_row.keys())}")
                    logger.info(f"  - 'score' value: {matching_row.get('score')}")
                    logger.info(f"  - 'scores' value: {matching_row.get('scores')}")
                    logger.info(f"  - Final scores used: {scores}")
            else:
                scores = {}

            # Try both 'model_response' and 'responses' for compatibility
            response_value = ""
            if matching_row:
                response_value = matching_row.get("responses") or matching_row.get("model_response") or ""

            # Strip property index suffix from question_id to get base conversation ID
            # Properties have compound IDs like "48-0", "48-1" (conversation-property_index)
            # Conversations should only have the base ID like "48"
            base_question_id = question_id.split('-')[0] if '-' in question_id else question_id

            conv = ConversationRecord(
                question_id=base_question_id,
                model=model,
                prompt=matching_row.get("prompt", "") if matching_row else "",
                responses=response_value,
                scores=scores,
                meta={}
            )
            conversations.append(conv)

        # NEW: Handle side-by-side specific logic if detected
        # If method is side_by_side, we need to reconstruct the conversation records to have
        # model=[model_a, model_b] and scores=[score_a, score_b] for SideBySideMetrics to work
        
        # Auto-detect side_by_side if not explicitly set but data looks like it
        if req.method == "single_model" and req.operationalRows:
            first_row = req.operationalRows[0]
            if "model_a" in first_row and "model_b" in first_row:
                logger.info("🔄 Auto-detected side_by_side method from operationalRows columns")
                req.method = "side_by_side"

        if req.method == "side_by_side":
            logger.info("🔄 Reconstructing conversations for side-by-side metrics...")
            
            # Group properties by base question_id to identify pairs
            properties_by_qid = {}
            for p in properties:
                if p.question_id not in properties_by_qid:
                    properties_by_qid[p.question_id] = []
                properties_by_qid[p.question_id].append(p)
            
            sxs_conversations = []
            
            # Pre-index operational rows for faster lookup
            import time
            t0 = time.time()
            operational_rows_map = {}
            for row in req.operationalRows:
                row_qid = str(row.get("question_id", ""))
                operational_rows_map[row_qid] = row
                # Also index by base ID if it's a compound ID (e.g. "48-0" -> "48")
                if '-' in row_qid:
                    base_id = row_qid.split('-')[0]
                    if base_id not in operational_rows_map:
                         operational_rows_map[base_id] = row
            
            logger.info(f"⏱️ Indexed {len(req.operationalRows)} operational rows in {time.time() - t0:.4f}s")
            t1 = time.time()

            sxs_conversations = []
            
            for qid, props in properties_by_qid.items():
                # Find matching operational row using lookup map
                matching_row = operational_rows_map.get(qid)
                
                # If not found by exact match, try base ID match (if qid has suffix)
                if not matching_row and '-' in qid:
                    matching_row = operational_rows_map.get(qid.split('-')[0])
                
                if matching_row:
                    # Extract models
                    model_a = matching_row.get("model_a")
                    model_b = matching_row.get("model_b")
                    
                    # If models not in row, try to infer from properties
                    if not model_a or not model_b:
                        unique_models = list(set(p.model for p in props))
                        if len(unique_models) >= 2:
                            model_a = unique_models[0]
                            model_b = unique_models[1]
                        else:
                            # Fallback
                            model_a = "model_a"
                            model_b = "model_b"
                    
                    # Extract scores
                    # Check for score_a/score_b columns first
                    score_a = matching_row.get("score_a", {})
                    score_b = matching_row.get("score_b", {})

                    # If empty, check if 'scores' or 'score' contains combined info
                    if not score_a and not score_b:
                        combined_score = matching_row.get("score") or matching_row.get("scores")
                        if combined_score:
                            # Handle list format [score_a, score_b]
                            if isinstance(combined_score, list) and len(combined_score) == 2:
                                score_a = combined_score[0] if isinstance(combined_score[0], dict) else {}
                                score_b = combined_score[1] if isinstance(combined_score[1], dict) else {}
                            elif isinstance(combined_score, dict):
                                # If it's a dict, duplicate it for both
                                score_a = combined_score
                                score_b = combined_score
                            else:
                                score_a = {}
                                score_b = {}
                    
                    # Extract winner to meta
                    meta = {}
                    if "winner" in matching_row:
                        meta["winner"] = matching_row["winner"]
                    elif "score" in matching_row and isinstance(matching_row["score"], dict) and "winner" in matching_row["score"]:
                        meta["winner"] = matching_row["score"]["winner"]
                    
                    # Create SxS conversation record
                    conv = ConversationRecord(
                        question_id=qid,
                        model=[model_a, model_b],
                        prompt=matching_row.get("prompt", ""),
                        responses=[matching_row.get("model_a_response", ""), matching_row.get("model_b_response", "")],
                        scores=[score_a, score_b],
                        meta=meta
                    )
                    sxs_conversations.append(conv)
            
            if sxs_conversations:
                logger.info(f"✅ Created {len(sxs_conversations)} side-by-side conversation records in {time.time() - t1:.4f}s")
                conversations = sxs_conversations

        logger.info(f"✅ Matched {matches_found}/{len(property_keys)} conversations with operationalRows")

        # Enhanced logging for debugging quality metrics
        if matches_found > 0 and conversations:
            # Log sample conversation scores
            sample_conv = conversations[0]
            logger.info(f"📊 Score field verification:")
            logger.info(f"  - Sample conversation has scores: {bool(sample_conv.scores)}")
            logger.info(f"  - Scores type: {type(sample_conv.scores)}")
            logger.info(f"  - Scores content: {sample_conv.scores}")
            if isinstance(sample_conv.scores, dict):
                logger.info(f"  - Score keys: {list(sample_conv.scores.keys())}")
        else:
            logger.warning("⚠️ No conversations matched with operationalRows - quality metrics will be empty!")
        
        # Create PropertyDataset with matching conversations and properties
        dataset = PropertyDataset(
            conversations=conversations,
            all_models=list(all_models),
            properties=properties,
            clusters=[],  # Will be populated by clustering
            model_stats={}
        )
        
        logger.info(f"PropertyDataset created with:")
        logger.info(f"  - {len(dataset.properties)} properties")
        logger.info(f"  - {len(dataset.conversations)} conversations") 
        logger.info(f"  - Models: {dataset.all_models}")
        
        # Debug: Check scores in conversations
        if dataset.conversations:
            sample_conv = dataset.conversations[0]
            logger.info(f"🔍 Sample conversation:")
            logger.info(f"  - question_id: {sample_conv.question_id}")
            logger.info(f"  - model: {sample_conv.model}")
            logger.info(f"  - scores type: {type(sample_conv.scores)}")
            logger.info(f"  - scores value: {sample_conv.scores}")
            logger.info(f"  - scores keys: {sample_conv.scores.keys() if isinstance(sample_conv.scores, dict) else 'N/A'}")
        
        if dataset.properties:
            logger.debug(f"Sample properties:")
            for i, prop in enumerate(dataset.properties[:3]):
                logger.debug(f"  Property {i}: id={prop.id}, question_id={prop.question_id}, model={prop.model}")
                logger.debug(f"    description: {prop.property_description}")
        
        # Run clustering only (no extraction)
        # Convert groupBy parameter to groupby_column (none -> None for no grouping)
        groupby_column = None if req.params.groupBy == "none" else req.params.groupBy
        
        logger.debug(f"Clustering parameters:")
        logger.debug(f"  - groupBy from request: {req.params.groupBy}")
        logger.debug(f"  - groupby_column for clusterer: {groupby_column}")
        logger.debug(f"  - min_cluster_size: {req.params.minClusterSize}")
        logger.debug(f"  - embedding_model: {req.params.embeddingModel}")
        
        clusterer = get_clusterer(
            method="hdbscan",
            min_cluster_size=req.params.minClusterSize,
            embedding_model=req.params.embeddingModel,
            assign_outliers=False,
            include_embeddings=False,
            cache_embeddings=True,
            groupby_column=groupby_column,
        )
        
        # Run clustering
        clustered_dataset = await clusterer.run(dataset, column_name="property_description")
        
    finally:
        # Restore original cache/env settings (no-op for DiskCache)
        os.environ["STRINGSIGHT_DISABLE_CACHE"] = original_cache_setting
        # Restore global caches
        try:
            _llm_utils._default_cache = _orig_default_cache
            _llm_utils._default_llm_utils = _orig_default_llm_utils
        except Exception:
            pass
        try:
            if hasattr(_cu, "_cache"):
                _cu._cache = _orig_embed_cache
        except Exception:
            pass

    # Convert clusters to API format
    clusters: List[Dict[str, Any]] = []
    for cluster in clustered_dataset.clusters:
        clusters.append({
            "id": cluster.id,
            "label": cluster.label,
            "size": cluster.size,
            "property_descriptions": cluster.property_descriptions,
            "property_ids": cluster.property_ids,
            "question_ids": cluster.question_ids,
            "meta": cluster.meta,
        })
    
    # Compute metrics using FunctionalMetrics or SideBySideMetrics
    from stringsight.metrics.functional_metrics import FunctionalMetrics
    from stringsight.metrics.side_by_side import SideBySideMetrics
    
    # Choose metrics computer based on method
    if req.method == "side_by_side":
        logger.info("🚀 Using SideBySideMetrics for computation")
        metrics_computer = SideBySideMetrics(
            output_dir=None,
            compute_bootstrap=True,  # Disable bootstrap for API speed
            log_to_wandb=False,
            generate_plots=False
        )
    else:
        logger.info("🚀 Using FunctionalMetrics for computation")
        metrics_computer = FunctionalMetrics(
            output_dir=None,
            compute_bootstrap=True,  # Disable bootstrap for API speed
            log_to_wandb=False,
            generate_plots=False
        )
    
    # Debug: Check what's in clustered_dataset before metrics
    logger.info(f"🔍 Before FunctionalMetrics:")
    logger.info(f"  - Clusters: {len(clustered_dataset.clusters)}")
    logger.info(f"  - Conversations: {len(clustered_dataset.conversations)}")
    logger.info(f"  - Properties: {len(clustered_dataset.properties)}")
    if clustered_dataset.conversations:
        sample = clustered_dataset.conversations[0]
        logger.info(f"  - Sample conv scores: {sample.scores}")
    
    # Run metrics computation on the clustered dataset
    clustered_dataset = metrics_computer.run(clustered_dataset)
    
    # Extract the computed metrics
    model_cluster_scores_dict = clustered_dataset.model_stats.get("model_cluster_scores", {})
    cluster_scores_dict = clustered_dataset.model_stats.get("cluster_scores", {})
    model_scores_dict = clustered_dataset.model_stats.get("model_scores", {})

    # Debug: Log what was extracted from model_stats
    logger.info(f"📈 After FunctionalMetrics computation:")
    logger.info(f"  - model_cluster_scores type: {type(model_cluster_scores_dict)}")
    logger.info(f"  - cluster_scores type: {type(cluster_scores_dict)}")
    logger.info(f"  - model_scores type: {type(model_scores_dict)}")

    if hasattr(model_cluster_scores_dict, 'shape'):
        logger.info(f"  - model_cluster_scores shape: {model_cluster_scores_dict.shape}")
        logger.info(f"  - model_cluster_scores columns: {list(model_cluster_scores_dict.columns)}")
    if hasattr(cluster_scores_dict, 'shape'):
        logger.info(f"  - cluster_scores shape: {cluster_scores_dict.shape}")
        logger.info(f"  - cluster_scores columns: {list(cluster_scores_dict.columns)}")
    if hasattr(model_scores_dict, 'shape'):
        logger.info(f"  - model_scores shape: {model_scores_dict.shape}")
        logger.info(f"  - model_scores columns: {list(model_scores_dict.columns)}")
        # Check if quality columns exist
        quality_cols = [col for col in model_scores_dict.columns if col.startswith('quality_')]
        logger.info(f"  - model_scores quality columns: {quality_cols}")

    # Convert to the format expected by the rest of the code
    # FunctionalMetrics returns DataFrames, convert back to nested dicts
    if hasattr(model_cluster_scores_dict, 'to_dict'):
        # It's a DataFrame, need to restructure it
        import pandas as pd
        df = model_cluster_scores_dict
        scores = {"model_cluster_scores": {}, "cluster_scores": {}, "model_scores": {}}
        
        # Convert DataFrame back to nested dict structure
        for _, row in df.iterrows():
            model = row['model']
            cluster = row['cluster']
            if model not in scores["model_cluster_scores"]:
                scores["model_cluster_scores"][model] = {}
            
            # Extract all metrics from the row
            metrics = {
                "size": row.get('size'),
                "proportion": row.get('proportion'),
                "proportion_delta": row.get('proportion_delta'),
                "quality": {},
                "quality_delta": {},
                "metadata": row.get('metadata', {})
            }
            
            # Extract quality metrics
            for col in df.columns:
                if col.startswith('quality_') and not col.startswith('quality_delta_'):
                    metric_name = col.replace('quality_', '')
                    if not any(x in metric_name for x in ['_ci_', '_significant']):
                        metrics["quality"][metric_name] = row[col]
                elif col.startswith('quality_delta_'):
                    metric_name = col.replace('quality_delta_', '')
                    if not any(x in metric_name for x in ['_ci_', '_significant']):
                        metrics["quality_delta"][metric_name] = row[col]
            
            scores["model_cluster_scores"][model][cluster] = metrics
        
        # Process cluster_scores
        if hasattr(cluster_scores_dict, 'to_dict'):
            df = cluster_scores_dict
            for _, row in df.iterrows():
                cluster = row['cluster']
                metrics = {
                    "size": row.get('size'),
                    "proportion": row.get('proportion'),
                    "quality": {},
                    "quality_delta": {}
                }
                for col in df.columns:
                    if col.startswith('quality_') and not col.startswith('quality_delta_'):
                        metric_name = col.replace('quality_', '')
                        if not any(x in metric_name for x in ['_ci_', '_significant']):
                            metrics["quality"][metric_name] = row[col]
                    elif col.startswith('quality_delta_'):
                        metric_name = col.replace('quality_delta_', '')
                        if not any(x in metric_name for x in ['_ci_', '_significant']):
                            metrics["quality_delta"][metric_name] = row[col]
                scores["cluster_scores"][cluster] = metrics
        
        # Process model_scores
        if hasattr(model_scores_dict, 'to_dict'):
            df = model_scores_dict
            scores["model_scores"] = {}
            for _, row in df.iterrows():
                model = row['model']
                metrics = {
                    "size": row.get('size'),
                    "proportion": row.get('proportion'),
                    "quality": {},
                    "quality_delta": {}
                }
                for col in df.columns:
                    if col.startswith('quality_') and not col.startswith('quality_delta_'):
                        metric_name = col.replace('quality_', '')
                        if not any(x in metric_name for x in ['_ci_', '_significant']):
                            metrics["quality"][metric_name] = row[col]
                    elif col.startswith('quality_delta_'):
                        metric_name = col.replace('quality_delta_', '')
                        if not any(x in metric_name for x in ['_ci_', '_significant']):
                            metrics["quality_delta"][metric_name] = row[col]
                scores["model_scores"][model] = metrics
        else:
            # Already in dict format, just assign it
            scores["model_scores"] = model_scores_dict
    else:
        # Already in dict format
        scores = {
            "model_cluster_scores": model_cluster_scores_dict,
            "cluster_scores": cluster_scores_dict,
            "model_scores": model_scores_dict
        }
    
    # Get total conversations
    total_conversations = compute_total_conversations_by_model(req.properties)
    
    # Enrich clusters with the metrics
    enriched = enrich_clusters_with_metrics(clusters, scores)

    # Attach overall proportion and per-property model info for UI consumption
    try:
        cluster_scores = scores.get("cluster_scores", {})
        # Build a map of property_id -> { model, property_description }
        prop_by_id: Dict[str, Dict[str, Any]] = {}
        for p in req.properties:
            pid = str(p.get("id"))
            if not pid:
                continue
            prop_by_id[pid] = {
                "property_id": pid,
                "model": str(p.get("model", "")),
                "property_description": p.get("property_description"),
            }
        for c in enriched:
            label = c.get("label")
            cs = cluster_scores.get(label, {}) if isinstance(cluster_scores, dict) else {}
            # Overall proportion across all models (size / total unique convs in subset)
            c_meta = dict(c.get("meta", {}))
            if isinstance(cs.get("proportion"), (int, float)):
                c_meta["proportion_overall"] = float(cs["proportion"])  
            # Attach property_items with model next to each description
            items: List[Dict[str, Any]] = []
            property_ids_list = c.get("property_ids", []) or []
            
            # Debug: Check for duplicates in property_ids
            if len(property_ids_list) != len(set(str(pid) for pid in property_ids_list)):
                logger.debug(f"Cluster {label} has duplicate property_ids!")
                logger.debug(f"  - property_ids: {property_ids_list}")
                logger.debug(f"  - unique count: {len(set(str(pid) for pid in property_ids_list))}")
                logger.debug(f"  - total count: {len(property_ids_list)}")
            
            # Deduplicate property_ids while preserving order
            seen_pids = set()
            for pid in property_ids_list:
                pid_str = str(pid)
                if pid_str not in seen_pids:
                    seen_pids.add(pid_str)
                    rec = prop_by_id.get(pid_str)
                    if rec:
                        items.append(rec)
            if items:
                c_meta["property_items"] = items
            c["meta"] = c_meta
    except Exception:
        # Best-effort enrichment; do not fail clustering if this post-process fails
        pass
    
    # Sort by size desc
    enriched = sorted(enriched, key=lambda c: c.get("size", 0), reverse=True)
    
    # Calculate total unique conversations in the dataset for the frontend
    total_unique_conversations = len(set(str(p.get("question_id", "")) for p in req.properties if p.get("question_id")))
    
    # Save full pipeline results to disk with timestamped directory
    results_dir: Optional[Path] = None
    results_dir_name: Optional[str] = None
    try:
        import json

        # Always create timestamp for summary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get base results directory (may be on persistent disk)
        base_results_dir = _get_results_dir()

        # Create results directory - use provided output_dir if available
        if req.output_dir:
            # Use the output_dir from the request
            results_dir = base_results_dir / req.output_dir
            results_dir_name = req.output_dir
        else:
            # Generate a new directory name with timestamp
            # Use filename from operationalRows if available, otherwise use "clustering"
            base_filename = "clustering"
            if req.operationalRows and len(req.operationalRows) > 0:
                # Extract original filename from __source_filename field
                first_row = req.operationalRows[0]
                if "__source_filename" in first_row:
                    base_filename = str(first_row["__source_filename"])
                    # Remove any path components and extension if present
                    base_filename = Path(base_filename).stem

            results_dir = base_results_dir / f"{base_filename}_{timestamp}"
            results_dir_name = f"{base_filename}_{timestamp}"

        results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving clustering results to {results_dir}")
        
        # Save full PropertyDataset as JSON
        full_dataset_path = results_dir / "full_dataset.json"
        clustered_dataset.save(str(full_dataset_path))
        logger.info(f"✓ Saved full dataset: {full_dataset_path}")
        
        # Save conversations in proper format (conversation.jsonl)
        # Convert PropertyDataset to DataFrame and then format conversations
        try:
            import pandas as pd
            # Get the method from the dataset (check if it's side_by_side or single_model)
            method = "side_by_side" if any(isinstance(conv.model, list) for conv in clustered_dataset.conversations) else "single_model"
            conv_df = clustered_dataset.to_dataframe(type="base", method=method)
            # Format conversations using the formatter
            formatted_conversations = format_conversations(conv_df, method)
            # Save as JSONL
            conversation_path = results_dir / "conversation.jsonl"
            with open(conversation_path, 'w') as f:
                for conv in formatted_conversations:
                    f.write(json.dumps(conv, default=str) + '\n')
            logger.info(f"✓ Saved conversations: {conversation_path}")
        except Exception as e:
            logger.warning(f"Failed to save conversation.jsonl: {e}")
        
        # Save clusters as JSON
        clusters_path = results_dir / "clusters.json"
        with open(clusters_path, 'w') as f:
            json.dump(enriched, f, indent=2, default=str)
        logger.info(f"✓ Saved clusters: {clusters_path}")
        
        # Save properties as JSONL
        properties_path = results_dir / "parsed_properties.jsonl"
        with open(properties_path, 'w') as f:
            for p in req.properties:
                f.write(json.dumps(p, default=str) + '\n')
        logger.info(f"✓ Saved properties: {properties_path}")
        
        # Save metrics scores
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(scores, f, indent=2, default=str)
        logger.info(f"✓ Saved metrics: {metrics_path}")
        
        # Save summary
        summary_path = results_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("StringSight Clustering Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total conversations: {total_unique_conversations}\n")
            f.write(f"Total properties: {len(req.properties)}\n")
            f.write(f"Total clusters: {len(enriched)}\n")
            f.write(f"Models: {', '.join(clustered_dataset.all_models)}\n\n")
            f.write(f"Clustering parameters:\n")
            f.write(f"  - Min cluster size: {req.params.minClusterSize}\n")
            f.write(f"  - Embedding model: {req.params.embeddingModel}\n")
            f.write(f"  - Group by: {req.params.groupBy}\n\n")
            f.write(f"Output files:\n")
            f.write(f"  - full_dataset.json: Complete dataset with all data\n")
            f.write(f"  - clusters.json: Cluster definitions with metrics\n")
            f.write(f"  - parsed_properties.jsonl: Property objects\n")
            f.write(f"  - metrics.json: Computed metrics\n")
        logger.info(f"✓ Saved summary: {summary_path}")
        
        logger.info(f"✅ All results saved to: {results_dir}")
        
    except Exception as e:
        # Don't fail the request if saving fails
        logger.warning(f"Failed to save results to disk: {e}")
    
    # Transform metrics to frontend format (JSONL-style arrays)
    # Convert nested dict format to flat array format expected by frontend
    
    # Build cluster_id lookup map from enriched clusters
    cluster_id_map = {c.get("label"): c.get("id") for c in enriched}
    
    model_cluster_scores_array = []
    model_cluster_scores_dict = scores.get("model_cluster_scores", {})
    
    # Debug: log what we're transforming
    if model_cluster_scores_dict:
        sample_model = list(model_cluster_scores_dict.keys())[0]
        sample_cluster = list(model_cluster_scores_dict[sample_model].keys())[0]
        sample_metrics = model_cluster_scores_dict[sample_model][sample_cluster]
        logger.info(f"  - Sample cluster: {sample_cluster}")
        logger.info(f"  - Sample metrics keys: {list(sample_metrics.keys())}")
        logger.info(f"  - Sample quality: {sample_metrics.get('quality')}")
        logger.info(f"  - Sample quality_delta: {sample_metrics.get('quality_delta')}")
    
    for model_name, clusters_dict in model_cluster_scores_dict.items():
        for cluster_name, metrics in clusters_dict.items():
            row = {
                "model": model_name,
                "cluster": cluster_name,
                "cluster_id": cluster_id_map.get(cluster_name),  # Add cluster_id for frontend matching
                "size": metrics.get("size"),  # Add size (number of properties in this model-cluster combo)
                "proportion": metrics.get("proportion", 0.0),
                "proportion_delta": metrics.get("proportion_delta"),
            }
            
            # Flatten quality metrics: {"helpfulness": 0.8} -> quality_helpfulness: 0.8
            quality = metrics.get("quality")
            if quality and isinstance(quality, dict):
                for metric_name, metric_value in quality.items():
                    row[f"quality_{metric_name}"] = metric_value
            else:
                logger.debug(f"No quality dict for {model_name}/{cluster_name}: {quality}")
            
            # Flatten quality_delta metrics
            quality_delta = metrics.get("quality_delta")
            if quality_delta and isinstance(quality_delta, dict):
                for metric_name, metric_value in quality_delta.items():
                    row[f"quality_delta_{metric_name}"] = metric_value
            else:
                logger.debug(f"No quality_delta dict for {model_name}/{cluster_name}: {quality_delta}")
            
            # Add metadata (contains behavior_type, group, etc.)
            row["metadata"] = metrics.get("metadata", {})
            
            model_cluster_scores_array.append(row)
    
    # Log sample of transformed array
    if model_cluster_scores_array:
        logger.info(f"✅ Transformed {len(model_cluster_scores_array)} model_cluster_scores rows")
        logger.info(f"  - Sample row keys: {list(model_cluster_scores_array[0].keys())}")
        logger.info(f"  - Sample row: {model_cluster_scores_array[0]}")
    
    # Transform cluster_scores to array format
    cluster_scores_array = []
    cluster_scores_dict = scores.get("cluster_scores", {})
    for cluster_name, metrics in cluster_scores_dict.items():
        row = {
            "cluster": cluster_name,
            "cluster_id": cluster_id_map.get(cluster_name),  # Add cluster_id
            "size": metrics.get("size", 0),
            "proportion": metrics.get("proportion", 0.0),
        }
        
        # Flatten quality metrics
        quality = metrics.get("quality")
        if quality and isinstance(quality, dict):
            for metric_name, metric_value in quality.items():
                row[f"quality_{metric_name}"] = metric_value
        
        # Flatten quality_delta metrics
        quality_delta = metrics.get("quality_delta")
        if quality_delta and isinstance(quality_delta, dict):
            for metric_name, metric_value in quality_delta.items():
                row[f"quality_delta_{metric_name}"] = metric_value
        
        # Add metadata
        row["metadata"] = metrics.get("metadata", {})
        
        cluster_scores_array.append(row)

    # Transform model_scores to array format
    model_scores_array = []
    model_scores_dict = scores.get("model_scores", {})

    # Check if model_scores_dict is a DataFrame (from FunctionalMetrics)
    if hasattr(model_scores_dict, 'to_dict'):
        # It's a DataFrame, convert to dict first
        import pandas as pd
        df = model_scores_dict

        logger.info(f"🔧 Transforming model_scores DataFrame to array format:")
        logger.info(f"  - DataFrame shape: {df.shape}")
        logger.info(f"  - DataFrame columns: {list(df.columns)}")

        for _, row in df.iterrows():
            model_name = row['model']

            model_row = {
                "model": model_name,
                "size": row.get('size', 0),
                # Don't include 'cluster' field for model_scores (it's aggregated across all clusters)
            }

            # Flatten quality metrics: quality_helpfulness -> quality_helpfulness
            for col in df.columns:
                if col.startswith('quality_') and not col.startswith('quality_delta_'):
                    if not any(x in col for x in ['_ci_', '_significant']):
                        model_row[col] = row[col]
                elif col.startswith('quality_delta_'):
                    if not any(x in col for x in ['_ci_', '_significant']):
                        model_row[col] = row[col]

            # Add confidence intervals if they exist
            for col in df.columns:
                if '_ci_lower' in col or '_ci_upper' in col or '_significant' in col:
                    model_row[col] = row[col]

            model_scores_array.append(model_row)
    else:
        # It's already a dict, transform it similar to cluster_scores
        for model_name, metrics in model_scores_dict.items():
            if not isinstance(metrics, dict):
                continue

            row = {
                "model": model_name,
                "size": metrics.get("size", 0),
            }

            # Flatten quality metrics
            quality = metrics.get("quality")
            if quality and isinstance(quality, dict):
                for metric_name, metric_value in quality.items():
                    row[f"quality_{metric_name}"] = metric_value

            # Flatten quality_delta metrics
            quality_delta = metrics.get("quality_delta")
            if quality_delta and isinstance(quality_delta, dict):
                for metric_name, metric_value in quality_delta.items():
                    row[f"quality_delta_{metric_name}"] = metric_value

            # Add confidence intervals if they exist
            quality_ci = metrics.get("quality_ci", {})
            for metric_name, ci_dict in quality_ci.items():
                if isinstance(ci_dict, dict):
                    row[f"quality_{metric_name}_ci_lower"] = ci_dict.get("lower")
                    row[f"quality_{metric_name}_ci_upper"] = ci_dict.get("upper")

            quality_delta_ci = metrics.get("quality_delta_ci", {})
            for metric_name, ci_dict in quality_delta_ci.items():
                if isinstance(ci_dict, dict):
                    row[f"quality_delta_{metric_name}_ci_lower"] = ci_dict.get("lower")
                    row[f"quality_delta_{metric_name}_ci_upper"] = ci_dict.get("upper")

            # Add significance flags if they exist
            quality_delta_significant = metrics.get("quality_delta_significant", {})
            for metric_name, is_significant in quality_delta_significant.items():
                row[f"quality_delta_{metric_name}_significant"] = is_significant

            model_scores_array.append(row)

    # Log the transformed model_scores
    if model_scores_array:
        logger.info(f"✅ Transformed {len(model_scores_array)} model_scores rows")
        logger.info(f"  - Sample row keys: {list(model_scores_array[0].keys())}")
        logger.info(f"  - Sample row: {model_scores_array[0]}")
    else:
        logger.warning("⚠️ No model_scores computed - this may indicate missing quality metrics in the data")

    # Persist flattened metrics in expected JSONL format for downstream endpoints/loaders
    try:
        if results_dir is not None:
            import pandas as pd
            mc_df = pd.DataFrame(model_cluster_scores_array)
            (results_dir / "model_cluster_scores_df.jsonl").write_text(
                mc_df.to_json(orient='records', lines=True)
            )
            cs_df = pd.DataFrame(cluster_scores_array)
            (results_dir / "cluster_scores_df.jsonl").write_text(
                cs_df.to_json(orient='records', lines=True)
            )
            # Save model_scores as well
            if model_scores_array:
                ms_df = pd.DataFrame(model_scores_array)
                (results_dir / "model_scores_df.jsonl").write_text(
                    ms_df.to_json(orient='records', lines=True)
                )
            logger.info(f"✓ Saved metrics JSONL files under: {results_dir}")
    except Exception as e:
        logger.warning(f"Failed to save metrics JSONL files: {e}")

    return {
        "clusters": enriched,
        "total_conversations_by_model": total_conversations,
        "total_unique_conversations": total_unique_conversations,
        "results_dir": results_dir_name,
        "metrics": {
            "model_cluster_scores": model_cluster_scores_array,
            "cluster_scores": cluster_scores_array,
            "model_scores": model_scores_array  # Now properly computed and transformed
        }
    }


class ClusterMetricsRequest(BaseModel):
    clusters: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    operationalRows: List[Dict[str, Any]]
    included_property_ids: Optional[List[str]] = None
    score_columns: Optional[List[str]] = None  # NEW: List of score column names to convert to dict format
    method: Optional[str] = "single_model"  # NEW: Method for score column conversion


@app.post("/cluster/metrics")
def cluster_metrics(req: ClusterMetricsRequest) -> Dict[str, Any]:
    """Recompute cluster metrics for a filtered subset without reclustering."""
    # NEW: Preprocess operationalRows to handle score_columns conversion
    score_columns_to_use = req.score_columns
    
    # Auto-detect score columns if not provided (same logic as /cluster/run)
    if not score_columns_to_use and req.operationalRows:
        import pandas as pd
        operational_df = pd.DataFrame(req.operationalRows)

        # Frontend may send either 'score' (singular) or 'scores' (plural)
        score_column_name = None
        if 'scores' in operational_df.columns:
            score_column_name = 'scores'
        elif 'score' in operational_df.columns:
            score_column_name = 'score'

        if score_column_name:
            # Check if it's actually a dict (not a string or number)
            sample_score = operational_df[score_column_name].iloc[0] if len(operational_df) > 0 else None
            if not isinstance(sample_score, dict):
                logger.info(f"'{score_column_name}' column exists but is not a dict - will attempt to detect score columns")
            else:
                logger.info(f"'{score_column_name}' column already in nested dict format - no conversion needed")
                score_columns_to_use = None
                # Normalize to 'score' for consistency
                if score_column_name == 'scores':
                    operational_df.rename(columns={'scores': 'score'}, inplace=True)
                    req.operationalRows = operational_df.to_dict('records')
        else:
            # Try to detect score columns based on naming patterns
            potential_score_cols = []
            score_related_keywords = ['score', 'rating', 'quality', 'helpfulness', 'accuracy', 'correctness', 'fluency', 'coherence', 'relevance']
            
            for col in operational_df.columns:
                if not pd.api.types.is_numeric_dtype(operational_df[col]):
                    continue
                if col in ['question_id', 'id', 'size', 'cluster_id'] or col.endswith('_id'):
                    continue
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in score_related_keywords):
                    potential_score_cols.append(col)
            
            if potential_score_cols:
                logger.info(f"Auto-detected potential score columns: {potential_score_cols}")
                score_columns_to_use = potential_score_cols
            else:
                logger.info("No score columns detected")
    
    # Convert score columns if needed
    if score_columns_to_use:
        logger.info(f"Converting score columns to dict format: {score_columns_to_use}")
        import pandas as pd
        from stringsight.core.preprocessing import convert_score_columns_to_dict
        
        # Convert to DataFrame for processing
        operational_df = pd.DataFrame(req.operationalRows)
        
        # Convert score columns to dict format
        operational_df = convert_score_columns_to_dict(
            operational_df,
            score_columns=score_columns_to_use,
            method=req.method
        )
        
        # Convert back to dict list
        req.operationalRows = operational_df.to_dict('records')
        logger.info(f"✓ Score columns converted successfully")
    
    long_df = prepare_long_frame(
        clusters=req.clusters,
        properties=req.properties,
        operational_rows=req.operationalRows,
        included_property_ids=req.included_property_ids,
    )
    total_conversations = compute_total_conversations_by_model(req.properties)
    scores = compute_subset_metrics(long_df, total_conversations)
    enriched = enrich_clusters_with_metrics(req.clusters, scores)
    enriched = sorted(enriched, key=lambda c: c.get("size", 0), reverse=True)
    
    # Calculate total unique conversations in the dataset for the frontend
    total_unique_conversations = len(set(str(p.get("question_id", "")) for p in req.properties if p.get("question_id")))
    
    return {
        "clusters": enriched,
        "total_conversations_by_model": total_conversations,
        "total_unique_conversations": total_unique_conversations
    }


@app.post("/detect-and-validate")
def detect_and_validate(
    file: UploadFile | None = File(default=None),
    payload: RowsPayload | None = Body(default=None),
) -> Dict[str, Any]:
    if not file and not payload:
        raise HTTPException(status_code=400, detail="Provide either a file or a rows payload.")

    if file:
        df = _load_dataframe_from_upload(file)
        method = detect_method(list(df.columns))
    else:
        assert payload is not None
        df = _load_dataframe_from_rows(payload.rows)
        method = payload.method or detect_method(list(df.columns))

    columns = list(df.columns)
    if method is None:
        return {
            "method": None,
            "valid": False,
            "missing": [],
            "row_count": int(len(df)),
            "columns": columns,
            "preview": df.head(50).to_dict(orient="records"),
        }

    missing = validate_required_columns(df, method)
    return {
        "method": method,
        "valid": len(missing) == 0,
        "missing": missing,
        "row_count": int(len(df)),
        "columns": columns,
        "preview": df.head(50).to_dict(orient="records"),
    }


@app.post("/conversations")
def conversations(
    file: UploadFile | None = File(default=None),
    payload: RowsPayload | None = Body(default=None),
) -> Dict[str, Any]:
    df, method = _resolve_df_and_method(file, payload)
    # Normalize score columns for convenience in clients
    try:
        df = explode_score_columns(df, method)
    except Exception:
        pass
    traces = format_conversations(df, method)
    return {"method": method, "conversations": traces}


@app.post("/read-path")
def read_path(req: ReadRequest) -> Dict[str, Any]:
    """Read a dataset from a server path, auto-detect/validate, return preview and method."""
    path = _resolve_within_base(req.path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")
    try:
        df = _load_dataframe_from_path(str(path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    method = req.method or detect_method(list(df.columns))
    if method is None:
        raise HTTPException(status_code=422, detail="Unable to detect dataset method from columns.")

    missing = validate_required_columns(df, method)
    if missing:
        raise HTTPException(status_code=422, detail={"error": f"Missing required columns for {method}", "missing": missing})

    # Optionally flatten scores
    try:
        df = explode_score_columns(df, method)
    except Exception:
        pass

    out_df = df.head(req.limit) if isinstance(req.limit, int) and req.limit > 0 else df
    return {
        "method": method,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "preview": out_df.to_dict(orient="records"),
    }


@app.post("/list-path")
def list_path(req: ListRequest) -> Dict[str, Any]:
    """List files and folders at a server directory path.

    Returns entries with `name`, `path`, `type` ("file"|"dir"), `modified` (ISO timestamp), and `size` (bytes for files).
    If `exts` is provided, filters files by allowed extensions (case-insensitive).
    """
    base = _resolve_within_base(req.path)
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {base}")

    allowed_exts = set(e.lower() for e in (req.exts or []))
    items: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(str(base))):
        if name.startswith('.'):  # hide hidden files/dirs
            continue
        full = base / name
        try:
            # Get modification time
            mtime = os.path.getmtime(str(full))
            modified = datetime.fromtimestamp(mtime).isoformat()
            
            if full.is_dir():
                items.append({"name": name, "path": str(full), "type": "dir", "modified": modified})
            else:
                ext = full.suffix.lower()
                if allowed_exts and ext not in allowed_exts:
                    continue
                size = os.path.getsize(str(full))
                items.append({"name": name, "path": str(full), "type": "file", "size": size, "modified": modified})
        except (OSError, IOError):
            # If we can't get file info, skip it
            continue

    return {"entries": items}


@app.post("/results/load")
def results_load(req: ResultsLoadRequest) -> Dict[str, Any]:
    """Load a results directory and return metrics plus optional dataset with pagination.

    Supports both JSON metrics (model_cluster_scores.json, cluster_scores.json,
    model_scores.json) and JSONL DataFrame exports (model_cluster_scores_df.jsonl,
    cluster_scores_df.jsonl, model_scores_df.jsonl). If a `full_dataset.json`
    file is present, returns its `conversations`, `properties`, and `clusters`.

    Request path can be:
    - Relative path from results directory (e.g., "frontend/conversation_...")
    - Absolute path within BASE_BROWSE_DIR

    Implements pagination to reduce initial load time and memory usage:
    - conversations_page/conversations_per_page for conversations pagination
    - properties_page/properties_per_page for properties pagination
    - load_metrics_only flag to skip loading conversations/properties entirely
    """
    # Try to resolve relative to results directory first (for job.result_path compatibility)
    path_obj = Path(req.path)
    if not path_obj.is_absolute():
        # Try relative to results directory first
        results_base = _get_results_dir()
        candidate = (results_base / req.path).resolve()
        if candidate.exists() and candidate.is_dir():
            results_dir = candidate
        else:
            # Fallback to original behavior (relative to CWD/BASE_BROWSE_DIR)
            results_dir = _resolve_within_base(req.path)
    else:
        results_dir = _resolve_within_base(req.path)

    if not results_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {results_dir}")

    # Load metrics (always cached for fast access)
    model_cluster_scores: Optional[List[Dict[str, Any]]] = None
    cluster_scores: Optional[List[Dict[str, Any]]] = None
    model_scores: Optional[List[Dict[str, Any]]] = None

    # Use cached JSONL reading for metrics files
    p = results_dir / "model_cluster_scores_df.jsonl"
    if p.exists():
        model_cluster_scores = _get_cached_jsonl(p)

    p = results_dir / "cluster_scores_df.jsonl"
    if p.exists():
        cluster_scores = _get_cached_jsonl(p)

    p = results_dir / "model_scores_df.jsonl"
    if p.exists():
        model_scores = _get_cached_jsonl(p)


    # Load conversations and properties
    conversations: List[Dict[str, Any]] = []
    properties: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []

    # Try lightweight JSONL first (much faster than full_dataset.json)
    lightweight_conv = results_dir / "clustered_results_lightweight.jsonl"
    if lightweight_conv.exists():
        try:
            # Simple approach: just read what we need with nrows limit
            # This is faster than counting + reading separately
            conversations = _read_jsonl_as_list(lightweight_conv, nrows=req.max_conversations)
            logger.info(f"Loaded {len(conversations)} conversations")
        except Exception as e:
            logger.warning(f"Failed to load lightweight conversations: {e}")

    # Load properties from parsed_properties.jsonl
    props_file = results_dir / "parsed_properties.jsonl"
    if props_file.exists():
        try:
            # Simple approach: just read what we need with nrows limit
            properties = _read_jsonl_as_list(props_file, nrows=req.max_properties)
            logger.info(f"Loaded {len(properties)} properties")
        except Exception as e:
            logger.warning(f"Failed to load properties: {e}")

    # Load clusters from clusters.jsonl or clusters.json
    # This is critical because if we load conversations/properties from JSONL,
    # we skip the full_dataset.json block below, so we must load clusters here.
    clusters_file_jsonl = results_dir / "clusters.jsonl"
    clusters_file_json = results_dir / "clusters.json"
    
    if clusters_file_jsonl.exists():
        try:
            clusters = _read_jsonl_as_list(clusters_file_jsonl)
            logger.info(f"Loaded {len(clusters)} clusters from jsonl")
        except Exception as e:
            logger.warning(f"Failed to load clusters from jsonl: {e}")
    elif clusters_file_json.exists():
        try:
            clusters = _read_json_safe(clusters_file_json)
            logger.info(f"Loaded {len(clusters)} clusters from json")
        except Exception as e:
            logger.warning(f"Failed to load clusters from json: {e}")

    # Fallback to full_dataset.json only if JSONL files don't exist
    if not conversations and not properties:
        full = results_dir / "full_dataset.json"
        if full.exists():
            payload = _read_json_safe(full)
            if isinstance(payload, dict):
                try:
                    c = payload.get("conversations")
                    p = payload.get("properties")
                    cl = payload.get("clusters")
                    if isinstance(c, list):
                        conversations_total = len(c)
                        start_idx = (req.conversations_page - 1) * req.conversations_per_page
                        end_idx = start_idx + req.conversations_per_page
                        if req.max_conversations:
                            end_idx = min(end_idx, req.max_conversations)
                        conversations = c[start_idx:end_idx]
                        conversations_has_more = end_idx < conversations_total
                    if isinstance(p, list):
                        properties_total = len(p)
                        start_idx = (req.properties_page - 1) * req.properties_per_page
                        end_idx = start_idx + req.properties_per_page
                        if req.max_properties:
                            end_idx = min(end_idx, req.max_properties)
                        properties = p[start_idx:end_idx]
                        properties_has_more = end_idx < properties_total
                    if isinstance(cl, list):
                        clusters = cl
                except Exception:
                    pass

    # Load clusters from full_dataset.json if available (clusters are small)
    if not clusters:
        full = results_dir / "full_dataset.json"
        if full.exists():
            try:
                payload = _read_json_safe(full)
                if isinstance(payload, dict):
                    cl = payload.get("clusters")
                    if isinstance(cl, list):
                        clusters = cl
            except Exception:
                pass

    return {
        "path": str(results_dir),
        "metrics": {
            "model_cluster_scores": model_cluster_scores or [],
            "cluster_scores": cluster_scores or [],
            "model_scores": model_scores or []
        },
        "conversations": conversations,
        "properties": properties,
        "clusters": clusters,
    }


@app.get("/results/stream/properties")
def stream_properties(
    path: str = Query(..., description="Results directory path"),
    offset: int = Query(0, description="Starting row offset"),
    limit: int = Query(1000, description="Number of rows to stream")
):
    """Stream properties data as JSONL for progressive loading.

    This endpoint streams properties line-by-line, allowing the frontend to
    start rendering results before the entire dataset is loaded.

    Usage:
        GET /results/stream/properties?path=/path/to/results&offset=0&limit=1000

    Returns:
        Streaming response with one JSON object per line (JSONL format)
    """
    import json

    results_dir = _resolve_within_base(path)
    if not results_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {results_dir}")

    props_file = results_dir / "parsed_properties.jsonl"
    if not props_file.exists():
        raise HTTPException(status_code=404, detail="Properties file not found")

    def generate_properties():
        """Generator function that yields properties line-by-line."""
        with props_file.open("r", encoding="utf-8") as f:
            # Skip to offset
            for _ in range(offset):
                next(f, None)

            # Stream up to limit
            count = 0
            for line in f:
                if count >= limit:
                    break
                line = line.strip()
                if line:
                    yield line + "\n"
                    count += 1

    return StreamingResponse(
        generate_properties(),
        media_type="application/x-ndjson",
        headers={
            "X-Total-Offset": str(offset),
            "X-Chunk-Size": str(limit)
        }
    )


@app.get("/results/stream/conversations")
def stream_conversations(
    path: str = Query(..., description="Results directory path"),
    offset: int = Query(0, description="Starting row offset"),
    limit: int = Query(1000, description="Number of rows to stream")
):
    """Stream conversations data as JSONL for progressive loading.

    This endpoint streams conversations line-by-line, allowing the frontend to
    start rendering results before the entire dataset is loaded.

    Usage:
        GET /results/stream/conversations?path=/path/to/results&offset=0&limit=1000

    Returns:
        Streaming response with one JSON object per line (JSONL format)
    """
    import json

    results_dir = _resolve_within_base(path)
    if not results_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {results_dir}")

    conv_file = results_dir / "clustered_results_lightweight.jsonl"
    if not conv_file.exists():
        raise HTTPException(status_code=404, detail="Conversations file not found")

    def generate_conversations():
        """Generator function that yields conversations line-by-line."""
        with conv_file.open("r", encoding="utf-8") as f:
            # Skip to offset
            for _ in range(offset):
                next(f, None)

            # Stream up to limit
            count = 0
            for line in f:
                if count >= limit:
                    break
                line = line.strip()
                if line:
                    yield line + "\n"
                    count += 1

    return StreamingResponse(
        generate_conversations(),
        media_type="application/x-ndjson",
        headers={
            "X-Total-Offset": str(offset),
            "X-Chunk-Size": str(limit)
        }
    )


# -----------------------------
# DataFrame operations
# -----------------------------

def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@app.post("/df/select")
def df_select(req: DFSelectRequest) -> Dict[str, Any]:
    df = _df_from_rows(req.rows)
    # Include filters (AND across columns, OR within column values)
    for col, values in (req.include or {}).items():
        if col in df.columns and values:
            # Be robust to type mismatches by comparing as strings too
            try:
                mask = df[col].isin(values)
            except Exception:
                mask = df[col].astype(str).isin([str(v) for v in values])
            df = df[mask]
    # Exclude filters
    for col, values in (req.exclude or {}).items():
        if col in df.columns and values:
            try:
                mask = ~df[col].isin(values)
            except Exception:
                mask = ~df[col].astype(str).isin([str(v) for v in values])
            df = df[mask]
    return {"rows": df.to_dict(orient="records")}


@app.post("/df/groupby/preview")
def df_groupby_preview(req: DFGroupPreviewRequest) -> Dict[str, Any]:
    try:
        logger.debug(f"BACKEND: df_groupby_preview called with by='{req.by}'")
        logger.debug(f"BACKEND: rows count: {len(req.rows)}")
        logger.debug(f"BACKEND: numeric_cols: {req.numeric_cols}")
        
        df = _df_from_rows(req.rows)
        logger.debug(f"BACKEND: DataFrame shape: {df.shape}")
        logger.debug(f"BACKEND: DataFrame columns: {list(df.columns)}")
        
        if req.by not in df.columns:
            logger.error(f"BACKEND: Column '{req.by}' not found in data")
            raise HTTPException(status_code=400, detail=f"Column not found: {req.by}")
        
        # Determine numeric columns
        num_cols = req.numeric_cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        logger.debug(f"BACKEND: Numeric columns determined: {num_cols}")
        
        # Aggregate
        logger.debug(f"BACKEND: Grouping by column '{req.by}'")
        grouped = df.groupby(req.by, dropna=False)
        preview = []
        for value, sub in grouped:
            means = {c: float(sub[c].mean()) for c in num_cols if c in sub.columns}
            preview.append({"value": value, "count": int(len(sub)), "means": means})
            logger.debug(f"BACKEND: Group '{value}': {len(sub)} items, means: {means}")
        
        logger.debug(f"BACKEND: Returning {len(preview)} groups")
        return {"groups": preview}
        
    except Exception as e:
        import traceback
        logger.error(f"BACKEND ERROR in df_groupby_preview:")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/df/groupby/rows")
def df_groupby_rows(req: DFGroupRowsRequest) -> Dict[str, Any]:
    df = _df_from_rows(req.rows)
    if req.by not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {req.by}")
    sub = df[df[req.by] == req.value]
    start = max((req.page - 1), 0) * max(req.page_size, 1)
    end = start + max(req.page_size, 1)
    return {"total": int(len(sub)), "rows": sub.iloc[start:end].to_dict(orient="records")}


@app.post("/df/custom")
def df_custom(req: DFCustomRequest) -> Dict[str, Any]:
    df = _df_from_rows(req.rows)
    code = (req.code or "").strip()
    if not code:
        return {"rows": req.rows}
    # Whitelist execution environment
    local_env = {"pd": pd, "df": df}
    try:
        result = eval(code, {"__builtins__": {}}, local_env)
        if isinstance(result, pd.DataFrame):
            return {"rows": result.to_dict(orient="records")}
        else:
            return {"error": "Expression must return a pandas DataFrame."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/auto-detect-columns")
def auto_detect_columns(req: AutoDetectRequest) -> Dict[str, Any]:
    """Auto-detect likely column mappings from a sample of data."""
    try:
        from stringsight.core.flexible_data_loader import auto_detect_columns
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(req.rows)
        
        # Run auto-detection
        suggestions = auto_detect_columns(df)
        
        return {
            "success": True,
            "suggestions": suggestions
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestions": {
                'prompt_col': '',
                'response_cols': [],
                'model_cols': [],
                'score_cols': [],
                'method': 'single_model'
            }
        }


@app.post("/validate-flexible-mapping")
def validate_flexible_mapping(req: FlexibleDataRequest) -> Dict[str, Any]:
    """Validate a flexible column mapping against the data."""
    try:
        from stringsight.core.flexible_data_loader import validate_data_format
        
        # Convert to DataFrame for validation
        df = pd.DataFrame(req.rows)
        
        # Validate the mapping
        is_valid, errors = validate_data_format(
            df=df,
            prompt_col=req.mapping.prompt_col,
            response_cols=req.mapping.response_cols,
            model_cols=req.mapping.model_cols,
            score_cols=req.mapping.score_cols
        )
        
        return {
            "valid": is_valid,
            "errors": errors
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"]
        }


@app.post("/process-flexible-data")
def api_process_flexible_data(req: FlexibleDataRequest) -> Dict[str, Any]:
    """Process data using flexible column mapping and return operational format."""
    try:
        from stringsight.core.flexible_data_loader import process_flexible_data
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(req.rows)
        
        # Process the data
        operational_df = process_flexible_data(
            df=df,
            prompt_col=req.mapping.prompt_col,
            response_cols=req.mapping.response_cols,
            model_cols=req.mapping.model_cols,
            score_cols=req.mapping.score_cols,
            method=req.mapping.method
        )
        
        # Convert back to records
        processed_rows = operational_df.to_dict(orient="records")
        
        return {
            "success": True,
            "rows": processed_rows,
            "method": req.mapping.method,
            "columns": operational_df.columns.tolist()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "rows": [],
            "method": req.mapping.method,
            "columns": []
        }


@app.post("/flexible-conversations")
def flexible_conversations(req: FlexibleDataRequest) -> Dict[str, Any]:
    """Process flexible data and return formatted conversations."""
    try:
        # First process the data to operational format
        process_result = api_process_flexible_data(req)
        
        if not process_result["success"]:
            return process_result
        
        # Now format as conversations using the existing logic
        df = pd.DataFrame(process_result["rows"])
        method = process_result["method"]
        
        # Use existing conversation formatting
        traces = format_conversations(df, method)
        
        return {
            "success": True,
            "method": method,
            "conversations": traces
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": req.mapping.method,
            "conversations": []
        }


@app.get("/prompts")
def list_prompts() -> Dict[str, Any]:
    """Return only 'default' and 'agent' prompt choices with metadata and defaults."""
    from stringsight import prompts as _prompts
    from stringsight.prompts import get_system_prompt as _get

    # Build entries for aliases; provide defaults for both methods so UI can prefill
    default_single = getattr(_prompts, "single_model_default_task_description", None)
    default_sbs = getattr(_prompts, "sbs_default_task_description", None)
    agent_single = getattr(_prompts, "agent_system_prompt_custom_task_description", None)
    agent_sbs = getattr(_prompts, "agent_sbs_system_prompt_custom_task_description", None)

    out: List[Dict[str, Any]] = []
    out.append({
        "name": "default",
        "label": "Default",
        "has_task_description": True,
        "default_task_description_single": default_single,
        "default_task_description_sbs": default_sbs,
        "preview": (_get("single_model", "default") or "")[:180],
    })
    out.append({
        "name": "agent",
        "label": "Agent",
        "has_task_description": True,
        "default_task_description_single": agent_single,
        "default_task_description_sbs": agent_sbs,
        "preview": (_get("single_model", "agent") or "")[:180],
    })
    return {"prompts": out}


@app.get("/prompt-text")
def prompt_text(name: str, task_description: Optional[str] = None, method: Optional[str] = None) -> Dict[str, Any]:
    """Return full text of a prompt by name or alias (default/agent), formatted.

    If 'name' is an alias, 'method' determines the template ('single_model' or 'side_by_side').
    Defaults to 'single_model' when omitted.
    """
    from stringsight.prompts import get_system_prompt as _get
    m = method or "single_model"
    try:
        value = _get(m, name, task_description)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": name, "text": value}


# -----------------------------
# Explain (tidy → side-by-side)
# -----------------------------

class TidyRow(BaseModel):
    """A single tidy row for single-model data.

    Fields:
        question_id: Optional stable ID used to pair A/B responses; pairs by prompt when absent.
        prompt: The task text.
        model: Model name (e.g., 'gpt-4.1').
        model_response: The model's response; accepts string or OAI/chat-like structure.
        score: Optional dict of metric name → value.

    Additional keys are allowed and passed through to the DataFrame.
    """
    question_id: Optional[str] = None
    prompt: str
    model: str
    model_response: Any
    score: Optional[Dict[str, float]] = None

    class Config:
        extra = "allow"


class ExplainSideBySideTidyRequest(BaseModel):
    """Request payload to run side-by-side analysis from tidy rows.

    Attributes:
        method: Must be "side_by_side".
        model_a: First model to compare; must exist in the tidy data.
        model_b: Second model to compare; must exist in the tidy data.
        data: List of tidy rows.
        score_columns: Optional list of metric column names if not using a 'score' dict per row.
        sample_size: Optional down-sampling for speed.
        output_dir: Optional output directory for artifacts.
    """
    method: Literal["side_by_side"]
    model_a: str
    model_b: str
    data: List[TidyRow]
    score_columns: Optional[List[str]] = None
    sample_size: Optional[int] = None
    output_dir: Optional[str] = None


@app.post("/api/explain/side-by-side")
async def explain_side_by_side_tidy(req: ExplainSideBySideTidyRequest) -> Dict[str, Any]:
    """Convert tidy data to side-by-side, run explain(), and return results.

    Returns a dictionary with:
        clustered_df: List of row dicts from the clustered DataFrame
        model_stats: Dict of DataFrame-like lists for model/cluster scores
    """
    rows_count = len(req.data) if getattr(req, "data", None) else 0
    logger.info(f"BACKEND: /api/explain/side-by-side models={req.model_a} vs {req.model_b} rows={rows_count}")
    if req.model_a == req.model_b:
        logger.warning("model_a equals model_b; tidy pairing may yield zero pairs.")
    if req.method != "side_by_side":
        raise HTTPException(status_code=422, detail="method must be 'side_by_side'")
    if not req.model_a or not req.model_b:
        raise HTTPException(status_code=422, detail="model_a and model_b are required")
    if not req.data:
        raise HTTPException(status_code=422, detail="data (non-empty) is required")

    # Construct DataFrame from tidy rows (extra fields preserved)
    df = pd.DataFrame([r.dict() for r in req.data])
    logger.debug(f"DataFrame shape: {df.shape}; columns: {list(df.columns)}")
    if "model" in df.columns:
        try:
            models = sorted(df["model"].dropna().astype(str).unique().tolist())
            logger.debug(f"Unique models in data: {models}")
        except Exception:
            pass
    join_col = "question_id" if ("question_id" in df.columns and df["question_id"].notna().any()) else "prompt"
    if join_col in df.columns and "model" in df.columns:
        try:
            model_sets = df.groupby(join_col)["model"].apply(lambda s: set(s.astype(str)))
            est_pairs = int(sum(1 for s in model_sets if req.model_a in s and req.model_b in s))
            logger.info(f"Estimated pairs on '{join_col}': {est_pairs}")
        except Exception:
            pass

    # Delegate tidy→SxS conversion and full pipeline to library
    t0 = time.perf_counter()
    clustered_df, model_stats = await public_api.explain_async(
        df=df,
        method="side_by_side",
        model_a=req.model_a,
        model_b=req.model_b,
        score_columns=req.score_columns,
        sample_size=req.sample_size,
        output_dir=req.output_dir,
    )
    dt = time.perf_counter() - t0
    stats_keys = list(model_stats.keys()) if isinstance(model_stats, dict) else []
    logger.info(f"explain() completed in {dt:.2f}s; rows_out={len(clustered_df)}; model_stats_keys={stats_keys}")

    return {
        "clustered_df": clustered_df.to_dict(orient="records"),
        "model_stats": {k: v.to_dict(orient="records") for k, v in (model_stats or {}).items()},
    }

# Alias without /api prefix for clients calling /explain/side-by-side
app.add_api_route("/explain/side-by-side", explain_side_by_side_tidy, methods=["POST"])


@app.post("/extract/single")
async def extract_single(req: ExtractSingleRequest) -> Dict[str, Any]:
    """Run extraction→parsing→validation for a single row."""
    # Build a one-row DataFrame
    df = pd.DataFrame([req.row])
    method = req.method or detect_method(list(df.columns))
    if method is None:
        raise HTTPException(status_code=422, detail="Unable to detect dataset method from columns.")

    # Validate required columns for clarity before running
    missing = validate_required_columns(df, method)
    if missing:
        raise HTTPException(status_code=422, detail={
            "error": f"Missing required columns for {method}",
            "missing": missing,
            "available": list(df.columns),
        })

    try:
        result = await public_api.extract_properties_only_async(
            df,
            method=method,
            system_prompt=req.system_prompt,
            task_description=req.task_description,
            model_name=req.model_name or "gpt-4.1",
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.95,
            max_tokens=req.max_tokens or 16000,
            max_workers=req.max_workers or 64,
            include_scores_in_prompt=False if req.include_scores_in_prompt is None else req.include_scores_in_prompt,
            use_wandb=req.use_wandb or False,
            output_dir=req.output_dir,
            return_debug=req.return_debug or False,
        )
    except ValueError as e:
        # Surface configuration / validation errors (e.g., missing API keys) to the frontend
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during single-row extraction")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    if isinstance(result, tuple):
        dataset, failures = result
    else:
        dataset, failures = result, []

    # Return parsed properties for this single row
    props = [p.to_dict() for p in dataset.properties]
    return {
        "properties": props,
        "counts": {"properties": len(props)},
        "failures": failures[:5] if req.return_debug else []
    }


@app.post("/extract/batch")
async def extract_batch(req: ExtractBatchRequest) -> Dict[str, Any]:
    """Run extraction→parsing→validation for all rows and return properties table."""
    df = pd.DataFrame(req.rows)

    # Apply sample_size if specified
    if req.sample_size and req.sample_size < len(df):
        df = df.sample(n=req.sample_size, random_state=42)
        logger.info(f"Sampled {req.sample_size} rows from {len(req.rows)} total rows")

    method = req.method or detect_method(list(df.columns))
    if method is None:
        raise HTTPException(status_code=422, detail="Unable to detect dataset method from columns.")

    # Validate required columns for clarity before running
    missing = validate_required_columns(df, method)
    if missing:
        raise HTTPException(status_code=422, detail={
            "error": f"Missing required columns for {method}",
            "missing": missing,
            "available": list(df.columns),
        })

    try:
        result = await public_api.extract_properties_only_async(
            df,
            method=method,
            system_prompt=req.system_prompt,
            task_description=req.task_description,
            model_name=req.model_name or "gpt-4.1",
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.95,
            max_tokens=req.max_tokens or 16000,
            max_workers=req.max_workers or 64,
            include_scores_in_prompt=False if req.include_scores_in_prompt is None else req.include_scores_in_prompt,
            use_wandb=req.use_wandb or False,
            output_dir=req.output_dir,
            return_debug=req.return_debug or False,
        )
    except ValueError as e:
        # Surface configuration / validation errors (e.g., missing API keys) to the frontend
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during batch extraction")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    if isinstance(result, tuple):
        dataset, failures = result
    else:
        dataset, failures = result, []

    # Convert to properties-only table, dropping any failed parses
    props = [p.to_dict() for p in getattr(dataset, 'properties', [])]
    # Enrich with original UI row index by aligning property (question_id, model) with df index and model columns
    try:
        if '__index' in df.columns:
            idx_map: Dict[tuple[str, str], int] = {}
            if method == 'single_model' and 'model' in df.columns:
                # Vectorized: ~10x faster than iterrows()
                idx_map = dict(zip(
                    zip(df.index.astype(str), df['model'].astype(str)),
                    df['__index'].astype(int)
                ))
            elif method == 'side_by_side' and 'model_a' in df.columns and 'model_b' in df.columns:
                # Vectorized: create both model_a and model_b mappings
                indices_int = df['__index'].astype(int).tolist()
                indices_str = df.index.astype(str).tolist()
                model_a_strs = df['model_a'].astype(str).tolist()
                model_b_strs = df['model_b'].astype(str).tolist()
                idx_map = {
                    **{(idx, model_a): ui for idx, model_a, ui in zip(indices_str, model_a_strs, indices_int)},
                    **{(idx, model_b): ui for idx, model_b, ui in zip(indices_str, model_b_strs, indices_int)}
                }
            for p in props:
                key = (str(p.get('question_id')), str(p.get('model')))
                if key in idx_map:
                    p['row_index'] = idx_map[key]
    except Exception:
        pass
    props_df = pd.DataFrame(props)
    rows = props_df.to_dict(orient="records") if not props_df.empty else []
    columns = props_df.columns.tolist() if not props_df.empty else []

    # Quick stats derived from parsing stage if available
    parse_failures = len(failures)
    empty_lists = 0
    try:
        # LLMJsonParser saves parsing_stats.json when output_dir is set; we keep it best-effort here
        parse_failures = 0
    except Exception:
        pass

    return {
        "rows": rows,
        "columns": columns,
        "counts": {"conversations": int(len(df)), "properties": int(len(rows))},
        "stats": {"parse_failures": parse_failures, "empty_lists": empty_lists},
        "failures": failures[:20] if req.return_debug else []
    }


# -----------------------------
# Async batch job API (in-memory)
# -----------------------------


@dataclass
class ExtractJob:
    id: str
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    count_done: int = 0
    count_total: int = 0
    error: Optional[str] = None
    properties: List[Dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False  # Flag to signal cancellation


_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, ExtractJob] = {}


@dataclass
class ClusterJob:
    id: str
    state: str = "queued"  # queued | running | completed | error | cancelled
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_path: Optional[str] = None
    cancelled: bool = False


_CLUSTER_JOBS_LOCK = threading.Lock()
_CLUSTER_JOBS: Dict[str, ClusterJob] = {}


def _run_extract_job(job: ExtractJob, req: ExtractJobStartRequest):
    """Sync wrapper for async extraction - runs in background thread."""
    try:
        asyncio.run(_run_extract_job_async(job, req))
    except Exception as e:
        logger.error(f"Error in background extract job: {e}")
        job.state = "error"
        job.error = str(e)

async def _run_extract_job_async(job: ExtractJob, req: ExtractJobStartRequest):
    try:
        with _JOBS_LOCK:
            job.state = "running"
            # Check if already cancelled before starting
            if job.cancelled:
                job.state = "cancelled"
                return

        df = pd.DataFrame(req.rows)

        # Apply sample_size if specified
        if req.sample_size and req.sample_size < len(df):
            df = df.sample(n=req.sample_size, random_state=42)
            logger.info(f"Sampled {req.sample_size} rows from {len(req.rows)} total rows")

        method = req.method or detect_method(list(df.columns))
        if method is None:
            raise RuntimeError("Unable to detect dataset method from columns.")

        total = len(df)
        with _JOBS_LOCK:
            job.count_total = total
            # Check cancellation again before expensive operation
            if job.cancelled:
                job.state = "cancelled"
                return

        # Define progress callback to update job status in real-time
        def update_progress(completed: int, total: int):
            with _JOBS_LOCK:
                if job:
                    job.count_done = completed
                    job.progress = completed / total if total > 0 else 0.0

        # Process all rows at once - NO CHUNKING
        # The extractor already uses parallel workers internally
        # Note: We can't interrupt this mid-process, but user can cancel before it starts

        # Create dataset and extractor manually to pass progress callback
        from stringsight.core.data_objects import PropertyDataset
        from stringsight.extractors import get_extractor
        from stringsight.postprocess import LLMJsonParser, PropertyValidator
        from stringsight.prompts import get_system_prompt

        system_prompt = get_system_prompt(method, req.system_prompt, req.task_description)
        dataset = PropertyDataset.from_dataframe(df, method=method)

        extractor = get_extractor(
            model_name=req.model_name or "gpt-4.1",
            system_prompt=system_prompt,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.95,
            max_tokens=req.max_tokens or 16000,
            max_workers=req.max_workers or 64,
            include_scores_in_prompt=False if req.include_scores_in_prompt is None else req.include_scores_in_prompt,
            verbose=False,
            use_wandb=False,
        )

        # Run extraction with progress callback
        extracted_dataset = await extractor.run(dataset, progress_callback=update_progress)

        # Determine output directory for saving parsing failures
        # Use req.output_dir if provided, otherwise create a directory in results
        base_results_dir = _get_results_dir()
        if req.output_dir:
            output_dir = str(base_results_dir / req.output_dir)
        else:
            # Create a directory for this extract job to save parsing failures
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(base_results_dir / f"extract_{job.id}_{timestamp}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Parsing failures will be saved to: {output_dir}")

        # Run parsing and validation
        parser = LLMJsonParser(fail_fast=False, verbose=False, use_wandb=False, output_dir=output_dir)
        parsed_dataset = parser.run(extracted_dataset)

        validator = PropertyValidator(verbose=False, use_wandb=False, output_dir=output_dir)
        result = validator.run(parsed_dataset)

        # result is a PropertyDataset (or (PropertyDataset, failures) in other contexts)
        if isinstance(result, tuple):
            dataset = result[0]
        else:
            dataset = result

        # Drop parsing failures by only including successfully parsed properties
        props = [p.to_dict() for p in getattr(dataset, 'properties', [])]

        # Add original row index by aligning with df index and model columns
        try:
            if '__index' in df.columns:
                idx_map: Dict[tuple[str, str], int] = {}
                if method == 'single_model' and 'model' in df.columns:
                    rows_list = df.to_dict('records')
                    for ridx, r in enumerate(rows_list):
                        idx_map[(str(ridx), str(r.get('model', '')))] = int(r['__index'])
                elif method == 'side_by_side' and 'model_a' in df.columns and 'model_b' in df.columns:
                    rows_list = df.to_dict('records')
                    for ridx, r in enumerate(rows_list):
                        ui = int(r['__index'])
                        idx_map[(str(ridx), str(r.get('model_a', '')))] = ui
                        idx_map[(str(ridx), str(r.get('model_b', '')))] = ui
                for p in props:
                    key = (str(p.get('question_id')), str(p.get('model')))
                    if key in idx_map:
                        p['row_index'] = idx_map[key]
        except Exception:
            pass

        with _JOBS_LOCK:
            job.properties = props
            job.count_done = total
            job.state = "done"
            job.progress = 1.0
    except Exception as e:
        with _JOBS_LOCK:
            job.state = "error"
            job.error = str(e)


@app.post("/extract/jobs/start")
def extract_jobs_start(req: ExtractJobStartRequest) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = ExtractJob(id=job_id)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    t = threading.Thread(target=_run_extract_job, args=(job, req), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/extract/jobs/status")
def extract_jobs_status(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job.id,
            "state": job.state,
            "progress": job.progress,
            "count_done": job.count_done,
            "count_total": job.count_total,
            "error": job.error,
        }


@app.get("/extract/jobs/result")
def extract_jobs_result(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.state not in ["done", "cancelled"]:
            raise HTTPException(status_code=409, detail=f"job not done (state: {job.state})")
        return {"properties": job.properties, "count": len(job.properties), "cancelled": job.state == "cancelled"}


@app.post("/extract/jobs/cancel")
def extract_jobs_cancel(job_id: str = Body(..., embed=True)) -> Dict[str, Any]:
    """Cancel a running extraction job.

    This will set the cancellation flag. If the job hasn't started processing yet,
    it will be cancelled immediately. If it's already processing, it will complete
    the current batch and then stop (since we process all rows at once, it will
    finish the current extraction).

    Returns any properties that have been extracted so far.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        if job.state in ["done", "error", "cancelled"]:
            # Already finished, return current state
            return {
                "job_id": job_id,
                "state": job.state,
                "message": f"Job already in state: {job.state}",
                "properties_count": len(job.properties)
            }

        # Set cancellation flag
        job.cancelled = True
        job.state = "cancelled"

        return {
            "job_id": job_id,
            "state": "cancelled",
            "message": "Cancellation requested",
            "properties_count": len(job.properties)
        }


@app.post("/extract/stream")
async def extract_stream(req: ExtractBatchRequest):
    """Stream property extraction results as they complete.

    This endpoint extracts properties and streams them back line-by-line as JSONL,
    allowing the frontend to display results progressively instead of waiting for
    the entire batch to complete.

    The streaming happens at the LLM call level - as each conversation's properties
    are extracted, they're immediately streamed back to the client.
    """
    import json

    df = pd.DataFrame(req.rows)
    method = req.method or detect_method(list(df.columns))
    if method is None:
        raise HTTPException(status_code=422, detail="Unable to detect dataset method from columns.")

    # Validate required columns
    missing = validate_required_columns(df, method)
    if missing:
        raise HTTPException(status_code=422, detail={
            "error": f"Missing required columns for {method}",
            "missing": missing,
            "available": list(df.columns),
        })

    async def generate_properties():
        """Generator that yields properties as they're extracted."""
        from stringsight.core.data_objects import PropertyDataset
        from stringsight.extractors import get_extractor
        from stringsight.postprocess import LLMJsonParser, PropertyValidator

        # Create dataset
        dataset = PropertyDataset.from_dataframe(df, method=method)

        # Create extractor
        extractor = get_extractor(
            model_name=req.model_name or "gpt-4.1",
            system_prompt=req.system_prompt or "default",
            prompt_builder=None,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.95,
            max_tokens=req.max_tokens or 16000,
            max_workers=req.max_workers or 64,
            include_scores_in_prompt=req.include_scores_in_prompt or False,
            verbose=False,
            use_wandb=False,
        )

        # Extract properties (this runs in parallel internally)
        extracted_dataset = await extractor.run(dataset)

        # Parse properties
        parser = LLMJsonParser(fail_fast=False, verbose=False, use_wandb=False)
        parsed_dataset = parser.run(extracted_dataset)

        # Validate properties
        validator = PropertyValidator(verbose=False, use_wandb=False)
        validated_dataset = validator.run(parsed_dataset)

        # Build index map ONCE before streaming (not inside the loop!)
        idx_map: Dict[tuple[str, str], int] = {}
        if '__index' in df.columns:
            if method == 'single_model' and 'model' in df.columns:
                # Vectorized: ~10x faster than iterrows()
                idx_map = dict(zip(
                    zip(df.index.astype(str), df['model'].astype(str)),
                    df['__index'].astype(int)
                ))
            elif method == 'side_by_side' and 'model_a' in df.columns and 'model_b' in df.columns:
                # Vectorized: create both model_a and model_b mappings
                indices_int = df['__index'].astype(int).tolist()
                indices_str = df.index.astype(str).tolist()
                model_a_strs = df['model_a'].astype(str).tolist()
                model_b_strs = df['model_b'].astype(str).tolist()
                idx_map = {
                    **{(idx, model_a): ui for idx, model_a, ui in zip(indices_str, model_a_strs, indices_int)},
                    **{(idx, model_b): ui for idx, model_b, ui in zip(indices_str, model_b_strs, indices_int)}
                }

        # Stream properties as JSONL
        for prop in validated_dataset.properties:
            if prop.property_description is not None:  # Only stream valid properties
                prop_dict = prop.to_dict()
                # Add row_index if available
                if idx_map:
                    key = (str(prop_dict.get('question_id')), str(prop_dict.get('model')))
                    if key in idx_map:
                        prop_dict['row_index'] = idx_map[key]

                yield json.dumps(prop_dict) + "\n"

    return StreamingResponse(
        generate_properties(),
        media_type="application/x-ndjson",
        headers={"X-Extraction-Method": method}
    )


# ============================================================================
# Cluster Job Queue System
# ============================================================================

def _run_cluster_job(job: ClusterJob, req: ClusterRunRequest):
    """Sync wrapper for async clustering - runs in background thread."""
    try:
        asyncio.run(_run_cluster_job_async(job, req))
    except Exception as e:
        logger.error(f"Error in background cluster job: {e}")
        with _CLUSTER_JOBS_LOCK:
            job.state = "error"
            job.error = str(e)


async def _run_cluster_job_async(job: ClusterJob, req: ClusterRunRequest):
    """Run clustering in background thread."""
    try:
        # Import here to avoid circular dependencies
        from stringsight.core.data_objects import PropertyDataset, Property, ConversationRecord
        from stringsight.clusterers import get_clusterer
        import os

        with _CLUSTER_JOBS_LOCK:
            job.state = "running"
            job.progress = 0.1
            if job.cancelled:
                job.state = "cancelled"
                return

        # Preserve original cache setting
        original_cache_setting = os.environ.get("STRINGSIGHT_DISABLE_CACHE", "0")
        os.environ["STRINGSIGHT_DISABLE_CACHE"] = original_cache_setting

        # Force-drop any pre-initialized global LMDB caches
        from stringsight.core import llm_utils as _llm_utils
        from stringsight.clusterers import clustering_utils as _cu
        _orig_default_cache = getattr(_llm_utils, "_default_cache", None)
        _orig_default_llm_utils = getattr(_llm_utils, "_default_llm_utils", None)
        _orig_embed_cache = getattr(_cu, "_cache", None)
        try:
            _llm_utils._default_cache = None
            _llm_utils._default_llm_utils = None
        except Exception:
            pass
        try:
            if hasattr(_cu, "_cache"):
                _cu._cache = None
        except Exception:
            pass

        # Preprocess operationalRows to handle score_columns conversion
        score_columns_to_use = req.score_columns

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.15

        # Auto-detect score columns if not provided
        if not score_columns_to_use and req.operationalRows:
            import pandas as pd
            operational_df = pd.DataFrame(req.operationalRows)

            score_column_name = None
            if 'scores' in operational_df.columns:
                score_column_name = 'scores'
            elif 'score' in operational_df.columns:
                score_column_name = 'score'

            if score_column_name:
                sample_score = operational_df[score_column_name].iloc[0] if len(operational_df) > 0 else None
                if not isinstance(sample_score, dict):
                    logger.info(f"'{score_column_name}' column exists but is not a dict - will attempt to detect score columns")
                else:
                    logger.info(f"'{score_column_name}' column already in nested dict format - no conversion needed")
                    score_columns_to_use = None
                    if score_column_name == 'scores':
                        operational_df.rename(columns={'scores': 'score'}, inplace=True)
            else:
                potential_score_cols = []
                score_related_keywords = ['score', 'rating', 'quality', 'helpfulness', 'accuracy', 'correctness', 'fluency', 'coherence', 'relevance']

                for col in operational_df.columns:
                    if not pd.api.types.is_numeric_dtype(operational_df[col]):
                        continue
                    if col in ['question_id', 'id', 'size', 'cluster_id'] or col.endswith('_id'):
                        continue
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in score_related_keywords):
                        potential_score_cols.append(col)

                if potential_score_cols:
                    logger.info(f"Auto-detected potential score columns: {potential_score_cols}")
                    score_columns_to_use = potential_score_cols
                else:
                    logger.info("No score columns detected")

            if score_column_name == 'scores':
                logger.info("🔄 Normalizing 'scores' column to 'score' for backend compatibility")
                req.operationalRows = operational_df.to_dict('records')

        # Convert score columns if needed
        if score_columns_to_use:
            logger.info(f"Converting score columns to dict format: {score_columns_to_use}")
            import pandas as pd
            from stringsight.core.preprocessing import convert_score_columns_to_dict

            operational_df = pd.DataFrame(req.operationalRows)
            operational_df = convert_score_columns_to_dict(
                operational_df,
                score_columns=score_columns_to_use,
                method=req.method
            )
            req.operationalRows = operational_df.to_dict('records')
            logger.info(f"✓ Score columns converted successfully")

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.2

        # Convert properties data to Property objects
        properties: List[Property] = []
        for p in req.properties:
            try:
                raw_question_id = str(p.get("question_id", ""))
                base_question_id = raw_question_id.split('-')[0] if '-' in raw_question_id else raw_question_id

                prop = Property(
                    id=str(p.get("id", "")),
                    question_id=base_question_id,
                    model=str(p.get("model", "")),
                    property_description=p.get("property_description"),
                    category=p.get("category"),
                    reason=p.get("reason"),
                    evidence=p.get("evidence"),
                    behavior_type=p.get("behavior_type"),
                    raw_response=p.get("raw_response"),
                    contains_errors=p.get("contains_errors"),
                    unexpected_behavior=p.get("unexpected_behavior"),
                    meta=p.get("meta", {})
                )
                properties.append(prop)
            except Exception as e:
                logger.warning(f"Skipping invalid property: {e}")
                continue

        if not properties:
            with _CLUSTER_JOBS_LOCK:
                job.state = "completed"
                job.progress = 1.0
                job.result = {"clusters": []}
            return

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.25

        # Create minimal conversations that match the properties
        conversations: List[ConversationRecord] = []
        all_models = set()
        property_keys = {(prop.question_id, prop.model) for prop in properties}

        logger.info(f"Found {len(property_keys)} unique (question_id, model) pairs from {len(properties)} properties")

        # Create exactly one conversation per unique (question_id, model) pair
        matches_found = 0
        for question_id, model in property_keys:
            all_models.add(model)

            # Find matching operational row for this conversation
            matching_row = None
            for row in req.operationalRows:
                row_qid = str(row.get("question_id", ""))
                row_model = str(row.get("model", ""))

                # Try exact match first
                if row_qid == question_id and row_model == model:
                    matching_row = row
                    matches_found += 1
                    break

                # If no exact match, try matching on base question_id (strip suffix after '-')
                row_qid_base = row_qid.split('-')[0] if '-' in row_qid else row_qid
                question_id_base = question_id.split('-')[0] if '-' in question_id else question_id

                if (row_qid_base == question_id or row_qid == question_id_base) and row_model == model:
                    matching_row = row
                    matches_found += 1
                    break

            # Create minimal conversation (use empty data if no matching row found)
            if matching_row:
                scores = matching_row.get("score") or matching_row.get("scores") or {}
            else:
                scores = {}

            # Try both 'model_response' and 'responses' for compatibility
            response_value = ""
            if matching_row:
                response_value = matching_row.get("responses") or matching_row.get("model_response") or ""

            # Strip property index suffix from question_id to get base conversation ID
            base_question_id = question_id.split('-')[0] if '-' in question_id else question_id

            conv = ConversationRecord(
                question_id=base_question_id,
                model=model,
                prompt=matching_row.get("prompt", "") if matching_row else "",
                responses=response_value,
                scores=scores,
                meta={}
            )
            conversations.append(conv)

        # Handle side-by-side specific logic if detected
        if req.method == "single_model" and req.operationalRows:
            first_row = req.operationalRows[0]
            if "model_a" in first_row and "model_b" in first_row:
                logger.info("🔄 Auto-detected side_by_side method from operationalRows columns")
                req.method = "side_by_side"

        if req.method == "side_by_side":
            logger.info("🔄 Reconstructing conversations for side-by-side metrics...")

            # Group properties by base question_id to identify pairs
            properties_by_qid = {}
            for p in properties:
                if p.question_id not in properties_by_qid:
                    properties_by_qid[p.question_id] = []
                properties_by_qid[p.question_id].append(p)

            # Pre-index operational rows for faster lookup
            operational_rows_map = {}
            for row in req.operationalRows:
                row_qid = str(row.get("question_id", ""))
                operational_rows_map[row_qid] = row
                # Also index by base ID if it's a compound ID
                if '-' in row_qid:
                    base_id = row_qid.split('-')[0]
                    if base_id not in operational_rows_map:
                        operational_rows_map[base_id] = row

            sxs_conversations = []

            for qid, props in properties_by_qid.items():
                # Find matching operational row using lookup map
                matching_row = operational_rows_map.get(qid)

                # If not found by exact match, try base ID match
                if not matching_row and '-' in qid:
                    matching_row = operational_rows_map.get(qid.split('-')[0])

                if matching_row:
                    # Extract models
                    model_a = matching_row.get("model_a")
                    model_b = matching_row.get("model_b")

                    # If models not in row, try to infer from properties
                    if not model_a or not model_b:
                        unique_models = list(set(p.model for p in props))
                        if len(unique_models) >= 2:
                            model_a = unique_models[0]
                            model_b = unique_models[1]
                        else:
                            model_a = "model_a"
                            model_b = "model_b"

                    # Extract scores
                    score_a = matching_row.get("score_a", {})
                    score_b = matching_row.get("score_b", {})

                    # If empty, check if 'scores' or 'score' contains combined info
                    if not score_a and not score_b:
                        combined_score = matching_row.get("score") or matching_row.get("scores")
                        if combined_score:
                            if isinstance(combined_score, list) and len(combined_score) == 2:
                                score_a = combined_score[0] if isinstance(combined_score[0], dict) else {}
                                score_b = combined_score[1] if isinstance(combined_score[1], dict) else {}
                            elif isinstance(combined_score, dict):
                                score_a = combined_score
                                score_b = combined_score
                            else:
                                score_a = {}
                                score_b = {}

                    # Extract winner to meta
                    meta = {}
                    if "winner" in matching_row:
                        meta["winner"] = matching_row["winner"]
                    elif "score" in matching_row and isinstance(matching_row["score"], dict) and "winner" in matching_row["score"]:
                        meta["winner"] = matching_row["score"]["winner"]

                    # Create SxS conversation record
                    conv = ConversationRecord(
                        question_id=qid,
                        model=[model_a, model_b],
                        prompt=matching_row.get("prompt", ""),
                        responses=[matching_row.get("model_a_response", ""), matching_row.get("model_b_response", "")],
                        scores=[score_a, score_b],
                        meta=meta
                    )
                    sxs_conversations.append(conv)

            if sxs_conversations:
                logger.info(f"✅ Created {len(sxs_conversations)} side-by-side conversation records")
                conversations = sxs_conversations

        logger.info(f"✅ Matched {matches_found}/{len(property_keys)} conversations with operationalRows")

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.3

        # Create PropertyDataset
        dataset = PropertyDataset(
            conversations=conversations,
            all_models=list(all_models),
            properties=properties,
            clusters=[],
            model_stats={}
        )

        # Get clustering parameters
        params = req.params
        min_cluster_size = params.minClusterSize if params and params.minClusterSize else 3
        embedding_model = params.embeddingModel if params else "text-embedding-3-small"
        groupby_column = None if params.groupBy == "none" else params.groupBy

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.35

        # Run clustering
        logger.info(f"Starting clustering with {len(properties)} properties, min_cluster_size={min_cluster_size}")

        clusterer = get_clusterer(
            method="hdbscan",
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            assign_outliers=False,
            include_embeddings=False,
            cache_embeddings=True,
            groupby_column=groupby_column,
        )

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.4

        clustered = await clusterer.run(dataset, column_name="property_description")

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.7

        logger.info(f"✓ Clustering complete - found {len(clustered.clusters)} clusters")

        # Save results to disk if output_dir specified
        results_dir_name = None
        results_dir_full_path = None
        if req.output_dir:
            base_results_dir = _get_results_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir_name = f"{req.output_dir}_{timestamp}"
            results_dir = base_results_dir / results_dir_name
            results_dir_full_path = str(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save clusters, properties, and conversations
            clusters_file = results_dir / "clusters.jsonl"
            properties_file = results_dir / "properties.jsonl"
            conversations_file = results_dir / "conversation.jsonl"

            import json
            from dataclasses import asdict

            with open(clusters_file, 'w') as f:
                for cluster in clustered.clusters:
                    f.write(json.dumps(cluster.to_dict()) + '\n')

            with open(properties_file, 'w') as f:
                for prop in properties:
                    f.write(json.dumps(prop.to_dict()) + '\n')

            # Convert conversations to dataframe format with correct column names
            conv_rows = []
            for conv in conversations:
                if isinstance(conv.model, str):
                    # Single model format
                    conv_row = {
                        'question_id': conv.question_id,
                        'prompt': conv.prompt,
                        'model': conv.model,
                        'model_response': conv.responses,
                        'score': conv.scores,
                        **conv.meta
                    }
                else:
                    # Side-by-side format
                    if isinstance(conv.scores, list) and len(conv.scores) == 2:
                        scores_a, scores_b = conv.scores[0], conv.scores[1]
                    else:
                        scores_a, scores_b = {}, {}

                    conv_row = {
                        'question_id': conv.question_id,
                        'prompt': conv.prompt,
                        'model_a': conv.model[0],
                        'model_b': conv.model[1],
                        'model_a_response': conv.responses[0],
                        'model_b_response': conv.responses[1],
                        'score_a': scores_a,
                        'score_b': scores_b,
                        'winner': conv.meta.get('winner'),
                        **{k: v for k, v in conv.meta.items() if k != 'winner'}
                    }
                conv_rows.append(conv_row)

            with open(conversations_file, 'w') as f:
                for row in conv_rows:
                    f.write(json.dumps(row) + '\n')

            logger.info(f"✓ Results saved to {results_dir}")

            with _CLUSTER_JOBS_LOCK:
                job.result_path = str(results_dir_name)

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.75

        # Compute metrics using FunctionalMetrics or SideBySideMetrics
        from stringsight.metrics.functional_metrics import FunctionalMetrics
        from stringsight.metrics.side_by_side import SideBySideMetrics

        # Choose metrics computer based on method
        if req.method == "side_by_side":
            logger.info("🚀 Using SideBySideMetrics for computation")
            metrics_computer = SideBySideMetrics(
                output_dir=None,
                compute_bootstrap=True,
                log_to_wandb=False,
                generate_plots=False
            )
        else:
            logger.info("🚀 Using FunctionalMetrics for computation")
            metrics_computer = FunctionalMetrics(
                output_dir=None,
                compute_bootstrap=True,
                log_to_wandb=False,
                generate_plots=False
            )

        # Run metrics computation on the clustered dataset
        clustered = metrics_computer.run(clustered)

        # Extract the computed metrics from model_stats
        model_cluster_scores_df = clustered.model_stats.get("model_cluster_scores", None)
        cluster_scores_df = clustered.model_stats.get("cluster_scores", None)
        model_scores_df = clustered.model_stats.get("model_scores", None)

        # Convert DataFrames to list of dicts for JSON serialization
        model_cluster_scores_array = []
        cluster_scores_array = []
        model_scores_array = []

        if model_cluster_scores_df is not None and hasattr(model_cluster_scores_df, 'to_dict'):
            model_cluster_scores_array = model_cluster_scores_df.to_dict('records')

        if cluster_scores_df is not None and hasattr(cluster_scores_df, 'to_dict'):
            cluster_scores_array = cluster_scores_df.to_dict('records')

        if model_scores_df is not None and hasattr(model_scores_df, 'to_dict'):
            model_scores_array = model_scores_df.to_dict('records')

        logger.info(f"✓ Metrics computed: {len(model_cluster_scores_array)} model_cluster_scores, "
                   f"{len(cluster_scores_array)} cluster_scores, {len(model_scores_array)} model_scores")

        # Save metrics if output_dir specified
        if req.output_dir and results_dir_name:
            results_dir = _get_results_dir() / results_dir_name

            import json
            if model_cluster_scores_array:
                with open(results_dir / "model_cluster_scores_df.jsonl", 'w') as f:
                    for item in model_cluster_scores_array:
                        f.write(json.dumps(item) + '\n')

            if cluster_scores_array:
                with open(results_dir / "cluster_scores.jsonl", 'w') as f:
                    for item in cluster_scores_array:
                        f.write(json.dumps(item) + '\n')

            if model_scores_array:
                with open(results_dir / "model_scores.jsonl", 'w') as f:
                    for item in model_scores_array:
                        f.write(json.dumps(item) + '\n')

            logger.info("✓ Metrics saved to disk")

        with _CLUSTER_JOBS_LOCK:
            job.progress = 0.9

        # Build enriched response
        enriched = []
        total_conversations = {}
        for model in all_models:
            model_convs = [c for c in conversations if c.model == model]
            total_conversations[model] = len(model_convs)

        total_unique_conversations = len({c.question_id for c in conversations})

        for cluster in clustered.clusters:
            cluster_dict = cluster.to_dict()
            enriched.append(cluster_dict)

        # Build final result
        result = {
            "clusters": enriched,
            "total_conversations_by_model": total_conversations,
            "total_unique_conversations": total_unique_conversations,
            "results_dir": results_dir_name,
            "metrics": {
                "model_cluster_scores": model_cluster_scores_array,
                "cluster_scores": cluster_scores_array,
                "model_scores": model_scores_array,
            }
        }

        # Mark job as completed
        with _CLUSTER_JOBS_LOCK:
            job.state = "completed"
            job.progress = 1.0
            job.result = result

        logger.info(f"✓ Cluster job {job.id} completed successfully")

    except Exception as e:
        logger.error(f"Error in background cluster job: {e}", exc_info=True)
        with _CLUSTER_JOBS_LOCK:
            job.state = "error"
            job.error = str(e)


@app.post("/cluster/job/start")
async def cluster_job_start(req: ClusterRunRequest) -> Dict[str, Any]:
    """Start a clustering job in the background."""
    job_id = str(uuid.uuid4())
    job = ClusterJob(id=job_id)

    with _CLUSTER_JOBS_LOCK:
        _CLUSTER_JOBS[job_id] = job

    # Start background thread
    thread = threading.Thread(target=_run_cluster_job, args=(job, req), daemon=True)
    thread.start()

    logger.info(f"Started cluster job {job_id}")

    return {
        "job_id": job_id,
        "state": job.state,
        "progress": job.progress
    }


@app.get("/cluster/job/status/{job_id}")
def cluster_job_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a clustering job."""
    with _CLUSTER_JOBS_LOCK:
        job = _CLUSTER_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return {
            "job_id": job_id,
            "status": job.state,
            "progress": job.progress,
            "error_message": job.error
        }


@app.get("/cluster/job/result/{job_id}")
def cluster_job_result(job_id: str) -> Dict[str, Any]:
    """Get the result of a completed clustering job."""
    with _CLUSTER_JOBS_LOCK:
        job = _CLUSTER_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.state != "completed":
            raise HTTPException(status_code=400, detail=f"Job is not completed yet (state: {job.state})")

        return {
            "job_id": job_id,
            "result": job.result,
            "result_path": job.result_path
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",  # Keep application logs
        access_log=False   # Disable access logs (the noisy GET requests)
    )

