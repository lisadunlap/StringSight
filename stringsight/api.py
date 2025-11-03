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
import io
import os
import time

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
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
from stringsight.email_service import send_results_email
import threading, uuid
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib

logger = get_logger(__name__)

# -------------------------------------------------------------------------
# Render persistent disk configuration
# -------------------------------------------------------------------------
def _get_persistent_data_dir() -> Path:
    """Get the base directory for persistent data (results, cache) on Render.
    
    If RENDER_DISK_PATH is set, use that as the base for all persistent data.
    Otherwise, default to the current working directory (local development).
    """
    render_disk = os.environ.get("RENDER_DISK_PATH")
    if render_disk:
        base = Path(render_disk).resolve()
        logger.info(f"Using Render persistent disk: {base}")
        return base
    return Path.cwd()

def _get_results_dir() -> Path:
    """Get the results directory, potentially on persistent disk."""
    base = _get_persistent_data_dir()
    return base / "results"

def _get_cache_dir() -> Path:
    """Get the cache directory, potentially on persistent disk."""
    # Check if RENDER_DISK_PATH is set and STRINGSIGHT_CACHE_DIR is not explicitly set
    # If so, automatically configure cache to use the persistent disk
    if os.environ.get("RENDER_DISK_PATH") and not os.environ.get("STRINGSIGHT_CACHE_DIR"):
        base = _get_persistent_data_dir()
        cache_dir = base / ".cache" / "stringsight"
        # Set the environment variable so the Cache class picks it up
        os.environ["STRINGSIGHT_CACHE_DIR"] = str(cache_dir)
        logger.info(f"Auto-configured cache directory to use persistent disk: {cache_dir}")
        return cache_dir
    # Otherwise, let Cache class handle it using STRINGSIGHT_CACHE_DIR env var or default
    return Path.cwd() / ".cache" / "stringsight"

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


class EmailResultsRequest(BaseModel):
    """Request to email clustering results to a user."""
    email: str
    results_dir: str
    experiment_name: str


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


class ExtractBatchRequest(BaseModel):
    rows: List[Dict[str, Any]]
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
    sample_size: Optional[int] = None  # Randomly sample N rows before extraction


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

# Initialize persistent disk configuration on startup
# This sets up environment variables for cache and results directories
_get_cache_dir()  # Call this to auto-configure cache if RENDER_DISK_PATH is set

# GZIP compression disabled - can add significant CPU overhead
# Uncomment below if network transfer is the bottleneck:
# from fastapi.middleware.gzip import GZipMiddleware
# app.add_middleware(GZipMiddleware, minimum_size=10000, compresslevel=1)

# Allow local Vite dev server by default - be permissive for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

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
        "openai/text-embedding-3-small",
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


@app.post("/cluster/run")
def cluster_run(req: ClusterRunRequest) -> Dict[str, Any]:
    """Run clustering directly on existing properties without re-running extraction.
    
    This is much more efficient than the full explain() pipeline since it skips
    the expensive LLM property extraction step and works with already-extracted properties.
    
    Note: Cache is disk-backed (DiskCache) and thread-safe.
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
    
    try:
        # Convert properties data to Property objects
        properties: List[Property] = []
        for p in req.properties:
            try:
                prop = Property(
                    id=str(p.get("id", "")),
                    question_id=str(p.get("question_id", "")),
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
                # This handles side-by-side format where question_id might be "0-0" vs "0"
                row_qid_base = row_qid.split('-')[0] if '-' in row_qid else row_qid
                if row_qid_base == question_id and row_model == model:
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
            scores = matching_row.get("score", {}) if matching_row else {}
            
            conv = ConversationRecord(
                question_id=question_id,
                model=model,
                prompt=matching_row.get("prompt", "") if matching_row else "",
                responses=matching_row.get("model_response", "") if matching_row else "",
                scores=scores,
                meta={}
            )
            conversations.append(conv)
        
        logger.info(f"✅ Matched {matches_found}/{len(property_keys)} conversations with operationalRows")
        
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
            cache_embeddings=False,
            groupby_column=groupby_column,
        )
        
        # Run clustering
        clustered_dataset = clusterer.run(dataset, column_name="property_description")
        
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
    
    # Compute metrics using FunctionalMetrics (without bootstrap for speed)
    from stringsight.metrics.functional_metrics import FunctionalMetrics
    
    # FunctionalMetrics needs PropertyDataset with clusters populated
    metrics_computer = FunctionalMetrics(
        output_dir=None,
        compute_bootstrap=False,  # Disable bootstrap for API speed
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
        logger.info(f"🔧 Transforming model_cluster_scores to array format:")
        logger.info(f"  - Sample model: {sample_model}")
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
                    row[f"quality_{metric_name}_delta"] = metric_value
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
                row[f"quality_{metric_name}_delta"] = metric_value
        
        # Add metadata
        row["metadata"] = metrics.get("metadata", {})
        
        cluster_scores_array.append(row)
    
    # Note: model_scores would need to be computed separately if needed
    # For now, we'll return an empty array as it's not computed in this endpoint
    
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
            "model_scores": []  # Not computed in clustering-only endpoint
        }
    }


class ClusterMetricsRequest(BaseModel):
    clusters: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    operationalRows: List[Dict[str, Any]]
    included_property_ids: Optional[List[str]] = None


@app.post("/cluster/metrics")
def cluster_metrics(req: ClusterMetricsRequest) -> Dict[str, Any]:
    """Recompute cluster metrics for a filtered subset without reclustering."""
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

    Request path must be within BASE_BROWSE_DIR (default: current working directory).

    Implements pagination to reduce initial load time and memory usage:
    - conversations_page/conversations_per_page for conversations pagination
    - properties_page/properties_per_page for properties pagination
    - load_metrics_only flag to skip loading conversations/properties entirely
    """
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
        "model_cluster_scores": model_cluster_scores or [],
        "cluster_scores": cluster_scores or [],
        "model_scores": model_scores or [],
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


@app.post("/results/email")
def email_results(req: EmailResultsRequest) -> Dict[str, Any]:
    """Email clustering results to a user.

    Sends a zip file of the results directory to the specified email address.
    Requires EMAIL_SMTP_SERVER, EMAIL_SENDER, and EMAIL_PASSWORD environment variables.

    Args:
        req: EmailResultsRequest with email, results_dir, and experiment_name

    Returns:
        Dict with 'success' boolean and 'message' string
    """
    results_dir = _resolve_within_base(req.results_dir)
    if not results_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {results_dir}")

    result = send_results_email(
        recipient_email=req.email,
        results_dir=str(results_dir),
        experiment_name=req.experiment_name
    )

    if not result['success']:
        raise HTTPException(status_code=500, detail=result['message'])

    return result


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
        model: Model name (e.g., 'gpt-4o').
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
def explain_side_by_side_tidy(req: ExplainSideBySideTidyRequest) -> Dict[str, Any]:
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
    clustered_df, model_stats = public_api.explain(
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
def extract_single(req: ExtractSingleRequest) -> Dict[str, Any]:
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

    result = public_api.extract_properties_only(
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
def extract_batch(req: ExtractBatchRequest) -> Dict[str, Any]:
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

    result = public_api.extract_properties_only(
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


class ExtractJobStartRequest(ExtractBatchRequest):
    pass  # Inherits all fields from ExtractBatchRequest


def _run_extract_job(job: ExtractJob, req: ExtractJobStartRequest):
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
        extracted_dataset = extractor.run(dataset, progress_callback=update_progress)

        # Run parsing and validation
        parser = LLMJsonParser(fail_fast=False, verbose=False, use_wandb=False)
        parsed_dataset = parser.run(extracted_dataset)

        validator = PropertyValidator(verbose=False, use_wandb=False)
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
        extracted_dataset = extractor.run(dataset)

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

