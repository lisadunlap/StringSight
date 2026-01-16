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

from typing import Any, Dict, List, Literal, cast
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
from stringsight.metrics.cluster_subset import enrich_clusters_with_metrics, compute_total_conversations_by_model, prepare_long_frame, compute_subset_metrics
from stringsight.logging_config import get_logger
from stringsight.schemas import ClusterRunRequest
from stringsight.constants import DEFAULT_MAX_WORKERS
import threading, uuid
from dataclasses import dataclass, field
from functools import lru_cache
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

def _get_cached_jsonl(path: Path, nrows: int | None = None) -> List[Dict[str, Any]]:
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


def _read_jsonl_as_list(path: Path, nrows: int | None = None) -> List[Dict[str, Any]]:
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
    method: Literal["single_model", "side_by_side"] | None = None


class ReadRequest(BaseModel):
    """Request body for reading a dataset from the server filesystem.

    Use with caution – this assumes the server has access to the path.
    """
    path: str
    method: Literal["single_model", "side_by_side"] | None = None
    limit: int | None = None  # return all rows if None


class ListRequest(BaseModel):
    path: str  # directory to list (server-side)
    exts: List[str] | None = None  # e.g., [".jsonl", ".json", ".csv"]


class ResultsLoadRequest(BaseModel):
    """Request to load a results directory from the server filesystem.

    Attributes:
        path: Absolute or base-relative path to the results directory, which must
              be within BASE_BROWSE_DIR (defaults to current working directory).
        max_conversations: Maximum number of conversations to load (default: all).
                          Use this to limit memory usage for large datasets.
        max_properties: Maximum number of properties to load (default: all).
        conversations_page: Page number for conversations (1-indexed).
        conversations_per_page: Number of conversations per page.
        properties_page: Page number for properties (1-indexed).
        properties_per_page: Number of properties per page.
    """
    path: str
    max_conversations: int | None = None
    max_properties: int | None = None
    conversations_page: int = 1
    conversations_per_page: int = 100
    properties_page: int = 1
    properties_per_page: int = 100






# -----------------------------
# Extraction endpoints schemas
# -----------------------------

class ExtractSingleRequest(BaseModel):
    row: Dict[str, Any]
    method: Literal["single_model", "side_by_side"] | None = None
    system_prompt: str | None = None
    task_description: str | None = None
    model_name: str | None = "gpt-4.1"
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    max_tokens: int | None = 16000
    max_workers: int | None = DEFAULT_MAX_WORKERS
    include_scores_in_prompt: bool | None = False
    use_wandb: bool | None = False
    output_dir: str | None = None
    return_debug: bool | None = False


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
    numeric_cols: List[str] | None = None


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

# GZIP compression enabled for improved network performance
# Uses moderate compression level (5) to balance CPU and transfer speed
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

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

# Import all refactored routers
from stringsight.routers import (
    health,
    dataframe,
    prompts,
    explain,
    validation,
    extraction,
    clustering,
)

# Include existing jobs router
app.include_router(jobs_router)

# Include all refactored routers
app.include_router(health.router)
app.include_router(dataframe.router)
app.include_router(prompts.router)
app.include_router(explain.router)
app.include_router(validation.router)
app.include_router(extraction.router)
app.include_router(clustering.router)

# NOTE:
# All of the primary API endpoints are implemented in `stringsight/routers/*` and
# are registered above via `app.include_router(...)`.
#
# Historically, this module also contained large, duplicate endpoint definitions
# (e.g. `/health`, `/cluster/run`, `/extract/*`, `/results/load`, etc.). Those
# duplicates have been removed so the router implementations are the single
# source of truth.


# -----------------------------
# Async batch job API (in-memory)
# -----------------------------


@dataclass
class ClusterJob:
    id: str
    state: str = "queued"  # queued | running | completed | error | cancelled
    progress: float = 0.0
    error: str | None = None
    result: Dict[str, Any | None] | None = None
    result_path: str | None = None
    cancelled: bool = False


_CLUSTER_JOBS_LOCK = threading.Lock()
_CLUSTER_JOBS: Dict[str, ClusterJob] = {}


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
        from stringsight.core.caching import UnifiedCache

        _orig_default_cache: UnifiedCache | None = getattr(_llm_utils, "_default_cache", None)
        _orig_default_llm_utils = getattr(_llm_utils, "_default_llm_utils", None)
        _orig_embed_cache = getattr(_cu, "_cache", None)
        try:
            if hasattr(_llm_utils, "_default_cache"):
                _llm_utils._default_cache = None  # type: ignore
            if hasattr(_llm_utils, "_default_llm_utils"):
                _llm_utils._default_llm_utils = None  # type: ignore
        except Exception:
            pass
        try:
            if hasattr(_cu, "_cache"):
                _cu._cache = None  # type: ignore
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
                score_columns=score_columns_to_use if score_columns_to_use else [],
                method=req.method or "single_model"
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
        all_models: set[str] = set()
        property_keys: set[tuple[str, str]] = {
            (prop.question_id, cast(str, prop.model)) for prop in properties
        }

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
            properties_by_qid: Dict[str, List[Property]] = {}
            for prop in properties:
                if prop.question_id not in properties_by_qid:
                    properties_by_qid[prop.question_id] = []
                properties_by_qid[prop.question_id].append(prop)

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
                    model_a_str = model_a if isinstance(model_a, str) else str(model_a)
                    model_b_str = model_b if isinstance(model_b, str) else str(model_b)
                    conv = ConversationRecord(
                        question_id=qid,
                        model=[model_a_str, model_b_str],
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

        clustered = await clusterer.run(dataset)  # type: ignore[misc]

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
            metrics_computer: SideBySideMetrics = SideBySideMetrics(
                output_dir=None,
                compute_bootstrap=True,
                log_to_wandb=False,
                generate_plots=False
            )
        else:
            logger.info("🚀 Using FunctionalMetrics for computation")
            metrics_computer_temp = FunctionalMetrics(
                output_dir=None,
                compute_bootstrap=True,
                log_to_wandb=False,
                generate_plots=False
            )
            metrics_computer = metrics_computer_temp  # type: ignore[assignment]

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

