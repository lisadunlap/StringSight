"""
File I/O, path browsing, and validation endpoints.

Endpoints for:
- Reading and validating datasets from files or payloads
- Browsing server filesystem
- Loading results directories
- Converting datasets to conversation format
"""

from typing import Dict, List, Any, Literal
from datetime import datetime
from pathlib import Path
import os

import pandas as pd
from fastapi import APIRouter, HTTPException, File, Body, UploadFile

from stringsight.schemas import RowsPayload, ReadRequest, ListRequest, ResultsLoadRequest
from stringsight.formatters import detect_method, validate_required_columns, format_conversations
from stringsight.utils.df_utils import explode_score_columns
from stringsight.utils.paths import _get_results_dir
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["validation"])


# -----------------------------
# Helper functions
# -----------------------------

def _resolve_within_base(user_path: str) -> Path:
    """Resolve a user-supplied path and ensure it is within the allowed base."""
    from pathlib import Path
    base = Path(os.environ.get("BASE_BROWSE_DIR", ".")).resolve()
    requested = (base / user_path).resolve()

    # Security: ensure the requested path is within base
    if not str(requested).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")

    return requested


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
            if nrows is not None and i >= nrows:
                break
            rows.append(json.loads(line))
    return rows


def _get_cached_jsonl(path: Path, nrows: int | None = None) -> List[Dict[str, Any]]:
    """Read JSONL file with caching (delegates to core/api_cache.py)."""
    from stringsight.core.api_cache import get_cached_jsonl
    return get_cached_jsonl(path, nrows)


def _load_dataframe_from_upload(upload: UploadFile) -> pd.DataFrame:
    """Load a DataFrame from an uploaded file (CSV or JSONL)."""
    filename = (upload.filename or "").lower()
    raw = upload.file.read()

    # Decode text formats
    if filename.endswith(".csv"):
        from io import StringIO
        return pd.read_csv(StringIO(raw.decode("utf-8")))
    elif filename.endswith(".jsonl"):
        from io import StringIO
        return pd.read_json(StringIO(raw.decode("utf-8")), lines=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .jsonl")


def _load_dataframe_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Load a DataFrame from a list of row dictionaries."""
    return pd.DataFrame(rows)


def _load_dataframe_from_path(path: str) -> pd.DataFrame:
    """Load a DataFrame from a file path (CSV or JSONL)."""
    p = path.lower()
    if p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif p.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .jsonl")


def _resolve_df_and_method(
    file: UploadFile | None,
    payload: RowsPayload | None,
) -> tuple[pd.DataFrame, str]:
    """Resolve DataFrame and method from either file upload or payload."""
    if not file and not payload:
        raise HTTPException(status_code=400, detail="Provide either a file upload or a rows payload.")

    if file:
        df = _load_dataframe_from_upload(file)
        detected = detect_method(list(df.columns))
        method = detected or (payload.method if payload else None)
    else:
        assert payload is not None
        df = _load_dataframe_from_rows(payload.rows)
        method = payload.method or detect_method(list(df.columns))

    if method is None:
        raise HTTPException(status_code=422, detail="Unable to detect method from columns")

    return df, method


# -----------------------------
# Validation endpoints
# -----------------------------

@router.post("/detect-and-validate")
def detect_and_validate(
    file: UploadFile | None = File(default=None),
    payload: RowsPayload | None = Body(default=None),
) -> Dict[str, Any]:
    """Auto-detect dataset method and validate required columns.

    Accepts either a file upload or a rows payload.
    Returns method, validation status, missing columns, and data preview.
    """
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


@router.post("/conversations")
def conversations(
    file: UploadFile | None = File(default=None),
    payload: RowsPayload | None = Body(default=None),
) -> Dict[str, Any]:
    """Convert dataset to conversation format for UI display.

    Normalizes score columns and formats conversations based on method.
    """
    df, method = _resolve_df_and_method(file, payload)

    # Normalize score columns for convenience in clients
    # Ensure method is a string literal type
    from typing import cast
    method_str = cast(Literal["single_model", "side_by_side"], method if isinstance(method, str) else (method.value if hasattr(method, 'value') else "single_model"))
    try:
        df = explode_score_columns(df, method_str)
    except Exception:
        pass

    traces = format_conversations(df, method_str)
    return {"method": method, "conversations": traces}


# -----------------------------
# Path browsing endpoints
# -----------------------------

@router.post("/read-path")
def read_path(req: ReadRequest) -> Dict[str, Any]:
    """Read a dataset from a server path, auto-detect/validate, return preview and method.

    Validates required columns and optionally flattens score columns.
    """
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
        raise HTTPException(
            status_code=422,
            detail={"error": f"Missing required columns for {method}", "missing": missing}
        )

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


@router.post("/list-path")
def list_path(req: ListRequest) -> Dict[str, Any]:
    """List files and folders at a server directory path.

    Returns entries with name, path, type (file/dir), modified timestamp, and size.
    Optionally filters files by allowed extensions.
    """
    base = _resolve_within_base(req.path)
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {base}")

    allowed_exts = set(e.lower() for e in (req.exts or [])) if req.exts else set()
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
                items.append({
                    "name": name,
                    "path": str(full),
                    "type": "dir",
                    "modified": modified
                })
            else:
                ext = full.suffix.lower()
                if allowed_exts and ext not in allowed_exts:
                    continue
                size = os.path.getsize(str(full))
                items.append({
                    "name": name,
                    "path": str(full),
                    "type": "file",
                    "size": size,
                    "modified": modified
                })
        except (OSError, IOError):
            # If we can't get file info, skip it
            continue

    return {"entries": items}


# -----------------------------
# Results loading endpoints
# -----------------------------

@router.post("/results/load")
def results_load(req: ResultsLoadRequest) -> Dict[str, Any]:
    """Load a results directory and return metrics plus optional dataset with pagination.

    Supports both JSON metrics (model_cluster_scores.json, cluster_scores.json,
    model_scores.json) and JSONL DataFrame exports (model_cluster_scores_df.jsonl,
    cluster_scores_df.jsonl, model_scores_df.jsonl). If a full_dataset.json
    file is present, returns its conversations, properties, and clusters.

    Request path can be:
    - Relative path from results directory (e.g., "frontend/conversation_...")
    - Absolute path within BASE_BROWSE_DIR

    Implements pagination to reduce initial load time and memory usage:
    - conversations_page/conversations_per_page for conversations pagination
    - properties_page/properties_per_page for properties pagination
    - max_conversations/max_properties for hard caps
    """
    # Try to resolve relative to results directory first (for job.result_path compatibility)
    path_obj = Path(req.path)
    results_dir: Path

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
    model_cluster_scores: List[Dict[str, Any | None]] | None = None
    cluster_scores: List[Dict[str, Any | None]] | None = None
    model_scores: List[Dict[str, Any | None]] | None = None

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
            conversations = _read_jsonl_as_list(lightweight_conv, nrows=req.max_conversations)
            logger.info(f"Loaded {len(conversations)} conversations")
        except Exception as e:
            logger.warning(f"Failed to load lightweight conversations: {e}")

    # Load properties from parsed_properties.jsonl
    props_file = results_dir / "parsed_properties.jsonl"
    if props_file.exists():
        try:
            properties = _read_jsonl_as_list(props_file, nrows=req.max_properties)
            logger.info(f"Loaded {len(properties)} properties")
        except Exception as e:
            logger.warning(f"Failed to load properties: {e}")

    # Load clusters from clusters.jsonl or clusters.json
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
                    p_data = payload.get("properties")
                    cl = payload.get("clusters")

                    if isinstance(c, list):
                        conversations_total = len(c)
                        start_idx = (req.conversations_page - 1) * req.conversations_per_page
                        end_idx = start_idx + req.conversations_per_page
                        if req.max_conversations:
                            end_idx = min(end_idx, req.max_conversations)
                        conversations = c[start_idx:end_idx]

                    if isinstance(p_data, list):
                        properties_total = len(p_data)
                        start_idx = (req.properties_page - 1) * req.properties_per_page
                        end_idx = start_idx + req.properties_per_page
                        if req.max_properties:
                            end_idx = min(end_idx, req.max_properties)
                        properties = p_data[start_idx:end_idx]

                    if isinstance(cl, list):
                        clusters = cl
                except Exception:
                    pass

    # Load clusters from full_dataset.json if not loaded yet
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
