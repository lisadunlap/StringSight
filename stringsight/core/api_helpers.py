"""
Shared helper functions for API endpoints.

This module contains common utilities used across multiple API routers,
including file I/O, path resolution, and DataFrame loading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal
from pathlib import Path
import os

import pandas as pd
from fastapi import UploadFile, HTTPException

from stringsight.formatters import Method, detect_method
from stringsight.logging_config import get_logger

logger = get_logger(__name__)


def get_base_browse_dir() -> Path:
    """Return the base directory allowed for server-side browsing.

    Defaults to the current working directory. You can override by setting
    environment variable `BASE_BROWSE_DIR` to an absolute path.

    Returns:
        Base directory path for browsing
    """
    env = os.environ.get("BASE_BROWSE_DIR")
    base = Path(env).expanduser().resolve() if env else Path.cwd()
    return base


def resolve_within_base(user_path: str) -> Path:
    """Resolve a user-supplied path and ensure it is within the allowed base.

    Args:
        user_path: Path provided by the client (file or directory)

    Returns:
        Absolute `Path` guaranteed to be within the base directory

    Raises:
        HTTPException: if the path is invalid or escapes the base directory
    """
    base = get_base_browse_dir()
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


def read_json_safe(path: Path) -> Any:
    """Read a JSON file from disk into a Python object.

    Args:
        path: Path to the JSON file

    Returns:
        Parsed JSON object
    """
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataframe_from_upload(upload: UploadFile) -> pd.DataFrame:
    """Load a DataFrame from an uploaded file (CSV or JSONL).

    Args:
        upload: Uploaded file from FastAPI

    Returns:
        Loaded DataFrame

    Raises:
        HTTPException: If file format is unsupported or loading fails
    """
    fname = upload.filename or ""
    if fname.endswith(".csv"):
        df = pd.read_csv(upload.file)
    elif fname.endswith(".jsonl"):
        df = pd.read_json(upload.file, lines=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use .csv or .jsonl")
    return df


def load_dataframe_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Load a DataFrame from a list of dicts (rows)."""
    return pd.DataFrame(rows)


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    """Load a DataFrame from a file path on the server filesystem.

    Args:
        path: File path (CSV or JSONL)

    Returns:
        Loaded DataFrame

    Raises:
        HTTPException: If file format is unsupported or loading fails
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use .csv or .jsonl")


def resolve_df_and_method(
    upload: UploadFile | None = None,
    rows: List[Dict[str, Any]] | None = None,
    path: str | None = None,
    method: Literal["single_model", "side_by_side"] | None = None,
) -> tuple[pd.DataFrame, Method]:
    """Resolve the DataFrame and method from one of three input sources.

    Args:
        upload: Optional uploaded file
        rows: Optional list of row dicts
        path: Optional file path on server
        method: Optional explicit method override

    Returns:
        Tuple of (DataFrame, Method)

    Raises:
        HTTPException: If no valid input source is provided
    """
    if upload is not None:
        df = load_dataframe_from_upload(upload)
    elif rows is not None:
        df = load_dataframe_from_rows(rows)
    elif path is not None:
        df = load_dataframe_from_path(path)
    else:
        raise HTTPException(status_code=400, detail="No data source provided")

    # Auto-detect method if not provided
    result_method: Literal["single_model", "side_by_side"]
    if method is None:
        detected = detect_method(df)
        if detected is None:
            raise HTTPException(status_code=400, detail="Could not detect method from data")
        result_method = detected
    else:
        # Method is already a string literal from the request
        result_method = method

    return df, result_method


# ============================================================================
# DEDUPLICATION HELPERS - Eliminate ~980 lines of duplicated code
# ============================================================================


def detect_and_convert_score_columns(
    operational_rows: List[Dict[str, Any]],
    score_columns: List[str] | None,
    method: str
) -> tuple[List[Dict[str, Any]], List[str] | None]:
    """
    Auto-detect score columns and convert to nested dict format.

    This function handles the complex logic of detecting and converting score columns
    that is duplicated 4x in api.py (lines 461-546, 1417-1481, 2813-2900, etc.).

    Handles:
    - Auto-detection of 'score'/'scores' columns already in nested dict format
    - Normalization of 'scores' -> 'score' for backend compatibility
    - Auto-detection of score-related columns (helpfulness, accuracy, etc.)
    - Conversion to nested dict format using convert_score_columns_to_dict()

    Args:
        operational_rows: List of conversation dicts
        score_columns: Explicit list of score column names (or None to auto-detect)
        method: 'single_model' or 'side_by_side'

    Returns:
        (processed_rows, detected_score_columns) - Tuple of processed rows and
        the score columns that were detected/used (None if already in dict format)
    """
    if not operational_rows:
        return operational_rows, score_columns

    operational_df = pd.DataFrame(operational_rows)
    score_columns_to_use = score_columns

    # Check if 'score' or 'scores' column already exists in nested dict format
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
                logger.info("Normalizing 'scores' column to 'score' for backend compatibility")
                return operational_df.to_dict('records'), None
            return operational_rows, None

    # Auto-detect score columns if not provided
    if not score_columns_to_use:
        potential_score_cols = []
        score_related_keywords = [
            'score', 'rating', 'quality', 'helpfulness',
            'accuracy', 'correctness', 'fluency', 'coherence', 'relevance'
        ]

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

    # Convert score columns if needed
    if score_columns_to_use:
        logger.info(f"Converting score columns to dict format: {score_columns_to_use}")
        from stringsight.core.preprocessing import convert_score_columns_to_dict

        operational_df = convert_score_columns_to_dict(
            operational_df,
            score_columns=score_columns_to_use,
            method=method
        )

        logger.info("Score columns converted successfully")
        return operational_df.to_dict('records'), score_columns_to_use

    return operational_rows, None


def reconstruct_sxs_conversations(
    properties: List[Any],  # List[Property]
    operational_rows: List[Dict[str, Any]]
) -> List[Any]:  # List[ConversationRecord]
    """
    Reconstruct side-by-side conversation records from properties.

    This function handles the complex logic of reconstructing SxS conversations
    that is duplicated 2x in api.py (lines 671-777, etc.).

    Groups properties by question_id and creates paired conversation records
    with [model_a, model_b] format required for SxS metrics.

    Handles:
    - Grouping properties by question_id
    - Fast lookup via pre-indexed operational_rows (O(1) instead of O(n))
    - Model extraction from operational rows or inferred from properties
    - Score extraction (score_a/score_b or combined list/dict format)
    - Winner metadata extraction from multiple possible locations

    Args:
        properties: List of Property objects to group
        operational_rows: List of operational row dicts with model_a/model_b data

    Returns:
        List of ConversationRecord objects in SxS format with [model_a, model_b]
    """
    from stringsight.core.data_objects import ConversationRecord
    import time

    # Group properties by question_id
    properties_by_qid: Dict[str, List[Any]] = {}
    for prop in properties:
        if prop.question_id not in properties_by_qid:
            properties_by_qid[prop.question_id] = []
        properties_by_qid[prop.question_id].append(prop)

    # Pre-index operational rows for O(1) lookup instead of O(n) search
    t0 = time.time()
    operational_rows_map = {}
    for row in operational_rows:
        row_qid = str(row.get("question_id", ""))
        operational_rows_map[row_qid] = row
        # Also index by base ID if it's a compound ID (e.g. "48-0" -> "48")
        if '-' in row_qid:
            base_id = row_qid.split('-')[0]
            if base_id not in operational_rows_map:
                operational_rows_map[base_id] = row

    logger.info(f"Indexed {len(operational_rows)} operational rows in {time.time() - t0:.4f}s")

    sxs_conversations = []
    t1 = time.time()

    for qid, props in properties_by_qid.items():
        # Find matching operational row using O(1) lookup map
        matching_row = operational_rows_map.get(qid)

        # If not found by exact match, try base ID match (if qid has suffix)
        if not matching_row and '-' in qid:
            matching_row = operational_rows_map.get(qid.split('-')[0])

        if not matching_row:
            continue

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
                # Fallback - skip this conversation
                continue

        # Extract scores - handle multiple formats
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
                    # If it's a dict, duplicate it for both (fallback)
                    score_a = combined_score
                    score_b = combined_score
                else:
                    score_a = {}
                    score_b = {}

        # Extract winner to meta - check multiple possible locations
        meta = {}
        if "winner" in matching_row:
            meta["winner"] = matching_row["winner"]
        elif "score" in matching_row and isinstance(matching_row["score"], dict):
            if "winner" in matching_row["score"]:
                meta["winner"] = matching_row["score"]["winner"]

        # Create SxS conversation record
        conv = ConversationRecord(
            question_id=qid,
            model=[model_a, model_b],
            prompt=matching_row.get("prompt", ""),
            responses=[
                matching_row.get("model_a_response", ""),
                matching_row.get("model_b_response", "")
            ],
            scores=[score_a, score_b],
            meta=meta
        )
        sxs_conversations.append(conv)

    logger.info(f"Created {len(sxs_conversations)} side-by-side conversation records in {time.time() - t1:.4f}s")
    return sxs_conversations


def enrich_properties_with_row_index(
    properties: List[Dict[str, Any]],
    df: pd.DataFrame,
    method: str
) -> List[Dict[str, Any]]:
    """
    Add original UI row index to properties by matching on question_id and model.

    This function handles the row index enrichment logic that is duplicated 3x
    in api.py (extraction endpoints).

    Uses vectorized pandas operations for 10x faster performance vs iterrows().
    Requires df to have '__index' column with original row positions.

    Args:
        properties: List of property dicts to enrich
        df: Original DataFrame with '__index' column
        method: 'single_model' or 'side_by_side'

    Returns:
        Properties list with 'row_index' field added to matching properties
    """
    if '__index' not in df.columns:
        return properties

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

    # Enrich properties with row_index
    for p in properties:
        key = (str(p.get('question_id')), str(p.get('model')))
        if key in idx_map:
            p['row_index'] = idx_map[key]

    return properties


def create_progress_callback(job_lock, job_ref):
    """
    Factory for creating progress update callbacks for background jobs.

    This pattern is duplicated 6x in api.py (extraction and clustering jobs).

    Args:
        job_lock: Threading lock for safe job access
        job_ref: Reference to job object/dict with progress/count attributes

    Returns:
        Callable(completed: int, total: int) that updates job progress atomically
    """
    def update_progress(completed: int, total: int):
        with job_lock:
            if job_ref:
                job_ref.progress = completed / total if total > 0 else 0.0
                if hasattr(job_ref, 'count_done'):
                    job_ref.count_done = completed
                if hasattr(job_ref, 'count_total'):
                    job_ref.count_total = total

    return update_progress
