"""
Property extraction and labeling endpoints.

Endpoints for:
- Single and batch property extraction
- Fixed-taxonomy labeling
- Background job management for extraction
- Streaming extraction results
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
import asyncio
import threading
import uuid
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from stringsight.schemas import ExtractSingleRequest, ExtractBatchRequest, ExtractJobStartRequest, LabelRequest
from stringsight.formatters import detect_method, validate_required_columns
from stringsight.utils.paths import _get_results_dir
from stringsight.logging_config import get_logger
import stringsight.public as public_api

logger = get_logger(__name__)

router = APIRouter(tags=["extraction"])


# -----------------------------
# Job management data structures
# -----------------------------

@dataclass
class ExtractJob:
    """Background extraction job state."""
    id: str
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    count_done: int = 0
    count_total: int = 0
    error: str | None = None
    properties: List[Dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False  # Flag to signal cancellation


_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, ExtractJob] = {}


# -----------------------------
# Helper functions for row index enrichment
# -----------------------------

def _enrich_properties_with_row_index(
    properties: List[Dict[str, Any]],
    df: pd.DataFrame,
    method: str
) -> List[Dict[str, Any]]:
    """Add UI row indices to properties using vectorized pandas operations.

    This is one of our deduplication helpers - used to be duplicated 3x in api.py.
    """
    try:
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

        for p in properties:
            key = (str(p.get('question_id')), str(p.get('model')))
            if key in idx_map:
                p['row_index'] = idx_map[key]

    except Exception:
        pass

    return properties


# -----------------------------
# Label endpoint
# -----------------------------

@router.post("/label/run")
async def label_run(req: LabelRequest) -> Dict[str, Any]:
    """Run fixed-taxonomy labeling pipeline.

    This endpoint runs the label() function which:
    1. Uses FixedAxesLabeler to assign conversation responses to predefined taxonomy labels
    2. Skips clustering (each taxonomy label becomes its own cluster via DummyClusterer)
    3. Computes metrics per label and model

    Only supports single_model format data.

    Returns:
        Dictionary with:
        - properties: List of property dicts with assigned labels
        - clusters: List of cluster dicts (one per taxonomy label)
        - metrics: Dict with model_cluster_scores, cluster_scores, model_scores DataFrames
        - total_conversations_by_model: Count of conversations per model
        - total_unique_conversations: Total unique conversations
    """
    t_start = time.perf_counter()
    timings = {}

    # Validate taxonomy
    if not req.taxonomy or len(req.taxonomy) == 0:
        raise HTTPException(status_code=400, detail="Taxonomy must contain at least one label")

    # Build DataFrame from rows
    df = pd.DataFrame(req.rows)
    timings['df_creation'] = time.perf_counter() - t_start

    # Normalize column names to match what label() expects
    column_mapping = {
        'responses': 'model_response',  # Frontend sends 'responses', backend expects 'model_response'
        'scores': 'score',  # Frontend sends 'scores', backend expects 'score'
    }

    for frontend_col, backend_col in column_mapping.items():
        if frontend_col in df.columns and backend_col not in df.columns:
            df = df.rename(columns={frontend_col: backend_col})
            logger.info(f"Renamed column '{frontend_col}' -> '{backend_col}' for label() compatibility")
    
    timings['column_mapping'] = time.perf_counter() - t_start

    # Apply sample_size if specified
    if req.sample_size and req.sample_size < len(df):
        df = df.sample(n=req.sample_size, random_state=42)
        logger.info(f"Sampled {req.sample_size} rows from {len(req.rows)} total rows for labeling")
    
    timings['sampling'] = time.perf_counter() - t_start

    # Validate that we have single_model format data
    method = detect_method(list(df.columns))
    if method == "side_by_side":
        raise HTTPException(
            status_code=400,
            detail="label() only supports single_model format data. Found side_by_side format."
        )

    # Check required columns
    if "prompt" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required 'prompt' column",
                "available": list(df.columns),
            }
        )

    # Check for model column
    model_col = req.model_column or "model"
    if model_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Missing required model column: '{model_col}'",
                "available": list(df.columns),
                "hint": "Specify 'model_column' in the request if your model column has a different name"
            }
        )

    # Auto-detect response column if not specified
    response_col = req.model_response_column
    if not response_col or response_col == "model_response":
        response_aliases = ["model_response", "responses", "response", "output", "completion", "text"]
        found_col = None
        for alias in response_aliases:
            if alias in df.columns:
                found_col = alias
                break

        if found_col:
            response_col = found_col
            if found_col != "model_response":
                logger.info(f"Auto-detected response column: '{found_col}' (will be mapped to 'model_response')")
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Could not find response column",
                    "available": list(df.columns),
                    "tried_aliases": response_aliases,
                    "hint": "Specify 'model_response_column' in the request to indicate which column contains model responses"
                }
            )
    
    timings['preprocessing_total'] = time.perf_counter() - t_start
    logger.info(f"[TIMING] Label endpoint preprocessing: {timings}")

    try:
        # Call the label() function from public API
        t_before_label = time.perf_counter()
        clustered_df, model_stats = await asyncio.to_thread(
            public_api.label,
            df,
            taxonomy=req.taxonomy,
            sample_size=None,  # Already sampled above
            score_columns=req.score_columns,
            prompt_column=req.prompt_column or "prompt",
            model_column=model_col,
            model_response_column=response_col,
            question_id_column=req.question_id_column,
            model_name=req.model_name or "gpt-4.1",
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            max_tokens=req.max_tokens or 2048,
            max_workers=req.max_workers or 64,
            metrics_kwargs=req.metrics_kwargs or {},
            use_wandb=req.use_wandb or False,
            wandb_project=req.wandb_project,
            verbose=req.verbose or False,
            output_dir=req.output_dir,
            extraction_cache_dir=req.extraction_cache_dir,
            metrics_cache_dir=req.metrics_cache_dir,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during label() pipeline")
        raise HTTPException(status_code=500, detail=f"Labeling failed: {e}")

    timings['label_pipeline'] = time.perf_counter() - t_before_label
    timings['total'] = time.perf_counter() - t_start
    logger.info(f"[TIMING] Label endpoint total: {timings['total']:.2f}s; label_pipeline: {timings['label_pipeline']:.2f}s")
    logger.info(f"label() completed; rows_out={len(clustered_df)}; taxonomy_labels={len(req.taxonomy)}")

    # Extract properties from clustered_df
    properties = []

    # Define core output columns that should be displayed in the table
    core_columns = {
        "id", "question_id", "category", "property_description", "reason", "evidence", "model",
        "cluster_id", "cluster_label"
    }
    
    for _, row in clustered_df.iterrows():
        # Start with all columns from the row
        prop: Dict[str, Any] = {}
        
        # Only include columns that are in core_columns
        for col in clustered_df.columns:
            if col not in core_columns:
                continue
            value = row.get(col)
            # Handle pandas NaN/None values for scalars
            # For array-like values, check using a different method
            if isinstance(value, (list, tuple, np.ndarray)):
                # For array-like values, just pass them through
                prop[col] = value
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                prop[col] = None
            else:
                # Convert non-JSON-serializable types to strings
                if isinstance(value, (str, int, float, bool)):
                    prop[col] = value
                elif isinstance(value, dict):
                    prop[col] = value
                else:
                    prop[col] = str(value)
        
        # Ensure question_id is always a string
        if "question_id" in prop:
            prop["question_id"] = str(prop["question_id"]) if prop["question_id"] is not None else ""

        # Add property_id as alias for id (frontend expects property_id)
        if "id" in prop:
            prop["property_id"] = prop["id"]

        properties.append(prop)

    # Enrich properties with UI row index (same as extraction endpoint)
    properties = _enrich_properties_with_row_index(properties, df, method="single_model")

    # Extract clusters - each taxonomy label is a cluster
    clusters = []
    cluster_id = 0
    for label_name, label_desc in req.taxonomy.items():
        # Get all properties for this label
        label_properties = [p for p in properties if p.get("cluster_label") == label_name]
        label_count = len(label_properties)

        # Extract property_ids and property_descriptions for ClusterSidecard display
        property_ids = [p.get("property_id") for p in label_properties if p.get("property_id") is not None]
        property_descriptions = [p.get("property_description") for p in label_properties if p.get("property_description")]

        cluster = {
            "cluster_id": cluster_id,
            "cluster_label": label_name,
            "cluster_description": label_desc,
            "size": label_count,
            "properties": label_properties,
            "property_ids": property_ids,
            "property_descriptions": property_descriptions
        }
        clusters.append(cluster)
        cluster_id += 1

    # Calculate conversation counts
    total_conversations_by_model: Dict[str, int] = {}
    for _, row in clustered_df.iterrows():
        model = row.get("model", "unknown")
        total_conversations_by_model[model] = total_conversations_by_model.get(model, 0) + 1

    total_unique_conversations = len(clustered_df["question_id"].unique()) if "question_id" in clustered_df.columns else len(clustered_df)

    # Format metrics if present
    metrics_output = None
    if model_stats:
        metrics_output = {
            k: v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v
            for k, v in model_stats.items()
        }

    return {
        "properties": properties,
        "clusters": clusters,
        "metrics": metrics_output,
        "total_conversations_by_model": total_conversations_by_model,
        "total_unique_conversations": total_unique_conversations,
    }


# -----------------------------
# Single extraction endpoint
# -----------------------------

@router.post("/extract/single")
async def extract_single(req: ExtractSingleRequest) -> Dict[str, Any]:
    """Run extraction→parsing→validation for a single row."""
    # Build a one-row DataFrame
    df = pd.DataFrame([req.row])
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


# -----------------------------
# Batch extraction endpoint
# -----------------------------

@router.post("/extract/batch")
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

    # Validate required columns
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during batch extraction")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    if isinstance(result, tuple):
        dataset, failures = result
    else:
        dataset, failures = result, []

    # Convert to properties-only table
    props = [p.to_dict() for p in getattr(dataset, 'properties', [])]

    # Enrich with original UI row index using our deduplication helper
    props = _enrich_properties_with_row_index(props, df, method)

    props_df = pd.DataFrame(props)
    rows = props_df.to_dict(orient="records") if not props_df.empty else []
    columns = props_df.columns.tolist() if not props_df.empty else []

    # Quick stats
    parse_failures = len(failures)
    empty_lists = 0

    return {
        "rows": rows,
        "columns": columns,
        "counts": {"conversations": int(len(df)), "properties": int(len(rows))},
        "stats": {"parse_failures": parse_failures, "empty_lists": empty_lists},
        "failures": failures[:20] if req.return_debug else []
    }


# -----------------------------
# Background job endpoints
# -----------------------------

def _run_extract_job(job: ExtractJob, req: ExtractJobStartRequest):
    """Sync wrapper for async extraction - runs in background thread."""
    try:
        asyncio.run(_run_extract_job_async(job, req))
    except Exception as e:
        logger.error(f"Error in background extract job: {e}")
        job.state = "error"
        job.error = str(e)


async def _run_extract_job_async(job: ExtractJob, req: ExtractJobStartRequest):
    """Async extraction job runner."""
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

        # Determine output directory for saving parsing failures
        base_results_dir = _get_results_dir()
        if req.output_dir:
            output_dir = str(base_results_dir / req.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(base_results_dir / f"extract_{job.id}_{timestamp}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Parsing failures will be saved to: {output_dir}")

        # Run parsing and validation
        parser = LLMJsonParser(fail_fast=False, verbose=False, use_wandb=False, output_dir=output_dir)
        parsed_dataset = parser.run(extracted_dataset)

        validator = PropertyValidator(verbose=False, use_wandb=False, output_dir=output_dir)
        result = validator.run(parsed_dataset)

        if isinstance(result, tuple):
            dataset = result[0]
        else:
            dataset = result

        # Drop parsing failures by only including successfully parsed properties
        props = [p.to_dict() for p in getattr(dataset, 'properties', [])]

        # Enrich with original UI row index using our deduplication helper
        props = _enrich_properties_with_row_index(props, df, method)

        with _JOBS_LOCK:
            job.properties = props
            job.count_done = total
            job.state = "done"
            job.progress = 1.0
    except Exception as e:
        with _JOBS_LOCK:
            job.state = "error"
            job.error = str(e)


@router.post("/extract/jobs/start")
def extract_jobs_start(req: ExtractJobStartRequest) -> Dict[str, Any]:
    """Start a background extraction job and return job ID."""
    job_id = str(uuid.uuid4())
    job = ExtractJob(id=job_id)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    t = threading.Thread(target=_run_extract_job, args=(job, req), daemon=True)
    t.start()
    return {"job_id": job_id}


@router.get("/extract/jobs/status")
def extract_jobs_status(job_id: str) -> Dict[str, Any]:
    """Get status of a background extraction job."""
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


@router.get("/extract/jobs/result")
def extract_jobs_result(job_id: str) -> Dict[str, Any]:
    """Get results of a completed extraction job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.state not in ["done", "cancelled"]:
            raise HTTPException(status_code=409, detail=f"job not done (state: {job.state})")
        return {"properties": job.properties, "count": len(job.properties), "cancelled": job.state == "cancelled"}


@router.post("/extract/jobs/cancel")
def extract_jobs_cancel(job_id: str) -> Dict[str, Any]:
    """Cancel a running extraction job.

    Sets the cancellation flag. If the job hasn't started processing yet,
    it will be cancelled immediately. If it's already processing, it will complete
    the current batch and then stop.

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


# -----------------------------
# Streaming extraction endpoint
# -----------------------------

@router.post("/extract/stream")
async def extract_stream(req: ExtractBatchRequest):
    """Stream property extraction results as they complete.

    This endpoint extracts properties and streams them back line-by-line as JSONL,
    allowing the frontend to display results progressively instead of waiting for
    the entire batch to complete.

    The streaming happens at the LLM call level - as each conversation's properties
    are extracted, they're immediately streamed back to the client.
    """
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
