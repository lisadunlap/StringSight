"""
Explain endpoint for side-by-side analysis.

Converts tidy format data to side-by-side format and runs explain pipeline.
"""

from typing import Dict, Any
import time

import pandas as pd
from fastapi import APIRouter, HTTPException

from stringsight.schemas import ExplainSideBySideTidyRequest
from stringsight.logging_config import get_logger
import stringsight.public as public_api

logger = get_logger(__name__)

router = APIRouter(tags=["explain"])


@router.post("/api/explain/side-by-side")
@router.post("/explain/side-by-side")  # Alias without /api prefix
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
