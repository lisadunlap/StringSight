"""
DataFrame manipulation endpoints.

Utility endpoints for filtering, grouping, and transforming DataFrames.
"""

from typing import Dict, List, Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from stringsight.schemas import DFSelectRequest, DFGroupPreviewRequest, DFCustomRequest
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/df", tags=["dataframe"])


def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper to convert list of row dicts to DataFrame."""
    return pd.DataFrame(rows)


@router.post("/select")
def df_select(req: DFSelectRequest) -> Dict[str, Any]:
    """Filter DataFrame rows based on include/exclude criteria.

    Include filters are AND across columns, OR within column values.
    Exclude filters remove matching rows.
    """
    df = _df_from_rows(req.rows)

    # Include filters (AND across columns, OR within column values)
    for col, values in (req.include or {}).items():
        if col in df.columns and values:
            try:
                mask = df[col].isin(values)
            except Exception:
                # Be robust to type mismatches by comparing as strings
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


@router.post("/groupby/preview")
def df_groupby_preview(req: DFGroupPreviewRequest) -> Dict[str, Any]:
    """Preview grouped data with counts and means for numeric columns."""
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


@router.post("/custom")
def df_custom(req: DFCustomRequest) -> Dict[str, Any]:
    """Execute custom pandas code on DataFrame.

    WARNING: Uses eval() in a sandboxed environment. Only for trusted inputs.
    """
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
