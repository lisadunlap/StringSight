"""
DataFrame manipulation endpoints.

Utility endpoints for filtering, grouping, transforming DataFrames,
and natural-language-to-pandas code generation.
"""

from typing import Dict, List, Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from stringsight.schemas import (
    DFSelectRequest, DFGroupPreviewRequest, DFCustomRequest,
    NL2PandasRequest, NL2PandasResponse,
    DFCorrelationsRequest, DFCorrelationsResponse, DFCorrelationEntry,
)
from stringsight.core.llm_utils import single_completion
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/df", tags=["dataframe"])


NL2PANDAS_SYSTEM_PROMPT = """\
You are a pandas code generation assistant. The user will describe a data view \
they want in natural language. You must return a single valid Python pandas \
expression that evaluates to a DataFrame.

Rules:
- The DataFrame variable is `df`. The pandas module is `pd`.
- Your expression MUST evaluate to a pandas DataFrame (not a Series, scalar, etc.).
- CRITICAL: Column names must be copied EXACTLY from the schema below, including \
spaces, parentheses, and special characters. Do NOT abbreviate or simplify them.
- Do NOT use exec(), import, open(), or any side-effectful operations.
- Return ONLY the pandas expression on the first line, with NO markdown fencing.
- On the second line, return a brief one-sentence explanation starting with "Explanation: ".

AVAILABLE COLUMNS (use these names EXACTLY):
{schema}

Sample values:
{samples}
"""


def _build_nl2pandas_prompt(req: NL2PandasRequest) -> str:
    """Build the system prompt with schema and sample values injected.

    Args:
        req: The NL2PandasRequest containing columns, dtypes, and sample_values.

    Returns:
        Formatted system prompt string.
    """
    schema_lines = [f"  - \"{col}\" (dtype: {req.dtypes.get(col, 'unknown')})" for col in req.columns]
    sample_lines = []
    for col in req.columns:
        vals = req.sample_values.get(col, [])
        if vals:
            sample_lines.append(f"  - {col}: {vals[:5]}")
    return NL2PANDAS_SYSTEM_PROMPT.format(
        schema="\n".join(schema_lines),
        samples="\n".join(sample_lines),
    )


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
    except KeyError as e:
        return {"error": f"Column not found: {e}. Available columns: {list(df.columns)}"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/nl2pandas", response_model=NL2PandasResponse)
def nl2pandas(req: NL2PandasRequest) -> NL2PandasResponse:
    """Generate a pandas expression from a natural language query.

    Calls an LLM with the DataFrame schema and sample values to produce
    a pandas expression the user can review before executing.

    Args:
        req: NL2PandasRequest with query, columns, dtypes, and sample_values.

    Returns:
        NL2PandasResponse with the generated code and a short explanation.
    """
    system_prompt = _build_nl2pandas_prompt(req)
    response = single_completion(req.query, system_prompt=system_prompt)

    if not response:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")

    lines = response.strip().splitlines()
    code = lines[0].strip()
    # Strip markdown code fencing if the LLM wrapped it
    if code.startswith("```"):
        code = code.lstrip("`").removeprefix("python").strip()
    if code.endswith("```"):
        code = code.rstrip("`").strip()

    explanation = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.lower().startswith("explanation:"):
            explanation = stripped[len("explanation:"):].strip()
            break
        elif stripped:
            explanation = stripped
            break

    return NL2PandasResponse(code=code, explanation=explanation)


@router.post("/correlations", response_model=DFCorrelationsResponse)
def df_correlations(req: DFCorrelationsRequest) -> DFCorrelationsResponse:
    """Compute Pearson correlations between numeric columns and score columns.

    Args:
        req: DFCorrelationsRequest with rows, score_columns, and numeric_columns.

    Returns:
        DFCorrelationsResponse with a list of pairwise correlation entries.
    """
    df = _df_from_rows(req.rows)

    all_cols = list(set(req.score_columns + req.numeric_columns))
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

    # Coerce to numeric, dropping non-parseable values
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    results: list[DFCorrelationEntry] = []
    for nc in req.numeric_columns:
        for sc in req.score_columns:
            if nc == sc:
                continue
            pair = df[[nc, sc]].dropna()
            count = len(pair)
            if count < 3:
                continue
            r = float(pair[nc].corr(pair[sc]))
            results.append(DFCorrelationEntry(
                column=nc, score_column=sc, pearson_r=round(r, 4), count=count
            ))

    return DFCorrelationsResponse(correlations=results)
