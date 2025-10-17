"""
Trace formatting utilities for frontend-facing APIs.

This module centralizes logic to:
- Detect dataset method (single_model vs side_by_side) from columns
- Validate required columns
- Convert rows into OpenAI-style conversation traces (trace-only, no metadata)

These helpers are intentionally isolated from the Gradio app so a React app
can consume the same JSON shape without pulling in UI code.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import pandas as pd

from stringsight.core.data_objects import (
    check_and_convert_to_oai_format,
)


Method = Literal["single_model", "side_by_side"]


REQUIRED_COLUMNS: Dict[Method, List[str]] = {
    "single_model": ["prompt", "model", "model_response"],
    "side_by_side": [
        "prompt",
        "model_a",
        "model_b",
        "model_a_response",
        "model_b_response",
    ],
}


def detect_method(columns: List[str]) -> Optional[Method]:
    """Return the best-matching method based on available columns.

    Args:
        columns: List of column names present in the dataset

    Returns:
        "single_model" | "side_by_side" if a set of required columns is satisfied,
        otherwise None.
    """
    col_set = set(columns)
    if set(REQUIRED_COLUMNS["side_by_side"]).issubset(col_set):
        return "side_by_side"
    if set(REQUIRED_COLUMNS["single_model"]).issubset(col_set):
        return "single_model"
    return None


def validate_required_columns(df: pd.DataFrame, method: Method) -> List[str]:
    """Return the list of missing required columns for the given method.

    Empty list indicates the DataFrame satisfies the requirement.
    """
    required = set(REQUIRED_COLUMNS[method])
    missing = [c for c in REQUIRED_COLUMNS[method] if c not in df.columns]
    return missing


def format_single_trace_from_row(row: pd.Series) -> Dict[str, object]:
    """Format a single-model row into a trace-only conversation object.

    Expects required columns to be present; callers should validate first.
    """
    prompt: str = row["prompt"]
    response = row["model_response"]
    messages, _ = check_and_convert_to_oai_format(prompt, response)
    return {
        "question_id": str(row.name),
        "prompt": prompt,
        "messages": messages,
    }


def format_side_by_side_trace_from_row(row: pd.Series) -> Dict[str, object]:
    """Format a side-by-side row into a pair of traces (trace-only).

    Expects required columns to be present; callers should validate first.
    """
    prompt: str = row["prompt"]

    resp_a = row["model_a_response"]
    resp_b = row["model_b_response"]

    messages_a, _ = check_and_convert_to_oai_format(prompt, resp_a)
    messages_b, _ = check_and_convert_to_oai_format(prompt, resp_b)

    return {
        "question_id": str(row.name),
        "prompt": prompt,
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "messages_a": messages_a,
        "messages_b": messages_b,
    }


def format_conversations(df: pd.DataFrame, method: Method) -> List[Dict[str, object]]:
    """Format an entire DataFrame into a list of traces for the given method."""
    if method == "single_model":
        return [format_single_trace_from_row(row) for _, row in df.iterrows()]
    else:
        return [format_side_by_side_trace_from_row(row) for _, row in df.iterrows()]


def format_trace_with_metadata(trace: Dict[str, object], metadata: Dict[str, object] | None = None) -> Dict[str, object]:
    """Optional wrapper – attach metadata above the trace without changing its schema.

    For the MVP we keep metadata empty; this is a future-proof hook.
    """
    out = {"trace": trace}
    if metadata:
        out["metadata"] = metadata
    return out


__all__ = [
    "Method",
    "REQUIRED_COLUMNS",
    "detect_method",
    "validate_required_columns",
    "format_single_trace_from_row",
    "format_side_by_side_trace_from_row",
    "format_conversations",
    "format_trace_with_metadata",
]


