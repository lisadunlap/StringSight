"""DataFrame utilities for StringSight APIs.

Currently includes helpers to normalize/flatten score columns.
"""
from __future__ import annotations

from typing import List, Literal
import pandas as pd

Method = Literal["single_model", "side_by_side"]


def _flatten_dict_series(series: pd.Series, prefix: str | None = None) -> pd.DataFrame:
    """Flatten a Series of dicts into a DataFrame.

    Non-dict values become empty rows.
    """
    def _ensure_dict(v):
        return v if isinstance(v, dict) else {}

    df = pd.json_normalize(series.map(_ensure_dict))
    if prefix:
        df = df.add_prefix(f"{prefix}_")
    return df


def explode_score_columns(df: pd.DataFrame, method: Method) -> pd.DataFrame:
    """Explode score columns into scalar columns and drop original score fields.

    - For single_model: `score` → columns (optionally prefixed `score_`)
    - For side_by_side: `score_a`, `score_b` → columns (prefixed `score_a_`, `score_b_`)
    """
    out = df.copy()
    if method == "single_model" and "score" in out.columns:
        flat = _flatten_dict_series(out["score"], prefix="score")
        out = pd.concat([out.drop(columns=["score"]), flat], axis=1)
    elif method == "side_by_side":
        if "score_a" in out.columns:
            flat_a = _flatten_dict_series(out["score_a"], prefix="score_a")
            out = pd.concat([out.drop(columns=["score_a"]), flat_a], axis=1)
        if "score_b" in out.columns:
            flat_b = _flatten_dict_series(out["score_b"], prefix="score_b")
            out = pd.concat([out.drop(columns=["score_b"]), flat_b], axis=1)
    return out


__all__ = ["explode_score_columns"]


