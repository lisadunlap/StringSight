"""
Convenience functions for common use cases.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from ..core.data_objects import PropertyDataset
from ..pipeline import Pipeline
from .helpers import run_pipeline_smart
from .sync_api import explain


def explain_side_by_side(
    df: pd.DataFrame,
    system_prompt: str | None = None,
    tidy_side_by_side_models: Tuple[str, str] | None = None,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function for side-by-side model comparison.

    Args:
        df: DataFrame with columns: model_a, model_b, model_a_response, model_b_response, winner
        system_prompt: System prompt for extraction (if None, will be auto-determined)
        **kwargs: Additional arguments passed to explain()

    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(
        df,
        method="side_by_side",
        system_prompt=system_prompt,
        tidy_side_by_side_models=tidy_side_by_side_models,
        **kwargs,
    )


def explain_single_model(
    df: pd.DataFrame,
    system_prompt: str | None = None,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function for single model analysis.

    Args:
        df: DataFrame with columns: model, model_response, score
        system_prompt: System prompt for extraction (if None, will be auto-determined)
        **kwargs: Additional arguments passed to explain()

    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(df, method="single_model", system_prompt=system_prompt, **kwargs)


def explain_with_custom_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline,
    method: str = "single_model"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Explain model behavior using a custom pipeline.

    Args:
        df: DataFrame with conversation data
        pipeline: Custom pipeline to use
        method: "side_by_side" or "single_model"

    Returns:
        Tuple of (clustered_df, model_stats)
    """
    dataset = PropertyDataset.from_dataframe(df)
    result_dataset = run_pipeline_smart(pipeline, dataset)
    return result_dataset.to_dataframe(), result_dataset.model_stats
