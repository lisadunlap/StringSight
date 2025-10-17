"""Logic for the **View Examples** tab – dropdown population + example renderer."""
from __future__ import annotations

from typing import Any, List, Tuple, Optional

import gradio as gr
import ast

from .state import app_state
from .utils import (
    get_unique_values_for_dropdowns,
    get_example_data,
    format_examples_display,
    search_clusters_by_text,
)

__all__: List[str] = [
    "get_dropdown_choices",
    "update_example_dropdowns",
    "view_examples",
    "get_filter_options",
    "update_filter_dropdowns",
]


# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------

def get_dropdown_choices(selected_models: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
    if app_state["clustered_df"] is None:
        return [], [], []

    choices = get_unique_values_for_dropdowns(app_state["clustered_df"])
    prompts = ["All Prompts"] + choices["prompts"]
    # If a sidebar selection is provided, filter models to that subset (ignoring the pseudo 'all')
    if selected_models:
        subset = [m for m in choices["models"] if m in [sm for sm in selected_models if sm != "all"]]
        models = ["All Models"] + (subset if subset else choices["models"])  # fallback to all available if subset empty
    else:
        models = ["All Models"] + choices["models"]
    properties = ["All Clusters"] + choices["properties"]
    return prompts, models, properties


def update_example_dropdowns(selected_models: Optional[List[str]] = None) -> Tuple[Any, Any, Any]:
    prompts, models, properties = get_dropdown_choices(selected_models)
    # If exactly one concrete model selected in sidebar, preselect it; else default to All Models
    preselect_model = "All Models"
    if selected_models:
        concrete = [m for m in selected_models if m != "all"]
        if len(concrete) == 1 and concrete[0] in models:
            preselect_model = concrete[0]
    return (
        gr.update(choices=prompts, value="All Prompts" if prompts else None),
        gr.update(choices=models, value=(preselect_model if models else None)),
        gr.update(choices=properties, value="All Clusters" if properties else None),
    )


# ---------------------------------------------------------------------------
# Example viewer
# ---------------------------------------------------------------------------

def view_examples(
    selected_prompt: str,
    selected_model: str,
    selected_property: str,
    max_examples: int = 5,
    use_accordion: bool = True,
    pretty_print_dicts: bool = True,
    search_term: str = "",
    show_unexpected_behavior: bool = False,
    selected_models_sidebar: Optional[List[str]] = None,
    selected_tags_sidebar: Optional[List[str]] = None,
) -> str:
    if app_state["clustered_df"] is None:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ Please load data first "
            "using the 'Load Data' tab</p>"
        )

    # Apply search filter first if search term is provided
    df = app_state["clustered_df"]

    # Apply sidebar-selected model filter if provided (ignoring pseudo 'all') before dropdown filters
    if selected_models_sidebar:
        concrete = [m for m in selected_models_sidebar if m != "all"]
        if concrete:
            df = df[df["model"].isin(concrete)]
            if df.empty:
                return "<p style='color: #e74c3c; padding: 20px;'>❌ No examples for the selected model subset.</p>"
    if search_term and isinstance(search_term, str) and search_term.strip():
        df = search_clusters_by_text(df, search_term.strip(), 'all')
        if df.empty:
            return f"<p style='color: #e74c3c; padding: 20px;'>❌ No clusters found matching '{search_term}'</p>"

    # Optional tags filter (sidebar): include rows whose first meta value is in selected tags
    if selected_tags_sidebar and len(selected_tags_sidebar) > 0 and 'meta' in df.columns:
        def _parse_meta(obj: Any) -> Any:
            if isinstance(obj, str):
                try:
                    return ast.literal_eval(obj)
                except Exception:
                    return obj
            return obj

        def _first_val(obj: Any) -> Any:
            if obj is None:
                return None
            obj = _parse_meta(obj)
            if isinstance(obj, dict):
                for _, v in obj.items():
                    return v
                return None
            if isinstance(obj, (list, tuple)):
                return obj[0] if len(obj) > 0 else None
            return obj

        parsed_meta = df['meta'].apply(_parse_meta)
        non_null_parsed = [m for m in parsed_meta.tolist() if m is not None]
        all_empty_dicts = (
            len(non_null_parsed) > 0 and all(isinstance(m, dict) and len(m) == 0 for m in non_null_parsed)
        )

        if not all_empty_dicts:
            allowed = set(map(str, selected_tags_sidebar))
            df = df[df['meta'].apply(_first_val).astype(str).isin(allowed)]
        if df.empty:
            return "<p style='color: #e74c3c; padding: 20px;'>❌ No examples found for selected tags</p>"

    examples = get_example_data(
        df,
        selected_prompt if selected_prompt != "All Prompts" else None,
        selected_model if selected_model != "All Models" else None,
        selected_property if selected_property != "All Clusters" else None,
        max_examples,
        show_unexpected_behavior=show_unexpected_behavior,
        randomize=(
            (selected_prompt == "All Prompts") and
            (selected_model == "All Models") and
            (selected_property == "All Clusters") and
            (not search_term or not str(search_term).strip())
        ),
    )

    return format_examples_display(
        examples,
        selected_prompt,
        selected_model,
        selected_property,
        use_accordion=use_accordion,
        pretty_print_dicts=pretty_print_dicts,
    )


# ---------------------------------------------------------------------------
# Filter dropdown helpers for frequency comparison
# ---------------------------------------------------------------------------

def get_filter_options() -> Tuple[List[str], List[str]]:
    if not app_state["model_stats"]:
        return ["All Models"], ["All Metrics"]

    available_models = ["All Models"] + list(app_state["model_stats"].keys())

    quality_metrics = set()
    for model_data in app_state["model_stats"].values():
        clusters = model_data.get("fine", []) + model_data.get("coarse", [])
        for cluster in clusters:
            quality_score = cluster.get("quality_score", {})
            if isinstance(quality_score, dict):
                quality_metrics.update(quality_score.keys())

    available_metrics = ["All Metrics"] + sorted(list(quality_metrics))

    return available_models, available_metrics


def update_filter_dropdowns() -> Tuple[Any, Any]:
    models, metrics = get_filter_options()
    return (
        gr.update(choices=models, value="All Models" if models else None),
        gr.update(choices=metrics, value="All Metrics" if metrics else None),
    ) 