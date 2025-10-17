"""
Utilities for the "Load Data" tab – loading pipeline results and scanning for
available experiment folders.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Loading utilities updated for FunctionalMetrics
# ---------------------------------------------------------------------------

from .state import app_state
from .data_loader import (
    load_pipeline_results,
    scan_for_result_subfolders,
    validate_results_directory,
)

# Metrics helpers
from .metrics_adapter import get_all_models

__all__ = [
    "load_data",
    "get_available_experiments",
    "get_experiment_choices",
    "refresh_experiment_dropdown",
    "load_experiment_data",
]


def load_data(results_dir: str, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> Tuple[str, str, str]:
    """Load pipeline results from *results_dir* and update the shared *app_state*.

    Returns a tuple of (summary_markdown, models_info_markdown, models_checkbox_update).
    """
    # Set loading flag to prevent redundant overview updates during data loading
    # (Temporarily disabled for debugging)
    # app_state["is_loading_data"] = True
    
    try:
        # 1. Validate directory structure
        progress(0.05, "Validating results directory…")
        is_valid, error_msg = validate_results_directory(results_dir)
        if not is_valid:
            # Clear loading flag on validation error (disabled for debugging)
            # app_state["is_loading_data"] = False
            return "", f"❌ Error: {error_msg}", ""

        # 2. Handle optional sub-folder selection (first match for now)
        progress(0.15, "Scanning for experiment subfolders…")
        subfolders = scan_for_result_subfolders(results_dir)
        final_dir = results_dir
        if subfolders and "." not in subfolders:
            final_dir = str(Path(results_dir) / subfolders[0])

        # 3. Load results into memory
        progress(0.35, "Loading pipeline results… This may take a moment")
        clustered_df, metrics, model_cluster_df, results_path = load_pipeline_results(final_dir)

        # 4. Stash in global state so other tabs can use it
        progress(0.6, "Preparing application state…")
        app_state["clustered_df"] = clustered_df
        app_state["metrics"] = metrics
        app_state["model_cluster_df"] = model_cluster_df
        # Temporary alias for legacy modules
        app_state["model_stats"] = metrics
        app_state["results_path"] = results_path
        app_state["available_models"] = get_all_models(metrics)
        app_state["current_results_dir"] = final_dir

        # 5. Compose status messages
        progress(0.8, "Finalizing summary…")
        n_models = len(metrics.get("model_cluster_scores", {}))
        n_properties = len(clustered_df)

        # Render as Markdown, not as a plain text block.
        summary = (
            "**Data Summary:**\n"
            f"- **Models:** {n_models}\n"
            f"- **Properties:** {n_properties:,}\n"
            f"- **Results Directory:** `{Path(final_dir).name}`"
        )
        # Check for cluster columns
        if ("cluster_id" in clustered_df.columns or 
            "property_description_cluster_id" in clustered_df.columns):
            cluster_id_col = ("cluster_id" if "cluster_id" in clustered_df.columns 
                          else "property_description_cluster_id")
            n_clusters = clustered_df[cluster_id_col].nunique()

        model_choices = app_state["available_models"]
        models_info = f"Available models: {', '.join(model_choices)}"

        # Gradio update object for the CheckboxGroup
        # Default: select all concrete models but leave the aggregate "all" unchecked
        selected_values = [m for m in model_choices if m != "all"]
        progress(1.0, "Dataset loaded")
        return summary, models_info, gr.update(choices=model_choices, value=selected_values)

    except Exception as e:
        # Clear loading flag on error (disabled for debugging)
        # app_state["is_loading_data"] = False
        error_msg = f"❌ Error loading results: {e}"
        return "", error_msg, gr.update(choices=[], value=[])


def get_available_experiments(base_dir: str) -> List[str]:
    """Return experiment sub-directories that contain the expected result files, sorted by modification time (most recent first)."""
    if not base_dir or not os.path.exists(base_dir):
        return []

    experiments: List[Tuple[str, float]] = []
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                if (
                    os.path.exists(os.path.join(item_path, "model_stats.json"))
                    or os.path.exists(os.path.join(item_path, "clustered_results_lightweight.jsonl"))
                ):
                    # Get modification time of the directory
                    mod_time = os.path.getmtime(item_path)
                    experiments.append((item, mod_time))
    except Exception as e:
        print(f"Error scanning experiments: {e}")

    # Sort by modification time (most recent first), then return just the names
    experiments.sort(key=lambda x: x[1], reverse=True)
    return [exp[0] for exp in experiments]


def get_experiment_choices() -> List[str]:
    """Return dropdown choices for the experiment selector."""
    from . import state
    if not state.BASE_RESULTS_DIR:
        return []
    experiments = get_available_experiments(state.BASE_RESULTS_DIR)
    return ["Select an experiment..."] + experiments


def refresh_experiment_dropdown() -> gr.update:
    """Gradio helper to refresh the experiment dropdown choices."""
    choices = get_experiment_choices()
    return gr.update(choices=choices, value="Select an experiment...")


def load_experiment_data(experiment_name: str) -> Tuple[str, str, str]:
    """Wrapper used by Gradio events to load a *selected* experiment."""
    from . import state
    if not state.BASE_RESULTS_DIR or experiment_name == "Select an experiment...":
        # Don't set loading flag for invalid selections
        return "", "Please select a valid experiment", gr.update(choices=[], value=[])

    # Set loading flag to prevent redundant overview updates during data loading
    # (Temporarily disabled for debugging)
    # app_state["is_loading_data"] = True
    
    experiment_path = os.path.join(state.BASE_RESULTS_DIR, experiment_name)
    print(f"🔍 Loading experiment: {experiment_name} from {experiment_path}")
    return load_data(experiment_path) 