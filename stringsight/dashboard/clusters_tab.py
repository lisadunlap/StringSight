"""Helpers for the **View Clusters** tab – both the interactive HTML and
fallback dataframe view."""
from typing import List

import pandas as pd
import ast

from .state import app_state
from .utils import (
    search_clusters_by_text,
    search_clusters_only,
    create_interactive_cluster_viewer,
    get_cluster_statistics,
    format_cluster_dataframe,
    extract_allowed_tag,
)

__all__ = ["view_clusters_interactive", "view_clusters_table"]


# ---------------------------------------------------------------------------
# Interactive HTML view
# ---------------------------------------------------------------------------

def view_clusters_interactive(
    selected_models: List[str],
    cluster_level: str,
    search_term: str = "",
    selected_tags: List[str] | None = None,
) -> str:
    if app_state["clustered_df"] is None:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ Please load data first "
            "using the 'Load Data' tab</p>"
        )

    df = app_state["clustered_df"].dropna(subset=["property_description"]).copy()

    # Apply search filter first
    if search_term and search_term.strip():
        df = search_clusters_only(df, search_term.strip(), cluster_level)

    # Optional tags filter – only keep rows whose meta resolves to an allowed tag in selected_tags
    if selected_tags and len(selected_tags) > 0 and 'meta' in df.columns:
        def _first_allowed_tag(obj):
            return extract_allowed_tag(obj)

        # Check if all meta are empty dicts (means no tags)
        def _parse_try(obj):
            if isinstance(obj, str):
                try:
                    return ast.literal_eval(obj)
                except Exception:
                    return obj
            return obj

        parsed_meta = df['meta'].apply(_parse_try)
        non_null_parsed = [m for m in parsed_meta.tolist() if m is not None]
        all_empty_dicts = (
            len(non_null_parsed) > 0 and all(isinstance(m, dict) and len(m) == 0 for m in non_null_parsed)
        )
        if not all_empty_dicts:
            allowed = set(map(str, selected_tags))
            df = df[df['meta'].apply(_first_allowed_tag).astype(str).isin(allowed)]

    # Exclude clusters with labels containing the word "Outliers" (affects clusters tab only)
    for _label_col in ("property_description_cluster_label", "cluster_label"):
        if _label_col in df.columns:
            df = df[~df[_label_col].astype(str).str.contains("Outliers", na=False)]

    # Build interactive viewer
    cluster_html = create_interactive_cluster_viewer(df, selected_models, cluster_level)

    # Statistics summary at the top
    stats = get_cluster_statistics(df, selected_models)
    if not stats:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ No cluster data available</p>"
        )

    # Get additional metrics from cluster_scores
    cluster_scores = app_state.get("metrics", {}).get("cluster_scores", {})
    
    # Calculate average quality scores and frequency
    total_frequency = 0
    quality_scores_list = []
    metric_names = set()
    
    for cluster_name, cluster_data in cluster_scores.items():
        # Skip clusters whose name contains "Outliers" in clusters tab stats
        if "Outliers" in str(cluster_name):
            continue
        total_frequency += cluster_data.get("proportion", 0) * 100
        quality_scores = cluster_data.get("quality", {})
        if quality_scores:
            quality_scores_list.extend(quality_scores.values())
            metric_names.update(quality_scores.keys())
    
    avg_quality = sum(quality_scores_list) / len(quality_scores_list) if quality_scores_list else 0
    metrics_suffix = f" ({', '.join(sorted(metric_names))})" if metric_names else ""

    stats_html = f"""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 15px 0;">Cluster Statistics</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px;">
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['total_properties']:,}</div>
                <div style="opacity: 0.9;">Total Properties</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['total_models']}</div>
                <div style="opacity: 0.9;">Models</div>
            </div>
    """

    if "clusters" in stats:
        stats_html += f"""
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['clusters']}</div>
                <div style="opacity: 0.9;">Behavior Clusters</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['avg_properties_per_cluster']:.1f}</div>
                <div style="opacity: 0.9;">Avg Properties/Cluster</div>
            </div>
        """

    stats_html += """
        </div>
    </div>
    """
    
    # No longer need note about coarse clusters
    if False:
        stats_html += """
        <div style="
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            ⚠️ <strong>Note:</strong> Coarse clusters not available in this dataset. Showing fine clusters instead.
        </div>
        """

    # Additional filter chips
    filter_info = ""
    if search_term and search_term.strip():
        filter_info += f"""
        <div style="
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            🔍 <strong>Search Filter:</strong> "{search_term}"
        </div>
        """

    if selected_models:
        filter_info += f"""
        <div style="
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            <strong>Selected Models:</strong> {', '.join(selected_models)}
        </div>
        """

    if selected_tags and len(selected_tags) > 0:
        filter_info += f"""
        <div style="
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            🏷️ <strong>Tag Filter:</strong> {', '.join(selected_tags)}
        </div>
        """

    return stats_html + filter_info + cluster_html


# ---------------------------------------------------------------------------
# Dataframe fallback view
# ---------------------------------------------------------------------------

def view_clusters_table(
    selected_models: List[str],
    cluster_level: str,
    search_term: str = "",
) -> pd.DataFrame:
    if app_state["clustered_df"] is None:
        return pd.DataFrame({"Message": ["Please load data first using the 'Load Data' tab"]})

    df = app_state["clustered_df"].copy()

    if search_term and search_term.strip():
        df = search_clusters_only(df, search_term.strip(), cluster_level)

    # Exclude clusters with labels containing the word "Outliers" (affects clusters tab only)
    for _label_col in ("property_description_cluster_label", "cluster_label"):
        if _label_col in df.columns:
            df = df[~df[_label_col].astype(str).str.contains("Outliers", na=False)]

    formatted_df = format_cluster_dataframe(df, selected_models, cluster_level)

    if formatted_df.empty:
        if search_term and search_term.strip():
            return pd.DataFrame({"Message": [f"No results found for search term '{search_term}'. Try a different search term."]})
        elif selected_models:
            available_models = df["model"].unique().tolist() if "model" in df.columns else []
            return pd.DataFrame({"Message": [
                f"No data found for selected models: {', '.join(selected_models)}. "
                f"Available models: {', '.join(available_models)}"
            ]})
        else:
            return pd.DataFrame({"Message": [
                "No data available. Please check your data files and try reloading."
            ]})

    return formatted_df 