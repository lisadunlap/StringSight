"""
Utility functions for Gradio pipeline results app.

This module contains common utility functions used across different components.
"""

import numpy as np
import pandas as pd
import json
import markdown
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import html
import ast
import re

# Conversation rendering helpers are now in a dedicated module for clarity
from . import conversation_display as _convdisp
from .conversation_display import (
    convert_to_openai_format,
    display_openai_conversation_html,
    pretty_print_embedded_dicts,
)

# NEW IMPLEMENTATION ---------------------------------------------------
from .metrics_adapter import get_model_clusters, get_all_models

# ---------------------------------------------------------------------------
# NEW helper utilities for FunctionalMetrics format
# ---------------------------------------------------------------------------


# Allowed cluster tags across the entire app
ALLOWED_TAGS: set[str] = {
    "Positive",
    "Negative (critical)",
    "Negative (non-critical)",
    "Style",
}


def _is_nan(value: Any) -> bool:
    try:
        return isinstance(value, float) and np.isnan(value)
    except Exception:
        return False


def _parse_meta_obj(meta_obj: Any) -> Any:
    """Normalize and parse metadata objects.

    - Parse stringified containers (dict/list)
    - Treat NaN-like values as None
    - Return as-is otherwise
    """
    if meta_obj is None:
        return None
    if _is_nan(meta_obj):
        return None
    if isinstance(meta_obj, str):
        s = meta_obj.strip()
        if s in ("", "None", "N/A", "null"):
            return None
        try:
            return ast.literal_eval(meta_obj)
        except Exception:
            return meta_obj
    return meta_obj


def extract_allowed_tag(meta_obj: Any) -> Optional[str]:
    """Extract the first tag value from metadata and return it only if in ALLOWED_TAGS.

    Rules:
    - If metadata is missing, NaN, or all empty dicts, return None
    - If the extracted value is not in ALLOWED_TAGS, return None
    """
    meta_obj = _parse_meta_obj(meta_obj)
    if meta_obj is None:
        return None
    if isinstance(meta_obj, dict):
        # Empty dict means no tag
        if len(meta_obj) == 0:
            return None
        for _, v in meta_obj.items():
            tag = str(v)
            return tag if tag in ALLOWED_TAGS else None
        return None
    if isinstance(meta_obj, (list, tuple)):
        if len(meta_obj) == 0:
            return None
        tag = str(meta_obj[0])
        return tag if tag in ALLOWED_TAGS else None
    # Scalar string/other
    tag = str(meta_obj)
    return tag if tag in ALLOWED_TAGS else None


def normalize_text_for_search(text: Any) -> str:
    """Lowercase and strip common Markdown/HTML formatting and punctuation for robust search.

    - Unwrap markdown links: [label](url) -> label
    - Remove inline code/backticks and strikethrough markers
    - Unwrap emphasis/bold/italics: *, **, _, __
    - Strip simple HTML tags
    - Remove all punctuation including commas, periods, quotes, etc.
    - Collapse whitespace
    """
    if text is None:
        return ""
    s = str(text)
    # Strip HTML tags first
    s = re.sub(r"<[^>]+>", " ", s)
    # Markdown links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)
    # Inline code `code` -> code
    s = re.sub(r"`([^`]*)`", r"\1", s)
    # Bold/italic wrappers (**text** | __text__ | *text* | _text_) -> text
    s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)
    s = re.sub(r"(\*|_)(.*?)\1", r"\2", s)
    # Strikethrough ~~text~~ -> text
    s = re.sub(r"~~(.*?)~~", r"\1", s)
    # Remove remaining markdown emphasis chars/backticks/tilde
    s = re.sub(r"[*_`~]", "", s)
    # Remove all punctuation (including commas, periods, quotes, parentheses, etc.)
    s = re.sub(r"[^\w\s]", " ", s)
    # Normalize whitespace and lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def format_confidence_interval(ci: dict | None, decimals: int = 3) -> str:
    """Return a pretty string for a CI dict of the form {"lower": x, "upper": y}."""
    if not ci or not isinstance(ci, dict):
        return "N/A"
    lower, upper = ci.get("lower"), ci.get("upper")
    if lower is None or upper is None:
        return "N/A"
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def get_confidence_interval_width(ci: dict | None) -> float | None:
    """Return CI width (upper-lower) if possible."""
    if not ci or not isinstance(ci, dict):
        return None
    lower, upper = ci.get("lower"), ci.get("upper")
    if lower is None or upper is None:
        return None
    return upper - lower


def has_confidence_intervals(record: dict | None) -> bool:
    """Simple check whether any *_ci key with lower/upper exists in a metrics record."""
    if not record or not isinstance(record, dict):
        return False
    for k, v in record.items():
        if k.endswith("_ci") and isinstance(v, dict) and {"lower", "upper"}.issubset(v.keys()):
            return True
    return False


def extract_quality_score(quality_field: Any) -> float | None:
    """Given a quality field that may be a dict of metric values or a scalar, return its mean."""
    if quality_field is None:
        return None
    if isinstance(quality_field, (int, float)):
        return float(quality_field)
    if isinstance(quality_field, dict) and quality_field:
        return float(np.mean(list(quality_field.values())))
    return None

# ---------------------------------------------------------------------------
# UPDATED: get_top_clusters_for_model for FunctionalMetrics format
# ---------------------------------------------------------------------------


def get_top_clusters_for_model(metrics: Dict[str, Any], model_name: str, top_n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
    """Return the top N clusters (by salience) for a given model.

    Args:
        metrics: The FunctionalMetrics dictionary (3-file format) loaded via data_loader.
        model_name: Name of the model to inspect.
        top_n: Number of clusters to return.

    Returns:
        List of (cluster_name, cluster_dict) tuples sorted by descending proportion_delta.
    """
    clusters_dict = get_model_clusters(metrics, model_name)
    if not clusters_dict:
        return []
    
    # Filter out "No properties" clusters
    clusters_dict = {k: v for k, v in clusters_dict.items() if k != "No properties"}
    
    # Filter out "Outliers" cluster for overview tab
    clusters_dict = {k: v for k, v in clusters_dict.items() if "Outliers" not in k}
    
    sorted_items = sorted(
        clusters_dict.items(), key=lambda kv: kv[1].get("proportion_delta", 0), reverse=True
    )
    return sorted_items[:top_n]


def compute_model_rankings_new(metrics: Dict[str, Any]) -> List[tuple]:
    """Compute rankings of models based on mean salience (proportion_delta).

    Args:
        metrics: The FunctionalMetrics dict loaded by data_loader.

    Returns:
        List[Tuple[str, Dict[str, float]]]: sorted list of (model_name, summary_dict)
    """
    model_scores: Dict[str, Dict[str, float]] = {}
    for model in get_all_models(metrics):
        clusters = get_model_clusters(metrics, model)
        # Filter out "No properties" clusters
        clusters = {k: v for k, v in clusters.items() if k != "No properties"}
        if not clusters:
            continue
        saliences = [c.get("proportion_delta", 0.0) for c in clusters.values()]
        model_scores[model] = {
            "avg_salience": float(np.mean(saliences)),
            "median_salience": float(np.median(saliences)),
            "num_clusters": len(saliences),
            "top_salience": float(max(saliences)),
            "std_salience": float(np.std(saliences)),
        }
    return sorted(model_scores.items(), key=lambda x: x[1]["avg_salience"], reverse=True)


def create_model_summary_card_new(
    model_name: str,
    metrics: Dict[str, Any],
    top_n: int = 3,
    score_significant_only: bool = False,
    quality_significant_only: bool = False,
    sort_by: str = "quality_asc",
    min_cluster_size: int = 1,
    selected_tags: Optional[List[str]] = None,
) -> str:
    """Generate a **styled** HTML summary card for a single model.

    The new implementation recreates the legacy card design the user prefers:
    • Card header with battle count
    • Each cluster displayed as a vertically-spaced block (NOT a table)
    • Frequency, distinctiveness factor and CI inline; quality score right-aligned
    """

    clusters_dict = get_model_clusters(metrics, model_name)
    if not clusters_dict:
        return f"<div style='padding:20px'>No cluster data for {model_name}</div>"

    # Filter out "No properties" clusters
    clusters_dict = {k: v for k, v in clusters_dict.items() if k != "No properties"}

    # Filter out "Outliers" cluster for overview tab
    clusters_dict = {k: v for k, v in clusters_dict.items() if "Outliers" not in k}

    # Helper: extract allowed tag from metadata
    def _extract_tag(meta_obj: Any) -> Optional[str]:
        return extract_allowed_tag(meta_obj)

    # Helper: sanitize label that might include dict-like suffixes
    def _sanitize_label(label: str) -> str:
        if not isinstance(label, str):
            return str(label)
        lbl = re.sub(r"\s*\(\s*\{[^}]*\}\s*\)\s*$", "", label)
        lbl = re.sub(r"\s*\{[^}]*\}\s*$", "", lbl)
        lbl = re.sub(r"\s*\(\s*[^(){}:]+\s*:\s*[^(){}]+\)\s*$", "", lbl)
        return lbl.strip()

    # Build consistent colors for tags for this card
    # Fixed mapping for known tags
    tag_to_color: Dict[str, str] = {
        "Style": "#9467bd",  # purple
        "Positive": "#28a745",  # green
        "Negative (non-critical)": "#ff7f0e",  # orange
        "Negative (critical)": "#dc3545",  # red
    }
    unique_tags: List[str] = []
    label_to_tag: Dict[str, str] = {}
    # Detect "all empty dicts" across metadata
    cluster_meta_values: List[Any] = []
    for c in clusters_dict.values():
        meta_obj = c.get("metadata") if isinstance(c, dict) else None
        meta_obj = _parse_meta_obj(meta_obj)
        cluster_meta_values.append(meta_obj)
    non_null_meta = [m for m in cluster_meta_values if m is not None]
    all_meta_empty_dicts = (
        len(non_null_meta) > 0 and all(isinstance(m, dict) and len(m) == 0 for m in non_null_meta)
    )
    if not all_meta_empty_dicts:
        for c in clusters_dict.values():
            tag_val = _extract_tag(c.get("metadata")) if isinstance(c, dict) else None
            if tag_val and tag_val not in unique_tags:
                unique_tags.append(tag_val)
        # tag_to_color already contains all allowed tags with fixed colors

    # Filter clusters ----------------------------------------------------
    all_clusters = [c for c in clusters_dict.values() if c.get("size", 0) >= min_cluster_size]

    # Optional: filter clusters by sidebar-selected tags
    if selected_tags:
        def _cluster_tag(c: dict) -> Optional[str]:
            return _extract_tag(c.get("metadata")) if isinstance(c, dict) else None
        allowed = set(map(str, selected_tags))
        all_clusters = [c for c in all_clusters if (t := _cluster_tag(c)) and str(t) in allowed]

    if score_significant_only:
        if model_name == "all":
            # For "all" model, we don't have proportion_delta_significant, so skip this filter
            pass
        else:
            all_clusters = [c for c in all_clusters if c.get("proportion_delta_significant", False)]
    # Only apply quality significance filter if quality metrics actually exist
    if quality_significant_only:
        # Check if we have any quality metrics in the data
        from .overview_tab import has_quality_metrics
        if has_quality_metrics():
            all_clusters = [c for c in all_clusters if any(c.get("quality_delta_significant", {}).values())]
        # If no quality metrics exist, ignore quality_significant_only filter

    if not all_clusters:
        return f"<div style='padding:20px'>No clusters pass filters for {model_name}</div>"

    # Count significant properties ---------------------------------------
    significant_frequency_count = 0
    significant_quality_count = 0
    
    for cluster in clusters_dict.values():
        if cluster.get("size", 0) >= min_cluster_size:
            # Count frequency significance
            if model_name != "all" and cluster.get("proportion_delta_significant", False):
                significant_frequency_count += 1
            
            # Count quality significance (sum across all metrics)
            quality_delta_significant = cluster.get("quality_delta_significant", {})
            significant_quality_count += sum(quality_delta_significant.values())

    # Sort ---------------------------------------------------------------
    def _mean_quality(c: dict[str, Any]) -> float:
        vals = list(c.get("quality", {}).values())
        return float(np.mean(vals)) if vals else 0.0

    # Build sort key map based on whether quality metrics are available
    sort_key_map = {
        "frequency_desc": (lambda c: c.get("proportion", 0), True),
        "frequency_asc": (lambda c: c.get("proportion", 0), False),
        "salience_desc": (lambda c: c.get("proportion_delta", 0) if model_name != "all" else c.get("proportion", 0), True),
        "salience_asc": (lambda c: c.get("proportion_delta", 0) if model_name != "all" else c.get("proportion", 0), False),
    }
    
    # Only add quality-based sorting if quality metrics exist
    from .overview_tab import has_quality_metrics
    if has_quality_metrics():
        sort_key_map.update({
            "quality_asc": (_mean_quality, False),
            "quality_desc": (_mean_quality, True),
        })
    
    # If sort_by requests quality sorting but no quality metrics exist, fall back to salience
    if sort_by in ["quality_asc", "quality_desc"] and not has_quality_metrics():
        sort_by = "salience_desc"

    key_fn, reverse = sort_key_map.get(sort_by, (lambda c: c.get("proportion_delta", 0) if model_name != "all" else c.get("proportion", 0), True))
    sorted_clusters = sorted(all_clusters, key=key_fn, reverse=reverse)[:top_n]

    # Determine total conversations for this model ----------------
    if model_name == "all":
        # For "all" model, sum the individual model totals to avoid double-counting
        model_scores = metrics.get("model_scores", {})
        total_battles = sum(model_data.get("size", 0) for model_data in model_scores.values())
    else:
        model_scores_entry = metrics.get("model_scores", {}).get(model_name, {})
        total_battles = model_scores_entry.get("size")
        if total_battles is None:
            # Fallback: deduplicate example IDs across clusters
            total_battles = sum(c.get("size", 0) for c in clusters_dict.values())

    # Card header --------------------------------------------------------
    display_model_name = ("All Models" if str(model_name).lower() == "all" else model_name)
    html_parts: list[str] = [f"""
    <div style="padding: 12px 8px; margin-bottom: 12px; border-bottom: 1px solid #e6e8eb;">
      <h3 style="margin-top:0; font-size: 18px;">{html.escape(display_model_name)}</h3>
      <p style="margin: 4px 0 8px 0; color:#555; font-size:13px;">
        {total_battles} battles · Top clusters by frequency
      </p>
      <p style="margin: 0 0 12px 0; color:#666; font-size:12px;">
        {significant_frequency_count} significant frequency properties · {significant_quality_count} significant quality properties
      </p>
    """]

    # Cluster blocks -----------------------------------------------------
    for i, cluster in enumerate(sorted_clusters):
        raw_name = next(k for k, v in clusters_dict.items() if v is cluster)
        # Do not pre-escape here; markdown renderer handles escaping. Pre-escaping causes
        # entities like &#x27; to render literally due to double-escaping.
        name = _sanitize_label(raw_name)
        prop = cluster.get("proportion", 0)
        freq_pct = prop * 100
        size = cluster.get("size", 0)

        # Tag badge from metrics metadata (no DataFrame fallback)
        tag_val = _extract_tag(cluster.get("metadata"))
        if not tag_val:
            tag_val = label_to_tag.get(raw_name) or label_to_tag.get(_sanitize_label(raw_name))
        tag_badge_html = ""
        stripe_color = "#4c6ef5"
        if tag_val:
            color = tag_to_color.get(tag_val, '#4c6ef5')
            tag_badge_html = (
                f"<span style=\"display:inline-block; margin-left:8px; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; background:{color}12; color:{color}; border:1px solid {color}26;\">{html.escape(str(tag_val))}</span>"
            )
            stripe_color = color

        # Check significance flags
        is_proportion_significant = False
        if model_name != "all":
            is_proportion_significant = cluster.get("proportion_delta_significant", False)
        
        quality_delta_significant = cluster.get("quality_delta_significant", {})
        is_quality_significant = any(quality_delta_significant.values())

        # Create significance indicators
        significance_indicators = []
        if is_proportion_significant:
            significance_indicators.append('<span style="display:inline-block; padding:1px 6px; border-radius:999px; font-size:10px; font-weight:700; line-height:1; color:#cc6699; border:1px solid #cc669933; background:#cc669912;">F</span>')
        if is_quality_significant:
            significance_indicators.append('<span style="display:inline-block; padding:1px 6px; border-radius:999px; font-size:10px; font-weight:700; line-height:1; color:#007bff; border:1px solid #007bff33; background:#007bff12; margin-left:6px;">Q</span>')
        
        significance_html = " ".join(significance_indicators) if significance_indicators else ""

        # Distinctiveness / frequency delta display
        if model_name == "all":
            # For "all" model, proportion_delta doesn't make sense, so show proportion instead
            distinct_factor = prop
            distinct_text = f"{freq_pct:.1f}% of all conversations"
            freq_with_delta_text = f"{freq_pct:.1f}%"
        else:
            sal = cluster.get("proportion_delta", 0)
            distinct_factor = 1 + (sal / prop) if prop else 1
            # Show delta in percentage points instead of raw proportion
            sal_pct = sal * 100.0
            freq_with_delta_text = f"{freq_pct:.1f}% ({sal_pct:+.1f}%)"
            distinct_text = f"{freq_with_delta_text}"

        # Confidence interval (frequency based)
        ci = cluster.get("proportion_ci")
        ci_str = format_confidence_interval(ci) if ci else "N/A"

        # Quality display – show average score and delta per metric
        quality_scores = cluster.get("quality", {}) or {}
        quality_delta = cluster.get("quality_delta", {}) or {}
        quality_display_html = ""

        metric_names: list[str] = sorted(set(quality_scores.keys()) | set(quality_delta.keys()))
        if metric_names:
            parts: list[str] = []
            for metric_name in metric_names:
                score_val = quality_scores.get(metric_name)
                delta_val = quality_delta.get(metric_name)
                score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "N/A"
                if isinstance(delta_val, (int, float)):
                    # Use grey for values very close to zero
                    if abs(delta_val) < 0.001:
                        color = "#AAAAAA"
                    else:
                        color = "#28a745" if delta_val > 0 else "#dc3545"
                    parts.append(
                        f"<div>{metric_name}: {score_str} <span style=\"color:{color}; font-weight:500;\">({delta_val:+.3f})</span></div>"
                    )
                else:
                    parts.append(f"<div>{metric_name}: {score_str}</div>")
            quality_display_html = "".join(parts)
        else:
            quality_display_html = '<span style="color:#666;">No quality data</span>'

        # Get light color for this cluster
        cluster_color = get_light_color_for_cluster(name, i)

        html_parts.append(f"""
        <div style="background:#fbfcfe; border:1px solid #edf1f5; border-left: 3px solid {stripe_color}; padding: 10px 10px; margin: 10px 0; border-radius: 8px; box-shadow: 0 1px 2px rgba(16,24,40,0.06);">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap: 12px;">
            <div style="flex:1; min-width:0;">
              <div style="margin-bottom:4px; font-size:14px;">
                {(_convdisp._markdown(str(name), pretty_print_dicts=False).replace('<p>', '<span>').replace('</p>', '</span>'))}
              </div>
            </div>
            <div style="font-size:12px; font-weight:normal; white-space:nowrap; text-align:right;">
              {quality_display_html}
            </div>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px; gap: 12px;">
            <div style="font-size:12px; color:#555; display:flex; align-items:center; flex-wrap:wrap; gap:6px;">
              <span>{freq_with_delta_text} frequency ({size} out of {total_battles} total)</span>
            </div>
            <div style="text-align:right; display:flex; align-items:center; gap:8px;">{(tag_badge_html if tag_badge_html else '')}{significance_html}</div>
          </div>
        </div>
        """)

    # Close card div -----------------------------------------------------
    html_parts.append("</div>")

    return "\n".join(html_parts)


def format_cluster_dataframe(clustered_df: pd.DataFrame, 
                           selected_models: Optional[List[str]] = None,
                           cluster_level: str = 'fine') -> pd.DataFrame:
    """Format cluster DataFrame for display in Gradio."""
    df = clustered_df.copy()
    
    
    # Filter by models if specified
    if selected_models:
        df = df[df['model'].isin(selected_models)]
    
    # Select relevant columns based on cluster level using correct column names from pipeline
    if cluster_level == 'fine':
        id_col = 'property_description_cluster_id'
        label_col = 'property_description_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'cluster_id'
        alt_label_col = 'cluster_label'
    else:
        id_col = 'property_description_cluster_id'
        label_col = 'property_description_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'cluster_id'
        alt_label_col = 'cluster_label'
    
    # Try both naming patterns
    if id_col in df.columns and label_col in df.columns:
        # Use the expected naming pattern
        cols = ['question_id', 'model', 'property_description', id_col, label_col, 'score']
    elif alt_id_col in df.columns and alt_label_col in df.columns:
        # Use the alternative naming pattern
        cols = ['question_id', 'model', 'property_description', alt_id_col, alt_label_col, 'score']
    else:
        # Fall back to basic columns if cluster columns are missing
        cols = ['question_id', 'model', 'property_description', 'score']
    
    # Keep only existing columns
    available_cols = [col for col in cols if col in df.columns]
    df = df[available_cols]
    

    
    return df


def truncate_cluster_name(cluster_desc: str, max_length: int = 50) -> str:
    """Truncate cluster description to fit in table column."""
    if len(cluster_desc) <= max_length:
        return cluster_desc
    return cluster_desc[:max_length-3] + "..."

def create_frequency_comparison_table(model_stats: Dict[str, Any], 
                                     selected_models: List[str],
                                     cluster_level: str = "fine",  # Ignored – kept for backward-compat
                                     top_n: int = 50,
                                     selected_model: str | None = None,
                                     selected_quality_metric: str | None = None) -> pd.DataFrame:
    """Create a comparison table for the new FunctionalMetrics format.

    The old signature is kept (cluster_level arg is ignored) so that callers
    can be updated incrementally.
    """

    if not selected_models:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Collect per-model, per-cluster rows
    # ------------------------------------------------------------------
    all_rows: List[dict] = []
    for model in selected_models:
        model_clusters = get_model_clusters(model_stats, model)  # type: ignore[arg-type]
        if not model_clusters:
            continue

        # Optional filter by a single model after the fact
        if selected_model and model != selected_model:
            continue

        for cluster_name, cdata in model_clusters.items():
            # Filter out "No properties" clusters
            if cluster_name == "No properties":
                continue
                
            # Basic numbers
            freq_pct = cdata.get("proportion", 0.0) * 100.0
            prop_ci = cdata.get("proportion_ci")

            # Quality per metric dicts ------------------------------------------------
            quality_dict = cdata.get("quality", {}) or {}
            quality_ci_dict = cdata.get("quality_ci", {}) or {}

            # Significance flags
            sal_sig = bool(cdata.get("proportion_delta_significant", False))
            quality_sig_flags = cdata.get("quality_delta_significant", {}) or {}

            all_rows.append({
                "cluster": cluster_name,
                "model": model,
                "frequency": freq_pct,
                "proportion_ci": prop_ci,
                "quality": quality_dict,
                "quality_ci": quality_ci_dict,
                "score_significant": sal_sig,
                "quality_significant_any": any(quality_sig_flags.values()),
                "quality_significant_metric": quality_sig_flags.get(selected_quality_metric) if selected_quality_metric else None,
            })

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)

    # Aggregate frequency across models ----------------------------------
    freq_sum = df_all.groupby("cluster")["frequency"].sum().sort_values(ascending=False)
    top_clusters = freq_sum.head(top_n).index.tolist()

    df_top = df_all[df_all["cluster"].isin(top_clusters)].copy()

    table_rows: List[dict] = []
    for clu in top_clusters:
        subset = df_top[df_top["cluster"] == clu]
        avg_freq = subset["frequency"].mean()

        # Aggregate CI (mean of bounds)
        ci_lowers = [ci.get("lower") for ci in subset["proportion_ci"] if isinstance(ci, dict)]
        ci_uppers = [ci.get("upper") for ci in subset["proportion_ci"] if isinstance(ci, dict)]
        freq_ci = {
            "lower": float(np.mean(ci_lowers)) if ci_lowers else None,
            "upper": float(np.mean(ci_uppers)) if ci_uppers else None,
        } if ci_lowers and ci_uppers else None

        # Quality aggregation -----------------------------------------------------
        q_vals: List[float] = []
        q_ci_l: List[float] = []
        q_ci_u: List[float] = []
        quality_sig_any = False
        for _, row in subset.iterrows():
            q_dict = row["quality"]
            if selected_quality_metric:
                if selected_quality_metric in q_dict:
                    q_vals.append(q_dict[selected_quality_metric])
                ci_metric = row["quality_ci"].get(selected_quality_metric) if isinstance(row["quality_ci"], dict) else None
                if ci_metric:
                    q_ci_l.append(ci_metric.get("lower"))
                    q_ci_u.append(ci_metric.get("upper"))
                quality_sig_any = quality_sig_any or bool(row["quality_significant_metric"])
            else:
                q_vals.extend(q_dict.values())
                for ci in row["quality_ci"].values():
                    if isinstance(ci, dict):
                        q_ci_l.append(ci.get("lower"))
                        q_ci_u.append(ci.get("upper"))
                quality_sig_any = quality_sig_any or row["quality_significant_any"]

        quality_val = float(np.mean(q_vals)) if q_vals else None
        quality_ci = {
            "lower": float(np.mean(q_ci_l)),
            "upper": float(np.mean(q_ci_u)),
        } if q_ci_l and q_ci_u else None

        score_sig = subset["score_significant"].any()

        table_rows.append({
            "Cluster": clu,
            "Frequency (%)": f"{avg_freq:.1f}",
            "Freq CI": format_confidence_interval(freq_ci),
            "Quality": f"{quality_val:.3f}" if quality_val is not None else "N/A",
            "Quality CI": format_confidence_interval(quality_ci) if quality_ci else "N/A",
            "Score Significance": "Yes" if score_sig else "No",
            "Quality Significance": "Yes" if quality_sig_any else "No",
        })

    return pd.DataFrame(table_rows)


def create_frequency_comparison_plots(model_stats: Dict[str, Any], 
                                     selected_models: List[str],
                                     cluster_level: str = 'fine',
                                     top_n: int = 50,
                                     show_confidence_intervals: bool = False) -> Tuple[go.Figure, go.Figure]:
    """Create frequency comparison plots (matching frequencies_tab.py exactly)."""
    

    
    # Use the same data preparation logic as the table function
    # Collect all clusters across all models for the chart (exact copy from frequencies_tab.py)
    all_clusters_data = []
    for model_name, model_data in model_stats.items():
        if model_name not in selected_models:
            continue
            
        clusters = model_data.get(cluster_level, [])
        for cluster in clusters:
            # Filter out "No properties" clusters
            if cluster.get('property_description') == "No properties":
                continue
                
            # Get confidence intervals for quality scores if available
            quality_score_ci = cluster.get('quality_score_ci', {})
            has_quality_ci = bool(quality_score_ci)
            
            # Get distinctiveness score confidence intervals (correct structure)
            score_ci = cluster.get('score_ci', {})
            ci_lower = score_ci.get('lower') if score_ci else None
            ci_upper = score_ci.get('upper') if score_ci else None
            
            all_clusters_data.append({
                'property_description': cluster['property_description'],
                'model': model_name,
                'frequency': cluster.get('proportion', 0) * 100,  # Convert to percentage
                'size': cluster.get('size', 0),
                'cluster_size_global': cluster.get('cluster_size_global', 0),
                'has_ci': has_confidence_intervals(cluster),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'has_quality_ci': has_quality_ci
            })
    
    if not all_clusters_data:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
        
    clusters_df = pd.DataFrame(all_clusters_data)
    
    # Get all unique clusters for the chart
    all_unique_clusters = clusters_df['property_description'].unique()
    total_clusters = len(all_unique_clusters)
    
    # Show all clusters by default
    top_n_for_chart = min(top_n, total_clusters)
    
    # Calculate total frequency per cluster and get top clusters
    cluster_totals = clusters_df.groupby('property_description')['frequency'].sum().sort_values(ascending=False)
    top_clusters = cluster_totals.head(top_n_for_chart).index.tolist()
    
    # Get quality scores for the same clusters to sort by quality
    quality_data_for_sorting = []
    for model_name, model_data in model_stats.items():
        if model_name not in selected_models:
            continue
        clusters = model_data.get(cluster_level, [])
        for cluster in clusters:
            # Filter out "No properties" clusters
            if cluster.get('property_description') == "No properties":
                continue
                
            if cluster['property_description'] in top_clusters:
                quality_data_for_sorting.append({
                    'property_description': cluster['property_description'],
                    'quality_score': extract_quality_score(cluster.get('quality_score', 0))
                })
    
    # Calculate average quality score per cluster and sort
    if quality_data_for_sorting:
        quality_df_for_sorting = pd.DataFrame(quality_data_for_sorting)
        avg_quality_per_cluster = quality_df_for_sorting.groupby('property_description')['quality_score'].mean().sort_values(ascending=True)  # Low to high
        top_clusters = avg_quality_per_cluster.index.tolist()
        # Reverse the order so low quality appears at top of chart
        top_clusters = top_clusters[::-1]
    
    # Filter data to only include top clusters
    chart_data = clusters_df[clusters_df['property_description'].isin(top_clusters)]
    
    if chart_data.empty:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
    
    # Get unique models for colors
    models = chart_data['model'].unique()
    # Use a color palette that avoids yellow - using Set1 which has better contrast
    colors = px.colors.qualitative.Set1[:len(models)]
    
    # Create horizontal bar chart for frequencies
    fig = go.Figure()
    
    # Add a bar for each model
    for i, model in enumerate(models):
        model_data = chart_data[chart_data['model'] == model]
        
        # Sort by cluster order (same as top_clusters)
        model_data = model_data.set_index('property_description').reindex(top_clusters).reset_index()
        
        # Fill NaN values with 0 for missing clusters
        model_data['frequency'] = model_data['frequency'].fillna(0)
        model_data['has_ci'] = model_data['has_ci'].fillna(False)
        # For CI columns, replace NaN with None using where() instead of fillna(None)
        model_data['ci_lower'] = model_data['ci_lower'].where(pd.notna(model_data['ci_lower']), None)
        model_data['ci_upper'] = model_data['ci_upper'].where(pd.notna(model_data['ci_upper']), None)
        
        # Ensure frequency is numeric and non-negative
        model_data['frequency'] = pd.to_numeric(model_data['frequency'], errors='coerce').fillna(0)
        model_data['frequency'] = model_data['frequency'].clip(lower=0)
        

                
        # Remove any rows where property_description is NaN (these are clusters this model doesn't appear in)
        model_data = model_data.dropna(subset=['property_description'])
        
        # Get confidence intervals for error bars
        ci_lower = []
        ci_upper = []
        for _, row in model_data.iterrows():
            freq_value = row.get('frequency', 0)
            if (row.get('has_ci', False) and 
                pd.notna(row.get('ci_lower')) and 
                pd.notna(row.get('ci_upper')) and
                freq_value > 0):  # Only calculate CIs for non-zero frequencies
                
                # IMPORTANT: These are distinctiveness score CIs, not frequency CIs
                # The distinctiveness score measures how much more/less frequently 
                # a model exhibits this behavior compared to the median model
                # We can use this to estimate uncertainty in the frequency measurement
                distinctiveness_ci_width = row['ci_upper'] - row['ci_lower']
                
                # Convert to frequency uncertainty (approximate)
                # A wider distinctiveness CI suggests more uncertainty in the frequency
                freq_uncertainty = distinctiveness_ci_width * freq_value * 0.1
                ci_lower.append(max(0, freq_value - freq_uncertainty))
                ci_upper.append(freq_value + freq_uncertainty)
            else:
                ci_lower.append(None)
                ci_upper.append(None)
        

        
        fig.add_trace(go.Bar(
            y=model_data['property_description'],
            x=model_data['frequency'],
            name=model,
            orientation='h',
            marker_color=colors[i],
            error_x=dict(
                type='data',
                array=[u - l if u is not None and l is not None else None for l, u in zip(ci_lower, ci_upper)],
                arrayminus=[f - l if f is not None and l is not None else None for f, l in zip(model_data['frequency'], ci_lower)],
                visible=show_confidence_intervals,
                thickness=1,
                width=3,
                color='rgba(0,0,0,0.3)'
            ),
            hovertemplate='<b>%{y}</b><br>' +
                        f'Model: {model}<br>' +
                        'Frequency: %{x:.1f}%<br>' +
                        'CI: %{customdata[0]}<extra></extra>',
            customdata=[[
                format_confidence_interval({
                    'lower': l, 
                    'upper': u
                }) if l is not None and u is not None else "N/A"
                for l, u in zip(ci_lower, ci_upper)
            ]]
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Model Frequencies in Top {len(top_clusters)} Clusters",
        xaxis_title="Frequency (%)",
        yaxis_title="Cluster Description",
        barmode='group',  # Group bars side by side
        height=max(600, len(top_clusters) * 25),  # Adjust height based on number of clusters
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis to show truncated cluster names
    fig.update_yaxes(
        tickmode='array',
        ticktext=[truncate_cluster_name(desc, 60) for desc in top_clusters],
        tickvals=top_clusters
    )
    
    # Create quality score chart
    # Get quality scores for the same clusters (single score per cluster)
    quality_data = []
    quality_cis = []  # Add confidence intervals for quality scores
    
    for cluster_desc in top_clusters:
        # Get the first available quality score for this cluster
        for model_name, model_data in model_stats.items():
            clusters = model_data.get(cluster_level, [])
            for cluster in clusters:
                if cluster['property_description'] == cluster_desc:
                    quality_score = extract_quality_score(cluster.get('quality_score', 0))
                    quality_data.append({
                        'property_description': cluster_desc,
                        'quality_score': quality_score
                    })
                    
                    # Get quality score confidence intervals
                    quality_ci = cluster.get('quality_score_ci', {})
                    if isinstance(quality_ci, dict) and quality_ci:
                        # Get the first available quality CI
                        for score_key, ci_data in quality_ci.items():
                            if isinstance(ci_data, dict):
                                ci_lower = ci_data.get('lower')
                                ci_upper = ci_data.get('upper')
                                if ci_lower is not None and ci_upper is not None:
                                    quality_cis.append({
                                        'property_description': cluster_desc,
                                        'ci_lower': ci_lower,
                                        'ci_upper': ci_upper
                                    })
                                    break
                        else:
                            quality_cis.append({
                                'property_description': cluster_desc,
                                'ci_lower': None,
                                'ci_upper': None
                            })
                    else:
                        quality_cis.append({
                            'property_description': cluster_desc,
                            'ci_lower': None,
                            'ci_upper': None
                        })
                    break
            if any(q['property_description'] == cluster_desc for q in quality_data):
                break
    
    if quality_data:
        quality_df = pd.DataFrame(quality_data)
        quality_cis_df = pd.DataFrame(quality_cis) if quality_cis else None
        
        # Create quality score chart with single bars
        fig_quality = go.Figure()
        
        # Prepare confidence intervals for error bars
        ci_lower = []
        ci_upper = []
        for _, row in quality_df.iterrows():
            cluster_desc = row['property_description']
            if quality_cis_df is not None:
                ci_row = quality_cis_df[quality_cis_df['property_description'] == cluster_desc]
                if not ci_row.empty:
                    ci_lower.append(ci_row.iloc[0]['ci_lower'])
                    ci_upper.append(ci_row.iloc[0]['ci_upper'])
                else:
                    ci_lower.append(None)
                    ci_upper.append(None)
            else:
                ci_lower.append(None)
                ci_upper.append(None)
        
        # Add a single bar for each cluster
        fig_quality.add_trace(go.Bar(
            y=[truncate_cluster_name(desc, 60) for desc in quality_df['property_description']],
            x=quality_df['quality_score'],
            orientation='h',
            marker_color='lightblue',  # Single color for all bars
            name='Quality Score',
            showlegend=False,
            error_x=dict(
                type='data',
                array=[u - l if u is not None and l is not None else None for l, u in zip(ci_lower, ci_upper)],
                arrayminus=[q - l if q is not None and l is not None else None for q, l in zip(quality_df['quality_score'], ci_lower)],
                visible=show_confidence_intervals,
                thickness=1,
                width=3,
                color='rgba(0,0,0,0.3)'
            ),
            hovertemplate='<b>%{y}</b><br>' +
                        'Quality Score: %{x:.3f}<br>' +
                        'CI: %{customdata[0]}<extra></extra>',
            customdata=[[
                format_confidence_interval({
                    'lower': l, 
                    'upper': u
                }) if l is not None and u is not None else "N/A"
                for l, u in zip(ci_lower, ci_upper)
            ]]
        ))
        
        # Update layout
        fig_quality.update_layout(
            title=f"Quality Scores",
            xaxis_title="Quality Score",
            yaxis_title="",  # No y-axis title to save space
            height=max(600, len(top_clusters) * 25),  # Same height as main chart
            showlegend=False,
            yaxis=dict(showticklabels=False)  # Hide y-axis labels to save space
        )
    else:
        # Create empty quality figure
        fig_quality = go.Figure()
        fig_quality.add_annotation(text="No quality score data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return fig, fig_quality


def search_clusters_by_text(clustered_df: pd.DataFrame, 
                          search_term: str,
                          search_in: str = 'description') -> pd.DataFrame:
    """Search clusters by text in descriptions or other fields."""
    if not search_term:
        return clustered_df.head(100)  # Return first 100 if no search
    
    norm_term = normalize_text_for_search(search_term)
    
    if search_in == 'description':
        series = clustered_df['property_description'].astype(str).apply(normalize_text_for_search)
        mask = series.str.contains(norm_term, na=False, regex=False)
    elif search_in == 'model':
        series = clustered_df['model'].astype(str).apply(normalize_text_for_search)
        mask = series.str.contains(norm_term, na=False, regex=False)
    elif search_in == 'cluster_label':
        # Use correct column names from pipeline
        fine_label_col = 'property_description_cluster_label'
        coarse_label_col = 'property_description_cluster_label'
        # Initialize mask aligned to clustered_df index to avoid boolean indexer misalignment
        mask = pd.Series(False, index=clustered_df.index)
        
        if fine_label_col in clustered_df.columns:
            series = clustered_df[fine_label_col].astype(str).apply(normalize_text_for_search)
            mask = mask | series.str.contains(norm_term, na=False, regex=False)
        if coarse_label_col in clustered_df.columns:
            series = clustered_df[coarse_label_col].astype(str).apply(normalize_text_for_search)
            mask = mask | series.str.contains(norm_term, na=False, regex=False)
    else:
        # Search in all text columns using correct column names
        text_cols = ['property_description', 'model', 
                    'property_description_cluster_label', 
                    'property_description_cluster_label']
        # Initialize mask aligned to clustered_df index to avoid boolean indexer misalignment
        mask = pd.Series(False, index=clustered_df.index)
        for col in text_cols:
            if col in clustered_df.columns:
                series = clustered_df[col].astype(str).apply(normalize_text_for_search)
                mask = mask | series.str.contains(norm_term, na=False, regex=False)
    
    return clustered_df[mask].head(100) 


def search_clusters_only(clustered_df: pd.DataFrame, 
                       search_term: str,
                       cluster_level: str = 'fine') -> pd.DataFrame:
    """Search only over cluster labels, not individual property descriptions."""
    if not search_term:
        return clustered_df
    
    norm_term = normalize_text_for_search(search_term)
    
    # Use the correct column names based on cluster level
    if cluster_level == 'fine':
        label_col = 'property_description_cluster_label'
        alt_label_col = 'cluster_label'
    else:
        label_col = 'property_description_cluster_label'
        alt_label_col = 'cluster_label'
    
    # Try both naming patterns
    if label_col in clustered_df.columns:
        series = clustered_df[label_col].astype(str).apply(normalize_text_for_search)
        mask = series.str.contains(norm_term, na=False, regex=False)
    elif alt_label_col in clustered_df.columns:
        series = clustered_df[alt_label_col].astype(str).apply(normalize_text_for_search)
        mask = series.str.contains(norm_term, na=False, regex=False)
    else:
        # If neither column exists, return empty DataFrame
        return pd.DataFrame()
    
    return clustered_df[mask]


def create_interactive_cluster_viewer(clustered_df: pd.DataFrame, 
                                    selected_models: Optional[List[str]] = None,
                                    cluster_level: str = 'fine') -> str:
    """Create interactive cluster viewer HTML similar to Streamlit version."""
    if clustered_df.empty:
        return "<p>No cluster data available</p>"
    
    df = clustered_df.copy()
    
    # Filter by models if specified
    if selected_models:
        df = df[df['model'].isin(selected_models)]
    
    if df.empty:
        return f"<p>No data found for selected models: {', '.join(selected_models or [])}</p>"
    
    # Get cluster scores data for quality and frequency information
    from .state import app_state
    cluster_scores = app_state.get("metrics", {}).get("cluster_scores", {})
    
    # Use the actual column names from the pipeline output (matching Streamlit version)
    if cluster_level == 'fine':
        id_col = 'property_description_cluster_id'
        label_col = 'property_description_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'cluster_id'
        alt_label_col = 'cluster_label'
    else:
        id_col = 'property_description_cluster_id'  
        label_col = 'property_description_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'cluster_id'
        alt_label_col = 'cluster_label'
    
    # Track if we fall back from coarse to fine
    fell_back_to_fine = False
    
    # Check if required columns exist
    # Try both naming patterns
    if id_col in df.columns and label_col in df.columns:
        # Use the expected naming pattern
        pass
    elif alt_id_col in df.columns and alt_label_col in df.columns:
        # Use the alternative naming pattern
        id_col = alt_id_col
        label_col = alt_label_col
    else:
        # If coarse clusters are not available, try to fall back to fine clusters
        if cluster_level == 'coarse':
            # Check if fine clusters are available
            fine_id_col = 'property_description_cluster_id'
            fine_label_col = 'property_description_cluster_label'
            fine_alt_id_col = 'cluster_id'
            fine_alt_label_col = 'cluster_label'
            
            if (fine_id_col in df.columns and fine_label_col in df.columns) or (fine_alt_id_col in df.columns and fine_alt_label_col in df.columns):
                # Fall back to fine clusters
                if fine_id_col in df.columns and fine_label_col in df.columns:
                    id_col = fine_id_col
                    label_col = fine_label_col
                else:
                    id_col = fine_alt_id_col
                    label_col = fine_alt_label_col
                cluster_level = 'fine'  # Update the cluster level for display
                fell_back_to_fine = True
            else:
                # No cluster columns available at all
                available_cols = list(df.columns)
                return f"""
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
                    <h4>❌ Missing cluster columns in data</h4>
                    <p><strong>Expected:</strong> {id_col}, {label_col} OR {alt_id_col}, {alt_label_col}</p>
                    <p><strong>Available columns:</strong> {', '.join(available_cols)}</p>
                    <p>Please ensure your data contains clustering results from the StringSight pipeline.</p>
                </div>
                """
        else:
            # For fine clusters, show the original error
            available_cols = list(df.columns)
            return f"""
            <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
                <h4>❌ Missing {cluster_level} cluster columns in data</h4>
                <p><strong>Expected:</strong> {id_col}, {label_col} OR {alt_id_col}, {alt_label_col}</p>
                <p><strong>Available columns:</strong> {', '.join(available_cols)}</p>
                <p>Please ensure your data contains clustering results from the StringSight pipeline.</p>
            </div>
            """
    
    # Group by cluster to get cluster information
    try:

        # If meta column exists, propagate it into the aggregation so we can tag clusters
        agg_spec = {
            'property_description': ['count', lambda x: x.unique().tolist()],
            'model': lambda x: x.unique().tolist()
        }
        if 'meta' in df.columns:
            agg_spec['meta'] = lambda x: x.iloc[0]
        cluster_groups = df.groupby([id_col, label_col]).agg(agg_spec).reset_index()
        
        # Flatten column names
        flat_cols = [id_col, label_col, 'size', 'property_descriptions', 'models']
        if 'meta' in df.columns:
            flat_cols.append('meta')
        cluster_groups.columns = flat_cols
        
        # Sort by size (largest first)
        cluster_groups = cluster_groups.sort_values('size', ascending=False)
        
        # Filter out "No properties" clusters
        cluster_groups = cluster_groups[cluster_groups[label_col] != "No properties"]
        

        
    except Exception as e:
        return f"""
        <div style="padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px;">
            <h4>❌ Error processing cluster data</h4>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please check your data format and try again.</p>
        </div>
        """
    
    if len(cluster_groups) == 0:
        return """
        <div style="padding: 20px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px;">
            <h4>ℹ️ No clusters found</h4>
            <p>No clusters match your current filters. Try selecting different models or adjusting your search.</p>
        </div>
        """
    
    # Helper to extract first value from meta for display
    def _extract_tag_from_meta(meta_obj: Any) -> Optional[str]:
        return extract_allowed_tag(meta_obj)

    # Build a stable color map for tags (if any)
    tag_to_color: dict[str, str] = {
        "Style": "#9467bd",  # purple
        "Positive": "#28a745",  # green
        "Negative (non-critical)": "#ff7f0e",  # orange
        "Negative (critical)": "#dc3545",  # red
    }
    if 'meta' in cluster_groups.columns:
        # If all meta objects are empty dicts, treat as no tags
        meta_vals = cluster_groups['meta'].tolist()
        parsed_meta = [_parse_meta_obj(m) for m in meta_vals]
        non_null_parsed = [m for m in parsed_meta if m is not None]
        all_empty_dicts = (
            len(non_null_parsed) > 0 and all(isinstance(m, dict) and len(m) == 0 for m in non_null_parsed)
        )
        if not all_empty_dicts:
            unique_tags = [t for t in (_extract_tag_from_meta(m) for m in meta_vals) if t]
            unique_tags = list(dict.fromkeys(unique_tags))  # preserve order, dedupe
            # tag_to_color already contains all allowed tags with fixed colors
    
    # Helper to remove embedded dicts like "({'group': 'Positive'})" from labels
    def _sanitize_cluster_label(label: str) -> str:
        if not isinstance(label, str):
            return str(label)
        # Remove ( { ... } ) at end
        label = re.sub(r"\s*\(\s*\{[^}]*\}\s*\)\s*$", "", label)
        # Remove trailing { ... }
        label = re.sub(r"\s*\{[^}]*\}\s*$", "", label)
        # Remove simple (key: value) trailer
        label = re.sub(r"\s*\(\s*[^(){}:]+\s*:\s*[^(){}]+\)\s*$", "", label)
        return label.strip()
    
    # Create HTML
    page_html = f"""
    <div style="max-width: 1600px; margin: 0 auto;">
        <p style="color: #666; margin-bottom: 20px;">
            Click on clusters below to explore their property descriptions. To see specific examples, copy the property description and paste it into the search bar for the examples tab.
            Showing {len(cluster_groups)} clusters sorted by size.
        </p>
    """
    
    for i, row in cluster_groups.iterrows():
        cluster_id = row[id_col]
        cluster_label = row[label_col]
        cluster_size = row['size']
        property_descriptions = row['property_descriptions']
        models_in_cluster = row['models']
        # Tag if meta exists in grouped data
        tag_badge_html = ""
        tag_value = None
        if 'meta' in cluster_groups.columns:
            tag_value = _extract_tag_from_meta(row.get('meta'))
            if tag_value:
                color = tag_to_color.get(tag_value, '#4c6ef5')
                tag_badge_html = (
                    f"<span style=\"display:inline-block; margin-left:10px; padding:3px 8px; "
                    f"border-radius:12px; font-size:11px; font-weight:600; "
                    f"background:{color}1A; color:{color}; border:1px solid {color}33;\">"
                    f"{html.escape(str(tag_value))}</span>"
                )
        # Use sanitized label for display then render markdown (no extra <strong>)
        label_display = _sanitize_cluster_label(str(cluster_label))
        label_html = (
            _convdisp._markdown(str(label_display), pretty_print_dicts=False)
            .replace('<p>', '<span>')
            .replace('</p>', '</span>')
        )
    
        # Get quality and frequency information from cluster_scores
        cluster_metrics = cluster_scores.get(cluster_label, {})
        frequency_pct = cluster_metrics.get("proportion", 0) * 100 if cluster_metrics else 0
        quality_scores = cluster_metrics.get("quality", {})
        quality_delta = cluster_metrics.get("quality_delta", {})
        
        # Build per-metric header display: "metric: score (delta)"
        header_quality_html = "<span style=\"color:#666;\">No quality data</span>"
        if quality_scores or quality_delta:
            metric_names = sorted(set(quality_scores.keys()) | set(quality_delta.keys()))
            line_parts: list[str] = []
            for metric_name in metric_names:
                score_val = quality_scores.get(metric_name)
                delta_val = quality_delta.get(metric_name)
                score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "N/A"
                if isinstance(delta_val, (int, float)):
                    # Use grey for values very close to zero (same as overview tab)
                    if abs(delta_val) < 0.001:
                        color = "#AAAAAA"
                    else:
                        color = "#28a745" if delta_val >= 0 else "#dc3545"
                    line_parts.append(f"<div>{metric_name}: {score_str} <span style=\"color: {color}; font-weight:500;\">({delta_val:+.3f})</span></div>")
                else:
                    line_parts.append(f"<div>{metric_name}: {score_str}</div>")
            header_quality_html = "".join(line_parts)
        
        # Format quality scores for detailed view
        quality_html = ""
        if quality_scores:
            quality_parts = []
            for metric_name, score in quality_scores.items():
                # Use grey for values very close to zero (same as overview tab)
                if abs(score) < 0.001:
                    color = "#AAAAAA"
                else:
                    color = "#28a745" if score >= 0 else "#dc3545"
                quality_parts.append(f'<span style="color:{color}; font-weight:500;">{metric_name}: {score:.3f}</span>')
            quality_html = " | ".join(quality_parts)
        else:
            quality_html = '<span style="color:#666;">No quality data</span>'
        
        # Format quality delta (relative to average)
        quality_delta_html = ""
        if quality_delta:
            delta_parts = []
            for metric_name, delta in quality_delta.items():
                # Use grey for values very close to zero
                if abs(delta) < 0.001:
                    color = "#AAAAAA"
                else:
                    color = "#28a745" if delta > 0 else "#dc3545"
                sign = "+" if delta >= 0 else ""
                delta_parts.append(f'<span style="color:{color}; font-weight:500;">{metric_name}: {sign}{delta:.3f}</span>')
            quality_delta_html = " | ".join(delta_parts)
        else:
            quality_delta_html = '<span style="color:#666;">No delta data</span>'
        
        # Format header quality score with visual indicators
        header_quality_text = header_quality_html
        
        # Get light color for this cluster (matching overview style)
        cluster_color = get_light_color_for_cluster(cluster_label, i)

        # Build per-model frequencies for this cluster (replace models list)
        metrics_all = app_state.get("metrics", {})
        model_cluster_scores = metrics_all.get("model_cluster_scores", {})
        model_freq_items: list[str] = []
        for m in models_in_cluster:
            m_dict = model_cluster_scores.get(m, {})
            c_dict = m_dict.get(cluster_label, {}) if isinstance(m_dict, dict) else {}
            prop = c_dict.get("proportion")
            if isinstance(prop, (int, float)):
                model_freq_items.append(f"{html.escape(str(m))}: {prop * 100:.1f}%")
        model_freqs_html = " | ".join(model_freq_items) if model_freq_items else "N/A"
        
        # Create expandable cluster card with overview-style design
        page_html += f"""
        <details style="margin: 15px 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <summary style="
                padding: 15px; 
                background: {get_light_color_for_cluster(cluster_label, i)};
                color: #333; 
                cursor: pointer; 
                font-weight: 400;
                font-size: 16px;
                user-select: none;
                list-style: none;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #dee2e6;
            ">
                <div style="max-width: 80%;">
                    <div style="margin-bottom: 4px; font-size: 14px;">
                        {label_html}
                    </div>
                    <span style="font-size: 12px; color: #555; display:inline-flex; align-items:center;">
                        {frequency_pct:.1f}% frequency ({cluster_size} properties) · {len(models_in_cluster)} models
                        {tag_badge_html}
                    </span>
                </div>
                <div style="font-size: 12px; font-weight: normal; text-align: right;">
                    <div style="margin-bottom: 4px; line-height: 1.2;">{header_quality_html}</div>
                    <div style="color: #6c757d;">
                        {frequency_pct:.1f}% frequency
                    </div>
                </div>
            </summary>
            
            <div style="padding: 20px; background: #f8f9fa;">
                <div style="margin-bottom: 15px;">
                    <strong>Cluster ID:</strong> {cluster_id}<br>
                    <strong>Size:</strong> {cluster_size} properties<br>
                    <strong>Model Frequencies:</strong> {model_freqs_html}<br>
                </div>
                
                <h4 style="color: #333; margin: 15px 0 10px 0;">
                    Property Descriptions ({len(property_descriptions)})
                </h4>
                
                <div style="max-height: 300px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
        """
        
        # Display property descriptions
        for i, desc in enumerate(property_descriptions, 1):
            page_html += f"""
                    <div style="
                        padding: 8px; 
                        margin: 2px 0; 
                        background: #f8f9fa; 
                        border-left: 3px solid #667eea;
                        border-radius: 2px;
                    ">
                        <strong>{i}.</strong> {desc}
                    </div>
            """
        
        page_html += """
                </div>
            </div>
        </details>
        """
    
    page_html += "</div>"
    return page_html


def get_cluster_statistics(clustered_df: pd.DataFrame, 
                         selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get cluster statistics for display."""
    if clustered_df.empty:
        return {}
    
    df = clustered_df.copy()
    
    # Filter by models if specified
    if selected_models:
        df = df[df['model'].isin(selected_models)]
    
    stats = {
        'total_properties': len(df),
        'total_models': df['model'].nunique() if 'model' in df.columns else 0,
    }
    
    # Fine cluster statistics - try both naming patterns
    fine_id_col = 'property_description_cluster_id'
    alt_fine_id_col = 'cluster_id'
    
    if fine_id_col in df.columns:
        stats['clusters'] = df[fine_id_col].nunique()
        cluster_sizes = df.groupby(fine_id_col).size()
        stats['min_properties_per_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    elif alt_fine_id_col in df.columns:
        stats['clusters'] = df[alt_fine_id_col].nunique()
        cluster_sizes = df.groupby(alt_fine_id_col).size()
        stats['min_properties_per_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    
    # Coarse cluster statistics - try both naming patterns
    coarse_id_col = 'property_description_cluster_id'
    alt_coarse_id_col = 'cluster_id'
    
    if coarse_id_col in df.columns:
        stats['coarse_clusters'] = df[coarse_id_col].nunique()
        cluster_sizes = df.groupby(coarse_id_col).size()
        stats['min_properties_per_coarse_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_coarse_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_coarse_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    elif alt_coarse_id_col in df.columns:
        stats['coarse_clusters'] = df[alt_coarse_id_col].nunique()
        cluster_sizes = df.groupby(alt_coarse_id_col).size()
        stats['min_properties_per_coarse_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_coarse_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_coarse_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    
    return stats


def get_unique_values_for_dropdowns(clustered_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get unique values for dropdown menus."""
    if clustered_df.empty:
        return {'prompts': [], 'models': [], 'properties': [], 'tags': []}
    
    # Get unique values, handling missing columns gracefully
    prompts = []
    if 'prompt' in clustered_df.columns:
        unique_prompts = clustered_df['prompt'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    elif 'question' in clustered_df.columns:
        unique_prompts = clustered_df['question'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    elif 'input' in clustered_df.columns:
        unique_prompts = clustered_df['input'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    elif 'user_prompt' in clustered_df.columns:
        unique_prompts = clustered_df['user_prompt'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    
    # Handle both single model and side-by-side datasets
    models = []
    if 'model' in clustered_df.columns:
        models = sorted(clustered_df['model'].dropna().unique().tolist())
    elif 'model_a' in clustered_df.columns and 'model_b' in clustered_df.columns:
        models_a = clustered_df['model_a'].dropna().unique().tolist()
        models_b = clustered_df['model_b'].dropna().unique().tolist()
        all_models = set(models_a + models_b)
        models = sorted(list(all_models))
    
    # Use fine cluster labels instead of property descriptions - try both naming patterns
    properties = []
    fine_label_col = 'property_description_cluster_label'
    alt_fine_label_col = 'cluster_label'
    
    if fine_label_col in clustered_df.columns:
        unique_properties = clustered_df[fine_label_col].dropna().unique().tolist()
        unique_properties = [prop for prop in unique_properties if prop != "No properties"]
        properties = [prop[:100] + "..." if len(prop) > 100 else prop for prop in sorted(unique_properties)]
    elif alt_fine_label_col in clustered_df.columns:
        unique_properties = clustered_df[alt_fine_label_col].dropna().unique().tolist()
        unique_properties = [prop for prop in unique_properties if prop != "No properties"]
        properties = [prop[:100] + "..." if len(prop) > 100 else prop for prop in sorted(unique_properties)]
    elif 'property_description' in clustered_df.columns:
        unique_properties = clustered_df['property_description'].dropna().unique().tolist()
        unique_properties = [prop for prop in unique_properties if prop != "No properties"]
        properties = [prop[:100] + "..." if len(prop) > 100 else prop for prop in sorted(unique_properties)]
    
    # Tags from meta first value if available (only ALLOWED_TAGS)
    tags: List[str] = []
    if 'meta' in clustered_df.columns:
        def _first_allowed(obj: Any) -> Optional[str]:
            return extract_allowed_tag(obj)

        # Compute candidate tags and check for all-empty-dict case
        parsed_meta_series = clustered_df['meta'].apply(_parse_meta_obj)
        non_null_parsed = [m for m in parsed_meta_series.tolist() if m is not None]
        all_empty_dicts = (
            len(non_null_parsed) > 0 and all(isinstance(m, dict) and len(m) == 0 for m in non_null_parsed)
        )

        if not all_empty_dicts:
            tag_series = clustered_df['meta'].apply(_first_allowed)
            tags = sorted({str(t) for t in tag_series.dropna().tolist() if t is not None and str(t) in ALLOWED_TAGS})
    
    return {
        'prompts': prompts,
        'models': models, 
        'properties': properties,
        'tags': tags,
    }

# ---------------------------------------------------------------------------
# Example data extraction (restored)
# ---------------------------------------------------------------------------

def get_example_data(
    clustered_df: pd.DataFrame,
    selected_prompt: str | None = None,
    selected_model: str | None = None,
    selected_property: str | None = None,
    max_examples: int = 5,
    show_unexpected_behavior: bool = False,
    randomize: bool = False,
) -> List[Dict[str, Any]]:
    """Return a list of example rows filtered by prompt / model / property.

    This function was accidentally removed during a refactor; it is required by
    *examples_tab.py* and other parts of the UI.
    
    Args:
        clustered_df: DataFrame containing the clustered results data
        selected_prompt: Prompt to filter by (None for all)
        selected_model: Model to filter by (None for all)
        selected_property: Property description to filter by (None for all)
        max_examples: Maximum number of examples to return
        show_unexpected_behavior: If True, filter to only show unexpected behavior
        randomize: If True, sample randomly from the filtered set instead of taking the first rows
        
    Returns:
        List of example dictionaries with extracted data
    """

    if clustered_df.empty:
        return []

    df = clustered_df.copy()

    # Filter by unexpected behavior if requested
    if show_unexpected_behavior:
        if "unexpected_behavior" in df.columns:
            # Assuming True/1 means unexpected behavior
            df = df[df["unexpected_behavior"].isin([True, 1, "True", "true"])]
        else:
            # If no unexpected_behavior column, return empty (or could return all)
            return []

    # Filter by prompt
    if selected_prompt:
        prompt_cols = ["prompt", "question", "input", "user_prompt"]
        for col in prompt_cols:
            if col in df.columns:
                df = df[df[col].str.contains(selected_prompt, case=False, na=False)]
                break

    # Filter by model - handle both single model and side-by-side datasets
    if selected_model:
        if "model" in df.columns:
            # Single model datasets
            df = df[df["model"] == selected_model]
        elif "model_a" in df.columns and "model_b" in df.columns:
            # Side-by-side datasets - filter where either model_a or model_b matches
            df = df[(df["model_a"] == selected_model) | (df["model_b"] == selected_model)]

    # Filter by property
    if selected_property:
        property_cols = ["property_description", "cluster", "cluster_label", "property_description_cluster_label"]
        for col in property_cols:
            if col in df.columns:
                df = df[df[col].str.contains(selected_property, case=False, na=False)]
                break

    # Limit to max_examples (randomized if requested)
    if randomize:
        if len(df) > max_examples:
            df = df.sample(n=max_examples)
        else:
            df = df.sample(frac=1)
    else:
        df = df.head(max_examples)

    examples: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        prompt_val = next(
            (row.get(col) for col in ["prompt", "question", "input", "user_prompt"] if row.get(col) is not None),
            "N/A",
        )

        # Check if this is a side-by-side dataset
        is_side_by_side = ('model_a_response' in row and 'model_b_response' in row and 
                          row.get('model_a_response') is not None and row.get('model_b_response') is not None)
        
        if is_side_by_side:
            # For side-by-side datasets, store both responses separately
            response_val = "SIDE_BY_SIDE"  # Special marker
            model_val = f"{row.get('model_a', 'Model A')} vs {row.get('model_b', 'Model B')}"
            
            # Extract scores for side-by-side format: score_a and score_b columns
            scores_a = row.get('score_a', {})
            scores_b = row.get('score_b', {})
            
            if scores_a or scores_b:
                scores_info = []
                if scores_a:
                    scores_info.append(f"{row.get('model_a', 'Model A')}: {scores_a}")
                if scores_b:
                    scores_info.append(f"{row.get('model_b', 'Model B')}: {scores_b}")
                response_val = f"SIDE_BY_SIDE with scores: {' | '.join(scores_info)}"
        else:
            # For single response datasets, use the existing logic
            response_val = next(
                (
                    row.get(col)
                    for col in [
                        "model_response",
                        "model_a_response",
                        "model_b_response",
                        "responses",
                        "response",
                        "output",
                    ]
                    if row.get(col) is not None
                ),
                "N/A",
            )
            model_val = row.get("model", "N/A")

        # Try both naming patterns for cluster data
        cluster_id = row.get("property_description_cluster_id", row.get("cluster_id", "N/A"))
        cluster_label = row.get("property_description_cluster_label", row.get("cluster_label", "N/A"))
        cluster_id = row.get("property_description_cluster_id", row.get("cluster_id", "N/A"))
        cluster_label = row.get("property_description_cluster_label", row.get("cluster_label", "N/A"))

        example_dict = {
            "id": row.get("id", "N/A"),
            "model": model_val,
            "prompt": prompt_val,
            "response": response_val,
            "property_description": row.get("property_description", "N/A"),
            "score": row.get("score", "N/A"),
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "category": row.get("category", "N/A"),
            "reason": row.get("reason", "N/A"),
            "evidence": row.get("evidence", "N/A"),
            "meta": row.get("meta", None),
            "raw_response": row.get("raw_response", "N/A"),
            "contains_errors": row.get("contains_errors", "N/A"),
            "unexpected_behavior": row.get("unexpected_behavior", "N/A"),
        }
        
        # Add side-by-side specific fields if applicable
        if is_side_by_side:
            example_dict.update({
                "is_side_by_side": True,
                "model_a": row.get("model_a", "Model A"),
                "model_b": row.get("model_b", "Model B"),
                "model_a_response": row.get("model_a_response", "N/A"),
                "model_b_response": row.get("model_b_response", "N/A"),
                "winner": row.get("winner", None),
                "score_a": scores_a,
                "score_b": scores_b,
            })
        else:
            example_dict["is_side_by_side"] = False
            
        examples.append(example_dict)

    return examples


def format_examples_display(examples: List[Dict[str, Any]], 
                          selected_prompt: str = None,
                          selected_model: str = None,
                          selected_property: str = None,
                          use_accordion: bool = True,
                          pretty_print_dicts: bool = True) -> str:
    """Format examples for HTML display with proper conversation rendering.
    
    Args:
        examples: List of example dictionaries
        selected_prompt: Currently selected prompt filter
        selected_model: Currently selected model filter  
        selected_property: Currently selected property filter
        use_accordion: If True, group system and info messages in collapsible accordions
        pretty_print_dicts: If True, pretty-print embedded dictionaries
    
    Returns:
        HTML string for display
    """
    from .conversation_display import convert_to_openai_format, display_openai_conversation_html
    from .side_by_side_display import display_side_by_side_responses
    
    if not examples:
        return "<p style='color: #e74c3c; padding: 20px;'>No examples found matching the current filters.</p>"

    # Create filter summary
    filter_parts = []
    if selected_prompt and selected_prompt != "All Prompts":
        filter_parts.append(f"Prompt: {selected_prompt}")
    if selected_model and selected_model != "All Models":
        filter_parts.append(f"Model: {selected_model}")
    if selected_property and selected_property != "All Clusters":
        filter_parts.append(f"Cluster: {selected_property}")
    
    filter_summary = ""
    if filter_parts:
        filter_summary = f"""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2196f3;">
            <strong>🔍 Active Filters:</strong> {" • ".join(filter_parts)}
        </div>
        """

    html_out = f"""
    <div class="examples-container" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <style>
            /* Make JSON/code wrappers transparent (fall back to white when inline-styled) */
            .examples-container pre,
            .examples-container .highlight,
            .examples-container .codehilite,
            .examples-container p pre,
            .examples-container li pre,
            .examples-container div pre {{
                background: transparent !important;
            }}
            .examples-container code {{ background: transparent !important; }}
        </style>
        <h3 style="color: #333; margin-bottom: 15px;">📋 Examples ({len(examples)} found)</h3>
{filter_summary}
    """
    
    for i, example in enumerate(examples, 1):
        # Check if this is a side-by-side example
        if example.get('is_side_by_side', False):
            # Use side-by-side display for comparison datasets
            # Determine which model(s) exhibit the property using clustered_df model + cluster label
            exhibits_a = False
            exhibits_b = False
            from .state import app_state as _app_state
            clustered_df = _app_state.get("clustered_df")
            if clustered_df is not None and len(clustered_df) > 0:
                label_val = example.get("cluster_label") or example.get("property_description")
                if label_val is not None and "model" in clustered_df.columns:
                    # Prefer matching by row identifier to avoid counting other conversations
                    id_val = example.get("question_id") or example.get("id")
                    id_col_to_use = None
                    for cand in ("question_id", "id"):
                        if cand in clustered_df.columns:
                            id_col_to_use = cand
                            break
                    label_cols = [
                        "property_description_cluster_label",
                        "cluster_label",
                    ]
                    for col in label_cols:
                        if col in clustered_df.columns:
                            label_mask = (clustered_df[col] == label_val)
                            if id_col_to_use is not None and id_val not in [None, "", "N/A"]:
                                common_mask = (clustered_df[id_col_to_use] == id_val)
                                exhibits_a = bool(((clustered_df["model"] == example.get("model_a")) & label_mask & common_mask).any())
                                exhibits_b = bool(((clustered_df["model"] == example.get("model_b")) & label_mask & common_mask).any())
                            else:
                                # Without a stable row identifier, avoid over-attributing across the dataset
                                exhibits_a = False
                                exhibits_b = False
                            break

            conversation_html = display_side_by_side_responses(
                model_a=example['model_a'],
                model_b=example['model_b'],
                model_a_response=example['model_a_response'],
                model_b_response=example['model_b_response'],
                use_accordion=use_accordion,
                pretty_print_dicts=pretty_print_dicts,
                score=example['score'],
                scores_a=example.get('score_a'),
                scores_b=example.get('score_b'),
                winner=example.get('winner'),
                evidence=example.get('evidence'),
                exhibits_a=exhibits_a,
                exhibits_b=exhibits_b,
            )
        else:
            # Convert response to OpenAI format for proper display (single model)
            response_data = example['response']
            if response_data != 'N/A':
                openai_conversation = convert_to_openai_format(response_data)
                conversation_html = display_openai_conversation_html(
                    openai_conversation,
                    use_accordion=use_accordion,
                    pretty_print_dicts=pretty_print_dicts,
                    evidence=example.get('evidence')
                )
            else:
                conversation_html = "<p style='color: #dc3545; font-style: italic;'>No response data available</p>"
        
        # Compact cluster badge for header row
        cluster_badge = ""
        if example['cluster_label'] != 'N/A':
            cluster_badge = (
                f"<span style=\"display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; background:#eef2ff; color:#4f46e5; border:1px solid #e0e7ff;\">"
                f"Cluster: {html.escape(str(example['cluster_label']))}"
                f"</span>"
            )
        
        # Tag badge derived from meta (first value)
        tag_badge = ""
        tag_value = None
        meta_obj = example.get('meta')
        tag_value = extract_allowed_tag(meta_obj)
        if tag_value is not None and str(tag_value).strip() != "":
            tag_badge = (
                f"<span style=\"display:inline-block; padding:2px 8px; border-radius:999px; background:#faf5ff; color:#6d28d9; border:1px solid #ede9fe;\">"
                f"Tag: {html.escape(str(tag_value))}"
                f"</span>"
            )
        
        # Score badge removed - we now show the prominent score box instead
        score_badge = ""
        
        # Create short preview of prompt for summary
        prompt_preview = example['prompt'][:80] + "..." if len(example['prompt']) > 80 else example['prompt']
        
        # Create expandable example card
        # First example is expanded by default
        open_attr = "open" if i == 1 else ""
        
        # Build prominent score box section (positioned after cluster details)
        score_section_html = ""
        raw_score = example.get('score')
        numeric_score: float | None = None
        

        # Check for alternative score column names if main score is not found
        if raw_score is None or raw_score == "N/A":
            alt_score_keys = ['quality_score', 'rating', 'evaluation_score', 'metric_score']
            for alt_key in alt_score_keys:
                if alt_key in example and example[alt_key] not in [None, "N/A", ""]:
                    raw_score = example[alt_key]
                    break
        
        # Extract numeric score - should already be properly parsed from backend
        score_dict = None
        if raw_score is not None and raw_score != "N/A" and raw_score != "":
            if isinstance(raw_score, (int, float)):
                numeric_score = float(raw_score)
            elif isinstance(raw_score, dict):
                # Score is already a dictionary
                score_dict = raw_score
        
        # If we have a score dictionary, display all metrics
        if score_dict and isinstance(score_dict, dict):
            numeric_metrics = {k: v for k, v in score_dict.items() if isinstance(v, (int, float))}
            if numeric_metrics:
                # Set numeric_score to any value so the display logic triggers
                numeric_score = 1.0  # Just a placeholder to trigger display
        
        # Create compact score display for all examples with scores
        if numeric_score is not None or (score_dict and isinstance(score_dict, dict)):
            # Use blue colors for multi-metric display
            color_bg = '#e3f2fd'  # Light blue background
            color_fg = '#495057'
            border_color = '#2196f3'  # Blue border
            
            # Handle score dictionary display
            if score_dict and isinstance(score_dict, dict):
                numeric_metrics = {k: v for k, v in score_dict.items() if isinstance(v, (int, float))}
                if numeric_metrics:
                    metrics_html_parts = []
                    for k, v in numeric_metrics.items():
                        # Use blue color for all metrics
                        metrics_html_parts.append(f"""
                        <div style="display: inline-block; margin: 2px 8px;">
                            <span style="font-weight: 600; color: #6c757d;">{k.title()}:</span>
                            <span style="color: #0066cc; font-weight: 700; font-family: 'SF Mono', monospace;">
                                {v:.3f}
                            </span>
                        </div>""")
                    
                    score_section_html = f"""
                    <div style="
                        margin: 12px 0; 
                        padding: 8px 12px; 
                        background: {color_bg}; 
                        border: 1px solid {border_color}; 
                        border-radius: 6px; 
                        border-left: 3px solid {border_color};
                    ">
                        <div style="display: flex; flex-wrap: wrap; align-items: center; font-size: 12px;">
                            {''.join(metrics_html_parts)}
                        </div>
                    </div>
                    """
                else:
                    score_section_html = ""
            else:
                # Single numeric score display (fallback)
                score_section_html = f"""
                <div style="
                    margin: 12px 0; 
                    padding: 8px 12px; 
                    background: {color_bg}; 
                    border: 1px solid {border_color}; 
                    border-radius: 6px; 
                    border-left: 3px solid {border_color};
                ">
                    <div style="font-size: 13px; color: {color_fg};">
                        <span style="font-weight: 600;">Quality Score:</span>
                        <span style="color: #0066cc; font-weight: 700; margin-left: 8px; font-family: 'SF Mono', monospace;">
                            {numeric_score:.3f}
                        </span>
                    </div>
                </div>
                """
        else:
            # No score data available
            score_section_html = ""

        html_out += f"""
        <details {open_attr} style="border: 1px solid #dee2e6; border-radius: 8px; margin-bottom: 15px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <summary style="
                padding: 15px; 
                cursor: pointer; 
                font-weight: 600; 
                color: #495057; 
                background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 8px 8px 0 0; 
                border-bottom: 1px solid #dee2e6;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <span>
                    <span style="background: #6c757d; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 10px;">#{i}</span>
                    {prompt_preview}
                </span>
                <span style="font-size: 12px; color: #6c757d;">
                    {example['model']}{score_badge}
                </span>
            </summary>
            
            <div style="padding: 20px;">
                <!-- Compact metadata badges row -->
                <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom: 16px; font-size:12px; color:#6b7280;">
                    <span style="display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6; border:1px solid #e5e7eb;">ID: {html.escape(str(example['id']))}</span>
                    <span style="display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6; border:1px solid #e5e7eb;">Model: {html.escape(str(example['model']))}</span>
                    {tag_badge}
                    {(f'<span style="display:inline-block; padding:2px 8px; border-radius:999px; background:#ecfdf5; color:#047857; border:1px solid #bbf7d0;">Category: {html.escape(str(example["category"]))}</span>' if example["category"] not in [None, "N/A", "None", "", "null"] and str(example["category"]).strip() != "" else '')}
                </div>

                <!-- Collapsible info section for Cluster / Tag / Property / Reason / Evidence -->
                {(
                    f'''<details style="margin-bottom:16px; border:1px solid #e5e7eb; border-radius:8px; background:#f9fafb;">
                        <summary style="cursor:pointer; padding:12px; font-weight:600; color:#374151; border-radius:8px;">
                            {(
                                (str(example.get("cluster_label", ""))[:100] + "..." if len(str(example.get("cluster_label", ""))) > 100 else str(example.get("cluster_label", "")))
                                if example.get("cluster_label") not in [None, "N/A", "None", "", "null"] and str(example.get("cluster_label", "")).strip() != ""
                                else "Cluster"
                            )}
                        </summary>
                        <div style="padding:0 12px 12px 12px; border-top:1px solid #e5e7eb;">
                            {(f'<div style="margin-top:12px;"><strong style="color:#374151;">Cluster</strong><div style="color:#4b5563; margin-top:4px;">{_convdisp._markdown(str(example["cluster_label"]))}</div></div>' if example.get("cluster_label") not in [None, "N/A", "None", "", "null"] and str(example.get("cluster_label", "")).strip() != "" else '')}
                            {(f'<div style="margin-top:12px;"><strong style="color:#374151;">Property</strong><div style="color:#4b5563; margin-top:4px;">{_convdisp._markdown(str(example["property_description"]))}</div></div>' if example["property_description"] not in [None, "N/A", "None", "", "null"] and str(example["property_description"]).strip() != "" else '')}
                            {(f'<div style="margin-top:12px;"><strong style="color:#374151;">Evidence</strong><div style="color:#4b5563; margin-top:4px;">{_convdisp._markdown(str(example["evidence"]))}</div></div>' if example["evidence"] not in [None, "N/A", "None", "", "null"] and str(example["evidence"]).strip() != "" else '')}
                        </div>
                    </details>'''
                 ) if any([
                    example.get("cluster_label") not in [None, "N/A", "None", "", "null"] and str(example.get("cluster_label", "")).strip() != "",
                    example.get("property_description") not in [None, "N/A", "None", "", "null"] and str(example.get("property_description", "")).strip() != "",
                    example.get("reason") not in [None, "N/A", "None", "", "null"] and str(example.get("reason", "")).strip() != "",
                    example.get("evidence") not in [None, "N/A", "None", "", "null"] and str(example.get("evidence", "")).strip() != "",
                 ]) else ''}

                <!-- Prominent Quality Score Box -->
                {score_section_html}

                <div style="margin-bottom: 15px;">
                    <div style="
                        border-radius: 6px; 
                        font-size: 15px; 
                        line-height: 1.5;
                        max-height: 600px;
                        overflow-y: auto;
                        overflow-x: hidden;
                        border: 1px solid #e9ecef;
                        padding: 12px;
                        background: #fefefe;
                    ">
                        {conversation_html}
                    </div>
                </div>
            </div>
        </details>
        """
    
    html_out += "</div>"
    return html_out

# ---------------------------------------------------------------------------
# Legacy function aliases (backward compatibility)
# ---------------------------------------------------------------------------

def compute_model_rankings(*args, **kwargs):
    """Legacy alias → forwards to compute_model_rankings_new."""
    return compute_model_rankings_new(*args, **kwargs)


def create_model_summary_card(*args, **kwargs):
    """Legacy alias → forwards to create_model_summary_card_new."""
    return create_model_summary_card_new(*args, **kwargs) 


def get_total_clusters_count(metrics: Dict[str, Any]) -> int:
    """Get the total number of clusters from the metrics data."""
    cluster_scores = metrics.get("cluster_scores", {})
    # Filter out "No properties" clusters
    cluster_scores = {k: v for k, v in cluster_scores.items() if k != "No properties"}
    return len(cluster_scores)


def get_light_color_for_cluster(cluster_name: str, index: int) -> str:
    """Generate a light dusty blue background for cluster boxes.
    
    Returns a consistent light dusty blue color for all clusters.
    """
    return "#f0f4f8"  # Very light dusty blue 

__all__ = [
    "get_model_clusters",
    "get_all_models", 
    "get_all_clusters",
    "format_confidence_interval",
    "get_confidence_interval_width",
    "has_confidence_intervals",
    "extract_quality_score",
    "get_top_clusters_for_model",
    "compute_model_rankings_new",
    "create_model_summary_card_new",
    "format_cluster_dataframe",
    "truncate_cluster_name",
    "create_frequency_comparison_table",
    "create_frequency_comparison_plots",
    "search_clusters_by_text",
    "search_clusters_only",
    "create_interactive_cluster_viewer",
    "get_cluster_statistics",
    "get_unique_values_for_dropdowns",
    "get_example_data",
    "format_examples_display",
    "compute_model_rankings",
    "create_model_summary_card",
    "get_total_clusters_count",
] 
