"""Logic helpers for the **Overview** tab."""
from typing import List, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import gradio as gr
from .state import app_state
from .utils import compute_model_rankings_new, create_model_summary_card_new
from .plotting import create_model_dataframe

__all__ = ["create_overview", "create_model_quality_plot", "create_model_quality_table", "get_available_model_quality_metrics", "has_quality_metrics"]


def create_overview(
    selected_models: List[str],
    top_n: int,
    score_significant_only: bool = False,
    quality_significant_only: bool = False,
    sort_by: str = "quality_asc",
    min_cluster_size: int = 1,
    selected_tags: Optional[List[str]] = None,
    progress: Optional[gr.Progress] = None,
) -> str:
    """Return the HTML snippet that summarises model performance."""
    if not app_state["metrics"]:
        return "Please load data first using the 'Load Data' tab."

    if not selected_models:
        return "Please select at least one model to display."

    # 1. Compute global rankings and filter to selection
    if progress:
        progress(0.05, "Computing model rankings…")
    model_rankings = compute_model_rankings_new(app_state["metrics"])
    filtered_rankings = [
        (name, stats) for name, stats in model_rankings if name in selected_models
    ]

    # Sort so "all" appears first, then the rest by their rankings
    all_models = [(name, stats) for name, stats in filtered_rankings if name == "all"]
    other_models = [(name, stats) for name, stats in filtered_rankings if name != "all"]
    filtered_rankings = all_models + other_models

    if not filtered_rankings:
        return "No data available for selected models."

    # 2. Assemble HTML
    overview_html = """
    <div style="width: 100%; margin: 0;">
        <h2 style="margin: 0 0 8px 0; font-size: 20px; color:#2c3e50;">Common Properties per model</h2>
        <p style="color: #666; margin-bottom: 10px;">
            Top distinctive clusters where each model shows unique behavioural patterns.
            Frequency shows what percentage of a model's battles resulted in that behavioural pattern.
        </p>
        <details style="margin-bottom:25px;">
            <summary style="cursor:pointer; color:#4c6ef5; font-weight:500;">What do these tags and numbers mean?</summary>
            <div style="margin-top:12px; font-size:14px; line-height:1.5; color:#333;">
                <strong>Frequency Delta</strong><br>
                For each cluster we compute how often <em>this model</em> appears in that cluster compared with the average across all models.<br>
                • A positive value (e.g. <code>+0.15</code>) means the model hits the behaviour more often than average.<br>
                • A negative value (e.g. <code>-0.08</code>) means it appears less often.<br>
                <strong>Quality Delta</strong><br>
                The difference between the cluster's quality score(s) for this model and the model's <em>overall</em> quality baseline, shown for each individual metric (e.g., helpfulness, accuracy).<br>
                Positive values (green) indicate the model performs better than its average in that behaviour; negative values (red) indicate that it performs worse.<br>
                <strong>Significance Tags (F/Q)</strong><br>
                <span style="color: #888; font-size: 13px;">
                    Statistical significance is determined using a bootstrap procedure on the conversations to obtain 95% confidence intervals.
                </span><br>
                The <span style="display:inline-block; padding:1px 6px; border-radius:999px; font-size:10px; font-weight:700; line-height:1; color:#cc6699; border:1px solid #cc669933; background:#cc669912;">F</span> and <span style="display:inline-block; padding:1px 6px; border-radius:999px; font-size:10px; font-weight:700; line-height:1; color:#007bff; border:1px solid #007bff33; background:#007bff12;">Q</span> tags indicate <em>statistical significance</em> based on bootstraped confidence intervals:<br>
                • <strong>F</strong> (pink): The proportion delta is statistically significant (confidence interval doesn't include zero)<br>
                • <strong>Q</strong> (blue): At least one quality metric delta is statistically significant<br>
                These tags help identify which behavioral patterns are reliably different from the model's baseline performance.<br><br>
            </div>
        </details>
    """

    total_models = max(1, len(filtered_rankings))
    for idx, (model_name, _) in enumerate(filtered_rankings):
        if progress:
            progress(0.1 + 0.8 * (idx / total_models), f"Rendering overview for {model_name}…")
        card_html = create_model_summary_card_new(
            model_name,
            app_state["metrics"],
            # top_n etc.
            top_n,
            score_significant_only=score_significant_only,
            quality_significant_only=quality_significant_only,
            sort_by=sort_by,
            min_cluster_size=min_cluster_size,
            selected_tags=selected_tags,
        )
        overview_html += card_html

    overview_html += "</div>"
    if progress:
        progress(1.0, "Overview ready")
    return overview_html


def create_model_quality_plot(
    selected_models: List[str], 
    quality_metric: str = "helpfulness", 
) -> go.Figure:
    """Create a bar plot of model-level quality scores with confidence intervals."""
    if not app_state["metrics"]:
        return None
    
    if not selected_models:
        return None
    
    # Handle case where quality_metric is None (during data source switching)
    if quality_metric is None:
        available_metrics = get_available_model_quality_metrics()
        if not available_metrics:
            return None  # No quality metrics available
        quality_metric = available_metrics[0]  # Use first available metric
    
    # Get model scores from metrics
    model_scores = app_state["metrics"].get("model_scores", {})
    if not model_scores:
        return None
    
    # Create model dataframe
    model_df = create_model_dataframe(model_scores)
    
    if model_df.empty:
        return None
    
    # Filter to selected models
    model_df = model_df[model_df['model'].isin(selected_models)]
    
    if model_df.empty:
        return None
    
    # Find the actual ABSOLUTE quality column (not delta) that matches the requested metric
    # We want raw quality scores, not deltas from baseline
    quality_col = None
    for col in model_df.columns:
        if (col.startswith("quality_") and 
            not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant")) and
            "delta" not in col.lower()):  # Explicitly exclude any delta columns
            # Check if the quality metric name is contained in the column name (case insensitive)
            col_name = col.replace("quality_", "").lower()
            if quality_metric.lower() in col_name:
                quality_col = col
                break
    
    # If no match found, use the first available absolute quality column
    if not quality_col:
        available_quality_cols = [col for col in model_df.columns 
                                if col.startswith("quality_") 
                                and not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant"))
                                and "delta" not in col.lower()]  # Explicitly exclude delta columns
        if not available_quality_cols:
            return None
        quality_col = available_quality_cols[0]  # Use first available absolute quality metric
    
    # Ensure quality values are numeric
    model_df[quality_col] = pd.to_numeric(model_df[quality_col], errors='coerce')
    
    # Check if we have any valid quality data
    if model_df[quality_col].isna().all():
        return None
    
    # Sort models by quality score (descending - best scores first)
    model_df = model_df.sort_values(by=quality_col, ascending=False).reset_index(drop=True)
    
    # Extract a clean metric name for display
    metric_display_name = quality_col.replace("quality_", "").split("(")[0].strip()
    
    # Create the plot
    fig = go.Figure()
    
    # Prepare error bar data if requested and available
    error_y = None
    ci_lower_col = f"{quality_col}_ci_lower"
    ci_upper_col = f"{quality_col}_ci_upper"
    if ci_lower_col in model_df.columns and ci_upper_col in model_df.columns:
        # Calculate error bar values (distance from mean to upper/lower bounds)
        error_y_upper = model_df[ci_upper_col] - model_df[quality_col]
        error_y_lower = model_df[quality_col] - model_df[ci_lower_col]
        error_y = dict(
            type='data',
            symmetric=False,
            array=error_y_upper,
            arrayminus=error_y_lower,
            visible=True,
            color="rgba(52, 73, 94, 0.7)",
            thickness=2.5,
            width=5
        )
    
    # Create a beautiful color gradient for the bars
    colors = px.colors.qualitative.Set3[:len(model_df)]
    
    # Add the bar chart with improved styling
    fig.add_trace(go.Bar(
        x=model_df['model'],
        y=model_df[quality_col],
        error_y=error_y,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.8)', width=2),
            opacity=0.8
        ),
        name=f'{metric_display_name} Score',
        text=[f"{val:.2f}" for val in model_df[quality_col]],
        textposition='outside',
        textfont=dict(size=14, color='darkblue', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>' +
                     f'{metric_display_name}: %{{y:.3f}}<br>' +
                     (
                         f'CI: [{model_df[ci_lower_col][0]:.2f}, {model_df[ci_upper_col][0]:.2f}]<br>'
                     ) +
                     '<extra></extra>',
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="darkblue",
            font=dict(size=14, color="darkblue")
        )
    ))
    
    # Enhanced layout with auto-sizing and improved styling
    fig.update_layout(
        # Auto-sizing configuration
        autosize=True,
        
        # Enhanced axis styling
        xaxis=dict(
            # No title for x-axis
            title=None,
            tickangle=45,
            tickfont=dict(size=14, color='#34495e', family='Arial'),
            gridcolor='rgba(189, 195, 199, 0.3)',
            gridwidth=1,
            showgrid=True,
            linecolor='#34495e',
            linewidth=2
        ),
        yaxis=dict(
            title=dict(
                text=f"{metric_display_name}",
                font=dict(size=18, color='#34495e', family='Arial')
            ),
            automargin=True,
            tickfont=dict(size=20, color='#34495e', family='Arial'),
            gridcolor='rgba(189, 195, 199, 0.3)',
            gridwidth=1,
            showgrid=True,
            linecolor='#34495e',
            linewidth=2
        ),
        
        # Enhanced styling
        showlegend=False,
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=60, autoexpand=True),
        font=dict(family="Arial, sans-serif", color='#2c3e50'),
        
        # No border - removed for cleaner look
    )

    fig.update_traces(
        textposition="outside",  # put labels above bars
        cliponaxis=False         # don’t cut them off
    )
    
    return fig


def create_model_quality_table(
    selected_models: List[str], 
    quality_metric: str = "helpfulness"
) -> str:
    """Create an HTML table of model-level quality scores."""
    if not app_state["metrics"]:
        return "No data loaded. Please load data first using the 'Load Data' tab."
    
    if not selected_models:
        return "Please select at least one model to display."
    
    # Handle case where quality_metric is None (during data source switching)
    if quality_metric is None:
        available_metrics = get_available_model_quality_metrics()
        if not available_metrics:
            return "No quality metrics available in the loaded data."
        quality_metric = available_metrics[0]  # Use first available metric
    
    # Get model scores from metrics
    model_scores = app_state["metrics"].get("model_scores", {})
    if not model_scores:
        return "No model scores available in the loaded data."
    
    # Create model dataframe
    model_df = create_model_dataframe(model_scores)
    
    if model_df.empty:
        return "No model data available."
    
    # Filter to selected models
    model_df = model_df[model_df['model'].isin(selected_models)]
    
    if model_df.empty:
        return "No data available for selected models."
    
    # Find the actual ABSOLUTE quality column (not delta) that matches the requested metric
    # We want raw quality scores, not deltas from baseline
    quality_col = None
    for col in model_df.columns:
        if (col.startswith("quality_") and 
            not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant")) and
            "delta" not in col.lower()):  # Explicitly exclude any delta columns
            # Check if the quality metric name is contained in the column name (case insensitive)
            col_name = col.replace("quality_", "").lower()
            if quality_metric.lower() in col_name:
                quality_col = col
                break
    
    # If no match found, use the first available absolute quality column
    if not quality_col:
        available_quality_cols = [col for col in model_df.columns 
                                if col.startswith("quality_") 
                                and not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant"))
                                and "delta" not in col.lower()]  # Explicitly exclude delta columns
        if not available_quality_cols:
            return "No quality metrics found in the data."
        quality_col = available_quality_cols[0]  # Use first available absolute quality metric
    
    # Ensure quality values are numeric
    model_df[quality_col] = pd.to_numeric(model_df[quality_col], errors='coerce')
    
    # Check if we have any valid quality data
    if model_df[quality_col].isna().all():
        return f"No valid quality data found for metric '{quality_metric}'."
    
    # Sort models by quality score (descending - best scores first)
    model_df = model_df.sort_values(by=quality_col, ascending=False).reset_index(drop=True)
    
    # Extract a clean metric name for display
    metric_display_name = quality_col.replace("quality_", "").split("(")[0].strip()
    
    # Define confidence interval column names
    ci_lower_col = f"{quality_col}_ci_lower"
    ci_upper_col = f"{quality_col}_ci_upper"
    
    # Debug: Check if confidence interval columns exist
    has_ci = ci_lower_col in model_df.columns and ci_upper_col in model_df.columns
    if not has_ci:
        # Try alternative naming pattern
        metric_name = quality_col.replace("quality_", "")
        alt_ci_lower = f"quality_{metric_name}_ci_lower"
        alt_ci_upper = f"quality_{metric_name}_ci_upper"
        if alt_ci_lower in model_df.columns and alt_ci_upper in model_df.columns:
            ci_lower_col = alt_ci_lower
            ci_upper_col = alt_ci_upper
            has_ci = True
    
    # Calculate ranks based on confidence intervals
    # A model's rank = 1 + number of models that are confidently better (non-overlapping CIs)
    ranks = []
    
    if has_ci:
        # Use confidence interval-based ranking
        for i, row in model_df.iterrows():
            # Get current model's quality score and confidence intervals
            current_score = row[quality_col]
            current_upper = row[ci_upper_col] if not pd.isna(row[ci_upper_col]) else current_score
            current_lower = row[ci_lower_col] if not pd.isna(row[ci_lower_col]) else current_score
            
            # Count how many models are confidently better
            confidently_better = 0
            for j, other_row in model_df.iterrows():
                if i != j:  # Don't compare with self
                    other_score = other_row[quality_col]
                    other_upper = other_row[ci_upper_col] if not pd.isna(other_row[ci_upper_col]) else other_score
                    other_lower = other_row[ci_lower_col] if not pd.isna(other_row[ci_lower_col]) else other_score
                    
                    # Check if other model's CI is completely above current model's CI
                    # This means the other model is confidently better
                    if other_lower > current_upper:
                        confidently_better += 1
            
            ranks.append(confidently_better + 1)  # Rank = 1 + number confidently better
    else:
        # Fallback to simple ranking by quality score (no confidence intervals)
        # Sort by quality score and assign ranks
        sorted_indices = model_df[quality_col].sort_values(ascending=False).index
        rank_dict = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        ranks = [rank_dict[idx] for idx in model_df.index]
    
    # Prepare table data
    table_rows = []
    for idx, row in model_df.iterrows():
        model_name = row['model']
        quality_score = row[quality_col]
        rank = ranks[idx]
        
        # Get confidence intervals if available
        ci_text = ""
        if ci_lower_col in model_df.columns and ci_upper_col in model_df.columns:
            ci_lower = row[ci_lower_col]
            ci_upper = row[ci_upper_col]
            ci_text = f" [{ci_lower:.3f}, {ci_upper:.3f}]"
        
        table_rows.append(f"""
        <tr>
            <td style=\"text-align: center; padding: 6px 8px; font-weight: bold; color: #2c3e50;\">{rank}</td>
            <td style=\"padding: 6px 8px; color: #2c3e50;\">{model_name}</td>
            <td style=\"text-align: center; padding: 6px 8px; color: #2c3e50;\">{quality_score:.3f}{ci_text}</td>
        </tr>
        """)
    
    # Create HTML table
    html_table = f"""
    <div style="width: 100%; margin: 0; max-height: 340px; overflow: auto;">
        <table style="width: 100%; border-collapse: collapse; background: white; border: 1px solid #ddd; border-radius: 4px; font-size: 13px;">
            <thead>
                <tr style="background: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                    <th style="padding: 6px 8px; text-align: center; font-weight: bold; color: #495057; border-right: 1px solid #dee2e6;">Rank</th>
                    <th style="padding: 6px 8px; text-align: left; font-weight: bold; color: #495057; border-right: 1px solid #dee2e6;">Model</th>
                    <th style="padding: 6px 8px; text-align: center; font-weight: bold; color: #495057;">{metric_display_name}</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        <p style="text-align: center; color: #6c757d; font-size: 11px; margin-top: 8px; font-family: Arial, sans-serif;">
            {f"Ranks based on confidence intervals (non-overlapping CIs). Models with overlapping CIs may have the same rank." if has_ci else "Ranks based on quality scores (confidence intervals not available)."}
        </p>
    </div>
    """
    
    return html_table


def has_quality_metrics() -> bool:
    """Check if quality metrics are actually available in the loaded data."""
    if not app_state["metrics"]:
        return False
    
    model_scores = app_state["metrics"].get("model_scores", {})
    if not model_scores:
        return False
    
    # Create model dataframe to get available columns
    model_df = create_model_dataframe(model_scores)
    
    if model_df.empty:
        return False
    
    # Find all ABSOLUTE quality columns (excluding CI, delta, and other suffix columns)
    quality_columns = [col for col in model_df.columns 
                      if col.startswith("quality_") 
                      and not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant"))
                      and "delta" not in col.lower()]
    
    return len(quality_columns) > 0


def get_available_model_quality_metrics() -> List[str]:
    """Get available quality metrics from the loaded model data."""
    if not has_quality_metrics():
        return []
    
    model_scores = app_state["metrics"].get("model_scores", {})
    
    # Create model dataframe to get available columns
    model_df = create_model_dataframe(model_scores)
    
    # Find all ABSOLUTE quality columns (excluding CI, delta, and other suffix columns)
    quality_columns = [col for col in model_df.columns 
                      if col.startswith("quality_") 
                      and not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant"))
                      and "delta" not in col.lower()]
    
    # Extract simplified metric names for dropdown choices
    # These will be matched against the full column names in create_model_quality_plot
    available_quality_metrics = []
    for col in quality_columns:
        # Remove "quality_" prefix and extract the main metric name
        metric_name = col.replace("quality_", "").split("(")[0].strip().lower()
        # Use common simplified names that users would expect
        if "help" in metric_name:
            available_quality_metrics.append("helpfulness")
        elif "understand" in metric_name:
            available_quality_metrics.append("understandability")
        elif "complete" in metric_name:
            available_quality_metrics.append("completeness")
        elif "concise" in metric_name:
            available_quality_metrics.append("conciseness")
        elif "harm" in metric_name:
            available_quality_metrics.append("harmlessness")
        else:
            # For other metrics, use the first word
            available_quality_metrics.append(metric_name.split()[0])
    
    # Remove duplicates while preserving order
    available_quality_metrics = list(dict.fromkeys(available_quality_metrics))
    
    return available_quality_metrics 