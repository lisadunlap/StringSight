import gradio as gr
import json
import pandas as pd
import markdown
import html
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional

# Import dashboard's conversation display utilities
try:
    # Try relative import (when imported as module)
    from .conversation_display import (
        convert_to_openai_format,
        display_openai_conversation_html
    )
    from .side_by_side_display import (
        display_side_by_side_responses,
        is_side_by_side_dataset,
        extract_side_by_side_data
    )
    USE_DASHBOARD_FORMAT = True
    USE_SIDE_BY_SIDE = True
except ImportError:
    try:
        # Try absolute import (when run as script)
        from stringsight.dashboard.conversation_display import (
            convert_to_openai_format,
            display_openai_conversation_html
        )
        from stringsight.dashboard.side_by_side_display import (
            display_side_by_side_responses,
            is_side_by_side_dataset,
            extract_side_by_side_data
        )
        USE_DASHBOARD_FORMAT = True
        USE_SIDE_BY_SIDE = True
    except ImportError:
        USE_DASHBOARD_FORMAT = False
        USE_SIDE_BY_SIDE = False
        print("Warning: Could not import dashboard display utilities, using fallback format")

# Import property extraction and evidence highlighting
try:
    from stringsight import prompts as stringsight_prompts
    from stringsight.public import extract_properties_only
    from stringsight.dashboard.examples_helpers import (
        annotate_text_with_evidence_placeholders,
        HIGHLIGHT_START,
        HIGHLIGHT_END,
        compute_highlight_spans,
    )
    from stringsight.formatters import detect_method
    USE_EXTRACTION = True
except ImportError as e:
    USE_EXTRACTION = False
    print(f"Warning: Could not import extraction utilities: {e}")

def load_data(file_path: str = "data/taubench/airline_data_oai_format.jsonl"):
    """Load the JSONL data file
    
    Args:
        file_path: Path to JSONL file with chat trajectories
    """
    import os
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        print(f"Current directory: {os.getcwd()}")
        
        # Suggest available files
        print("\nAvailable JSONL files with trajectories:")
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.endswith('.jsonl') and 'oai_format' in file:
                    print(f"  - {os.path.join(root, file)}")
        return None
    
    try:
        df = pd.read_json(file_path, lines=True)
        print(f"Successfully loaded {len(df)} trajectories from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"File exists but may have invalid JSON format")
        # Try to read first few lines to diagnose
        try:
            with open(file_path, 'r') as f:
                print(f"First line of file: {f.readline()[:200]}...")
        except:
            pass
        return None

def format_message_html(role: str, content: Any) -> str:
    """Format a single message as HTML
    
    Handles any data type by converting to string as fallback.
    """
    
    # Define colors for different roles
    role_colors = {
        "system": "#ff6b6b",      # Red
        "user": "#4ecdc4",        # Teal 
        "assistant": "#45b7d1",   # Blue
        "tool": "#96ceb4",        # Green
        "info": "#feca57"         # Yellow
    }
    
    # Ensure role is a string
    role = str(role) if role is not None else "unknown"
    
    # Get color for this role, default to gray
    color = role_colors.get(role.lower(), "#95a5a6")
    
    # Format content for HTML display - handle any data type
    try:
        if content is None:
            content_html = "<em>(No content)</em>"
        elif isinstance(content, dict):
            content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
        elif isinstance(content, (list, tuple)):
            # Convert lists/tuples to JSON for readability
            content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
        elif isinstance(content, str):
            # Convert markdown to HTML with proper extensions
            try:
                content_html = markdown.markdown(
                    content, 
                    extensions=[
                        'nl2br',           # Convert newlines to <br>
                        'fenced_code',     # Support ```code blocks```
                        'codehilite',      # Syntax highlighting for code blocks
                        'tables',          # Support tables
                        'toc',             # Table of contents
                        'sane_lists',      # Better list handling
                        'def_list',        # Definition lists
                        'attr_list',       # Attribute lists
                        'footnotes',       # Footnotes
                        'abbr',            # Abbreviations
                        'md_in_html'       # Allow HTML in markdown
                    ],
                    output_format='html5'
                )
            except Exception as e:
                # Fallback to basic markdown if advanced features fail
                try:
                    content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
                except:
                    # Final fallback: just escape and display as-is
                    content_html = f"<pre>{str(content)}</pre>"
        else:
            # For any other type (int, float, bool, custom objects), convert to string
            content_str = str(content)
            try:
                content_html = markdown.markdown(content_str, extensions=['nl2br', 'fenced_code'])
            except:
                # Final fallback: just escape and display as-is
                content_html = f"<pre>{content_str}</pre>"
    except Exception as e:
        # Absolute fallback: show error and raw content
        content_html = f"<div style='color: red;'><em>Error formatting content: {str(e)}</em></div><pre>{str(content)}</pre>"
    
    return f"""
    <div style="
        border-left: 4px solid {color};
        margin: 8px 0;
        background-color: white;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    ">
        <div style="
            color: #666;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        ">{role}</div>
        <div style="
            color: #333;
            line-height: 1.6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            {content_html}
        </div>
    </div>
    """

def display_trajectory(trajectory_index: int, df: pd.DataFrame) -> tuple:
    """Display a trajectory at the given index
    
    Handles unexpected data formats by converting to strings.
    """
    if df is None or trajectory_index >= len(df):
        return "", "", "", ""
    
    current_row = df.iloc[trajectory_index]
    
    # Get metadata - convert everything to string as fallback
    question_id = str(current_row.get('question_id', 'N/A')) if 'question_id' in current_row else 'N/A'
    model = str(current_row.get('model', 'N/A')) if 'model' in current_row else 'N/A'
    
    # Get conversation data from model_response field
    trace = current_row.get('model_response', [])
    
    # Handle trace in different formats
    if isinstance(trace, list):
        message_count = len(trace)
    elif isinstance(trace, str):
        # Trace might be a JSON string
        try:
            trace = json.loads(trace)
            message_count = len(trace) if isinstance(trace, list) else 1
        except:
            message_count = 1  # Single string message
    else:
        message_count = 1 if trace else 0
    
    # Format metadata
    metadata_html = f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px; font-size: 14px;">
        <div><strong>Question ID:</strong> {question_id}</div>
        <div><strong>Model:</strong> {model}</div>
        <div><strong>Messages:</strong> {message_count}</div>
    </div>
    """
    
    # Format prompt - handle any data type (MOVED BEFORE CONVERSATION)
    prompt_html = ""
    if 'prompt' in current_row and current_row['prompt'] is not None:
        prompt_value = current_row['prompt']
        if isinstance(prompt_value, str):
            prompt_display = prompt_value
        else:
            # Convert non-string prompts to JSON or string
            try:
                prompt_display = json.dumps(prompt_value, indent=2)
            except:
                prompt_display = str(prompt_value)
        
        # Format prompt as conversation message for consistent styling
        if USE_DASHBOARD_FORMAT:
            try:
                prompt_as_message = [{"role": "user", "content": prompt_display}]
                prompt_html = f"""
                <div style="margin-top: 20px;">
                    <h4 style="margin-bottom: 10px; color: #333;">Prompt</h4>
                    {display_openai_conversation_html(prompt_as_message, use_accordion=False, pretty_print_dicts=True, evidence=None)}
                </div>
                """
            except Exception as e:
                # Fallback to plain format
                prompt_html = f"""
                <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                    <h4>Prompt</h4>
                    <pre style='white-space: pre-wrap; word-wrap: break-word;'>{prompt_display}</pre>
                </div>
                """
        else:
            # Use format_message_html for fallback
            prompt_html = f"""
            <div style="margin-top: 20px;">
                <h4 style="margin-bottom: 10px; color: #333;">Prompt</h4>
                {format_message_html('user', prompt_display)}
            </div>
            """
    
    # Format scores - handle any data type (MOVED BEFORE CONVERSATION)
    scores_html = ""
    if 'score' in current_row and current_row['score'] is not None:
        score_data = current_row['score']
        scores_html = "<div style='margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;'><h4>Scores</h4>"
        
        if isinstance(score_data, dict):
            for metric, value in score_data.items():
                # Convert value to string if needed
                value_str = str(value) if not isinstance(value, str) else value
                scores_html += f"<p><strong>{metric}:</strong> {value_str}</p>"
        elif isinstance(score_data, (list, tuple)):
            # Handle scores as list
            try:
                scores_html += f"<pre>{json.dumps(score_data, indent=2)}</pre>"
            except:
                scores_html += f"<pre>{str(score_data)}</pre>"
        else:
            # Single score value
            scores_html += f"<p><strong>Score:</strong> {str(score_data)}</p>"
        
        scores_html += "</div>"
    
    # Format conversation trajectory - handle various data formats
    conversation_html = ""
    
    if USE_DASHBOARD_FORMAT:
        # Use dashboard's sophisticated display format
        try:
            # Convert trace to OpenAI format
            if isinstance(trace, list):
                openai_format = trace  # Already in list format
            elif isinstance(trace, str):
                # Try to parse as JSON
                try:
                    parsed = json.loads(trace)
                    if isinstance(parsed, list):
                        openai_format = parsed
                    else:
                        openai_format = convert_to_openai_format(trace)
                except:
                    openai_format = convert_to_openai_format(trace)
            else:
                openai_format = convert_to_openai_format(trace)
            
            # Use dashboard's display function
            conversation_html = display_openai_conversation_html(
                openai_format,
                use_accordion=True,
                pretty_print_dicts=True,
                evidence=None
            )
        except Exception as e:
            # Fallback to basic display on error
            conversation_html = f"<p style='color: red;'>Error rendering conversation: {str(e)}</p>"
            conversation_html += f"<pre>{html.escape(str(trace)[:500])}</pre>"
    else:
        # Fallback: Use original format_message_html
        if isinstance(trace, list):
            for i, message in enumerate(trace):
                if isinstance(message, dict):
                    role = str(message.get('role', 'unknown'))
                    content = message.get('content')
                    conversation_html += format_message_html(role, content)
                    
                    # Handle tool calls if present
                    if 'tool_calls' in message:
                        tool_calls = message['tool_calls']
                        if not isinstance(tool_calls, list):
                            tool_calls = [tool_calls]
                        
                        for tool_call in tool_calls:
                            try:
                                tool_call_json = json.dumps(tool_call, indent=2) if not isinstance(tool_call, str) else tool_call
                            except:
                                tool_call_json = str(tool_call)
                            
                            conversation_html += f"""
                            <div style="
                                border-left: 4px solid #e67e22;
                                padding-left: 20px;
                                margin: 5px 0 5px 20px;
                                background-color: #fdf6e3;
                                padding: 10px;
                                border-radius: 0 5px 5px 0;
                            ">
                                <h5 style="
                                    color: #666;
                                    font-size: 12px;
                                    font-weight: bold;
                                    margin-bottom: 5px;
                                    text-transform: uppercase;
                                ">TOOL CALL</h5>
                                <pre>{tool_call_json}</pre>
                            </div>
                            """
                    
                    # Handle tool call responses
                    if role == 'tool':
                        tool_call_id = str(message.get('tool_call_id', 'Unknown'))
                        name = str(message.get('name', 'Unknown'))
                        conversation_html += f"<p style='font-size: 12px; color: #666;'>Tool: {name} | Call ID: {tool_call_id}</p>"
                else:
                    # Message is not a dict - treat as plain content
                    conversation_html += format_message_html('info', message)
        elif isinstance(trace, str):
            # Trace is a single string
            conversation_html = format_message_html('info', trace)
        elif trace is not None:
            # Trace is some other type - convert to string
            conversation_html = format_message_html('info', trace)
    
    return metadata_html, prompt_html, scores_html, conversation_html

def compose_trajectory_html(trajectory_index: int, df: pd.DataFrame) -> str:
    """Compose a single HTML block for a trajectory combining metadata, prompt, scores, and conversation.
    
    Args:
        trajectory_index: Zero-based index of the trajectory row in the DataFrame.
        df: DataFrame containing at least `model_response` and optionally `prompt`, `score`, `question_id`, `model`.
    
    Returns:
        A single HTML string rendering the selected trajectory, suitable for a single `gr.HTML` output.
    """
    metadata_html, prompt_html, scores_html, conversation_html = display_trajectory(trajectory_index, df)
    # Wrap in a single container for consistent spacing and styling
    return f"""
    <div style="display: flex; flex-direction: column; gap: 16px;">
        {metadata_html}
        <!-- Plain prompt block (no chat bubble) -->
        <div style="margin-top: 8px; padding: 12px; background-color: #f0f4ff; border-left: 4px solid #4a6fe3; border-radius: 4px;">
            <h4 style="margin: 0 0 6px 0; color: #2c3e50;">Prompt</h4>
            <div style="white-space: pre-wrap; word-wrap: break-word; color: #2c3e50;">{html.escape(str(df.iloc[trajectory_index].get('prompt', '')))}</div>
        </div>
        <!-- Scores blue bar -->
        <div style="padding: 10px 12px; background-color: #e6f0ff; border: 1px solid #c6dcff; border-radius: 6px;">
            <div style="color: #1f5fbf; font-weight: 600;">Scores</div>
            <div style="color: #1f5fbf; font-size: 13px; margin-top: 4px;">
                {format_scores_inline(df.iloc[trajectory_index].get('score'))}
            </div>
        </div>
        <div>
            <h4 style="margin: 8px 0; color: #333;">Conversation</h4>
            {conversation_html}
        </div>
    </div>
    """

def format_scores_inline(score_obj: Any) -> str:
    """Format a score object as inline key:value entries.
    
    Args:
        score_obj: Ideally a dict of metric -> value; other types are stringified.
    
    Returns:
        HTML string with scores inline; empty string if not available.
    """
    if score_obj is None:
        return ""
    if isinstance(score_obj, dict):
        parts: List[str] = []
        for k, v in score_obj.items():
            try:
                val = f"{float(v):.3f}"
            except Exception:
                val = html.escape(str(v))
            parts.append(f"<span style=\"margin-right:12px;\"><strong>{html.escape(str(k))}</strong>: {val}</span>")
        return "".join(parts)
    return html.escape(str(score_obj))

def get_examples_dropdown_choices(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Build dropdown choices for Examples tab.
    
    Args:
        df: DataFrame containing at least optional columns: `prompt`, `model`, and optionally `cluster`/`property`.
    
    Returns:
        Tuple of (prompts, models, properties) lists prefixed with "All ..." options.
    """
    prompts: List[str] = ["All Prompts"]
    models: List[str] = ["All Models"]
    properties: List[str] = ["All Clusters"]

    if 'prompt' in df.columns:
        unique_prompts = [p for p in df['prompt'].dropna().unique().tolist()]
        prompts += unique_prompts
    if 'model' in df.columns:
        unique_models = sorted([str(m) for m in df['model'].dropna().unique().tolist()])
        models += unique_models
    # Support either 'cluster' or 'property' naming if present
    prop_col = 'cluster' if 'cluster' in df.columns else ('property' if 'property' in df.columns else None)
    if prop_col:
        unique_props = sorted([str(x) for x in df[prop_col].dropna().unique().tolist()])
        properties += unique_props
    return prompts, models, properties

def build_examples_html(
    df: pd.DataFrame,
    selected_prompt: str,
    selected_model: str,
    selected_property: str,
    max_examples: int,
    search_term: str
) -> str:
    """Render a combined examples list with styling similar to the examples tab.
    
    Args:
        df: Source DataFrame of examples
        selected_prompt: Selected prompt or "All Prompts"
        selected_model: Selected model or "All Models"
        selected_property: Selected property/cluster or "All Clusters"
        max_examples: Max number of examples to render
        search_term: Free-text search term to filter by prompt/response content
    
    Returns:
        HTML string with all examples concatenated.
    """
    filtered = df
    if selected_model and selected_model != "All Models" and 'model' in filtered.columns:
        filtered = filtered[filtered['model'] == selected_model]
    if selected_prompt and selected_prompt != "All Prompts" and 'prompt' in filtered.columns:
        filtered = filtered[filtered['prompt'] == selected_prompt]
    prop_col = 'cluster' if 'cluster' in filtered.columns else ('property' if 'property' in filtered.columns else None)
    if selected_property and selected_property != "All Clusters" and prop_col:
        filtered = filtered[filtered[prop_col] == selected_property]
    if search_term and isinstance(search_term, str) and search_term.strip():
        term = search_term.strip().lower()
        def _row_has_term(row: pd.Series) -> bool:
            in_prompt = str(row.get('prompt', '')).lower()
            in_resp = str(row.get('model_response', '')).lower()
            return (term in in_prompt) or (term in in_resp)
        filtered = filtered[filtered.apply(_row_has_term, axis=1)]

    if len(filtered) == 0:
        return "<p style='color:#e74c3c; padding: 12px;'>❌ No examples match the current filters.</p>"

    # Limit and build HTML
    out_parts: List[str] = [
        "<div><h3 style=\"color:#333; margin: 8px 0 12px 0;\">Examples</h3></div>"
    ]
    subset = filtered.head(max_examples)
    for idx, row in subset.iterrows():
        # Header with model and optional question id
        qid = row.get('question_id', None)
        model_name = row.get('model', '')
        header = f"<div style=\"display:flex; justify-content:space-between; color:#555; font-size:13px;\"><div><strong>Model:</strong> {html.escape(str(model_name))}</div>"
        if qid is not None:
            header += f"<div><strong>ID:</strong> {html.escape(str(qid))}</div>"
        header += "</div>"

        # Plain prompt
        prompt_block = ""
        if 'prompt' in row and row['prompt'] is not None:
            prompt_block = (
                "<div style=\"margin-top:8px; padding: 12px; background-color: #f0f4ff; border-left: 4px solid #4a6fe3; border-radius: 4px;\">"
                "<div style=\"margin:0 0 6px 0; color: #2c3e50; font-weight:600;\">Prompt</div>"
                f"<div style=\"white-space: pre-wrap; word-wrap: break-word; color: #2c3e50;\">{html.escape(str(row['prompt']))}</div>"
                "</div>"
            )

        # Scores blue bar
        score_block = (
            "<div style=\"padding: 10px 12px; background-color: #e6f0ff; border: 1px solid #c6dcff; border-radius: 6px; margin-top:8px;\">"
            "<div style=\"color: #1f5fbf; font-weight: 600;\">Scores</div>"
            f"<div style=\"color: #1f5fbf; font-size: 13px; margin-top: 4px;\">{format_scores_inline(row.get('score'))}</div>"
            "</div>"
        )

        # Conversation using existing renderer
        # Reuse display_trajectory to leverage conversation rendering and then compose minimal section
        # but ensure we don't duplicate prompt/scores blocks here
        conv_html = ""
        try:
            # Minimal reuse: only conversation_html from display_trajectory
            _, _, _, conversation_html = display_trajectory(idx, df)
            conv_html = (
                "<div style=\"margin-top:8px;\">"
                "<h4 style=\"margin: 8px 0; color: #333;\">Conversation</h4>"
                f"{conversation_html}"
                "</div>"
            )
        except Exception:
            conv_html = ""

        card = (
            "<div style=\"background:#fff; border:1px solid #e1e4e8; border-radius:8px; padding:14px; margin-bottom:14px;\">"
            f"{header}{prompt_block}{score_block}{conv_html}"
            "</div>"
        )
        out_parts.append(card)

    return "".join(out_parts)

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate standard confidence interval for a list of values.
    
    Args:
        values: List of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values or len(values) == 0:
        return 0.0, 0.0, 0.0
    
    from scipy import stats
    
    n = len(values)
    mean = np.mean(values)
    
    if n == 1:
        return mean, mean, mean
    
    # Calculate standard error
    std_err = stats.sem(values)
    
    # Use t-distribution for small samples, normal for large samples
    if n < 30:
        # t-distribution
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * std_err
    else:
        # Normal distribution
        z_val = stats.norm.ppf((1 + confidence) / 2)
        margin = z_val * std_err
    
    lower = mean - margin
    upper = mean + margin
    
    return mean, lower, upper

def extract_benchmark_scores_with_ci(df: pd.DataFrame, score_columns: List[str] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract benchmark scores per model with confidence intervals.
    
    Args:
        df: DataFrame with model and score data
        score_columns: List of column names to use as metrics (if None, auto-detect)
    
    Returns:
        Dict mapping model -> {metric: {'mean': float, 'lower_ci': float, 'upper_ci': float, 'count': int}}
    """
    if 'model' not in df.columns:
        return {}
    
    model_scores = {}
    
    # Auto-detect score columns if not provided
    if score_columns is None:
        score_columns = []
        
        # Check for 'score' column (dict or numeric)
        if 'score' in df.columns:
            score_columns.append('score')
        
        # Look for columns that might be metrics (numeric columns with reasonable names)
        potential_metrics = [col for col in df.columns 
                           if col not in ['model', 'prompt', 'question_id', 'model_response', 'trace']
                           and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        score_columns.extend(potential_metrics)
    
    if not score_columns:
        return {}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_scores[model] = {}
        
        for col in score_columns:
            if col not in df.columns:
                continue
            
            all_values = {}  # metric_name -> list of values
            
            for value in model_df[col]:
                if pd.isna(value):
                    continue
                    
                if isinstance(value, dict):
                    # Extract all numeric values from dict
                    for metric_name, metric_value in value.items():
                        try:
                            numeric_val = float(metric_value)
                            # Store with prefix to avoid collisions
                            key = f"{col}.{metric_name}" if col != 'score' else metric_name
                            if key not in all_values:
                                all_values[key] = []
                            all_values[key].append(numeric_val)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(value, (int, float)):
                    if col not in all_values:
                        all_values[col] = []
                    all_values[col].append(float(value))
            
            # Calculate mean and CI for each metric
            for metric_key, values in all_values.items():
                if not values:
                    continue
                
                mean_score, lower_ci, upper_ci = calculate_confidence_interval(values)
                count = len(values)
                    
                model_scores[model][metric_key] = {
                    'mean': mean_score,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'count': count
                }
    
    return model_scores

def extract_benchmark_scores(df: pd.DataFrame, score_columns: List[str] = None) -> Dict[str, Dict[str, float]]:
    """Extract benchmark scores per model from score columns.
    
    Args:
        df: DataFrame with model and score data
        score_columns: List of column names to use as metrics (if None, auto-detect)
    
    Returns:
        Dict mapping model -> {metric: average_score}
    """
    if 'model' not in df.columns:
        return {}
    
    model_scores = {}
    
    # Auto-detect score columns if not provided
    if score_columns is None:
        score_columns = []
        
        # Check for 'score' column (dict or numeric)
        if 'score' in df.columns:
            score_columns.append('score')
        
        # Look for columns that might be metrics (numeric columns with reasonable names)
        potential_metrics = [col for col in df.columns 
                           if col not in ['model', 'prompt', 'question_id', 'model_response', 'trace']
                           and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        score_columns.extend(potential_metrics)
    
    if not score_columns:
        return {}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_scores[model] = {}
        
        for col in score_columns:
            if col not in df.columns:
                continue
            
            scores_list = []
            
            for value in model_df[col]:
                if pd.isna(value):
                    continue
                    
                if isinstance(value, dict):
                    # Extract all numeric values from dict
                    for metric_name, metric_value in value.items():
                        try:
                            numeric_val = float(metric_value)
                            # Store with prefix to avoid collisions
                            key = f"{col}.{metric_name}" if col != 'score' else metric_name
                            if key not in model_scores[model]:
                                model_scores[model][key] = []
                            model_scores[model][key].append(numeric_val)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(value, (int, float)):
                    scores_list.append(float(value))
            
            # Average the scores for this column
            if scores_list:
                model_scores[model][col] = sum(scores_list) / len(scores_list)
        
        # Average any dict-based metrics that were collected
        for metric_key in list(model_scores[model].keys()):
            if isinstance(model_scores[model][metric_key], list):
                values = model_scores[model][metric_key]
                model_scores[model][metric_key] = sum(values) / len(values) if values else 0.0
    
    return model_scores

def build_benchmark_results_df(df: pd.DataFrame, score_columns: List[str] = None) -> pd.DataFrame:
    """Build a wide benchmark results DataFrame for display.
    
    Args:
        df: Pandas DataFrame expected to have a `model` column and one or more metric columns. 
            Metric columns may be numeric columns or a dict-like column (e.g., `score`) mapping metric name to value.
        score_columns: Optional list of column names to use as metrics. If None, metrics are auto-detected similarly
            to `extract_benchmark_scores`.
    
    Returns:
        A DataFrame with columns: `Model` plus one column per metric key. Each row corresponds to a model.
        - `Model` (str): model identifier from the `model` column
        - metric columns (float): mean value per metric for the model
    """
    scores_by_model: Dict[str, Dict[str, float]] = extract_benchmark_scores(df, score_columns)
    if not scores_by_model:
        return pd.DataFrame()

    # Collect all metric keys
    all_metrics = set()
    for scores in scores_by_model.values():
        all_metrics.update(scores.keys())
    ordered_metrics = sorted(all_metrics)

    # Build rows
    rows: List[Dict[str, Any]] = []
    for model, metrics in sorted(scores_by_model.items()):
        row: Dict[str, Any] = {"Model": model}
        for metric in ordered_metrics:
            # Do not fill defaults silently; if missing, leave as NaN
            row[metric] = metrics[metric] if metric in metrics else float("nan")
        rows.append(row)

    results_df = pd.DataFrame(rows, columns=["Model"] + ordered_metrics)

    # Round numeric metric columns to 3 decimals for display in the table
    if ordered_metrics:
        # Only round numeric columns to avoid altering non-numeric data
        numeric_metric_columns = [
            col for col in ordered_metrics if pd.api.types.is_numeric_dtype(results_df[col])
        ]
        if numeric_metric_columns:
            results_df[numeric_metric_columns] = results_df[numeric_metric_columns].round(3)

    return results_df

def calculate_win_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate win rates from side-by-side comparison data.
    
    Args:
        df: DataFrame with side-by-side comparison data
        
    Returns:
        Dictionary with win rate statistics
    """
    if not has_side_by_side_format(df) or 'winner' not in df.columns:
        return None
    
    stats = {
        'overall': {},
        'head_to_head': {},
        'total_comparisons': len(df)
    }
    
    # Get all unique models
    all_models = set()
    if 'model_a' in df.columns:
        all_models.update(df['model_a'].unique())
    if 'model_b' in df.columns:
        all_models.update(df['model_b'].unique())
    
    # Calculate overall win rates for each model
    for model in all_models:
        wins = 0
        losses = 0
        ties = 0
        total = 0
        
        for _, row in df.iterrows():
            model_a = row.get('model_a')
            model_b = row.get('model_b')
            winner = row.get('winner')
            
            # Skip if winner is not specified
            if pd.isna(winner):
                continue
            
            # Check if this model is involved in the comparison
            if model == model_a:
                total += 1
                if winner == 'model_a':
                    wins += 1
                elif winner == 'model_b':
                    losses += 1
                elif winner == 'tie':
                    ties += 1
            elif model == model_b:
                total += 1
                if winner == 'model_b':
                    wins += 1
                elif winner == 'model_a':
                    losses += 1
                elif winner == 'tie':
                    ties += 1
        
        if total > 0:
            stats['overall'][model] = {
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'total': total,
                'win_rate': wins / total,
                'loss_rate': losses / total,
                'tie_rate': ties / total
            }
    
    # Calculate head-to-head win rates
    for model_a in all_models:
        if model_a not in stats['head_to_head']:
            stats['head_to_head'][model_a] = {}
        
        for model_b in all_models:
            if model_a == model_b:
                continue
            
            # Count wins/losses/ties in direct matchups
            matchups = df[
                ((df['model_a'] == model_a) & (df['model_b'] == model_b)) |
                ((df['model_a'] == model_b) & (df['model_b'] == model_a))
            ]
            
            wins = 0
            losses = 0
            ties = 0
            
            for _, row in matchups.iterrows():
                winner = row.get('winner')
                if pd.isna(winner):
                    continue
                
                # Determine if model_a won
                if row['model_a'] == model_a:
                    if winner == 'model_a':
                        wins += 1
                    elif winner == 'model_b':
                        losses += 1
                    elif winner == 'tie':
                        ties += 1
                else:  # row['model_b'] == model_a
                    if winner == 'model_b':
                        wins += 1
                    elif winner == 'model_a':
                        losses += 1
                    elif winner == 'tie':
                        ties += 1
            
            total = wins + losses + ties
            if total > 0:
                stats['head_to_head'][model_a][model_b] = {
                    'wins': wins,
                    'losses': losses,
                    'ties': ties,
                    'total': total,
                    'win_rate': wins / total
                }
    
    return stats

def create_overview_tab(df: pd.DataFrame, score_columns: List[str] = None) -> str:
    """Create overview statistics HTML for the dataset.
    
    Shows:
    - Number of conversations per model
    - Number of unique prompts across all models
    - Benchmark results per model
    - Basic dataset statistics
    
    Args:
        df: DataFrame with conversation data
        score_columns: Optional list of column names to treat as metrics
    """
    if df is None or len(df) == 0:
        return "<p>No data available</p>"
    
    # Get basic statistics
    total_conversations = len(df)
    
    # Count conversations per model
    if 'model' in df.columns:
        model_counts = df['model'].value_counts().to_dict()
        num_models = len(model_counts)
    else:
        model_counts = {}
        num_models = 0
    
    # Count unique prompts
    if 'prompt' in df.columns:
        unique_prompts = df['prompt'].nunique()
    else:
        unique_prompts = 0
    
    # Count total messages across all conversations
    total_messages = 0
    if 'model_response' in df.columns:
        for resp in df['model_response']:
            if isinstance(resp, list):
                total_messages += len(resp)
            elif resp:
                total_messages += 1
    
    # Extract benchmark scores
    model_scores = extract_benchmark_scores(df, score_columns)
    
    # Build HTML
    overview_html = """
    <div style="padding: 20px; font-family: sans-serif;">
        <h2 style="color: #333; margin-bottom: 20px;">Dataset Overview</h2>
        
        <!-- Summary Cards -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px;">
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #007bff;">{total_conversations}</div>
                <div style="color: #6c757d; margin-top: 5px;">Total Conversations</div>
            </div>
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #28a745;">{num_models}</div>
                <div style="color: #6c757d; margin-top: 5px;">Models</div>
            </div>
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #fd7e14;">{unique_prompts}</div>
                <div style="color: #6c757d; margin-top: 5px;">Unique Prompts</div>
            </div>
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: #6f42c1;">{total_messages}</div>
                <div style="color: #6c757d; margin-top: 5px;">Total Messages</div>
            </div>
        </div>
    """.format(
        total_conversations=total_conversations,
        num_models=num_models,
        unique_prompts=unique_prompts,
        total_messages=total_messages
    )
    
    # Conversations per model table
    if model_counts:
        overview_html += """
        <h3 style="color: #333; margin-bottom: 15px;">Conversations per Model</h3>
        <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 30px;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 2px solid #dee2e6;">
                        <th style="text-align: left; padding: 10px; color: #495057; font-weight: 600;">Model</th>
                        <th style="text-align: right; padding: 10px; color: #495057; font-weight: 600;">Count</th>
                        <th style="text-align: right; padding: 10px; color: #495057; font-weight: 600;">Percentage</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Sort models by count (descending)
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        
        for model, count in sorted_models:
            percentage = (count / total_conversations) * 100
            overview_html += f"""
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 12px; color: #333;">
                        <code style="background: #f8f9fa; padding: 4px 8px; border-radius: 4px; font-size: 13px;">{html.escape(str(model))}</code>
                    </td>
                    <td style="padding: 12px; text-align: right; font-weight: 500; color: #007bff;">{count:,}</td>
                    <td style="padding: 12px; text-align: right; color: #6c757d;">{percentage:.1f}%</td>
                </tr>
            """
        
        overview_html += """
                </tbody>
            </table>
        </div>
        """
    
    # (Benchmark Results table replaced by a Gradio Dataframe in the UI)
    
    # Win rates section (for side-by-side datasets)
    win_rates = calculate_win_rates(df)
    if win_rates and win_rates['overall']:
        overview_html += """
        <h3 style="color: #333; margin-bottom: 15px;">Win Rates</h3>
        <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 30px; overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; min-width: 600px;">
                <thead>
                    <tr style="border-bottom: 2px solid #dee2e6;">
                        <th style="text-align: left; padding: 10px; color: #495057; font-weight: 600;">Model</th>
                        <th style="text-align: center; padding: 10px; color: #495057; font-weight: 600;">Win Rate</th>
                        <th style="text-align: center; padding: 10px; color: #495057; font-weight: 600;">Wins</th>
                        <th style="text-align: center; padding: 10px; color: #495057; font-weight: 600;">Losses</th>
                        <th style="text-align: center; padding: 10px; color: #495057; font-weight: 600;">Ties</th>
                        <th style="text-align: center; padding: 10px; color: #495057; font-weight: 600;">Total</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Sort models by win rate (descending)
        sorted_models = sorted(
            win_rates['overall'].items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        for model, stats in sorted_models:
            win_rate = stats['win_rate'] * 100
            wins = stats['wins']
            losses = stats['losses']
            ties = stats['ties']
            total = stats['total']
            
            overview_html += f"""
                <tr style="border-bottom: 1px solid #f1f3f5;">
                    <td style="padding: 12px; color: #333;">
                        <code style="background: #f8f9fa; padding: 4px 8px; border-radius: 4px; font-size: 13px;">{html.escape(str(model))}</code>
                    </td>
                    <td style="padding: 12px; text-align: center;">
                        <span style="font-weight: 600; color: #000;">{win_rate:.1f}%</span>
                    </td>
                    <td style="padding: 12px; text-align: center; color: #28a745;">{wins}</td>
                    <td style="padding: 12px; text-align: center; color: #dc3545;">{losses}</td>
                    <td style="padding: 12px; text-align: center; color: #6c757d;">{ties}</td>
                    <td style="padding: 12px; text-align: center; font-weight: 500;">{total}</td>
                </tr>
            """
        
        overview_html += """
                </tbody>
            </table>
        </div>
        """
        
        # Head-to-head matchup matrix
        if len(win_rates['head_to_head']) > 1:
            overview_html += """
            <h3 style="color: #333; margin-bottom: 15px;">Head-to-Head Win Rates</h3>
            <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 30px; overflow-x: auto;">
                <p style="color: #6c757d; font-size: 14px; margin-bottom: 15px;">
                    Win rate of row model vs column model (e.g., 60% means row model wins 60% of direct matchups)
                </p>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid #dee2e6;">
                            <th style="text-align: left; padding: 10px; color: #495057; font-weight: 600;">Model</th>
            """
            
            # Add column headers for each model
            models_list = sorted(win_rates['head_to_head'].keys())
            for model in models_list:
                model_short = model[:15] + "..." if len(model) > 15 else model
                overview_html += f"""
                            <th style="text-align: center; padding: 10px; color: #495057; font-weight: 600; font-size: 12px;">{html.escape(model_short)}</th>
                """
            
            overview_html += """
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Add rows for each model
            for row_model in models_list:
                row_model_short = row_model[:20] + "..." if len(row_model) > 20 else row_model
                overview_html += f"""
                    <tr style="border-bottom: 1px solid #f1f3f5;">
                        <td style="padding: 10px; color: #333;">
                            <code style="background: #f8f9fa; padding: 4px 8px; border-radius: 4px; font-size: 12px;">{html.escape(row_model_short)}</code>
                        </td>
                """
                
                for col_model in models_list:
                    if row_model == col_model:
                        # Diagonal - same model
                        overview_html += """
                        <td style="padding: 10px; text-align: center; background: #f8f9fa;">—</td>
                        """
                    else:
                        matchup = win_rates['head_to_head'].get(row_model, {}).get(col_model)
                        if matchup:
                            win_rate = matchup['win_rate'] * 100
                            wins = matchup['wins']
                            total = matchup['total']
                            
                            # Color code based on win rate
                            if win_rate >= 60:
                                bg_color = "#d4edda"
                                text_color = "#155724"
                            elif win_rate >= 40:
                                bg_color = "#fff3cd"
                                text_color = "#856404"
                            else:
                                bg_color = "#f8d7da"
                                text_color = "#721c24"
                            
                            overview_html += f"""
                            <td style="padding: 10px; text-align: center; background: {bg_color};">
                                <span style="font-weight: 600; color: {text_color};">{win_rate:.0f}%</span>
                                <br>
                                <span style="font-size: 11px; color: #6c757d;">({wins}/{total})</span>
                            </td>
                            """
                        else:
                            overview_html += """
                            <td style="padding: 10px; text-align: center; color: #adb5bd;">—</td>
                            """
                
                overview_html += """
                    </tr>
                """
            
            overview_html += """
                    </tbody>
                </table>
            </div>
            """
    
    # Dataset info
    overview_html += """
        <h3 style="color: #333; margin-bottom: 15px;">Dataset Information</h3>
        <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 8px; color: #6c757d; width: 40%;">Columns in dataset:</td>
                    <td style="padding: 8px; color: #333; font-weight: 500;">{num_columns}</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 8px; color: #6c757d;">Has prompt field:</td>
                    <td style="padding: 8px; color: #333; font-weight: 500;">{has_prompt}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #6c757d;">Has model_response field:</td>
                    <td style="padding: 8px; color: #333; font-weight: 500;">{has_response}</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 8px; color: #6c757d;">Has score field:</td>
                    <td style="padding: 8px; color: #333; font-weight: 500;">{has_score}</td>
                </tr>
            </table>
        </div>
    </div>
    """.format(
        num_columns=len(df.columns),
        has_prompt="✓ Yes" if 'prompt' in df.columns else "✗ No",
        has_response="✓ Yes" if 'model_response' in df.columns else "✗ No",
        has_score="✓ Yes" if 'score' in df.columns else "✗ No"
    )
    
    return overview_html

def create_benchmark_plot_single_metric(df: pd.DataFrame, metric: str, score_columns: List[str] = None) -> go.Figure:
    """Create interactive plot for a single metric with Wilson confidence intervals.
    
    Args:
        df: DataFrame with conversation data
        metric: Specific metric to plot
        score_columns: List of columns to use as metrics
        
    Returns:
        Plotly figure with benchmark comparison and error bars
    """
    model_scores_with_ci = extract_benchmark_scores_with_ci(df, score_columns)
    
    if not model_scores_with_ci:
        fig = go.Figure()
        fig.add_annotation(
            text="No benchmark scores available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    # Check if metric exists
    models_with_metric = []
    for model, scores in model_scores_with_ci.items():
        if metric in scores:
            models_with_metric.append(model)
    
    if not models_with_metric:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for metric: {metric}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400)
        return fig
    
    # Sort models by mean score (descending)
    models_sorted = sorted(
        models_with_metric,
        key=lambda m: model_scores_with_ci[m][metric]['mean'],
        reverse=True
    )
    
    # Extract data for plotting
    means = [model_scores_with_ci[m][metric]['mean'] for m in models_sorted]
    lower_cis = [model_scores_with_ci[m][metric]['lower_ci'] for m in models_sorted]
    upper_cis = [model_scores_with_ci[m][metric]['upper_ci'] for m in models_sorted]
    counts = [model_scores_with_ci[m][metric]['count'] for m in models_sorted]
    
    # Calculate ranks based on confidence intervals
    # A model's rank = 1 + number of models that are confidently better (non-overlapping CIs)
    ranks = []
    for i in range(len(models_sorted)):
        current_upper = upper_cis[i]
        
        # Count how many models are confidently better
        confidently_better = 0
        for j in range(len(models_sorted)):
            if i != j:
                other_lower = lower_cis[j]
                # Check if other model's CI is completely above current model's CI
                if other_lower > current_upper:
                    confidently_better += 1
        
        ranks.append(confidently_better + 1)
    
    # Create x-axis labels with ranks
    x_labels = [f"#{ranks[i]} {models_sorted[i]}" for i in range(len(models_sorted))]
    
    # Calculate error bar sizes
    error_minus = [means[i] - lower_cis[i] for i in range(len(means))]
    error_plus = [upper_cis[i] - means[i] for i in range(len(means))]
    
    # Create bar chart with error bars
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=x_labels,
        y=means,
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color='rgba(0,0,0,0.8)',
            thickness=2,
            width=8
        ),
        marker=dict(
            color='#636EFA',  # Plotly default blue
            line=dict(color='rgba(0,0,0,0.1)', width=1)
        ),
        text=[f"{m:.3f}" for m in means],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                     f'{metric}: %{{y:.3f}}<br>' +
                     '95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>' +
                     'n=%{customdata[2]}<br>' +
                     'Rank: %{customdata[3]}<extra></extra>',
        customdata=[[lower_cis[i], upper_cis[i], counts[i], ranks[i]] for i in range(len(models_sorted))]
    ))
    
    # Update layout
    metric_display = metric.replace('_', ' ').title()
    fig.update_layout(
        title={
            'text': f'{metric_display} Comparison (with 95% CI)',
            'font': {'size': 20, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Model",
        yaxis_title=metric_display,
        hovermode='closest',
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="sans-serif", size=12, color="#333"),
        margin=dict(t=100, b=100, l=80, r=40)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='lightgray',
        tickfont=dict(size=11),
        tickangle=-45 if len(models_sorted) > 3 else 0
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='lightgray',
        range=[0, min(max(upper_cis) * 1.15, max(means) * 1.2)]
    )
    
    return fig

def create_benchmark_radar_plot(df: pd.DataFrame, score_columns: List[str] = None) -> go.Figure:
    """Create radar/spider plot for benchmark metrics.
    
    Args:
        df: DataFrame with conversation data
        score_columns: List of columns to use as metrics
        
    Returns:
        Plotly figure with radar chart
    """
    import numpy as np
    
    model_scores = extract_benchmark_scores(df, score_columns)
    
    if not model_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No benchmark scores available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400)
        return fig
    
    # Get all metrics and models
    all_metrics = set()
    for scores in model_scores.values():
        all_metrics.update(scores.keys())
    all_metrics = sorted(all_metrics)
    models = sorted(model_scores.keys())
    
    if not all_metrics or len(all_metrics) < 3:
        # Radar plots need at least 3 metrics
        fig = go.Figure()
        fig.add_annotation(
            text="Radar plot requires at least 3 metrics",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400)
        return fig
    
    # Create radar chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, model in enumerate(models):
        scores = model_scores[model]
        metric_values = [scores.get(metric, 0) for metric in all_metrics]
        # Close the radar by repeating first value
        metric_values_closed = metric_values + [metric_values[0]]
        metrics_closed = all_metrics + [all_metrics[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=metric_values_closed,
            theta=metrics_closed,
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)],
            hovertemplate='<b>%{fullData.name}</b><br>%{theta}: %{r:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=False,
                showgrid=True,
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='lightgray',
                gridcolor='lightgray'
            )
        ),
        title={
            'text': 'Benchmark Metrics Radar Chart',
            'font': {'size': 20, 'color': '#333'},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=600,
        paper_bgcolor='white',
        font=dict(family="sans-serif", size=12, color="#333")
    )
    
    return fig

def get_potential_score_columns(df: pd.DataFrame) -> List[str]:
    """Get list of columns that could potentially be score/metric columns."""
    if df is None or len(df) == 0:
        return []
    
    potential_columns = []
    
    # Always include 'score' if it exists
    if 'score' in df.columns:
        potential_columns.append('score')
    
    # Add numeric columns (excluding IDs and other non-metric fields)
    exclude_patterns = ['id', 'index', 'question', 'prompt', 'model', 'response', 'trace', 'message']
    for col in df.columns:
        if col in potential_columns:
            continue
        
        # Check if column name suggests it's not a metric
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
        
        # Check if column is numeric or contains numeric dicts
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            potential_columns.append(col)
        elif df[col].dtype == 'object':
            # Check if it's a dict column with numeric values
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample, dict):
                potential_columns.append(col)
    
    return potential_columns

def count_prompts_seen_per_model(df: pd.DataFrame, unique_only: bool = True) -> Dict[str, int]:
    """Count prompts seen per model.
    
    Args:
        df: Pandas DataFrame with required columns:
            - 'model' (str): model identifier per row
            - 'prompt' (str): prompt text per row
        unique_only: If True, counts unique prompts per model (deduplicated by
            ['model', 'prompt']). If False, counts all prompt rows per model
            (including repeated prompts).
    
    Returns:
        Dict[str, int]: Mapping of model -> prompt count.
    
    Raises:
        ValueError: If the DataFrame does not include required columns 'model' and 'prompt'.
    """
    required_columns = {"model", "prompt"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing_columns)}")

    # Work only with rows where both 'model' and 'prompt' are present
    model_prompt_df = df[["model", "prompt"]].dropna()

    if unique_only:
        # Count unique prompts per model
        dedup_df = model_prompt_df.drop_duplicates(subset=["model", "prompt"]) 
        counts = dedup_df.groupby("model").size()
    else:
        # Count all prompts per model (including duplicates)
        counts = model_prompt_df.groupby("model").size()

    return {str(model): int(count) for model, count in counts.to_dict().items()}

def create_trajectory_options(df: pd.DataFrame) -> List[str]:
    """Create trajectory options for the dropdown
    
    Handles any data format by converting to strings.
    """
    if df is None:
        return []
    
    options = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Get question_id and model, converting to string as fallback
        question_id = str(row.get('question_id', 'N/A')) if 'question_id' in row else 'N/A'
        model = str(row.get('model', 'N/A')) if 'model' in row else 'N/A'
        
        # Truncate long IDs for readability
        if len(question_id) > 20:
            question_id = question_id[:17] + "..."
        if len(model) > 30:
            model = model[:27] + "..."
        
        options.append(f"Trajectory {i} (ID: {question_id}, Model: {model})")
    
    return options

def has_side_by_side_format(df: pd.DataFrame) -> bool:
    """Check if the dataset has side-by-side comparison format.
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if dataset has model_a and model_b columns
    """
    if df is None or len(df) == 0:
        return False
    
    # Check if required columns exist
    required_cols = ['model_a', 'model_b', 'model_a_response', 'model_b_response']
    return all(col in df.columns for col in required_cols)

def create_side_by_side_options(df: pd.DataFrame) -> List[str]:
    """Create options for side-by-side comparison dropdown.
    
    Args:
        df: DataFrame with side-by-side comparison data
        
    Returns:
        List of formatted option strings
    """
    if df is None:
        return []
    
    options = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Get question_id, converting to string as fallback
        question_id = str(row.get('question_id', row.get('id', i)))
        model_a = str(row.get('model_a', 'Model A'))
        model_b = str(row.get('model_b', 'Model B'))
        
        # Truncate long IDs for readability
        if len(question_id) > 20:
            question_id = question_id[:17] + "..."
        if len(model_a) > 20:
            model_a = model_a[:17] + "..."
        if len(model_b) > 20:
            model_b = model_b[:17] + "..."
        
        options.append(f"Comparison {i} (ID: {question_id}) | {model_a} vs {model_b}")
    
    return options

def display_side_by_side_comparison(comparison_index: int, df: pd.DataFrame) -> tuple:
    """Display a side-by-side comparison at the given index.
    
    Args:
        comparison_index: Index of comparison to display
        df: DataFrame with side-by-side comparison data
        
    Returns:
        Tuple of (metadata_html, prompt_html, comparison_html)
    """
    if df is None or comparison_index >= len(df):
        return "", "", ""
    
    current_row = df.iloc[comparison_index]
    
    # Get metadata
    question_id = str(current_row.get('question_id', current_row.get('id', 'N/A')))
    model_a = str(current_row.get('model_a', 'Model A'))
    model_b = str(current_row.get('model_b', 'Model B'))
    
    # Format metadata
    metadata_html = f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 20px; font-size: 14px;">
        <div><strong>Question ID:</strong> {question_id}</div>
        <div><strong>Model A:</strong> {model_a}</div>
        <div><strong>Model B:</strong> {model_b}</div>
    </div>
    """
    
    # Format prompt
    prompt_html = ""
    if 'prompt' in current_row and current_row['prompt'] is not None:
        prompt_value = current_row['prompt']
        if isinstance(prompt_value, str):
            prompt_display = prompt_value
        else:
            try:
                prompt_display = json.dumps(prompt_value, indent=2)
            except:
                prompt_display = str(prompt_value)
        
        # Format prompt as conversation message for consistent styling
        if USE_DASHBOARD_FORMAT:
            try:
                prompt_as_message = [{"role": "user", "content": prompt_display}]
                prompt_html = f"""
                <div style="margin-top: 20px;">
                    <h4 style="margin-bottom: 10px; color: #333;">Prompt</h4>
                    {display_openai_conversation_html(prompt_as_message, use_accordion=False, pretty_print_dicts=True, evidence=None)}
                </div>
                """
            except Exception as e:
                prompt_html = f"""
                <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                    <h4>Prompt</h4>
                    <pre style='white-space: pre-wrap; word-wrap: break-word;'>{prompt_display}</pre>
                </div>
                """
        else:
            prompt_html = f"""
            <div style="margin-top: 20px;">
                <h4 style="margin-bottom: 10px; color: #333;">Prompt</h4>
                {format_message_html('user', prompt_display)}
            </div>
            """
    
    # Create side-by-side comparison
    comparison_html = ""
    if USE_SIDE_BY_SIDE:
        try:
            side_by_side_data = extract_side_by_side_data(current_row)
            comparison_html = display_side_by_side_responses(
                model_a=side_by_side_data['model_a'],
                model_b=side_by_side_data['model_b'],
                model_a_response=side_by_side_data['model_a_response'],
                model_b_response=side_by_side_data['model_b_response'],
                use_accordion=True,
                pretty_print_dicts=True,
                winner=side_by_side_data.get('winner'),
                scores_a=side_by_side_data.get('score_a'),
                scores_b=side_by_side_data.get('score_b')
            )
        except Exception as e:
            comparison_html = f"<p style='color: red;'>Error rendering side-by-side comparison: {str(e)}</p>"
    else:
        # Fallback: display both responses using basic format
        model_a_resp = current_row.get('model_a_response', 'N/A')
        model_b_resp = current_row.get('model_b_response', 'N/A')
        comparison_html = f"""
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <h4>{model_a}</h4>
                {format_message_html('assistant', model_a_resp)}
            </div>
            <div style="flex: 1;">
                <h4>{model_b}</h4>
                {format_message_html('assistant', model_b_resp)}
            </div>
        </div>
        """
    
    return metadata_html, prompt_html, comparison_html

def get_available_prompts() -> List[Tuple[str, str]]:
    """Get available system prompts for property extraction.
    
    Returns:
        List of (display_name, prompt_name) tuples
    """
    if not USE_EXTRACTION:
        return []
    
    prompts_list = [
        ("Default - Single Model", "single_model_system_prompt"),
        ("Custom Task - Single Model", "single_model_system_prompt_custom"),
        ("Default - Side by Side", "sbs_system_prompt"),
        ("Custom Task - Side by Side", "sbs_system_prompt_custom"),
        ("Agent-Focused", "agent_system_prompt"),
        ("TAU Bench Comparison", "taubench_comparison_system_prompt"),
        ("Agentic SWE", "agentic_swe_system_prompt"),
        ("Tool-Focused", "agentic_tool_focused_prompt"),
        ("Reasoning-Focused", "agentic_reasoning_focused_prompt"),
        ("Reward Hacking", "agentic_reward_hacking_focused_prompt"),
    ]
    
    return prompts_list

def run_property_extraction(
    trajectory_index: int,
    prompt_name: str,
    task_description: str,
    df: pd.DataFrame
) -> Tuple[str, str]:
    """Run property extraction on a single trajectory and return formatted HTML.
    
    Args:
        trajectory_index: Index of trajectory in DataFrame
        prompt_name: Name of the system prompt to use
        task_description: Optional custom task description
        df: DataFrame with conversation data
        
    Returns:
        Tuple of (properties_html, highlighted_conversation_html)
    """
    if not USE_EXTRACTION:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ Extraction utilities not available</p>",
            ""
        )
    
    if df is None or trajectory_index >= len(df):
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ Invalid trajectory selection</p>",
            ""
        )
    
    try:
        # Get the current row
        current_row = df.iloc[trajectory_index:trajectory_index+1].copy()
        
        # Detect method from DataFrame columns
        method = detect_method(list(df.columns))
        if method is None:
            return (
                "<p style='color: #e74c3c; padding: 20px;'>❌ Unable to detect dataset format</p>",
                ""
            )
        
        # Get the system prompt
        system_prompt_text = getattr(stringsight_prompts, prompt_name, None)
        if system_prompt_text is None:
            return (
                f"<p style='color: #e74c3c; padding: 20px;'>❌ Prompt '{prompt_name}' not found</p>",
                ""
            )
        
        # Clean up task description (strip whitespace, treat empty as None)
        task_desc = task_description.strip() if task_description and task_description.strip() else None
        
        # Check if this prompt requires task_description
        requires_task_desc = "{task_description}" in system_prompt_text
        if requires_task_desc and not task_desc:
            return (
                "<p style='color: #e74c3c; padding: 20px;'>⚠️ This prompt requires a task description. Please provide one in the text box above.</p>",
                ""
            )
        
        # Run extraction
        result = extract_properties_only(
            current_row,
            method=method,
            system_prompt=system_prompt_text,
            task_description=task_desc,
            model_name="gpt-4o-mini",  # Using mini for demo/speed
            temperature=0.7,
            max_workers=1,
            include_scores_in_prompt=True,
            use_wandb=False,
            return_debug=False
        )
        
        # Handle result (can be PropertyDataset or tuple)
        if isinstance(result, tuple):
            dataset, failures = result
        else:
            dataset = result
            failures = []
        
        properties = dataset.properties if hasattr(dataset, 'properties') else []
        
        if not properties:
            return (
                "<p style='color: #ffa500; padding: 20px;'>ℹ️ No properties extracted. The model may not have found any notable behaviors.</p>",
                ""
            )
        
        # Format properties as HTML
        properties_html = format_properties_html(properties)
        
        # For new design we highlight in existing preview; return empty second value (unused)
        return properties_html, ""
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return (
            f"<p style='color: #e74c3c; padding: 20px;'>❌ Error during extraction: {html.escape(str(e))}</p>"
            f"<details><summary>Show details</summary><pre>{html.escape(error_details)}</pre></details>",
            ""
        )

def format_properties_html(properties: List[Any], include_click_instruction: bool = True) -> str:
    """Format extracted properties as HTML cards with click handlers.
    
    Args:
        properties: List of Property objects
        include_click_instruction: Whether to show instruction about clicking properties
        
    Returns:
        HTML string with formatted properties (with data attributes for click handling)
    """
    if not properties:
        return "<p style='color: #6c757d;'>No properties found</p>"
    
    html_parts = [
        "<div style='padding: 20px;'>",
        f"<h3 style='color: #333; margin-bottom: 20px;'>Extracted Properties ({len(properties)})</h3>"
    ]
    
    # Color palette for per-property highlight mapping (matches preview highlighting)
    palette = _color_palette()
    
    for i, prop in enumerate(properties, 1):
        # Get property attributes
        prop_desc = getattr(prop, 'property_description', 'N/A')
        category = getattr(prop, 'category', 'N/A')
        reason = getattr(prop, 'reason', 'N/A')
        evidence = getattr(prop, 'evidence', 'N/A')
        behavior_type = getattr(prop, 'behavior_type', 'N/A')
        contains_errors = getattr(prop, 'contains_errors', False)
        unexpected = getattr(prop, 'unexpected_behavior', False)
        model_name = getattr(prop, 'model', None)
        
        # Pick highlight color for this property (used in Evidence box and dot)
        color = palette[(i - 1) % len(palette)]
        
        # Build property card
        card_html = f"""
        <div class="property-card" style="
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #333; font-size: 16px;">Property #{i}</h4>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; align-items: center;">
                    <span title="Highlight color" style="display:inline-block; width:12px; height:12px; border-radius:50%; background:{color}; border:1px solid rgba(0,0,0,0.08);"></span>
                    <span style="
                        background: #f0f0f0;
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: 600;
                        color: #333;
                    ">{html.escape(str(behavior_type))}</span>
                    {f'<span style="background: #ffc107; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">⚠️ Contains Errors</span>' if contains_errors else ''}
                    {f'<span style="background: #ff6b6b; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; color: white;">🔍 Unexpected</span>' if unexpected else ''}
                </div>
            </div>
            
            {f'<div style="margin-bottom: 12px;"><strong style="color: #666;">Model:</strong> <code style="background: #f8f9fa; padding: 2px 8px; border-radius: 4px;">{html.escape(str(model_name))}</code></div>' if model_name else ''}
            
            <div style="margin-bottom: 12px;">
                <strong style="color: #666;">Description:</strong>
                <p style="margin: 4px 0 0 0; color: #333; line-height: 1.6;">{html.escape(str(prop_desc))}</p>
            </div>
            
            <div style="margin-bottom: 12px;">
                <strong style="color: #666;">Category:</strong>
                <span style="color: #495057; margin-left: 8px;">{html.escape(str(category))}</span>
            </div>
            
            <div style="margin-bottom: 12px;">
                <strong style="color: #666;">Reason:</strong>
                <p style="margin: 4px 0 0 0; color: #495057; line-height: 1.6; font-style: italic;">{html.escape(str(reason))}</p>
            </div>
            
            <div style="
                background: #f8f9fa;
                padding: 12px;
                border-radius: 6px;
                border-left: 4px solid {color};
            ">
                <strong style="color: #666;">Evidence</strong>
                <p style="margin: 4px 0 0 0; color: #495057; line-height: 1.6; font-family: monospace; font-size: 13px;">{html.escape(str(evidence))}</p>
            </div>
        </div>
        """
        
        html_parts.append(card_html)
    
    html_parts.append("</div>")
    
    return "".join(html_parts)

def format_conversation_with_highlights(
    row_data: pd.Series,
    properties: List[Any],
    method: str,
    selected_property_index: Optional[int] = None
) -> str:
    """Format conversation with evidence highlighting.
    
    Args:
        row_data: Row from DataFrame with conversation data
        properties: List of Property objects with evidence
        method: Dataset method (single_model or side_by_side)
        selected_property_index: If specified, only highlight this property's evidence
        
    Returns:
        HTML string with highlighted conversation (without prompt section)
    """
    try:
        # Filter properties if a specific one is selected
        props_to_highlight = properties
        if selected_property_index is not None and 0 <= selected_property_index < len(properties):
            props_to_highlight = [properties[selected_property_index]]
        
        # Get the conversation content
        if method == "single_model":
            model_response = row_data.get('model_response', [])
            model_name = row_data.get('model', 'Model')
            
            # Apply highlighting to the response text
            highlighted_response = apply_evidence_highlighting(model_response, props_to_highlight)
            
            # Format as conversation HTML (reuse existing display function)
            if USE_DASHBOARD_FORMAT:
                try:
                    if isinstance(highlighted_response, list):
                        openai_format = highlighted_response
                    else:
                        openai_format = convert_to_openai_format(highlighted_response)
                    
                    conversation_html = display_openai_conversation_html(
                        openai_format,
                        use_accordion=True,
                        pretty_print_dicts=True,
                        evidence=None
                    )
                except Exception:
                    conversation_html = f"<pre>{html.escape(str(highlighted_response))}</pre>"
            else:
                conversation_html = f"<pre>{html.escape(str(highlighted_response))}</pre>"
            
            highlight_note = ""
            if selected_property_index is not None:
                highlight_note = f"<div style='background: #fff176; padding: 8px; border-radius: 4px; margin-bottom: 10px; font-size: 13px;'><strong>📍 Showing highlights for Property #{selected_property_index + 1}</strong></div>"
            
            return f"""
            <div style="padding: 20px;">
                <h3 style="color: #333; margin-bottom: 15px;">Conversation with Evidence Highlighted</h3>
                {highlight_note}
                <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
                    <strong>Model:</strong> {html.escape(str(model_name))}
                </div>
                {conversation_html}
            </div>
            """
            
        elif method == "side_by_side":
            model_a = row_data.get('model_a', 'Model A')
            model_b = row_data.get('model_b', 'Model B')
            response_a = row_data.get('model_a_response', [])
            response_b = row_data.get('model_b_response', [])
            
            # Separate properties by model, applying filter if needed
            if selected_property_index is not None and 0 <= selected_property_index < len(properties):
                selected_prop = properties[selected_property_index]
                selected_model = getattr(selected_prop, 'model', '')
                if selected_model == model_a:
                    props_a = [selected_prop]
                    props_b = []
                elif selected_model == model_b:
                    props_a = []
                    props_b = [selected_prop]
                else:
                    props_a = []
                    props_b = []
            else:
                props_a = [p for p in props_to_highlight if getattr(p, 'model', '') == model_a]
                props_b = [p for p in props_to_highlight if getattr(p, 'model', '') == model_b]
            
            # Apply highlighting
            highlighted_a = apply_evidence_highlighting(response_a, props_a)
            highlighted_b = apply_evidence_highlighting(response_b, props_b)
            
            # Format both responses
            html_a = format_single_response_html(highlighted_a, model_a)
            html_b = format_single_response_html(highlighted_b, model_b)
            
            highlight_note = ""
            if selected_property_index is not None:
                highlight_note = f"<div style='background: #fff176; padding: 8px; border-radius: 4px; margin-bottom: 10px; font-size: 13px;'><strong>📍 Showing highlights for Property #{selected_property_index + 1}</strong></div>"
            
            return f"""
            <div style="padding: 20px;">
                <h3 style="color: #333; margin-bottom: 15px;">Conversations with Evidence Highlighted</h3>
                {highlight_note}
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    {html_a}
                    {html_b}
                </div>
            </div>
            """
        
        return "<p>Unable to format conversation</p>"
        
    except Exception as e:
        return f"<p style='color: #e74c3c;'>Error highlighting conversation: {html.escape(str(e))}</p>"

def _color_palette() -> List[str]:
    """Return up to 10 distinct highlight colors."""
    return [
        "#FFF176",  # yellow
        "#A5D6A7",  # green
        "#90CAF9",  # blue
        "#FFCC80",  # orange
        "#CE93D8",  # purple
        "#F48FB1",  # pink
        "#80DEEA",  # cyan
        "#B39DDB",  # indigo
        "#FFAB91",  # coral
        "#C5E1A5",  # light green
    ]


def _wrap_with_mark_colored(text: str, color: str) -> str:
    return (
        text.replace(
            HIGHLIGHT_START,
            f"<mark style=\"background-color: {color}; padding: 2px 4px; border-radius: 3px;\">",
        ).replace(HIGHLIGHT_END, "</mark>")
    )


def _render_model_response_block(content: Any, use_accordion: bool = True, evidence: Optional[List[str]] = None) -> str:
    """Render a model response using the dashboard chat bubble renderer - SAME FORMAT AS EXAMPLES TAB."""
    try:
        if USE_DASHBOARD_FORMAT:
            # Determine OpenAI-style messages list
            if isinstance(content, list):
                openai_format = content
            else:
                openai_format = convert_to_openai_format(content)
            # Sanitize unbalanced fenced code blocks in each message content
            sanitized_messages: List[Dict[str, Any]] = []
            for m in openai_format:
                if isinstance(m, dict):
                    c = m.get("content")
                    if isinstance(c, str):
                        msg = dict(m)
                        msg["content"] = _sanitize_markdown_fences(c)
                        sanitized_messages.append(msg)
                    else:
                        sanitized_messages.append(m)
                else:
                    sanitized_messages.append(m)
            # Use the exact same call as the Examples tab
            html_body = display_openai_conversation_html(
                sanitized_messages,
                use_accordion=use_accordion,
                pretty_print_dicts=True,
                evidence=evidence
            )
            # Clamp within a container to prevent overflow
            return (
                "<div class=\"ss-conv\" style=\"max-width:100%; box-sizing:border-box; overflow:hidden;\">"
                f"{html_body}"
                "</div>"
            )
        else:
            # Fallback: render with markdown (supports fenced code), ensure wrapping
            from markdown import markdown as _md
            def _md_render(txt: str) -> str:
                try:
                    return _md(
                        txt,
                        extensions=["fenced_code", "codehilite", "tables", "nl2br"],
                    )
                except Exception:
                    return f"<pre style=\"white-space:pre-wrap; word-wrap:break-word;\">{html.escape(txt)}</pre>"
            if isinstance(content, list):
                out_parts: List[str] = []
                for msg in content:
                    if isinstance(msg, dict):
                        role = html.escape(str(msg.get("role", "assistant")))
                        txt = msg.get("content", "")
                        if not isinstance(txt, str):
                            txt = str(txt)
                        txt = _sanitize_markdown_fences(txt)
                        out_parts.append(
                            f"<div style=\"margin-bottom:10px; max-width:100%; box-sizing:border-box;\"><div style=\"font-weight:600;color:#555;margin-bottom:4px;\">{role}</div><div style=\"overflow:auto;\">{_md_render(txt)}</div></div>"
                        )
                return "".join(out_parts) if out_parts else "<p style='color:#999'>No content</p>"
            else:
                txt = content if isinstance(content, str) else json.dumps(content, indent=2)
                txt = _sanitize_markdown_fences(txt)
                return f"<div style=\"overflow:auto; max-width:100%; box-sizing:border-box;\">{_md_render(txt)}</div>"
    except Exception as e:
        return f"<p style='color:#e74c3c;padding:12px;'>Error rendering: {html.escape(str(e))}</p>"


def apply_evidence_highlighting(conversation: Any, properties: List[Any]) -> Any:
    """Apply evidence highlighting to conversation text.
    
    Args:
        conversation: Conversation data (can be list of messages or string)
        properties: List of Property objects with evidence fields
        
    Returns:
        Conversation with highlighting placeholders inserted
    """
    if not properties:
        return conversation
    
    # Build color-coded evidence list per property
    colored_evidence: List[Tuple[Any, str]] = []  # (evidence, color)
    palette = _color_palette()
    for idx, prop in enumerate(properties[:10]):
        evidence = getattr(prop, 'evidence', None)
        if evidence:
            color = palette[idx % len(palette)]
            colored_evidence.append((evidence, color))
    if not colored_evidence:
        return conversation
    
    # Handle different conversation formats
    if isinstance(conversation, list):
        # List of message dicts - highlight content in each message
        highlighted_messages = []
        for msg in conversation:
            if isinstance(msg, dict):
                msg_copy = msg.copy()
                content = msg_copy.get('content', '')
                if isinstance(content, str):
                    # Sequentially apply different colors for each property's evidence
                    colored_content = content
                    for evidence, color in colored_evidence:
                        tmp = annotate_text_with_evidence_placeholders(
                            colored_content, evidence, n=3, overlap_threshold=0.5
                        )
                        colored_content = _wrap_with_mark_colored(tmp, color)
                    msg_copy['content'] = colored_content
                highlighted_messages.append(msg_copy)
            else:
                highlighted_messages.append(msg)
        return highlighted_messages
    
    elif isinstance(conversation, str):
        # Plain string - highlight directly
        colored_text = conversation
        for evidence, color in colored_evidence:
            tmp = annotate_text_with_evidence_placeholders(
                colored_text, evidence, n=3, overlap_threshold=0.5
            )
            colored_text = _wrap_with_mark_colored(tmp, color)
        return colored_text
    
    return conversation

def format_single_response_html(response: Any, model_name: str) -> str:
    """Format a single model response as HTML.
    
    Args:
        response: Response data (list of messages or string)
        model_name: Name of the model
        
    Returns:
        HTML string
    """
    try:
        if USE_DASHBOARD_FORMAT:
            if isinstance(response, list):
                openai_format = response
            else:
                openai_format = convert_to_openai_format(response)
            
            content_html = display_openai_conversation_html(
                openai_format,
                use_accordion=False,
                pretty_print_dicts=True,
                evidence=None
            )
        else:
            content_html = f"<pre>{html.escape(str(response))}</pre>"
        
        return f"""
        <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px;">
            <h4 style="color: #333; margin: 0 0 10px 0;">{html.escape(str(model_name))}</h4>
            {content_html}
        </div>
        """
    except Exception:
        return f"""
        <div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px;">
            <h4 style="color: #333; margin: 0 0 10px 0;">{html.escape(str(model_name))}</h4>
            <pre>{html.escape(str(response))}</pre>
        </div>
        """

def main(file_path: str = "data/taubench/airline_data_oai_format.jsonl"):
    """Create and launch the Gradio interface
    
    Args:
        file_path: Path to JSONL file with chat trajectories
    """
    # Load data
    df = load_data(file_path)
    if df is None or len(df) == 0:
        # Create error interface with proper syntax
        def error_message():
            return f"Error: Could not load data file '{file_path}'. Please check that the file exists and contains valid JSONL."
        
        return gr.Interface(
            fn=error_message,
            inputs=None,
            outputs=gr.Textbox(label="Error"),
            title="Chat Trajectory Viewer - Error"
        )
    
    # Create trajectory options
    trajectory_options = create_trajectory_options(df)
    
    def update_trajectory(trajectory_choice: str) -> str:
        """Update the conversation HTML when trajectory selection changes.
        
        Args:
            trajectory_choice: String like "Trajectory {i} ..."; we extract the {i}.
        
        Returns:
            Combined HTML string for the selected trajectory.
        """
        if not trajectory_choice:
            return ""
        try:
            index = int(trajectory_choice.split()[1])
        except (ValueError, IndexError):
            return ""
        return compose_trajectory_html(index, df)
    
    # Create the interface with tabs
    with gr.Blocks(title="Chat Trajectory Viewer", theme=gr.themes.Soft()) as demo:
        # Add custom CSS for better markdown rendering
        gr.HTML("""
        <style>
        /* Code block styling */
        pre {
            background-color: #f6f8fa !important;
            border: 1px solid #e1e4e8 !important;
            border-radius: 6px !important;
            padding: 16px !important;
            overflow-x: auto !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
            font-size: 14px !important;
            line-height: 1.45 !important;
            white-space: pre-wrap !important;      /* wrap long lines */
            word-break: break-word !important;     /* break long tokens */
            overflow-wrap: anywhere !important;    /* last-resort breaks */
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        
        code {
            background-color: #f6f8fa !important;
            border-radius: 3px !important;
            padding: 2px 4px !important;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
            font-size: 13px !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        /* Keep rendered conversation bubbles from overflow */
        .ss-conv, .ss-conv * {
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        .ss-conv pre, .ss-conv code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }
        
        /* Table styling */
        table {
            border-collapse: collapse !important;
            width: 100% !important;
            margin: 16px 0 !important;
        }
        
        th, td {
            border: 1px solid #e1e4e8 !important;
            padding: 8px 12px !important;
            text-align: left !important;
        }
        
        th {
            background-color: #f6f8fa !important;
            font-weight: 600 !important;
        }
        
        /* List styling */
        ul, ol {
            padding-left: 20px !important;
            margin: 8px 0 !important;
        }
        
        li {
            margin: 4px 0 !important;
        }
        
        /* Blockquote styling */
        blockquote {
            border-left: 4px solid #dfe2e5 !important;
            margin: 16px 0 !important;
            padding-left: 16px !important;
            color: #6a737d !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            margin: 16px 0 8px 0 !important;
            font-weight: 600 !important;
        }
        
        /* Links */
        a {
            color: #0366d6 !important;
            text-decoration: none !important;
        }
        
        a:hover {
            text-decoration: underline !important;
        }
        
        /* Horizontal rule */
        hr {
            border: none !important;
            border-top: 1px solid #e1e4e8 !important;
            margin: 16px 0 !important;
        }
        </style>
        """)
        
        gr.Markdown("# Chat Trajectory Viewer")
        gr.Markdown("Browse through conversation trajectories and explore dataset statistics")
        
        with gr.Tabs():
            # Overview Tab
            with gr.Tab("Overview"):
                # Score column selector
                potential_columns = get_potential_score_columns(df)
                
                if potential_columns:
                    with gr.Accordion("Customize Benchmark Metrics", open=False):
                        gr.Markdown("""
                        Select which columns should be used as benchmark metrics. 
                        By default, the system auto-detects numeric columns and 'score' dictionaries.
                        """)
                        
                        score_column_selector = gr.CheckboxGroup(
                            choices=potential_columns,
                            value=potential_columns,  # All selected by default
                            label="Metric Columns",
                            info="Select columns to include in benchmark results"
                        )
                        
                        plot_type_selector = gr.Radio(
                            choices=["Bar Chart", "Radar Chart", "Both"],
                            value="Bar Chart",
                            label="Plot Type",
                            info="Choose visualization type"
                        )
                        
                        refresh_overview_btn = gr.Button("Refresh Overview", variant="primary")
                
                overview_output = gr.HTML(
                    label="Dataset Statistics",
                    value=create_overview_tab(df, potential_columns)
                )
                with gr.Row():
                    benchmark_table = gr.Dataframe(
                        label="Benchmark Results (click column headers to sort)",
                        value=build_benchmark_results_df(df, potential_columns if potential_columns else None),
                        wrap=True,
                        interactive=True
                    )
                
                # Benchmark plots
                if potential_columns:
                    gr.Markdown("### Benchmark Metrics Visualization")
                    
                    # Get available metrics
                    model_scores_with_ci = extract_benchmark_scores_with_ci(df, potential_columns)
                    available_metrics = set()
                    for scores in model_scores_with_ci.values():
                        available_metrics.update(scores.keys())
                    available_metrics = sorted(available_metrics)
                    
                    if available_metrics:
                        with gr.Row():
                            metric_selector = gr.Dropdown(
                                choices=available_metrics,
                                value=available_metrics[0],
                                label="Select Metric to Plot",
                                info="Choose which metric to visualize with confidence intervals"
                            )
                    
                        with gr.Row():
                            benchmark_plot = gr.Plot(
                                label="Benchmark Comparison (with 95% Confidence Intervals)",
                                value=create_benchmark_plot_single_metric(df, available_metrics[0], potential_columns) if available_metrics else None
                            )
                        
                        with gr.Row():
                            benchmark_radar = gr.Plot(
                                label="Radar Chart (All Metrics)",
                                value=create_benchmark_radar_plot(df, potential_columns),
                                visible=False
                            )
                        
                        # Removed explicit Refresh Overview button; updates happen via control changes
                        
                        # Update plot when metric changes
                        metric_selector.change(
                            fn=lambda selected_metric: create_benchmark_plot_single_metric(df, selected_metric, potential_columns),
                            inputs=[metric_selector],
                            outputs=[benchmark_plot]
                        )
                        
                        # Also update on plot type change
                        plot_type_selector.change(
                            fn=lambda plot_type: (
                                gr.update(visible=(plot_type in ["Bar Chart", "Both"])),
                                gr.update(visible=(plot_type in ["Radar Chart", "Both"]))
                            ),
                            inputs=[plot_type_selector],
                            outputs=[benchmark_plot, benchmark_radar]
                        )
            
            # Examples Tab (replaces Trajectories)
            with gr.Tab("Examples"):
                # Filters row
                p_choices, m_choices, prop_choices = get_examples_dropdown_choices(df)
                with gr.Row():
                    prompt_dd = gr.Dropdown(choices=p_choices, value=(p_choices[0] if p_choices else None), label="Prompt")
                    model_dd = gr.Dropdown(choices=m_choices, value=(m_choices[0] if m_choices else None), label="Model")
                    property_dd = gr.Dropdown(choices=prop_choices, value=(prop_choices[0] if prop_choices else None), label="Cluster")
                with gr.Row():
                    max_examples_slider = gr.Slider(minimum=1, maximum=50, step=1, value=5, label="Max examples")
                    search_tb = gr.Textbox(label="Search", placeholder="Filter by text in prompt/response")
                with gr.Row():
                    examples_output = gr.HTML(label="Examples")

                def update_examples(selected_prompt: str, selected_model: str, selected_property: str, max_examples: int, search: str) -> str:
                    return build_examples_html(
                        df=df,
                        selected_prompt=selected_prompt,
                        selected_model=selected_model,
                        selected_property=selected_property,
                        max_examples=max_examples,
                        search_term=search or ""
                    )

                # Wire up changes
                for ctrl in [prompt_dd, model_dd, property_dd, max_examples_slider, search_tb]:
                    ctrl.change(
                        fn=update_examples,
                        inputs=[prompt_dd, model_dd, property_dd, max_examples_slider, search_tb],
                        outputs=[examples_output]
                    )

                # Initialize examples view
                examples_output.value = update_examples(
                    p_choices[0] if p_choices else "All Prompts",
                    m_choices[0] if m_choices else "All Models",
                    prop_choices[0] if prop_choices else "All Clusters",
                    5,
                    ""
                )
            
            # Property Extraction Demo Tab
            if USE_EXTRACTION:
                with gr.Tab("Property Extraction Demo"):
                    gr.Markdown("""
                    ### Property Extraction Demo
                    
                    Extract and explore properties from conversations in your dataset. 
                    Configure the extraction below, then click properties to highlight their evidence in the conversation.
                    """)
                    
                    # Get available prompts
                    available_prompts = get_available_prompts()
                    prompt_choices = [display_name for display_name, _ in available_prompts]
                    prompt_value_map = {display_name: prompt_name for display_name, prompt_name in available_prompts}
                    
                    # Two-column layout: left = extraction config & results; right = conversation filters & preview
                    with gr.Row():
                        with gr.Column(scale=5, min_width=360):
                            gr.Markdown("### Configuration")
                            # Extraction prompt selector
                            extraction_prompt_dropdown = gr.Dropdown(
                                choices=prompt_choices,
                                label="System Prompt",
                                value=prompt_choices[0] if prompt_choices else None,
                                info="Select the extraction prompt"
                            )
                            # Full prompt accordion
                            with gr.Accordion("📄 View Full System Prompt", open=False):
                                extraction_prompt_display = gr.Markdown(
                                    value=f"```text\n{getattr(stringsight_prompts, prompt_value_map.get(prompt_choices[0], 'single_model_system_prompt'), 'No prompt selected')}\n```" if prompt_choices else "No prompts available"
                                )
                            # Optional task description
                            extraction_task_description = gr.Textbox(
                                label="Task Description (Optional)",
                                placeholder="Enter a custom task description for prompts that support it...",
                                lines=2,
                                info="Required for 'Custom Task' prompts"
                            )
                            # Extract button + status
                            extraction_button = gr.Button("🔍 Extract Properties", variant="primary", size="lg")
                            extraction_status = gr.HTML(value="<p style='color: #6c757d; padding: 10px;'>Click 'Extract Properties' to begin</p>")
                            # Properties output
                            gr.Markdown("---")
                            gr.Markdown("### Extracted Properties")
                            extraction_properties_state = gr.State([])
                            extraction_row_data_state = gr.State(None)
                            extraction_method_state = gr.State(None)
                            extraction_properties_output = gr.HTML(label="Properties", value="<p style='color: #999; padding: 20px;'>Properties will appear here after extraction</p>")

                        with gr.Column(scale=7):
                            gr.Markdown("### Selected Conversation Preview")
                            # Conversation filters on the right
                            available_models = sorted(df['model'].dropna().unique().tolist()) if 'model' in df.columns else []
                            available_prompts = sorted(df['prompt'].dropna().unique().tolist()) if 'prompt' in df.columns else []
                            with gr.Row():
                                extraction_model_filter = gr.Dropdown(
                                    choices=["All Models"] + available_models,
                                    label="Model",
                                    value=(available_models[0] if available_models else None),
                                    info="Filter by model"
                                )
                                extraction_prompt_filter = gr.Dropdown(
                                    choices=["All Prompts"] + available_prompts,
                                    label="Prompt",
                                    value=(available_prompts[0] if available_prompts else None),
                                    info="Filter by prompt"
                                )
                            with gr.Accordion("Original Conversation", open=True) as extraction_preview_accordion:
                                extraction_preview_output = gr.HTML(
                                    value="<p style='color:#999;padding:12px;'>Use the filters above to show a response</p>"
                                )
                    
                    # Wire up the extraction button
                    def run_extraction_wrapper(model_choice: str, prompt_choice: str, prompt_display_name: str, task_desc: str):
                        """Run extraction based on filters (model, prompt)."""
                        if not prompt_display_name:
                            return (
                                "<p style='color: #e74c3c; padding: 20px;'>❌ Please select a prompt</p>",
                                "",
                                "",
                                gr.update(visible=False),
                                [],
                                None,
                                None
                            )
                        
                        # Get the actual prompt name
                        prompt_name = prompt_value_map.get(prompt_display_name)
                        if not prompt_name:
                            return (
                                f"<p style='color: #e74c3c; padding: 20px;'>❌ Invalid prompt selection: {prompt_display_name}</p>",
                                "",
                                "",
                                gr.update(visible=False),
                                [],
                                None,
                                None
                            )
                        
                        # Filter dataframe by model and prompt to pick first matching row
                        filtered = df
                        if model_choice and model_choice != "All Models" and 'model' in df.columns:
                            filtered = filtered[filtered['model'] == model_choice]
                        if prompt_choice and prompt_choice != "All Prompts" and 'prompt' in df.columns:
                            filtered = filtered[filtered['prompt'] == prompt_choice]
                        if len(filtered) == 0:
                            return (
                                "<p style='color:#e74c3c; padding: 20px;'>❌ No rows match the selected filters</p>",
                                "",
                                "",
                                gr.update(visible=False),
                                [],
                                None,
                                None
                            )
                        chosen_idx = filtered.index.tolist()[0]
                        current_row = df.iloc[chosen_idx:chosen_idx+1].copy()
                        method = detect_method(list(df.columns))
                        
                        if method is None:
                            return (
                                "<p style='color: #e74c3c; padding: 20px;'>❌ Unable to detect dataset format</p>",
                                "",
                                "",
                                gr.update(visible=False),
                                [],
                                None,
                                None
                            )
                        
                        # Get system prompt
                        system_prompt_text = getattr(stringsight_prompts, prompt_name, None)
                        if system_prompt_text is None:
                            return (
                                f"<p style='color: #e74c3c; padding: 20px;'>❌ Prompt '{prompt_name}' not found</p>",
                                "",
                                "",
                                gr.update(visible=False),
                                [],
                                None,
                                None
                            )
                        
                        # Validate task description
                        task_desc_clean = task_desc.strip() if task_desc and task_desc.strip() else None
                        requires_task_desc = "{task_description}" in system_prompt_text
                        if requires_task_desc and not task_desc_clean:
                            return (
                                "<p style='color: #e74c3c; padding: 20px;'>⚠️ This prompt requires a task description</p>",
                                "",
                                "",
                                gr.update(visible=False),
                                [],
                                None,
                                None
                            )
                        
                        # Run extraction
                        try:
                            result = extract_properties_only(
                                current_row,
                                method=method,
                                system_prompt=system_prompt_text,
                                task_description=task_desc_clean,
                                model_name="gpt-4o-mini",
                                temperature=0.7,
                                max_workers=1,
                                include_scores_in_prompt=True,
                                use_wandb=False,
                                return_debug=False
                            )
                            
                            if isinstance(result, tuple):
                                dataset, _ = result
                            else:
                                dataset = result
                            
                            properties = dataset.properties if hasattr(dataset, 'properties') else []
                            
                            if not properties:
                                return (
                                    "<p style='color: #ffa500; padding: 20px;'>ℹ️ No properties extracted</p>",
                                    "",
                                    "",
                                    gr.update(visible=False),
                                    [],
                                    None,
                                    None
                                )
                            
                            # Format outputs
                            props_html = format_properties_html(properties, include_click_instruction=False)
                            row_data = df.iloc[chosen_idx]
                            
                            # Render highlighted conversation
                            content = row_data.get('model_response', '')
                            highlighted = apply_evidence_highlighting(content, properties)
                            highlighted_html = _render_model_response_block(highlighted, use_accordion=True)
                            
                            final_status = f"<p style='color: #28a745; padding: 10px;'>✓ Extraction complete! Found {len(properties)} properties.</p>"
                            
                            return (
                                final_status,
                                props_html,
                                properties,  # Store in state
                                row_data,
                                method
                            )
                            
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            return (
                                f"<p style='color: #e74c3c; padding: 20px;'>❌ Error: {html.escape(str(e))}</p>"
                                f"<details><summary>Details</summary><pre>{html.escape(error_details)}</pre></details>",
                                "",
                                [],
                                None,
                                None
                            )
                    
                    extraction_button.click(
                        fn=run_extraction_wrapper,
                        inputs=[extraction_model_filter, extraction_prompt_filter, extraction_prompt_dropdown, extraction_task_description],
                        outputs=[
                            extraction_status,
                            extraction_properties_output,
                            extraction_properties_state,
                            extraction_row_data_state,
                            extraction_method_state
                        ]
                    )
                    
                    # When filters change post-extraction, update BOTH preview and highlighted conversation
                    def refresh_preview_and_highlights(model_choice: str, prompt_choice: str, properties: List[Any]):
                        """Update original preview when filters change (no post-extraction highlight)."""
                        filtered = df
                        if model_choice and model_choice != "All Models" and 'model' in df.columns:
                            filtered = filtered[filtered['model'] == model_choice]
                        if prompt_choice and prompt_choice != "All Prompts" and 'prompt' in df.columns:
                            filtered = filtered[filtered['prompt'] == prompt_choice]
                        if len(filtered) == 0:
                            no_data = "<p style='color:#999;padding:12px;'>No response for current filters</p>"
                            return no_data
                        
                        row = filtered.iloc[0]
                        content = row.get('model_response', '')
                        
                        # Render original conversation
                        original_html = _render_model_response_block(content, use_accordion=True)
                        return original_html
                    
                    extraction_model_filter.change(
                        fn=refresh_preview_and_highlights,
                        inputs=[extraction_model_filter, extraction_prompt_filter, extraction_properties_state],
                        outputs=[extraction_preview_output]
                    )
                    extraction_prompt_filter.change(
                        fn=refresh_preview_and_highlights,
                        inputs=[extraction_model_filter, extraction_prompt_filter, extraction_properties_state],
                        outputs=[extraction_preview_output]
                    )
                    
                    # Update prompt display when prompt selection changes
                    def update_prompt_display(prompt_display_name: str):
                        """Update the system prompt display when prompt selection changes."""
                        if not prompt_display_name:
                            return "No prompt selected"
                        
                        prompt_name = prompt_value_map.get(prompt_display_name)
                        if not prompt_name:
                            return f"Prompt not found: {prompt_display_name}"
                        
                        prompt_text = getattr(stringsight_prompts, prompt_name, None)
                        if prompt_text is None:
                            return f"Prompt '{prompt_name}' not available"
                        
                        # Format as markdown - properly display with line breaks
                        # Use triple backticks for code block which preserves formatting
                        return f"```text\n{prompt_text}\n```"
                    
                    extraction_prompt_dropdown.change(
                        fn=update_prompt_display,
                        inputs=[extraction_prompt_dropdown],
                        outputs=[extraction_prompt_display]
                    )
                    
                    # Initialize preview with first available conversation
                    if available_models or available_prompts:
                        initial_model = available_models[0] if available_models else None
                        initial_prompt = available_prompts[0] if available_prompts else None
                        extraction_preview_output.value = refresh_preview_and_highlights(initial_model, initial_prompt, [])
            
            # Side by Side Tab (only if dataset has this format)
            if has_side_by_side_format(df):
                side_by_side_options = create_side_by_side_options(df)
                
                with gr.Tab("Side by Side"):
                    gr.Markdown("### Compare model responses side-by-side")
                    
                    with gr.Row():
                        comparison_dropdown = gr.Dropdown(
                            choices=side_by_side_options,
                            label="Select Comparison",
                            value=side_by_side_options[0] if side_by_side_options else None
                        )
                    
                    with gr.Row():
                        sbs_metadata_output = gr.HTML(label="Metadata")
                    
                    with gr.Row():
                        sbs_prompt_output = gr.HTML(label="Prompt")
                    
                    with gr.Row():
                        sbs_comparison_output = gr.HTML(label="Model Comparison")
                    
                    # Set up the event handler
                    def update_side_by_side(comparison_choice: str):
                        """Update the display when comparison selection changes"""
                        if not comparison_choice:
                            return "", "", ""
                        
                        # Extract index from choice string
                        try:
                            index = int(comparison_choice.split()[1])
                        except (ValueError, IndexError):
                            return "", "", ""
                        
                        return display_side_by_side_comparison(index, df)
                    
                    comparison_dropdown.change(
                        fn=update_side_by_side,
                        inputs=[comparison_dropdown],
                        outputs=[sbs_metadata_output, sbs_prompt_output, sbs_comparison_output]
                    )
                    
                    # Initialize with first comparison
                    if side_by_side_options:
                        initial_sbs_metadata, initial_sbs_prompt, initial_sbs_comparison = display_side_by_side_comparison(0, df)
                        sbs_metadata_output.value = initial_sbs_metadata
                        sbs_prompt_output.value = initial_sbs_prompt
                        sbs_comparison_output.value = initial_sbs_comparison
    
    return demo

def _sanitize_markdown_fences(text: str) -> str:
    """Auto-close unbalanced fenced code blocks and normalize malformed fences.
    - Ensures an even count of ``` so the renderer doesn't treat rest of message as code.
    - Collapses stray backtick variations to standard ```.
    """
    try:
        # Normalize any ```sql or ```
        # Do not modify content inside code; just count raw fences
        fence_count = text.count("```")
        if fence_count % 2 == 1:
            # Append a closing fence at the end
            text = text + "\n```"
        return text
    except Exception:
        return text

if __name__ == "__main__":
    import sys
    
    # Minimal CLI parsing: positional file path and optional --share flag
    # Usage:
    #   python -m stringsight.dashboard.chat_viewer [file.jsonl] [--share]
    args = sys.argv[1:]
    share = False
    file_path = "data/taubench/airline_data_oai_format.jsonl"
    
    for token in list(args):
        if token == "--share":
            share = True
            args.remove(token)
    
    if len(args) >= 1:
        file_path = args[0]
    
    print(f"Loading trajectories from: {file_path}")
    print("\nTo use a different file, run:")
    print("  python -m stringsight.dashboard.chat_viewer path/to/your/file.jsonl [--share]")
    print()
    
    # Show which display format is being used
    if USE_DASHBOARD_FORMAT:
        print("✅ Using StringSight dashboard display format (chat bubbles, syntax highlighting, tool calls)")
    else:
        print("⚠️  Using fallback display format (dashboard utilities not available)")
    
    if USE_SIDE_BY_SIDE:
        print("✅ Side-by-side comparison support available")
    else:
        print("⚠️  Side-by-side comparison not available (dashboard utilities not imported)")
    
    if USE_EXTRACTION:
        print("✅ Property extraction demo available")
    else:
        print("⚠️  Property extraction demo not available (extraction utilities not imported)")
    print()
    
    demo = main(file_path)
    
    # Check if dataset has side-by-side format and print info
    import pandas as pd
    try:
        df_check = pd.read_json(file_path, lines=True)
        if has_side_by_side_format(df_check):
            print("✅ Dataset has side-by-side comparison format (model_a vs model_b)")
            print("   → Check the 'Side by Side' tab to compare model responses")
        else:
            print("ℹ️  Dataset has single-trajectory format")
            print("   → Use the 'Trajectories' tab to browse conversations")
        print()
    except:
        pass
    
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860, show_error=False)