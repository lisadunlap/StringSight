"""
Standalone Gradio app for loading arbitrary JSON/JSONL data and viewing:
- Overview metrics (model counts, unique/shared prompts, average scores)
- Examples cards (score + example text) with raw-row accordion for debugging

This app is intentionally separate from the main dashboard for debug purposes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
from pathlib import Path
import html
import ast

import gradio as gr
import pandas as pd
import plotly.express as px
from .utils import get_example_data, format_examples_display


def _read_input_file(path_str: str) -> Tuple[pd.DataFrame, List[str]]:

    warnings: List[str] = []
    if not path_str or not str(path_str).strip():
        raise ValueError("Please provide a file path.")

    path = Path(path_str).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            df = pd.DataFrame(data["data"])  # permissive: common wrapper
            warnings.append("Wrapped JSON detected; used 'data' field.")
        else:
            raise ValueError("JSON must be a list of objects (or have a 'data' list).")
    else:
        raise ValueError("Unsupported file type. Please provide a .json or .jsonl file.")

    if df is None or len(df) == 0:
        raise ValueError("Loaded file is empty.")

    return df, warnings


def _guess_response_column(columns: List[str]) -> Optional[str]:

    candidates = [
        "response",
        "model_response",
        "completion",
        "output",
        "assistant_response",
        "answer",
    ]
    for c in candidates:
        if c in columns:
            return c
    return None


def _guess_prompt_column(columns: List[str]) -> Optional[str]:

    candidates = [
        "prompt",
        "question",
        "input",
        "instruction",
        "user_prompt",
    ]
    for c in candidates:
        if c in columns:
            return c
    return None


def _build_overview(df: pd.DataFrame, model_col: Optional[str], prompt_col: str, score_cols: List[str], score_dict_col: Optional[str]) -> Tuple[str, Optional[Dict[str, float]]]:

    parts: List[str] = []

    # Model counts
    if model_col and model_col in df.columns:
        n_models = int(df[model_col].nunique(dropna=True))
        parts.append(f"<div><strong>Models:</strong> {n_models}</div>")
        # shared prompts across all selected models (in this app, selection is global; later can filter)
        # compute prompts that appear for all models
        unique_models = df[model_col].dropna().unique().tolist()
        if len(unique_models) > 0:
            by_prompt = df.groupby(prompt_col)[model_col].nunique()
            shared = int((by_prompt >= len(unique_models)).sum())
            parts.append(f"<div><strong>Prompts shared across all models:</strong> {shared}</div>")
        else:
            parts.append("<div><strong>Prompts shared across all models:</strong> N/A</div>")
    else:
        parts.append("<div><strong>Models:</strong> N/A</div>")
        parts.append("<div><strong>Prompts shared across all models:</strong> N/A</div>")

    # Unique prompts
    n_prompts = int(pd.Series(df[prompt_col]).nunique(dropna=True))
    parts.append(f"<div><strong>Unique prompts:</strong> {n_prompts}</div>")

    # Benchmark metrics: build averages dict
    averages: Dict[str, float] = {}
    if score_dict_col and score_dict_col in df.columns:
        # Expect dict-like
        def _extract_keys(obj: Any) -> Dict[str, float]:
            # Accept dict or stringified dict/JSON
            if isinstance(obj, dict):
                source = obj
            elif isinstance(obj, str):
                parsed: Any = None
                try:
                    parsed = json.loads(obj)
                except Exception:
                    try:
                        parsed = ast.literal_eval(obj)
                    except Exception:
                        parsed = None
                source = parsed if isinstance(parsed, dict) else None
            else:
                source = None
            if not isinstance(source, dict):
                return {}
            return {k: v for k, v in source.items() if isinstance(v, (int, float))}
        series = df[score_dict_col].apply(_extract_keys)
        # collect all keys
        all_keys = set()
        for d in series:
            all_keys.update(d.keys())
        for k in sorted(all_keys):
            vals = [d[k] for d in series if k in d]
            if len(vals) > 0:
                averages[k] = float(pd.Series(vals).mean())
    elif score_cols:
        for col in score_cols:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(numeric) > 0:
                    averages[col] = float(numeric.mean())

    # Overview HTML
    parts_html = "".join(parts)
    if averages:
        metrics_html = "<div style='margin-top:8px;'><strong>Benchmark metrics (averages):</strong> " + \
            ", ".join([f"{k}: {v:.3f}" for k, v in averages.items()]) + "</div>"
    else:
        metrics_html = "<div style='margin-top:8px; color:#666;'>No numeric metrics detected.</div>"

    html = f"<div style='padding:12px; line-height:1.6;'>{parts_html}{metrics_html}</div>"
    return html, (averages if averages else None)


def _build_metrics_plot(averages: Optional[Dict[str, float]]):

    if not averages:
        return None
    keys = list(averages.keys())
    vals = [averages[k] for k in keys]
    df = pd.DataFrame({"metric": keys, "value": vals})
    fig = px.bar(df, x="metric", y="value", title="Average metrics", text="value")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".3f", xaxis_title="", yaxis_title="")
    return fig


def create_app() -> gr.Blocks:

    with gr.Blocks(title="Example Viewer (Debug)") as demo:
        gr.Markdown("# 🧪 Example Viewer (Debug)")

        with gr.Row():
            file_path_tb = gr.Textbox(label="Insert filepath (.json/.jsonl)", placeholder="/path/to/data.jsonl")
            load_btn = gr.Button("Load", variant="primary")

        status_md = gr.Markdown("", elem_id="status")

        df_state = gr.State(value=None)  # holds the loaded DataFrame

        # Sample raw row (debug)
        with gr.Accordion("Sample row (debug)", open=False):
            sample_row_code = gr.Code(label="Raw row (JSON)", value="", language="json")

        # Column selection
        with gr.Accordion("Column mapping", open=True):
            with gr.Row():
                data_mode_dd = gr.Radio(choices=["Single", "Side-by-side"], value="Single", label="Data mode")
                prompt_col_dd = gr.Dropdown(label="Prompt column", choices=[], value=None)
                response_col_dd = gr.Dropdown(label="Response column", choices=[], value=None)
                model_col_dd = gr.Dropdown(label="Model column (optional)", choices=[], value=None)
            with gr.Row():
                score_dict_col_dd = gr.Dropdown(label="Score dict column (optional)", choices=[], value=None, info="If provided, it should be a dict per row of metric_name -> value")
                score_cols_cg = gr.CheckboxGroup(label="Numeric score columns (optional)", choices=[], value=[], info="Select one or more numeric columns to treat as metrics")
            # Side-by-side mappings (hidden unless Side-by-side mode)
            with gr.Row(visible=False) as sxs_row1:
                resp_a_col_dd = gr.Dropdown(label="Response A column", choices=[], value=None)
                resp_b_col_dd = gr.Dropdown(label="Response B column", choices=[], value=None)
                model_a_col_dd = gr.Dropdown(label="Model A column (optional)", choices=[], value=None)
                model_b_col_dd = gr.Dropdown(label="Model B column (optional)", choices=[], value=None)
            with gr.Row(visible=False) as sxs_row2:
                score_dict_a_dd = gr.Dropdown(label="Score A dict column (optional)", choices=[], value=None)
                score_dict_b_dd = gr.Dropdown(label="Score B dict column (optional)", choices=[], value=None)
            sxs_metrics_md = gr.Markdown(visible=False, value="")
            columns_md = gr.Markdown(visible=True, value="")
            with gr.Row():
                model_filter = gr.CheckboxGroup(label="Filter by model (optional)", choices=[], value=[])
                select_all_models_btn = gr.Button("Select All Models")

        # Overview section (no tabs)
        overview_html = gr.HTML(visible=True)
        with gr.Row():
            view_toggle = gr.Radio(choices=["Plot", "Table"], value="Plot", label="Benchmark view", visible=False)
        overview_plot = gr.Plot(visible=False)
        overview_table = gr.Dataframe(visible=False, wrap=True)

        # Examples section
        with gr.Row():
            max_examples = gr.Slider(minimum=1, maximum=200, step=1, value=20, label="Max examples")
            search_tb = gr.Textbox(label="Search", placeholder="Filter by text in prompt/response")
        with gr.Accordion("Score filter (optional)", open=False, visible=False) as score_filter_acc:
            with gr.Row():
                score_metric_dd = gr.Dropdown(label="Metric/column", choices=["— None —"], value="— None —")
                score_values_cg = gr.CheckboxGroup(label="Allowed values", choices=[], value=[])
        examples_html = gr.HTML(value="<p style='color:#666;padding:8px;'>Load data to view examples</p>")

        # Handlers
        NONE_OPTION = "— None —"

        def _on_load(path: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
            try:
                df, warns = _read_input_file(path)
            except Exception as e:
                return (
                    gr.update(value=f"❌ {e}", visible=True),
                    None,
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(value=""),
                )

            cols = df.columns.tolist()
            status = "\n".join([f"⚠️ {w}" for w in warns]) if warns else "✅ Loaded file successfully. Select columns to proceed."

            # guesses
            guess_prompt = _guess_prompt_column(cols)
            guess_resp = _guess_response_column(cols)

            # Guess mode and score columns
            default_mode = "Side-by-side" if ("model_a_response" in cols and "model_b_response" in cols) else "Single"
            # Guess score-dict column with priority: 'score' else first dict-like
            guess_score_dict: Optional[str] = None
            if 'score' in cols:
                guess_score_dict = 'score'
            else:
                for c in cols:
                    try:
                        if df[c].apply(lambda v: isinstance(v, dict)).any():
                            guess_score_dict = c
                            break
                    except Exception:
                        continue
            # Side-by-side guesses
            guess_resp_a = 'model_a_response' if 'model_a_response' in cols else None
            guess_resp_b = 'model_b_response' if 'model_b_response' in cols else None
            guess_model_a = 'model_a' if 'model_a' in cols else None
            guess_model_b = 'model_b' if 'model_b' in cols else None
            guess_score_a = 'score_a' if 'score_a' in cols else None
            guess_score_b = 'score_b' if 'score_b' in cols else None

            # Detect metric keys for SxS info display
            def _extract_keys_from_series(series):
                keys = set()
                for val in series:
                    src = None
                    if isinstance(val, dict):
                        src = val
                    elif isinstance(val, str):
                        try:
                            src = json.loads(val)
                        except Exception:
                            try:
                                src = ast.literal_eval(val)
                            except Exception:
                                src = None
                    if isinstance(src, dict):
                        keys.update(list(src.keys()))
                return sorted(keys)
            sxs_keys: List[str] = []
            try:
                if 'score_a' in cols:
                    sxs_keys.extend(_extract_keys_from_series(df['score_a']))
                if 'score_b' in cols:
                    sxs_keys.extend(_extract_keys_from_series(df['score_b']))
            except Exception:
                pass
            sxs_keys = sorted(list(dict.fromkeys(sxs_keys)))
            sxs_metrics_text = ("Detected SxS metrics: " + ", ".join(sxs_keys)) if sxs_keys else "Detected SxS metrics: none"
            columns_text = "**Available columns**: " + ", ".join(cols)

            # Build sample raw row JSON
            try:
                first_row_json = json.dumps(df.iloc[0].to_dict(), ensure_ascii=False, indent=2, default=str)
            except Exception:
                first_row_json = "{}"

            return (
                gr.update(value=status, visible=True),
                df,
                gr.update(value=default_mode),
                gr.update(choices=cols, value=guess_prompt),
                gr.update(choices=cols, value=guess_resp),
                gr.update(choices=[None] + cols, value=("model" if "model" in cols else None)),
                gr.update(choices=[NONE_OPTION] + cols, value=(guess_score_dict if guess_score_dict else NONE_OPTION)),
                gr.update(choices=cols, value=[]),
                gr.update(choices=[], value=[]),
                gr.update(value=first_row_json),
                gr.update(choices=cols, value=guess_resp_a),
                gr.update(choices=cols, value=guess_resp_b),
                gr.update(choices=[None] + cols, value=guess_model_a),
                gr.update(choices=[None] + cols, value=guess_model_b),
                gr.update(choices=[NONE_OPTION] + cols, value=(guess_score_a if guess_score_a else NONE_OPTION)),
                gr.update(choices=[NONE_OPTION] + cols, value=(guess_score_b if guess_score_b else NONE_OPTION)),
                gr.update(value=sxs_metrics_text),
                gr.update(value=columns_text),
            )

        load_btn.click(
            _on_load,
            inputs=[file_path_tb],
            outputs=[status_md, df_state, data_mode_dd, prompt_col_dd, response_col_dd, model_col_dd, score_dict_col_dd, score_cols_cg, model_filter, sample_row_code, resp_a_col_dd, resp_b_col_dd, model_a_col_dd, model_b_col_dd, score_dict_a_dd, score_dict_b_dd, sxs_metrics_md, columns_md],
        )

        # Toggle side-by-side mapping visibility
        def _toggle_mode(mode: str):
            is_sxs = (mode == "Side-by-side")
            return (
                gr.update(visible=not is_sxs),  # single response
                gr.update(visible=is_sxs),      # sxs_row1
                gr.update(visible=is_sxs),      # sxs_row2
                gr.update(visible=not is_sxs),  # score_dict_col_dd
                gr.update(visible=not is_sxs),  # score_cols_cg
                gr.update(visible=is_sxs),      # sxs_metrics_md
            )

        data_mode_dd.change(
            _toggle_mode,
            inputs=[data_mode_dd],
            outputs=[response_col_dd, sxs_row1, sxs_row2, score_dict_col_dd, score_cols_cg, sxs_metrics_md],
        )

        def _build_prompt_stats_card(df_in: pd.DataFrame, prompt_col: Optional[str], model_col: Optional[str], selected_models: List[str]) -> str:
            work = df_in
            if model_col and model_col in work.columns and selected_models:
                work = work[work[model_col].isin(selected_models)]
            # Prompt stats
            total_prompts = int(work[prompt_col].dropna().nunique()) if (prompt_col and prompt_col in work.columns) else 0
            total_samples = int(len(work))

            avg_prompts_per_model: Optional[float] = None
            shared_prompts_across_all: Optional[int] = None
            if model_col and model_col in work.columns and prompt_col and prompt_col in work.columns:
                models_present = work[model_col].dropna().unique().tolist()
                if len(models_present) > 0:
                    per_model_counts = work.groupby(model_col)[prompt_col].nunique()
                    if not per_model_counts.empty:
                        avg_prompts_per_model = float(per_model_counts.mean())
                    by_prompt = work.groupby(prompt_col)[model_col].nunique()
                    shared_prompts_across_all = int((by_prompt >= len(models_present)).sum())

            apm_display = f"{avg_prompts_per_model:.1f}" if isinstance(avg_prompts_per_model, (int, float)) else "N/A"
            shared_display = f"{shared_prompts_across_all}" if isinstance(shared_prompts_across_all, int) else "N/A"

            # Gradient stat card
            card = f"""
<div style="background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); padding: 16px 20px; border-radius: 12px; color: #13111c; box-shadow: 0 6px 16px rgba(0,0,0,0.05); margin-bottom: 12px;">
  <div style="font-weight:700; font-size:16px; margin-bottom: 8px; color:#1f2937;">Prompt Statistics</div>
  <div style="display:flex; gap:24px; flex-wrap:wrap;">
    <div style="min-width:160px;">
      <div style="font-weight:800; font-size:24px; color:#111827;">{int(total_samples)}</div>
      <div style="font-size:12px; color:#1f2937; opacity:0.9;">Total Samples</div>
    </div>
    <div style="min-width:160px;">
      <div style="font-weight:800; font-size:24px; color:#111827;">{int(total_prompts)}</div>
      <div style="font-size:12px; color:#1f2937; opacity:0.9;">Unique Prompts</div>
    </div>
    <div style="min-width:200px;">
      <div style="font-weight:800; font-size:24px; color:#111827;">{apm_display}</div>
      <div style="font-size:12px; color:#1f2937; opacity:0.9;">Avg Prompts / Model</div>
    </div>
    <div style="min-width:220px;">
      <div style="font-weight:800; font-size:24px; color:#111827;">{shared_display}</div>
      <div style="font-size:12px; color:#1f2937; opacity:0.9;">Prompts Shared Across All Models</div>
    </div>
  </div>
</div>
"""
            return card

        def _update_overview(df: Any, prompt_col: str, response_col: str, model_col: Optional[str], score_dict_col: Optional[str], score_cols: List[str], view: str, selected_models: List[str], data_mode: str, resp_a_col: Optional[str], resp_b_col: Optional[str]) -> Tuple[Any, Any, Any, Any]:
            if df is None:
                return gr.update(), gr.update(), gr.update(value="<p style='color:#666;padding:8px;'>No data loaded.</p>"), gr.update(visible=False)

            # Validation
            missing: List[str] = []
            # Always need prompt
            if not prompt_col or prompt_col not in df.columns:
                missing.append("prompt")
            # Require appropriate response fields based on mode
            if data_mode == "Single":
                if not response_col or response_col not in df.columns:
                    missing.append("response")
            else:
                if (not resp_a_col or resp_a_col not in df.columns) or (not resp_b_col or resp_b_col not in df.columns):
                    missing.append("response_a/response_b")
            if missing:
                return gr.update(), gr.update(), gr.update(value=f"<p style='color:#e74c3c;padding:8px;'>Missing required column(s): {', '.join(missing)}</p>"), gr.update(visible=False)

            # Apply model filter if provided
            work = df
            if model_col and model_col in work.columns and selected_models:
                work = work[work[model_col].isin(selected_models)]
                if len(work) == 0:
                    return gr.update(), gr.update(), gr.update(value="<p style='color:#e74c3c;padding:8px;'>No rows for selected model subset.</p>"), gr.update(visible=False)

            # Build top prompt statistics card
            html = _build_prompt_stats_card(
                work,
                (prompt_col if prompt_col in work.columns else None),
                (model_col if (model_col and model_col in work.columns) else None),
                selected_models or [],
            )

            # Compute per-model metric averages for plot/table
            def _parse_metrics_series(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
                metric_rows: List[Dict[str, Any]] = []
                metric_names: set[str] = set()
                # Ensure model column exists
                model_series = df_in[model_col] if (model_col and model_col in df_in.columns) else pd.Series(['all'] * len(df_in))
                for idx, row in df_in.iterrows():
                    # extract dict
                    metric_dict: Dict[str, float] = {}
                    sdc = None if (not score_dict_col or score_dict_col == NONE_OPTION) else score_dict_col
                    if sdc and sdc in df_in.columns:
                        val = row[score_dict_col]
                        src = None
                        if isinstance(val, dict):
                            src = val
                        elif isinstance(val, str):
                            try:
                                src = json.loads(val)
                            except Exception:
                                try:
                                    src = ast.literal_eval(val)
                                except Exception:
                                    src = None
                        if isinstance(src, dict):
                            metric_dict.update({k: float(v) for k, v in src.items() if isinstance(v, (int, float))})
                    if score_cols:
                        for c in score_cols:
                            if c in df_in.columns:
                                try:
                                    v = float(pd.to_numeric(row[c], errors='coerce'))
                                    if not pd.isna(v):
                                        metric_dict[c] = v
                                except Exception:
                                    pass
                    # Side-by-side: add contributions from score_a/score_b if present
                    if data_mode == "Side-by-side":
                        for col_name, model_name_col in [("score_a", "model_a"), ("score_b", "model_b")]:
                            if col_name in df_in.columns:
                                sval = row[col_name]
                                src = None
                                if isinstance(sval, dict):
                                    src = sval
                                elif isinstance(sval, str):
                                    try:
                                        src = json.loads(sval)
                                    except Exception:
                                        try:
                                            src = ast.literal_eval(sval)
                                        except Exception:
                                            src = None
                                if isinstance(src, dict):
                                    mdl = row.get(model_name_col) if model_name_col in df_in.columns else model_name_col
                                    for k, v in src.items():
                                        if isinstance(v, (int, float)):
                                            metric_rows.append({'model': str(mdl), 'metric': k, 'value': float(v)})
                                            metric_names.add(k)
                    if metric_dict:
                        model_name = str(model_series.loc[idx]) if idx in model_series.index else 'all'
                        for mname, mval in metric_dict.items():
                            metric_rows.append({'model': model_name, 'metric': mname, 'value': mval})
                            metric_names.add(mname)
                if not metric_rows:
                    return pd.DataFrame(columns=['model','metric','value']), []
                tidy = pd.DataFrame(metric_rows)
                # groupby mean
                tidy = tidy.groupby(['model','metric'])['value'].mean().reset_index()
                return tidy, sorted(metric_names)

            tidy_metrics, metric_list = _parse_metrics_series(work)

            if view == "Plot":
                if tidy_metrics is None or tidy_metrics.empty:
                    # Hide when no metrics to display
                    return gr.update(visible=False), gr.update(visible=False), gr.update(value=html, visible=True), gr.update(visible=False)
                else:
                    fig = px.bar(tidy_metrics, x="metric", y="value", color="model", barmode="group", text="value")
                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig.update_layout(yaxis_tickformat=".3f", xaxis_title="Metric", yaxis_title="Average")
                return gr.update(value=fig, visible=True), gr.update(visible=False), gr.update(value=html, visible=True), gr.update(visible=True)
            else:
                # table view: rows = model, columns = metrics
                if tidy_metrics is None or tidy_metrics.empty:
                    return gr.update(visible=False), gr.update(visible=False), gr.update(value=html, visible=True), gr.update(visible=False)
                else:
                    tbl = tidy_metrics.pivot(index='model', columns='metric', values='value').reset_index().rename_axis(None, axis=1)
                return gr.update(visible=False), gr.update(value=tbl, visible=True), gr.update(value=html, visible=True), gr.update(visible=True)

        for comp in [view_toggle, prompt_col_dd, response_col_dd, model_col_dd, score_dict_col_dd, score_cols_cg, model_filter, data_mode_dd, resp_a_col_dd, resp_b_col_dd]:
            comp.change(
                _update_overview,
                inputs=[df_state, prompt_col_dd, response_col_dd, model_col_dd, score_dict_col_dd, score_cols_cg, view_toggle, model_filter, data_mode_dd, resp_a_col_dd, resp_b_col_dd],
                outputs=[overview_plot, overview_table, overview_html, view_toggle],
            )

        def _normalize_dataframe(df: pd.DataFrame, prompt_col: str, response_col: str, model_col: Optional[str], score_dict_col: Optional[str], score_cols: List[str], data_mode: str, resp_a_col: Optional[str], resp_b_col: Optional[str], model_a_col: Optional[str], model_b_col: Optional[str], score_a_col: Optional[str], score_b_col: Optional[str]) -> pd.DataFrame:
            """Map user-selected columns into the format expected by utils.get_example_data.

            Produces columns: 'prompt', 'response', 'model', 'score' (dict of metric->value) when possible.
            """
            norm = df.copy()
            if prompt_col in norm.columns:
                norm['prompt'] = norm[prompt_col].astype(str)
            if data_mode == "Side-by-side":
                # Map A/B responses and models
                if resp_a_col and resp_a_col in norm.columns:
                    norm['model_a_response'] = norm[resp_a_col]
                if resp_b_col and resp_b_col in norm.columns:
                    norm['model_b_response'] = norm[resp_b_col]
                if model_a_col and model_a_col in norm.columns:
                    norm['model_a'] = norm[model_a_col]
                if model_b_col and model_b_col in norm.columns:
                    norm['model_b'] = norm[model_b_col]
            else:
                if response_col in norm.columns:
                    norm['response'] = norm[response_col]
            if model_col and model_col in norm.columns:
                norm['model'] = norm[model_col].astype(str)
            else:
                norm['model'] = 'all'

            # Build score dict
            score_series = None
            if data_mode == "Side-by-side":
                # Set score_a / score_b if provided
                if score_a_col and score_a_col in norm.columns:
                    norm['score_a'] = norm[score_a_col]
                if score_b_col and score_b_col in norm.columns:
                    norm['score_b'] = norm[score_b_col]
            if score_dict_col and score_dict_col in norm.columns:
                def _to_dict(val: Any) -> Dict[str, float]:
                    if isinstance(val, dict):
                        return {k: float(v) for k, v in val.items() if isinstance(v, (int, float))}
                    if isinstance(val, str):
                        parsed: Any = None
                        try:
                            parsed = json.loads(val)
                        except Exception:
                            try:
                                parsed = ast.literal_eval(val)
                            except Exception:
                                parsed = None
                        if isinstance(parsed, dict):
                            return {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
                    return {}
                score_series = norm[score_dict_col].apply(_to_dict)
            elif score_cols:
                def _build_from_cols(row: pd.Series) -> Dict[str, float]:
                    result: Dict[str, float] = {}
                    for c in score_cols:
                        if c in row and row[c] is not None and str(row[c]) != '':
                            try:
                                result[c] = float(pd.to_numeric(row[c], errors='coerce'))
                            except Exception:
                                pass
                    return result
                score_series = norm.apply(_build_from_cols, axis=1)
            if score_series is not None:
                norm['score'] = score_series
            return norm

        def _render_examples(df: Any, prompt_col: str, response_col: str, score_dict_col: Optional[str], score_cols: List[str], search: str, limit: int, model_col: Optional[str], selected_models: List[str], filter_metric: Optional[str], filter_values: List[str], data_mode: str, resp_a_col: Optional[str], resp_b_col: Optional[str], model_a_col: Optional[str], model_b_col: Optional[str], score_a_col: Optional[str], score_b_col: Optional[str]) -> Any:
            if df is None:
                return gr.update(value="<p style='color:#666;padding:8px;'>No data loaded.</p>")
            # Validate prompt
            if not prompt_col or prompt_col not in df.columns:
                return gr.update(value=f"<p style='color:#e74c3c;padding:8px;'>Missing required column: prompt</p>")
            # Validate responses depending on mode
            if data_mode == "Single":
                if not response_col or response_col not in df.columns:
                    return gr.update(value=f"<p style='color:#e74c3c;padding:8px;'>Missing required column: response</p>")
            else:
                if (not resp_a_col or resp_a_col not in df.columns) or (not resp_b_col or resp_b_col not in df.columns):
                    return gr.update(value=f"<p style='color:#e74c3c;padding:8px;'>Missing required columns: response_a/response_b</p>")

            work = df
            # Apply model filter
            if model_col and model_col in work.columns and selected_models:
                work = work[work[model_col].isin(selected_models)]
            if search and str(search).strip():
                query = str(search).strip().lower()
                mask = work[prompt_col].astype(str).str.lower().str.contains(query) | work[response_col].astype(str).str.lower().str.contains(query)
                work = work[mask]

            # Score filter (optional)
            if filter_metric and filter_metric != NONE_OPTION and filter_values:
                sdc = None if (not score_dict_col or score_dict_col == NONE_OPTION) else score_dict_col
                if sdc and sdc in work.columns:
                    def _match_row(val: Any) -> bool:
                        src = None
                        if isinstance(val, dict):
                            src = val
                        elif isinstance(val, str):
                            try:
                                src = json.loads(val)
                            except Exception:
                                try:
                                    src = ast.literal_eval(val)
                                except Exception:
                                    src = None
                        if isinstance(src, dict) and filter_metric in src:
                            return str(src.get(filter_metric)) in set(map(str, filter_values))
                        return False
                    work = work[work[sdc].apply(_match_row)]
                elif score_cols and filter_metric in score_cols and filter_metric in work.columns:
                    work = work[work[filter_metric].astype(str).isin(list(map(str, filter_values)))]
                else:
                    # Side-by-side support: check score_a/score_b dicts
                    def _dict_contains(dval: Any) -> bool:
                        src = None
                        if isinstance(dval, dict):
                            src = dval
                        elif isinstance(dval, str):
                            try:
                                src = json.loads(dval)
                            except Exception:
                                try:
                                    src = ast.literal_eval(dval)
                                except Exception:
                                    src = None
                        if isinstance(src, dict) and filter_metric in src:
                            return str(src.get(filter_metric)) in set(map(str, filter_values))
                        return False
                    cols_to_check = [c for c in ['score_a','score_b'] if c in work.columns]
                    if cols_to_check:
                        mask = False
                        for c in cols_to_check:
                            mask = (mask | work[c].apply(_dict_contains)) if isinstance(mask, pd.Series) else work[c].apply(_dict_contains)
                        work = work[mask]

            # Normalize columns and delegate rendering to shared utils for consistent formatting
            norm = _normalize_dataframe(work, prompt_col, response_col, model_col, (None if (not score_dict_col or score_dict_col == NONE_OPTION) else score_dict_col), score_cols, data_mode, resp_a_col, resp_b_col, model_a_col, model_b_col, score_a_col, score_b_col)
            norm_limited = norm.head(int(limit))
            examples = get_example_data(norm_limited, None, None, None, int(limit), show_unexpected_behavior=False, randomize=False)
            html_block = format_examples_display(examples, None, None, None, use_accordion=True, pretty_print_dicts=True)
            return gr.update(value=html_block)

        for comp in [prompt_col_dd, response_col_dd, score_dict_col_dd, score_cols_cg, search_tb, max_examples, model_filter, score_metric_dd, score_values_cg, data_mode_dd, resp_a_col_dd, resp_b_col_dd, model_a_col_dd, model_b_col_dd, score_dict_a_dd, score_dict_b_dd]:
            comp.change(
                _render_examples,
                inputs=[df_state, prompt_col_dd, response_col_dd, score_dict_col_dd, score_cols_cg, search_tb, max_examples, model_col_dd, model_filter, score_metric_dd, score_values_cg, data_mode_dd, resp_a_col_dd, resp_b_col_dd, model_a_col_dd, model_b_col_dd, score_dict_a_dd, score_dict_b_dd],
                outputs=[examples_html],
            )

        # Trigger once after loading too
        def _kick_examples(df: Any, prompt_col: Optional[str], response_col: Optional[str]):
            if df is None or not prompt_col or not response_col:
                return gr.update()
            return _render_examples(df, prompt_col, response_col, None, [], "", 20, None, [])

        load_btn.click(
            _kick_examples,
            inputs=[df_state, prompt_col_dd, response_col_dd],
            outputs=[examples_html],
        )

        # Populate score filter metric and values
        def _populate_score_filter(df: Any, model_col: Optional[str], selected_models: List[str], search: str, prompt_col: Optional[str], response_col: Optional[str], score_dict_col: Optional[str], score_cols: List[str]) -> Tuple[Any, Any, Any]:
            if df is None:
                return gr.update(choices=[NONE_OPTION], value=NONE_OPTION), gr.update(choices=[], value=[]), gr.update(visible=False)
            work = df
            if model_col and model_col in work.columns and selected_models:
                work = work[work[model_col].isin(selected_models)]
            if search and prompt_col and response_col and prompt_col in work.columns and response_col in work.columns:
                q = str(search).strip().lower()
                if q:
                    mask = work[prompt_col].astype(str).str.lower().str.contains(q) | work[response_col].astype(str).str.lower().str.contains(q)
                    work = work[mask]
            metrics_choices: List[str] = [NONE_OPTION]
            values_choices: List[str] = []
            sdc = None if (not score_dict_col or score_dict_col == NONE_OPTION) else score_dict_col
            if sdc and sdc in work.columns:
                # Collect keys across rows
                keys: set[str] = set()
                def _to_dict(val: Any) -> Dict[str, Any]:
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except Exception:
                            try:
                                return ast.literal_eval(val)
                            except Exception:
                                return {}
                    return {}
                series = work[sdc].apply(_to_dict)
                for d in series:
                    if isinstance(d, dict):
                        keys.update([k for k, v in d.items()])
                metrics_choices += sorted(keys)
            elif score_cols:
                metrics_choices += [c for c in score_cols if c in work.columns]
            # Side-by-side: include keys from score_a/score_b if present
            if ('score_a' in work.columns) or ('score_b' in work.columns):
                def _extract_keys_from(val: Any) -> List[str]:
                    if isinstance(val, dict):
                        return list(val.keys())
                    if isinstance(val, str):
                        try:
                            d = json.loads(val)
                        except Exception:
                            try:
                                d = ast.literal_eval(val)
                            except Exception:
                                d = None
                        return list(d.keys()) if isinstance(d, dict) else []
                    return []
                keys = set()
                for col in ['score_a','score_b']:
                    if col in work.columns:
                        keys.update([k for sub in work[col].apply(_extract_keys_from).tolist() for k in sub])
                metrics_choices = list(dict.fromkeys([NONE_OPTION] + sorted(keys)))
            # Toggle score filter accordion visibility via output
            has_metrics = len([m for m in metrics_choices if m != NONE_OPTION]) > 0
            return (
                gr.update(choices=metrics_choices, value=(metrics_choices[0] if metrics_choices else NONE_OPTION)),
                gr.update(choices=values_choices, value=[]),
                gr.update(visible=has_metrics),
            )

        # Update metric choices when columns or filters change
        for comp in [score_dict_col_dd, score_cols_cg, model_filter, search_tb, prompt_col_dd, response_col_dd, data_mode_dd]:
            comp.change(
                _populate_score_filter,
                inputs=[df_state, model_col_dd, model_filter, search_tb, prompt_col_dd, response_col_dd, score_dict_col_dd, score_cols_cg],
                outputs=[score_metric_dd, score_values_cg, score_filter_acc],
            )

        # Update value choices when metric changes
        def _populate_value_choices(df: Any, model_col: Optional[str], selected_models: List[str], search: str, prompt_col: Optional[str], response_col: Optional[str], score_dict_col: Optional[str], score_cols: List[str], metric: Optional[str]) -> Any:
            if df is None or not metric or metric == NONE_OPTION:
                return gr.update(choices=[], value=[])
            work = df
            if model_col and model_col in work.columns and selected_models:
                work = work[work[model_col].isin(selected_models)]
            if search and prompt_col and response_col and prompt_col in work.columns and response_col in work.columns:
                q = str(search).strip().lower()
                if q:
                    mask = work[prompt_col].astype(str).str.lower().str.contains(q) | work[response_col].astype(str).str.lower().str.contains(q)
                    work = work[mask]
            sdc = None if (not score_dict_col or score_dict_col == NONE_OPTION) else score_dict_col
            values: List[str] = []
            if sdc and sdc in work.columns:
                def _to_val(val: Any) -> Any:
                    if isinstance(val, dict):
                        return val.get(metric)
                    if isinstance(val, str):
                        try:
                            d = json.loads(val)
                        except Exception:
                            try:
                                d = ast.literal_eval(val)
                            except Exception:
                                d = None
                        if isinstance(d, dict):
                            return d.get(metric)
                    return None
                series = work[sdc].apply(_to_val)
                uniq = sorted({str(v) for v in series.dropna().tolist()})
                values = uniq
            elif score_cols and metric in work.columns:
                uniq = sorted({str(v) for v in work[metric].dropna().astype(str).tolist()})
                values = uniq
            return gr.update(choices=values, value=[])

        score_metric_dd.change(
            _populate_value_choices,
            inputs=[df_state, model_col_dd, model_filter, search_tb, prompt_col_dd, response_col_dd, score_dict_col_dd, score_cols_cg, score_metric_dd],
            outputs=[score_values_cg],
        )

        # Update model filter choices when model column changes
        def _on_model_col_change(df: Any, model_col: Optional[str]):
            if df is None or not model_col or model_col not in getattr(df, 'columns', []):
                return gr.update(choices=[], value=[])
            models = (
                df[model_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            return gr.update(choices=models, value=models)

        model_col_dd.change(
            _on_model_col_change,
            inputs=[df_state, model_col_dd],
            outputs=[model_filter],
        )

        def _select_all_models(current_choices: List[str]):
            return gr.update(value=current_choices)

        select_all_models_btn.click(_select_all_models, inputs=[model_filter], outputs=[model_filter])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.queue().launch(share=True, show_error=False)


