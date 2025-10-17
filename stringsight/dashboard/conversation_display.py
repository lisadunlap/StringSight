from __future__ import annotations

"""Conversation display helpers for dashboard.

This module encapsulates everything related to:
• safely parsing model responses (lists / dicts / JSON strings)
• pretty-printing embedded dictionaries for readability
• converting multiple conversation formats to the OpenAI chat list format
• rendering that list as HTML (including accordion grouping + raw JSON viewer).

Moving this logic out of utils.py keeps the latter lean and focussed on general
analytics utilities.
"""

from typing import List, Dict, Any
import ast
import json
import html
import markdown
import re

__all__: List[str] = [
    "convert_to_openai_format",
    "display_openai_conversation_html",
    "pretty_print_embedded_dicts",
]

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _find_balanced_spans(text: str):
    """Return (start, end) spans of balanced {...} or [...] regions in *text*."""
    spans, stack = [], []
    for i, ch in enumerate(text):
        if ch in "{[":
            stack.append((ch, i))
        elif ch in "]}" and stack:
            opener, start = stack.pop()
            if (opener, ch) in {("{", "}"), ("[", "]")} and not stack:
                spans.append((start, i + 1))
    return spans


def _try_parse_slice(slice_: str):
    """Attempt to parse *slice_* into a Python object; return None on failure."""
    try:
        return ast.literal_eval(slice_)
    except Exception:
        try:
            return json.loads(slice_)
        except Exception:
            return None


def _find_code_spans(text: str) -> List[tuple]:
    """Return spans for markdown code regions to be preserved as-is.

    Includes:
    - fenced code blocks delimited by ``` ... ```
    - inline code segments delimited by `...`
    """
    spans: List[tuple] = []

    # Fenced blocks ``` ... ``` (language spec allowed after opening fence)
    idx = 0
    while True:
        start = text.find("```", idx)
        if start == -1:
            break
        # Find the end fence
        end = text.find("```", start + 3)
        if end == -1:
            # Unclosed fence: treat rest of string as code
            spans.append((start, len(text)))
            break
        spans.append((start, end + 3))
        idx = end + 3

    # Inline code `...`
    for m in re.finditer(r"`[^`]*`", text, flags=re.DOTALL):
        spans.append((m.start(), m.end()))

    # Sort and merge overlapping spans
    spans.sort()
    merged: List[tuple] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def _is_inside_any_span(start: int, end: int, spans: List[tuple]) -> bool:
    for s, e in spans:
        if start >= s and end <= e:
            return True
    return False


def pretty_print_embedded_dicts(text: str) -> str:
    """Replace dicts, lists, or other complex structures with pretty-printed JSON, except inside code.

    Dict-like regions that fall within markdown code spans (inline backticks
    or fenced code blocks) are left untouched so code examples render verbatim.
    """
    if not text:
        return text

    code_spans = _find_code_spans(text)

    def _to_json_safe(obj: Any):
        """Recursively convert Python objects to JSON-serializable equivalents.

        - Ellipsis (…) or ... becomes "..."
        - Unsupported objects become str(obj)
        """
        if obj is ... or isinstance(obj, type(Ellipsis)):
            return "..."
        if isinstance(obj, dict):
            return {str(k): _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [_to_json_safe(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _is_complex_structure(obj):
        """Check if object is worth pretty-printing (not just a simple value)"""
        if isinstance(obj, dict):
            return len(obj) > 0
        elif isinstance(obj, list):
            return len(obj) > 0 and any(isinstance(item, (dict, list)) for item in obj)
        return False

    def _format_with_preserved_spacing(json_str):
        """Convert JSON string to HTML with preserved indentation and wrapping.

        Use a <pre> block with white-space: pre-wrap so that long tokens can wrap
        while preserving indentation and newlines without converting spaces to
        non-breaking spaces (which prevents wrapping).
        """
        formatted = html.escape(json_str, quote=False)
        return (
            "<pre style=\"font-family: monospace; line-height: 1.4; font-size: 14px; "
            "white-space: pre-wrap !important; word-break: break-word; overflow-wrap: anywhere; "
            "background: #ffffff; padding: 10px; border-radius: 4px; margin: 0;\">"
            f"{formatted}"
            "</pre>"
        )

    new_parts, last_idx = [], 0
    for start, end in _find_balanced_spans(text):
        candidate = text[start:end]
        parsed = _try_parse_slice(candidate)
        
        if _is_complex_structure(parsed) and not _is_inside_any_span(start, end, code_spans):
            new_parts.append(html.escape(text[last_idx:start], quote=False))
            pretty = json.dumps(_to_json_safe(parsed), indent=2, ensure_ascii=False)
            new_parts.append(_format_with_preserved_spacing(pretty))
            last_idx = end
    new_parts.append(html.escape(text[last_idx:], quote=False))
    return "".join(new_parts)

# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def convert_to_openai_format(response_data: Any):
    """Convert various response payloads into the OpenAI chat format list."""
    if isinstance(response_data, list):
        return response_data
    if isinstance(response_data, dict):
        # If it already looks like an OpenAI-style message, wrap it in a list
        if "role" in response_data and "content" in response_data:
            return [response_data]
        # Otherwise treat dict as assistant content (preserve structure for tool_calls)
        return [{"role": "assistant", "content": response_data}]
    if isinstance(response_data, str):
        # Try Python literal first (handles single quotes)
        try:
            parsed = ast.literal_eval(response_data)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        # Try JSON
        try:
            parsed = json.loads(response_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Fallback plain-text assistant message
        return [{"role": "assistant", "content": response_data}]
    # Fallback for any other type
    return [{"role": "assistant", "content": str(response_data)}]

# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _markdown(text: str, *, pretty_print_dicts: bool = True) -> str:
    """Render markdown, optionally pretty-printing any embedded dicts."""
    processed = pretty_print_embedded_dicts(text) if pretty_print_dicts else html.escape(text, quote=False)
    
    # Configure extensions for proper code block handling
    extensions = ["fenced_code"]
    extension_configs = {}
    
    try:
        import pygments
        extensions.append("codehilite")
        extension_configs['codehilite'] = {
            'css_class': 'highlight',
            'use_pygments': True,
            'guess_lang': True,
            'linenums': False
        }
    except ImportError:
        pass
    
    # Convert newlines to <br> only outside of code blocks
    # Process fenced code blocks first, then handle line breaks
    result = markdown.markdown(processed, extensions=extensions, extension_configs=extension_configs)

    # IMPORTANT: Avoid injecting <br> tags when lists are present, as this can
    # introduce empty bullets or odd spacing in nested lists.
    import re
    if re.search(r'<(ul|ol)\b', result):
        return result

    # Otherwise, add line breaks for non-code content only
    code_block_pattern = r'(<pre[^>]*>.*?</pre>|<code[^>]*>.*?</code>)'
    parts = re.split(code_block_pattern, result, flags=re.DOTALL)

    for i in range(0, len(parts), 2):  # Process non-code parts only
        if i < len(parts):
            parts[i] = re.sub(r'(?<!\n)\n(?!\n)', '<br>\n', parts[i])

    return ''.join(parts)


def display_openai_conversation_html(conversation_data: List[Dict[str, Any]], *, use_accordion: bool = True, pretty_print_dicts: bool = True, evidence: Any = None) -> str:
    """Convert an OpenAI-style conversation list into styled HTML for Gradio."""
    from .examples_helpers import annotate_text_with_evidence_placeholders, HIGHLIGHT_START, HIGHLIGHT_END
    if not conversation_data:
        return "<p>No conversation data available</p>"

    # Collapsed raw JSON section for debugging
    raw_json = json.dumps(conversation_data, indent=2, ensure_ascii=False)
    html_out = f"""
    <details style="margin: 8px 0;">
        <summary style="cursor: pointer; font-weight: 600;">
            Click to see raw response ({len(conversation_data)})
        </summary>
        <div style="padding: 8px 15px;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere; background: #ffffff; padding: 10px; border-radius: 4px;">{html.escape(raw_json, quote=False)}</pre>
        </div>
    </details>
    """

    # Very subtle background colors with darker borders for chat bubbles
    role_styles = {
        "system": {
            "bg": "#fafafa",  # very light gray
            "border": "#d1d5db",  # medium gray border
            "text": "#374151",  # dark gray text
            "align": "left"
        },
        "info": {
            "bg": "#f0f9ff",  # barely blue
            "border": "#bae6fd",  # light blue border
            "text": "#0c4a6e",  # darker blue text
            "align": "left"
        },
        "assistant": {
            "bg": "#fafafa",  # very light gray
            "border": "#d1d5db",  # medium gray border
            "text": "#1f2937",  # dark text
            "align": "left"
        },
        "tool": {
            "bg": "#f7fef9",  # barely green
            "border": "#bbf7d0",  # light green border
            "text": "#166534",  # darker green text
            "align": "left"
        },
        "user": {
            "bg": "#eff6ff",  # barely blue
            "border": "#bfdbfe",  # light blue border
            "text": "#1e40af",  # darker blue text
            "align": "left"
        },
    }

    def _maybe_annotate(content_str: str) -> str:
        if evidence is None or not isinstance(content_str, str) or not content_str.strip():
            return content_str
        return annotate_text_with_evidence_placeholders(content_str, evidence)

    def _replace_placeholders_with_mark(html_str: str) -> str:
        if not html_str:
            return html_str
        return (
            html_str
            .replace(HIGHLIGHT_START, "<mark class=\"evidence-highlight\">")
            .replace(HIGHLIGHT_END, "</mark>")
        )

    def _format_tool_calls(content: Dict[str, Any]) -> str:
        """Format tool calls in a more readable way."""
        if not isinstance(content, dict) or "tool_calls" not in content:
            return f"<code>{html.escape(json.dumps(content, ensure_ascii=False))}</code>"
        
        tool_calls = content["tool_calls"]
        if not isinstance(tool_calls, list):
            return f"<code>{html.escape(json.dumps(content, ensure_ascii=False))}</code>"
        
        html_parts = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            if not isinstance(tool_call, dict):
                continue
                
            # Extract tool call information
            name = tool_call.get("name", "Unknown tool")
            arguments = tool_call.get("arguments", "")
            tool_id = tool_call.get("id", tool_call.get("tool_call_id", ""))
            # Coerce call type to a safe uppercase string
            raw_call_type = tool_call.get("type", "function")
            call_type = str(raw_call_type or "function")
            
            # Parse arguments if they're a JSON string
            formatted_args = arguments
            if isinstance(arguments, str) and arguments.strip():
                try:
                    parsed_args = json.loads(arguments)
                    formatted_args = json.dumps(parsed_args, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    formatted_args = arguments
            elif isinstance(arguments, (dict, list, tuple, int, float, bool)) or arguments is None:
                # Stringify any non-string argument type
                try:
                    formatted_args = json.dumps(arguments, indent=2, ensure_ascii=False)
                except Exception:
                    formatted_args = str(arguments)
            
            # Format with preserved spacing for proper indentation
            if formatted_args and isinstance(formatted_args, str) and ('\n' in formatted_args or '  ' in formatted_args):
                escaped_args = html.escape(formatted_args, quote=False)
                formatted_args = (
                    "<pre style=\"font-family: monospace; line-height: 1.4; font-size: 14px; "
                    "white-space: pre-wrap !important; word-break: break-word; overflow-wrap: anywhere; "
                    "background: #ffffff; padding: 10px; border-radius: 4px; margin: 0;\">"
                    f"{escaped_args}"
                    "</pre>"
                )
            else:
                formatted_args = html.escape(str(formatted_args), quote=False)
            
            # Create the tool call display with subtle styling and better spacing
            tool_html = f"""
            <div style="border: 1px solid #fed7aa; border-radius: 8px; margin: 8px 0; padding: 12px; background: #fffbf5;">
                <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                    <span style="background: #fb923c; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                        {call_type.upper()}
                    </span>
                    <span style="font-weight: 600; color: #c2410c; font-size: 14px; flex: 1; min-width: 150px;">{html.escape(name)}</span>
                    {f'<span style="font-size: 11px; color: #6b7280; white-space: nowrap;">ID: {html.escape(tool_id)}</span>' if tool_id else ''}
                </div>
                
                {f'''<div style="margin-top: 12px;">
                    <div style="font-weight: 600; color: #6b7280; margin-bottom: 6px; font-size: 12px;">Arguments:</div>
                    <div style="font-size: 12px; line-height: 1.4; color: #374151;">{formatted_args}</div>
                </div>''' if formatted_args else ''}
            </div>
            """
            
            html_parts.append(tool_html)
        
        if len(tool_calls) > 1:
            return f"""
            <div style="border-left: 3px solid #fed7aa; padding-left: 12px; margin: 8px 0;">
                <div style="font-weight: 600; color: #c2410c; margin-bottom: 8px; font-size: 14px;">
                    {len(tool_calls)} tool call{'s' if len(tool_calls) != 1 else ''}:
                </div>
                {''.join(html_parts)}
            </div>
            """
        else:
            return ''.join(html_parts)

    def _format_msg(role: str, content: Any) -> str:
        # Check if this is a tool call by examining the content
        is_tool_call = False
        if isinstance(content, dict) and "tool_calls" in content:
            is_tool_call = True
        
        if isinstance(content, dict) or (isinstance(content, list) and content and all(isinstance(d, dict) for d in content)):
            if is_tool_call:
                # Render assistant text (if provided) plus styled tool calls
                text_html = ""
                if isinstance(content, dict) and isinstance(content.get("text"), str) and content.get("text").strip():
                    annotated = _maybe_annotate(content.get("text", ""))
                    text_html = _markdown(annotated, pretty_print_dicts=pretty_print_dicts)
                    text_html = _replace_placeholders_with_mark(text_html)
                content_html = text_html + _format_tool_calls(content)
            elif pretty_print_dicts:
                def _to_json_safe_inline(obj: Any):
                    if obj is ... or isinstance(obj, type(Ellipsis)):
                        return "..."
                    if isinstance(obj, dict):
                        return {str(k): _to_json_safe_inline(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_json_safe_inline(v) for v in obj]
                    if isinstance(obj, tuple):
                        return [_to_json_safe_inline(v) for v in obj]
                    if isinstance(obj, (str, int, float, bool)) or obj is None:
                        return obj
                    return str(obj)

                safe_json = html.escape(json.dumps(_to_json_safe_inline(content), indent=2, ensure_ascii=False), quote=False)
                content_html = (
                    f"<pre style='background: #ffffff; padding: 10px; border-radius: 4px; "
                    f"white-space: pre-wrap !important; word-break: break-word; overflow-wrap: anywhere;'>{safe_json}</pre>"
                )
            else:
                content_html = f"<code>{html.escape(json.dumps(content, ensure_ascii=False))}</code>"
        elif isinstance(content, str):
            # Insert highlight placeholders before markdown so offsets make sense in plain text
            annotated = _maybe_annotate(content)
            content_html = _markdown(annotated, pretty_print_dicts=pretty_print_dicts)
            # Convert placeholders to <mark> after markdown
            content_html = _replace_placeholders_with_mark(content_html)
        elif content is None:
            content_html = "<em>(No content)</em>"
        else:
            content_html = str(content)
        
        # Determine role display text and styling
        if is_tool_call:
            # Keep assistant styling; tool blocks are styled within
            role_display = "assistant"
            style = role_styles.get("assistant", role_styles["assistant"])
        else:
            role_display = role
            style = role_styles.get(role.lower(), {
                "bg": "#fafafa",
                "border": "#d1d5db",
                "text": "#374151", 
                "align": "left"
            })
        
        # Chat bubble design: all left-aligned with a thin darker role bar and slightly sharper corners
        align = style["align"]
        justify = "flex-start"
        margin_left = "0"
        margin_right = "0"
        max_width = "95%"
        
        return (
            f"<div style='display: flex; justify-content: {justify}; margin: 12px 0;'>"
            f"<div style='max-width: {max_width}; margin-left: {margin_left}; margin-right: {margin_right}; "
            f"background-color: {style['bg']}; border: 1px solid {style['border']}; border-left: 4px solid {style['text']}; "
            f"padding: 12px 16px; border-radius: 10px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);'>"
            f"<div style='font-weight: 600; color: {style['text']}; margin-bottom: 6px; text-transform: capitalize; "
            f"font-size: 13px; opacity: 0.8;'>{role_display}</div>"
            f"<div style='color: {style['text']}; line-height: 1.6; font-family: inherit; font-size: 15px;'>{content_html}</div>"
            "</div>"
            "</div>"
        )

    if use_accordion:
        system_msgs, info_msgs, other_msgs = [], [], []
        for m in conversation_data:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown").lower()
            content = m.get("content", "")
            if isinstance(content, dict) and "text" in content and "tool_calls" not in content:
                content = content["text"]
            if role == "system":
                system_msgs.append((role, content))
            elif role == "info":
                info_msgs.append((role, content))
            else:
                other_msgs.append((role, content))

        def _accordion(title: str, items: List):
            if not items:
                return ""
            inner = "".join(_format_msg(r, c) for r, c in items)
            return (
                f"<details style='margin: 8px 0;'>"
                f"<summary style='cursor: pointer; font-weight: 600;'>"
                f"{html.escape(title)} ({len(items)})"  # e.g. "Click to see system messages (3)"
                f"</summary>"
                f"<div style='padding: 8px 15px;'>{inner}</div>"
                "</details>"
            )

        html_out += _accordion("Click to see system messages", system_msgs)
        html_out += _accordion("Click to see info messages", info_msgs)
        for r, c in other_msgs:
            html_out += _format_msg(r, c)
    else:
        # No accordion: just render everything
        for m in conversation_data:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown").lower()
            content = m.get("content", "")
            if isinstance(content, dict) and "text" in content and "tool_calls" not in content:
                content = content["text"]
            html_out += _format_msg(role, content)

    # CSS for proper code block styling and summary hover effects
    css_styles = """
    <style>
    .evidence-highlight { 
        background: #fef08a; 
        padding: 0 2px; 
        border-radius: 2px;
        box-shadow: 0 0 0 1px rgba(234, 179, 8, 0.2);
    }
    :root {
        /* Code block color palette - GitHub Light inspired */
        --code-bg: transparent; /* make JSON/code wrapper background transparent */
        --code-text: #24292f;
        --code-comment: #6a737d;
        --code-keyword: #d73a49;
        --code-string: #032f62;
        --code-number: #005cc5;
        --code-operator: #24292f;
        --code-function: #6f42c1;
        --code-border: #d0d7de;
        
        /* Inline code colors - same light theme */
        --inline-code-bg: #f3f4f6;
        --inline-code-text: #24292f;
        --inline-code-border: #d1d5db;
        
        /* Code block structure */
        --code-border-radius: 8px;
        --code-padding: 16px;
        --code-font-size: 14px;
        --code-line-height: 1.5;
        --code-font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    }
    
    /* Base code styling */
    pre, code {
        font-family: var(--code-font-family) !important;
        font-size: var(--code-font-size) !important;
        line-height: var(--code-line-height) !important;
        font-variant-ligatures: normal !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* Fenced code blocks - light theme */
    .highlight, .codehilite, pre.highlight, pre.codehilite, 
    .language-python, .language-text, .language-bash {
        background: var(--code-bg) !important;
        color: var(--code-text) !important;
        border: 1px solid var(--code-border) !important;
        border-radius: var(--code-border-radius) !important;
        padding: var(--code-padding) !important;
        margin: 12px 0 !important;
        overflow-x: auto !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        position: relative !important;
        white-space: pre !important;
        display: block !important;
    }
    
    .highlight pre, .codehilite pre {
        background: transparent !important;
        color: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0 !important;
        overflow: visible !important;
        white-space: pre !important;
        display: block !important;
    }
    
    /* Ensure code blocks preserve formatting */
    .highlight code, .codehilite code {
        white-space: pre !important;
        display: block !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        border: none !important;
        font-size: inherit !important;
        line-height: inherit !important;
    }
    
    /* Add language label for fenced blocks */
    .highlight::before, .codehilite::before {
        content: 'python';
        position: absolute;
        top: 8px;
        right: 12px;
        background: rgba(0, 0, 0, 0.05);
        color: #586069;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Syntax highlighting for Python - Light theme */
    .highlight .k, .codehilite .k,    /* keywords */
    .highlight .kn, .codehilite .kn,  /* keyword.namespace */
    .highlight .kp, .codehilite .kp,  /* keyword.pseudo */
    .highlight .kr, .codehilite .kr,  /* keyword.reserved */
    .highlight .kt, .codehilite .kt   /* keyword.type */
    {
        color: var(--code-keyword) !important;
        font-weight: 600 !important;
    }
    
    .highlight .s, .codehilite .s,    /* strings */
    .highlight .s1, .codehilite .s1,  /* string.single */
    .highlight .s2, .codehilite .s2,  /* string.double */
    .highlight .se, .codehilite .se   /* string.escape */
    {
        color: var(--code-string) !important;
    }
    
    .highlight .c, .codehilite .c,    /* comments */
    .highlight .c1, .codehilite .c1,  /* comment.single */
    .highlight .cm, .codehilite .cm   /* comment.multiline */
    {
        color: var(--code-comment) !important;
        font-style: italic !important;
    }
    
    .highlight .m, .codehilite .m,    /* numbers */
    .highlight .mi, .codehilite .mi,  /* number.integer */
    .highlight .mf, .codehilite .mf,  /* number.float */
    .highlight .mo, .codehilite .mo   /* number.octal */
    {
        color: var(--code-number) !important;
        font-weight: 600 !important;
    }
    
    .highlight .nf, .codehilite .nf,  /* function names */
    .highlight .fm, .codehilite .fm   /* function.magic */
    {
        color: var(--code-function) !important;
        font-weight: 600 !important;
    }
    
    .highlight .o, .codehilite .o,    /* operators */
    .highlight .ow, .codehilite .ow   /* operator.word */
    {
        color: var(--code-operator) !important;
    }
    
    /* Inline code - light theme */
    p code, li code, div code, span code, 
    h1 code, h2 code, h3 code, h4 code, h5 code, h6 code {
        background: var(--inline-code-bg) !important;
        color: var(--inline-code-text) !important;
        border: 1px solid var(--inline-code-border) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        display: inline !important;
    }
    
    /* Code blocks inside paragraphs should not be treated as inline */
    p pre, li pre, div pre {
        background: var(--code-bg) !important;
        color: var(--code-text) !important;
        border: 1px solid var(--code-border) !important;
        border-radius: var(--code-border-radius) !important;
        padding: var(--code-padding) !important;
        margin: 8px 0 !important;
        white-space: pre !important;
        overflow-x: auto !important;
        display: block !important;
    }
    
    /* Scrollbar styling for code blocks - light theme */
    .highlight::-webkit-scrollbar, .codehilite::-webkit-scrollbar,
    pre::-webkit-scrollbar {
        height: 8px !important;
        background: #f1f3f4 !important;
        border-radius: 4px !important;
    }
    
    .highlight::-webkit-scrollbar-thumb, .codehilite::-webkit-scrollbar-thumb,
    pre::-webkit-scrollbar-thumb {
        background: #c1c8cd !important;
        border-radius: 4px !important;
    }
    
    .highlight::-webkit-scrollbar-thumb:hover, .codehilite::-webkit-scrollbar-thumb:hover,
    pre::-webkit-scrollbar-thumb:hover {
        background: #a8b3ba !important;
    }
    </style>
    """
    
    css_styles += "</style>"
    html_out = css_styles + html_out

    return html_out 