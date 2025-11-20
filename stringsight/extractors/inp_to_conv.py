import json
from typing import Any, Dict, List, Union


def format_oai_simple(row):
  return [{
    "role": "user", "content": row["prompt"]
  }, {
    "role": "assistant", "content": row["model_response"]
  }]


def openai_messages_to_conv(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of OpenAI ChatCompletion `messages` (including multimodal
    content and tool call structures) to the condensed conversation format
    consumed downstream.

    Input format (per OpenAI spec):
      - Each message is a dict with keys like `role`, `content`, optional
        `tool_calls` or `function_call`.
      - `content` may be:
        - a string
        - a list of parts, e.g. [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]

    Output format (internal):
      - A list of messages, each a dict with:
        - `role`: str
        - `content`: dict with a single ordered list of `segments`
          segments[i] is one of:
            - {"kind": "text", "text": str}
            - {"kind": "image", "image": Union[str, Dict[str, Any]]}
            - {"kind": "tool", "tool_calls": List[Dict[str, Any]]}

    Order preservation:
      - We iterate messages in order.
      - Within a message, we iterate the `content` parts in order, appending
        matching segments.
    
    Error handling:
      - If a message is not in the expected format (not a dict), it will be
        converted to a string and stored as a text segment with role "user".
    """
    conv: List[Dict[str, Any]] = []

    # Handle non-list input by converting to string
    if not isinstance(messages, list):
        return [{"role": "user", "content": {"segments": [{"kind": "text", "text": str(messages)}]}}]

    for msg in messages:
        # Handle non-dict messages by converting to string
        if not isinstance(msg, dict):
            conv.append({
                "role": "user",
                "content": {"segments": [{"kind": "text", "text": str(msg)}]}
            })
            continue
        
        # Try to get role, default to "user" if not present
        role = msg.get("role", "user")
        if not isinstance(role, str):
            role = str(role)
        
        new_msg: Dict[str, Any] = {"role": role, "content": {"segments": []}}
        segments: List[Dict[str, Any]] = new_msg["content"]["segments"]

        # Preserve optional identifiers.
        for opt in ("name", "id"):
            if opt in msg:
                new_msg[opt] = msg[opt]

        content = msg.get("content")

        # 1) New multimodal content array
        if isinstance(content, list):
            for part in content:
                ptype = part.get("type") if isinstance(part, dict) else None
                if ptype == "text":
                    segments.append({"kind": "text", "text": part.get("text", "")})
                elif ptype in ("image_url", "input_image"):
                    # Normalize to something we can later map to OpenAI format.
                    # Keep as provided to preserve fidelity and allow URLs or data URLs.
                    img_val: Union[str, Dict[str, Any], None] = None
                    if isinstance(part, dict):
                        # Common shapes: {"image_url": {"url": str}}, or {"url": str}
                        img_val = part.get("image_url") or part.get("url") or part
                    else:
                        img_val = part
                    segments.append({"kind": "image", "image": img_val})
                else:
                    # Unknown part type: store textual representation for visibility
                    segments.append({"kind": "text", "text": str(part)})

        # 2) Simple string
        elif isinstance(content, str):
            segments.append({"kind": "text", "text": content})

        # 3) Dict content (may include text / image / tool_calls)
        elif isinstance(content, dict):
            # Prefer explicit segments if provided
            if "segments" in content and isinstance(content["segments"], list):
                for seg in content["segments"]:
                    segments.append(seg)
            else:
                text_val = content.get("text")
                if isinstance(text_val, str) and text_val != "":
                    segments.append({"kind": "text", "text": text_val})
                images = content.get("image")
                if isinstance(images, list):
                    for img in images:
                        segments.append({"kind": "image", "image": img})
        
        # 4) Handle None or unexpected content types by converting to string
        else:
            if content is not None:
                segments.append({"kind": "text", "text": str(content)})

        # 5) Tool calls (multi-tool and legacy single-function)
        tool_calls_out: List[Dict[str, Any]] = []

        raw_tool_calls = msg.get("tool_calls", None)

        # Normalize tool_calls which may be:
        # - a list of dicts (standard)
        # - a JSON-encoded string of the above
        # - a single dict
        # - any other type, which we still preserve as a single call
        normalized_tool_calls: Any = raw_tool_calls
        if isinstance(raw_tool_calls, str):
            try:
                normalized_tool_calls = json.loads(raw_tool_calls)
            except Exception:
                # Keep raw string if not valid JSON
                normalized_tool_calls = raw_tool_calls

        if isinstance(normalized_tool_calls, dict):
            normalized_tool_calls = [normalized_tool_calls]

        if isinstance(normalized_tool_calls, list):
            for tc in normalized_tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = (fn.get("name") if isinstance(fn, dict) else None) or (
                    tc.get("name") if isinstance(tc, dict) else None
                )
                raw_args = (fn.get("arguments") if isinstance(fn, dict) else None) or (
                    tc.get("arguments") if isinstance(tc, dict) else ""
                )
                try:
                    args_val = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args_val = raw_args
                tc_out: Dict[str, Any] = {
                    "name": name,
                    "arguments": args_val,
                    "id": (tc.get("id") if isinstance(tc, dict) else None)
                    or (tc.get("tool_call_id") if isinstance(tc, dict) else None),
                }
                # Preserve any additional, non-standard keys.
                if isinstance(tc, dict):
                    for k, v in tc.items():
                        if k not in {"function", "id", "tool_call_id"}:
                            tc_out[k] = v
                tool_calls_out.append(tc_out)
        elif normalized_tool_calls is not None:
            # Preserve any non-standard shape as a single tool call record
            tool_calls_out.append(
                {
                    "name": None,
                    "arguments": normalized_tool_calls,
                    "id": msg.get("id"),
                }
            )

        elif "function_call" in msg and msg["function_call"]:
            fc = msg["function_call"]
            if isinstance(fc, dict):
                raw_args = fc.get("arguments", "")
                try:
                    args_val = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args_val = raw_args
                tool_calls_out.append({
                    "name": fc.get("name"),
                    "arguments": args_val,
                    "id": fc.get("id") or msg.get("id"),
                })
            else:
                # Handle non-dict function_call by converting to string
                tool_calls_out.append({
                    "name": None,
                    "arguments": str(fc),
                    "id": msg.get("id"),
                })

        if tool_calls_out:
            segments.append({"kind": "tool", "tool_calls": tool_calls_out})

        conv.append(new_msg)

    return conv