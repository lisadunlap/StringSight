# This contains functions to convert OAI conversation objects to strings
# This format is a list of dictionaries with the following keys:
# - role: "user", "assistant", "system", "tool", etc
# - content: the content of the message (can be a string or a dictionary with values "text", "image", or "tool_calls")
# - the tool_calls key (optional) in the content dictionary: a list of tool calls, each tool call is a dict with the following keys:
#   - name: the name of the tool
#   - arguments: the arguments to the tool
#   - tool_call_id: the id of the tool call
#   - anything else that is present in the tool call
# - name (optional): the name of the model or tool that sent the message (this should persist for the entire conversation)
# - id (optional): the id of the specific model or tool, this is meant to only be used once (e.g. the id of a specifically function call)
# - any other keys that are present in the conversation object (not shown in the string right now)

import pprint
import ast
import json
from typing import Any, Dict


def pretty_print_dict(val):
    """
    Pretty print a dictionary or a string that can be parsed as a dictionary.
    """
    if isinstance(val, dict):
        return pprint.pformat(val, width=80, compact=False)
    if isinstance(val, str):
        try:
            # Try to parse as dict
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return pprint.pformat(parsed, width=80, compact=False)
        except Exception:
            pass
    return str(val)


def convert_tool_calls_to_str(tool_calls: Any) -> str:
    """
    Convert tool call data into a readable string representation.

    Args:
        tool_calls:
            - Preferred: list of dicts, where each dict represents one tool call
              with at least `name`, `arguments`, and optional `id` and extra keys.
            - Also accepted:
              - JSON-encoded string of the above list or a single dict
              - Single dict representing one tool call
              - Any other type, which will be stringified as a fallback

    Returns:
        Human-readable string describing the tool calls, with arguments and any
        additional metadata pretty-printed when possible.
    """
    # If we received a JSON string, try to parse it to avoid losing structure.
    if isinstance(tool_calls, str):
        try:
            parsed = json.loads(tool_calls)
            tool_calls = parsed
        except Exception:
            # Fall back to raw string if not valid JSON
            return str(tool_calls)

    # Normalize a single dict into a list of dicts
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]

    # Handle non-list tool_calls (after normalization)
    if not isinstance(tool_calls, list):
        return str(tool_calls)

    # Create name and arguments string for each tool call
    tool_calls_str = []
    for tool_call in tool_calls:
        # Handle non-dict tool calls
        if not isinstance(tool_call, dict):
            tool_calls_str.append(str(tool_call))
            continue

        name = tool_call.get("name", "<no_name>")
        args = tool_call.get("arguments", {})
        # Pretty print arguments if dict or dict-string
        args_str = pretty_print_dict(args)
        base = f"call {name} with args {args_str} (id: {tool_call.get('id', '<no_id>')})"
        extra_keys = [key for key in tool_call.keys() if key not in ["name", "arguments", "id"]]
        if extra_keys:
            # Show extra key-value pairs, pretty-printed if dict
            extras = {k: tool_call[k] for k in extra_keys}
            base += f"\nadditional info: {pretty_print_dict(extras)}"
        tool_calls_str.append(base)
    return "\n".join(tool_calls_str)


def _render_image_value(image: Any) -> str:
    """
    Render an image value to a readable single-line string.
    Accepts URL/data-URL strings or dict-like objects with nested url.
    """
    # String: URL or data URL
    if isinstance(image, str):
        if image.startswith('data:'):
            return f"image: data-url (len={len(image)})"
        return f"image: {image}"

    # Dict: try common shapes
    if isinstance(image, dict):
        url = None
        if 'url' in image and isinstance(image['url'], str):
            url = image['url']
        elif 'image_url' in image and isinstance(image['image_url'], dict):
            inner = image['image_url']
            if 'url' in inner and isinstance(inner['url'], str):
                url = inner['url']
        if url is not None:
            if url.startswith('data:'):
                return f"image: data-url (len={len(url)})"
            return f"image: {url}"
        # Fallback pretty-print
        return f"image: {pretty_print_dict(image)}"

    # Fallback
    return f"image: {str(image)}"


def convert_content_to_str(content: Any) -> str:
    """
    Convert the content of a message to a string.
    Pretty print any dictionary values or stringified dictionaries.
    
    Handles any data format by converting to string as fallback.
    """
    # Handle None
    if content is None:
        return "(empty)"
    
    # Handle strings
    if isinstance(content, str):
        # Try to pretty print if it's a dict string
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                return pretty_print_dict(parsed)
        except Exception:
            pass
        return content
    
    # Handle dictionaries
    elif isinstance(content, dict):
        # Ordered segments path (preferred when present)
        if 'segments' in content and isinstance(content['segments'], list):
            out_lines = []
            for seg in content['segments']:
                if not isinstance(seg, dict) or 'kind' not in seg:
                    out_lines.append(str(seg))
                    continue
                kind = seg['kind']
                if kind == 'text':
                    out_lines.append(pretty_print_dict(seg.get('text', '')))
                elif kind == 'image':
                    out_lines.append(_render_image_value(seg.get('image')))
                elif kind == 'tool':
                    out_lines.append(convert_tool_calls_to_str(seg.get('tool_calls', [])))
                else:
                    out_lines.append(str(seg))
            return "\n".join(out_lines)

        ret = []
        for key, value in content.items():
            try:
                if key == 'text':
                    # Pretty print if text is a dict or dict-string
                    ret.append(pretty_print_dict(value))
                elif key == 'image':
                    if isinstance(value, list):
                        for img in value:
                            ret.append(_render_image_value(img))
                    else:
                        ret.append(_render_image_value(value))
                elif key == 'tool_calls':
                    ret.append(convert_tool_calls_to_str(value))
                else:
                    # Pretty print for any other dict or dict-string
                    ret.append(f"{key}: {pretty_print_dict(value)}")
            except Exception as e:
                # Fallback for any formatting errors
                ret.append(f"{key}: {str(value)}")
        return "\n".join(ret)
    
    # Handle lists and tuples
    elif isinstance(content, (list, tuple)):
        try:
            return json.dumps(content, indent=2)
        except:
            return str(content)
    
    # Fallback for any other type
    else:
        try:
            return pretty_print_dict(content)
        except:
            return str(content)


def conv_to_str(conv: Any) -> str:
    """
    Convert an OAI conversation object to a string.
    
    Handles any data format by converting to string as fallback.
    """
    def _format_message_content(msg: Dict[str, Any]) -> str:
        """
        Format the content and any associated tool calls for a single message.

        Args:
            msg: Message dictionary with optional keys:
                - "content": arbitrary payload (string, dict, list, or None)
                - "tool_calls": list/dict/JSON-string describing tool calls

        Returns:
            Combined string containing the rendered content and, if present,
            a "Tool calls:" section describing any tool invocations.
        """
        content = msg.get("content", "")
        content_str = convert_content_to_str(content)

        tool_calls_val = msg.get("tool_calls", None)
        tool_calls_str = ""
        if tool_calls_val is not None and tool_calls_val != []:
            tool_calls_str = convert_tool_calls_to_str(tool_calls_val)

        if tool_calls_str:
            if content_str:
                return f"{content_str}\n\nTool calls:\n{tool_calls_str}"
            return tool_calls_str
        return content_str

    # Handle non-list conversations: still wrap in BEGIN/END markers so that
    # even fallback string representations are clearly delimited.
    if not isinstance(conv, list):
        body = str(conv)
        return (
            "## BEGINNING OF CONVERSATION TO ANAZLYZE\n"
            f"{body}\n"
            "## END OF CONVERSATION TO ANAZLYZE"
        )

    ret = []
    for msg in conv:
        # Handle non-dict messages
        if not isinstance(msg, dict):
            ret.append(f"\n**message:**\n{str(msg)}")
            continue

        # Get role with fallback
        role = str(msg.get("role", "unknown"))

        # Special handling for tool messages
        if role == "tool":
            name = msg.get("name", "<unnamed_tool>")
            body = _format_message_content(msg)
            ret.append(f"**output of tool {name}**\n{body}")
        else:
            # Regular message formatting
            body = _format_message_content(msg)

            if "name" in msg:
                name = str(msg["name"])
                if "id" in msg:
                    msg_id = str(msg["id"])
                    ret.append(f"\n**{role} {name} (id: {msg_id}):**\n{body}")
                else:
                    ret.append(f"\n**{role} {name}**\n{body}")
            else:
                ret.append(f"\n**{role}:**\n{body}")

    body = "\n\n".join(ret)
    return (
        "## BEGINNING OF CONVERSATION TO ANAZLYZE\n"
        f"{body}\n"
        "## END OF CONVERSATION TO ANAZLYZE"
    )


def simple_to_oai_format(prompt: str, response: str) -> list:
    """
    Convert a simple prompt-response pair to OAI format.
    
    Args:
        prompt: The user's prompt/question
        response: The model's response
        
    Returns:
        List of dictionaries in OAI conversation format
    """
    return [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant", 
            "content": response
        }
    ]


def check_and_convert_to_oai_format(prompt: str, response: str) -> tuple[list, bool]:
    """
    Check if response is a string and convert to OAI format if needed.
    
    Args:
        prompt: The user's prompt/question
        response: The model's response (could be string or already OAI format)
        
    Returns:
        Tuple of (conversation_in_oai_format, was_converted)
    """
    # If response is already a list (OAI format), return as is
    if isinstance(response, list):
        return response, False
    
    # If response is a string, convert to OAI format
    if isinstance(response, str):
        return simple_to_oai_format(prompt, response), True
    
    # For other types, try to convert to string first
    try:
        response_str = str(response)
        return simple_to_oai_format(prompt, response_str), True
    except Exception:
        # If conversion fails, return as is
        return response, False