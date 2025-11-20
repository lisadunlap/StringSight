"""
Parser to convert OpenTelemetry/OpenInference trace format to OpenAI (OAI) format.

This module provides functionality to parse trace data from OpenTelemetry/OpenInference
formats (e.g., from Patronus AI, OpenInference, etc.) and convert them to the standard
OpenAI chat completion message format.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


def parse_trace_to_oai(trace: Union[Dict[str, Any], str]) -> List[Dict[str, Any]]:
    """
    Parse an OpenTelemetry/OpenInference trace into OpenAI (OAI) format.
    
    Args:
        trace: Trace data as a dictionary or JSON string. Expected structure:
            - Has 'spans' key with list of span objects
            - Each span has 'span_name', 'span_attributes', 'child_spans', etc.
            - LLM spans contain 'llm.input_messages' and 'llm.output_messages' in span_attributes
            - Tool spans contain tool call information
    
    Returns:
        List of messages in OAI format, where each message is a dict with:
            - 'role': 'user', 'assistant', 'system', or 'tool'
            - 'content': string or list of content parts
            - 'tool_calls': (optional) list of tool calls for assistant messages
            - 'tool_call_id': (optional) for tool messages
            - 'name': (optional) tool name for tool messages
    
    Example:
        >>> trace = {"spans": [...]}
        >>> messages = parse_trace_to_oai(trace)
        >>> # messages is now in OAI format
    """
    # Handle string input
    if isinstance(trace, str):
        try:
            trace = json.loads(trace)
        except json.JSONDecodeError:
            raise ValueError("Trace string is not valid JSON")
    
    if not isinstance(trace, dict):
        raise ValueError("Trace must be a dictionary or JSON string")
    
    # Extract spans from trace
    spans = trace.get('spans', [])
    if not spans:
        return []
    
    # Collect all events (LLM calls, tool calls, etc.) with timestamps
    events = []
    
    def extract_span_events(span: Dict[str, Any], parent_timestamp: Optional[str] = None):
        """Recursively extract events from spans and their children."""
        span_attributes = span.get('span_attributes', {})
        timestamp = span.get('timestamp', parent_timestamp)
        span_name = span.get('span_name', '')
        
        # Extract LLM messages - handle both nested and flattened formats
        # Flattened format: llm.input_messages.0.message.content, llm.input_messages.0.message.role, etc.
        input_messages = _extract_flattened_messages(span_attributes, 'llm.input_messages')
        if not input_messages:
            # Try nested format
            input_messages = _parse_messages_from_attr(span_attributes.get('llm.input_messages'))
        
        for msg in input_messages:
            events.append({
                'type': 'llm_input',
                'message': msg,
                'timestamp': timestamp,
                'span_name': span_name
            })
        
        # Extract output messages
        output_messages = _extract_flattened_messages(span_attributes, 'llm.output_messages')
        if not output_messages:
            # Try nested format
            output_messages = _parse_messages_from_attr(span_attributes.get('llm.output_messages'))
        
        for msg in output_messages:
            events.append({
                'type': 'llm_output',
                'message': msg,
                'timestamp': timestamp,
                'span_name': span_name,
                'span_attributes': span_attributes
            })
        
        # Extract tool calls from LLM output messages
        for msg in output_messages:
            if isinstance(msg, dict) and 'tool_calls' in msg:
                for tool_call in msg.get('tool_calls', []):
                    events.append({
                        'type': 'tool_call',
                        'tool_call': tool_call,
                        'timestamp': timestamp,
                        'span_name': span_name
                    })
        
        # Extract tool execution results
        if 'openinference.span.kind' in span_attributes:
            span_kind = span_attributes.get('openinference.span.kind')
            if span_kind == 'TOOL':
                tool_name = span_attributes.get('tool.name', span_name)
                tool_input = _parse_value(span_attributes.get('input.value'))
                tool_output = _parse_value(span_attributes.get('output.value'))
                
                # Extract tool call ID from input if available
                tool_call_id = None
                if isinstance(tool_input, dict):
                    tool_call_id = tool_input.get('tool_call_id') or tool_input.get('id')
                
                events.append({
                    'type': 'tool_result',
                    'tool_name': tool_name,
                    'tool_input': tool_input,
                    'tool_output': tool_output,
                    'tool_call_id': tool_call_id,
                    'timestamp': timestamp,
                    'span_name': span_name
                })
        
        # Recursively process child spans
        for child_span in span.get('child_spans', []):
            extract_span_events(child_span, timestamp)
    
    # Extract all events from all top-level spans
    for span in spans:
        extract_span_events(span)
    
    # Sort events by timestamp
    events.sort(key=lambda x: x.get('timestamp', ''))
    
    # Convert events to OAI format messages
    messages = []
    tool_call_counter = {}  # Track tool call IDs
    
    for event in events:
        if event['type'] == 'llm_input':
            msg = event['message']
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                # Skip system messages for now (can be added if needed)
                if role == 'system':
                    continue
                messages.append({
                    'role': role,
                    'content': content
                })
        
        elif event['type'] == 'llm_output':
            msg = event['message']
            if isinstance(msg, dict):
                role = msg.get('role', 'assistant')
                content = msg.get('content', '')
                tool_calls = msg.get('tool_calls')
                
                oai_msg = {
                    'role': role,
                    'content': content if content else None
                }
                
                if tool_calls:
                    # Convert tool calls to OAI format
                    oai_tool_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tool_call_id = tc.get('id') or tc.get('tool_call_id')
                            if not tool_call_id:
                                # Generate a unique ID
                                tool_call_id = f"call_{len(tool_call_counter)}"
                                tool_call_counter[tool_call_id] = True
                            
                            function_name = tc.get('function', {}).get('name') if isinstance(tc.get('function'), dict) else tc.get('name')
                            function_args = tc.get('function', {}).get('arguments') if isinstance(tc.get('function'), dict) else tc.get('arguments')
                            
                            # Parse arguments if it's a string
                            if isinstance(function_args, str):
                                try:
                                    function_args = json.loads(function_args)
                                except json.JSONDecodeError:
                                    pass
                            
                            oai_tool_calls.append({
                                'id': tool_call_id,
                                'type': 'function',
                                'function': {
                                    'name': function_name or 'unknown',
                                    'arguments': json.dumps(function_args) if not isinstance(function_args, str) else function_args
                                }
                            })
                    
                    if oai_tool_calls:
                        oai_msg['tool_calls'] = oai_tool_calls
                
                messages.append(oai_msg)
        
        elif event['type'] == 'tool_result':
            tool_name = event.get('tool_name', 'unknown')
            tool_output = event.get('tool_output')
            tool_call_id = event.get('tool_call_id')
            
            # Convert tool output to string if needed
            if tool_output is None:
                tool_output = ''
            elif not isinstance(tool_output, str):
                try:
                    tool_output = json.dumps(tool_output)
                except (TypeError, ValueError):
                    tool_output = str(tool_output)
            
            # Generate tool call ID if not present
            if not tool_call_id:
                tool_call_id = f"call_{len(tool_call_counter)}"
                tool_call_counter[tool_call_id] = True
            
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call_id,
                'name': tool_name,
                'content': tool_output
            })
    
    # Clean up messages: remove None content, merge consecutive messages of same role
    cleaned_messages = []
    for msg in messages:
        # Skip messages with no content and no tool calls
        if not msg.get('content') and not msg.get('tool_calls'):
            continue
        
        # Remove None content
        if msg.get('content') is None:
            msg.pop('content', None)
        
        cleaned_messages.append(msg)
    
    return cleaned_messages


def _parse_messages_from_attr(attr_value: Any) -> List[Dict[str, Any]]:
    """
    Parse messages from span attribute value.
    
    The attribute value can be:
    - A JSON string
    - A list of message dicts
    - A single message dict
    """
    if attr_value is None:
        return []
    
    # If it's already a list of dicts, return as is
    if isinstance(attr_value, list):
        return attr_value
    
    # If it's a dict, wrap in list
    if isinstance(attr_value, dict):
        return [attr_value]
    
    # If it's a string, try to parse as JSON
    if isinstance(attr_value, str):
        try:
            parsed = json.loads(attr_value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list (don't raise error)
            return []
    
    return []


def _parse_value(value: Any) -> Any:
    """Parse a value that might be a JSON string or already parsed."""
    if value is None:
        return None
    
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return the string as-is
            return value
    
    return value


def _extract_flattened_messages(span_attributes: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    """
    Extract messages from flattened span attributes.
    
    Handles format like:
    - llm.input_messages.0.message.content
    - llm.input_messages.0.message.role
    - llm.input_messages.1.message.content
    etc.
    """
    messages = []
    message_indices = set()
    
    # Find all message indices
    for key in span_attributes.keys():
        if key.startswith(prefix + '.'):
            # Extract index (e.g., "0" from "llm.input_messages.0.message.content")
            parts = key[len(prefix) + 1:].split('.')
            if parts and parts[0].isdigit():
                message_indices.add(int(parts[0]))
    
    # Build messages from flattened attributes
    for idx in sorted(message_indices):
        msg = {}
        
        # Extract role
        role_key = f"{prefix}.{idx}.message.role"
        if role_key in span_attributes:
            msg['role'] = span_attributes[role_key]
        
        # Extract content - handle both single content and list of content parts
        content_key = f"{prefix}.{idx}.message.content"
        if content_key in span_attributes:
            content = span_attributes[content_key]
            # Handle content that might be a list or string
            if isinstance(content, str):
                try:
                    # Try to parse as JSON (might be a list of content parts)
                    parsed = json.loads(content)
                    msg['content'] = parsed
                except json.JSONDecodeError:
                    msg['content'] = content
            elif isinstance(content, list):
                # Already a list (multimodal content)
                msg['content'] = content
            else:
                msg['content'] = content
        
        # Also check for content parts (multimodal format)
        # Format: llm.input_messages.0.message.content.0.type, llm.input_messages.0.message.content.0.text, etc.
        content_parts = []
        content_part_idx = 0
        while True:
            content_part_type_key = f"{prefix}.{idx}.message.content.{content_part_idx}.type"
            if content_part_type_key not in span_attributes:
                break
            
            content_part = {}
            content_part['type'] = span_attributes[content_part_type_key]
            
            # Extract text if present
            text_key = f"{prefix}.{idx}.message.content.{content_part_idx}.text"
            if text_key in span_attributes:
                content_part['text'] = span_attributes[text_key]
            
            # Extract image_url if present
            image_url_key = f"{prefix}.{idx}.message.content.{content_part_idx}.image_url"
            if image_url_key in span_attributes:
                image_url = span_attributes[image_url_key]
                if isinstance(image_url, str):
                    try:
                        image_url = json.loads(image_url)
                    except json.JSONDecodeError:
                        pass
                content_part['image_url'] = image_url
            
            if content_part:
                content_parts.append(content_part)
            content_part_idx += 1
        
        # If we found content parts, use them instead of single content
        if content_parts:
            msg['content'] = content_parts
        
        # Extract tool_calls if present
        tool_calls = []
        tool_call_idx = 0
        while True:
            tool_call_id_key = f"{prefix}.{idx}.message.tool_calls.{tool_call_idx}.id"
            if tool_call_id_key not in span_attributes:
                break
            
            tool_call = {}
            tool_call['id'] = span_attributes.get(tool_call_id_key)
            
            # Extract function name
            function_name_key = f"{prefix}.{idx}.message.tool_calls.{tool_call_idx}.function.name"
            if function_name_key in span_attributes:
                if 'function' not in tool_call:
                    tool_call['function'] = {}
                tool_call['function']['name'] = span_attributes[function_name_key]
            
            # Extract function arguments
            function_args_key = f"{prefix}.{idx}.message.tool_calls.{tool_call_idx}.function.arguments"
            if function_args_key in span_attributes:
                if 'function' not in tool_call:
                    tool_call['function'] = {}
                args = span_attributes[function_args_key]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                tool_call['function']['arguments'] = args
            
            if tool_call:
                tool_calls.append(tool_call)
            tool_call_idx += 1
        
        if tool_calls:
            msg['tool_calls'] = tool_calls
        
        if msg:
            messages.append(msg)
    
    return messages


def extract_tools_from_trace(trace: Union[Dict[str, Any], str]) -> List[Dict[str, Any]]:
    """
    Extract tool definitions from a trace.
    
    Args:
        trace: Trace data as a dictionary or JSON string
    
    Returns:
        List of tool definitions in OAI format, where each tool is a dict with:
            - 'type': 'function'
            - 'function': dict with 'name', 'description', 'parameters'
    """
    # Handle string input
    if isinstance(trace, str):
        try:
            trace = json.loads(trace)
        except json.JSONDecodeError:
            raise ValueError("Trace string is not valid JSON")
    
    if not isinstance(trace, dict):
        raise ValueError("Trace must be a dictionary or JSON string")
    
    tools = []
    tool_names_seen = set()
    
    def extract_tools_from_span(span: Dict[str, Any]):
        """Recursively extract tool definitions from spans."""
        span_attributes = span.get('span_attributes', {})
        
        # Check for tool information in span attributes
        if 'openinference.span.kind' in span_attributes:
            span_kind = span_attributes.get('openinference.span.kind')
            if span_kind == 'TOOL':
                tool_name = span_attributes.get('tool.name')
                tool_description = span_attributes.get('tool.description', '')
                tool_parameters = span_attributes.get('tool.parameters')
                
                if tool_name and tool_name not in tool_names_seen:
                    # Parse parameters if it's a string
                    if isinstance(tool_parameters, str):
                        try:
                            tool_parameters = json.loads(tool_parameters)
                        except json.JSONDecodeError:
                            tool_parameters = {}
                    elif tool_parameters is None:
                        tool_parameters = {}
                    
                    tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'description': tool_description,
                            'parameters': tool_parameters
                        }
                    })
                    tool_names_seen.add(tool_name)
        
        # Recursively process child spans
        for child_span in span.get('child_spans', []):
            extract_tools_from_span(child_span)
    
    # Extract tools from all top-level spans
    spans = trace.get('spans', [])
    for span in spans:
        extract_tools_from_span(span)
    
    return tools

