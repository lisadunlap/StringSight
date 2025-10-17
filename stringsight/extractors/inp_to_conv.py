import json

def format_oai_simple(row):
  return [{
    "role": "user", "content": row["prompt"]
  }, {
    "role": "assistant", "content": row["model_response"]
  }]

def openai_messages_to_conv(messages):
    """
    Convert a list of OpenAI ChatCompletion `messages` (including tool call
    structures) to the condensed conversation format consumed by
    `conv_to_str`.
    """
    conv = []
    for msg in messages:
        # Mandatory role.
        new_msg = {'role': msg['role']}

        # Preserve optional identifiers.
        for opt in ('name', 'id'):
            if opt in msg:
                new_msg[opt] = msg[opt]

        text_part = msg.get('content')
        tool_calls_out = []

        # New multi-tool format.
        if 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                fn = tc.get('function', {})
                name = fn.get('name') or tc.get('name')
                raw_args = fn.get('arguments') if fn else tc.get('arguments', '')
                try:
                    args_val = json.loads(raw_args)
                except Exception:
                    args_val = raw_args
                tc_out = {
                    'name': name,
                    'arguments': args_val,
                    'id': tc.get('id') or tc.get('tool_call_id'),
                }
                # Preserve any additional, non-standard keys.
                for k, v in tc.items():
                    if k not in {'function', 'id', 'tool_call_id'}:
                        tc_out[k] = v
                tool_calls_out.append(tc_out)

        # Legacy single-function structure.
        elif 'function_call' in msg and msg['function_call']:
            fc = msg['function_call']
            raw_args = fc.get('arguments', '')
            try:
                args_val = json.loads(raw_args)
            except Exception:
                args_val = raw_args
            tool_calls_out.append({
                'name': fc.get('name'),
                'arguments': args_val,
                'id': fc.get('id') or msg.get('id'),
            })

        # Assemble the content field.
        if tool_calls_out:
            content_dict = {}
            if text_part not in (None, ''):
                content_dict['text'] = text_part
            content_dict['tool_calls'] = tool_calls_out
            new_msg['content'] = content_dict
        else:
            new_msg['content'] = text_part

        conv.append(new_msg)
    return conv