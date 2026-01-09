"""
TAU2-style evaluation JSON converters.

This module contains utilities to convert TAU2-ish evaluation outputs (as seen in
`data/tau2/*.json`) into StringSight's expected tidy dataframe format.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, TypedDict

import pandas as pd


class OAIMessage(TypedDict, total=False):
    """
    A minimal conversation message structure compatible with StringSight.

    Note:
        Although this format is inspired by OpenAI chat messages, StringSight's
        renderers accept a slightly more permissive tool call shape. In
        particular, for tool-using assistant turns, `tool_calls` is represented
        as a list of dicts with `name` and dict `arguments`, matching what
        `stringsight/extractors/conv_to_str.py` expects.

    Keys:
    - role: "system" | "user" | "assistant" | "tool"
    - content: string content (may be omitted for tool-calling assistant turns)
    - tool_calls: list of tool calls for assistant messages (StringSight-style)
    - tool_call_id: tool call id for tool messages
    - name: tool name for tool messages (optional)
    """

    role: str
    content: str
    tool_calls: List[Dict[str, Any]]
    tool_call_id: str
    name: str


def tau2_json_to_stringsight_df(
    data: Mapping[str, Any],
    *,
    prompt_fields: Sequence[str] = ("known_info",),
    include_system_message: bool = True,
    include_scenario_message: bool = True,
    scenario_fields: Sequence[str] = ("domain", "reason_for_call", "known_info", "task_instructions"),
) -> pd.DataFrame:
    """
    Convert a TAU2-style evaluation JSON into a StringSight tidy dataframe.

    Expected input schema (minimum required):
    - data["info"]["agent_info"]["llm"]: str
    - data["info"]["environment_info"]["policy"]: str
    - data["info"]["user_info"]["global_simulation_guidelines"]: str
    - data["tasks"]: list of tasks, each containing:
      - task["id"]: str
      - task["user_scenario"]["instructions"]: dict containing at least:
        - "known_info": str  (used as the output dataframe's `prompt`)
        - plus any fields listed in `scenario_fields` (if include_scenario_message=True)
    - data["simulations"]: list of simulations, each containing:
      - sim["task_id"]: str (matches a task id)
      - sim["messages"]: list of message dicts in TAU2 format
      - sim["reward_info"]["reward"]: float

    Output dataframe columns:
    - prompt: str
      - By default: the scenario "known_info" field.
      - If prompt_fields contains multiple keys: a labeled, concatenated string containing those fields.
    - model: str
      - data["info"]["agent_info"]["llm"]
    - model_response: list[dict]
      - OpenAI-conversation-style messages:
        - optionally prepends a system message (policy + guidelines)
        - optionally prepends a synthetic user message containing scenario fields
        - followed by the recorded simulation trace converted to OAI format
    - reward: float
      - sim["reward_info"]["reward"]
    """

    tasks_by_id: Dict[str, Mapping[str, Any]] = {t["id"]: t for t in data["tasks"]}

    model_name: str = data["info"]["agent_info"]["llm"]
    policy: str = data["info"]["environment_info"]["policy"]
    guidelines: str = data["info"]["user_info"]["global_simulation_guidelines"]

    rows: List[Dict[str, Any]] = []
    for sim in data["simulations"]:
        task_id: str = sim["task_id"]
        task = tasks_by_id[task_id]
        instr: Mapping[str, Any] = task["user_scenario"]["instructions"]

        prompt: str = _format_prompt(instr, prompt_fields=prompt_fields)
        reward: float = sim["reward_info"]["reward"]

        oai_messages: List[OAIMessage] = []

        if include_system_message:
            oai_messages.append(
                {
                    "role": "system",
                    "content": f"{policy}\n\n{guidelines}",
                }
            )

        if include_scenario_message:
            scenario_text = _format_scenario_message(instr, scenario_fields=scenario_fields)
            oai_messages.append({"role": "user", "content": scenario_text})

        oai_messages.extend(_tau2_messages_to_oai(sim["messages"]))

        rows.append(
            {
                "prompt": prompt,
                "model": model_name,
                "model_response": oai_messages,
                "reward": reward,
            }
        )

    return pd.DataFrame(rows, columns=["prompt", "model", "model_response", "reward"])


def _format_scenario_message(instr: Mapping[str, Any], *, scenario_fields: Sequence[str]) -> str:
    """
    Format a synthetic user message from TAU2 scenario instructions.

    Args:
        instr: Task instructions mapping (e.g., task["user_scenario"]["instructions"]).
              Expected to contain keys listed in scenario_fields.
        scenario_fields: Ordered keys to include in the rendered message.

    Returns:
        A single string suitable for an OpenAI-format user message.
    """

    return _format_keyed_sections(instr, fields=scenario_fields, title="TAU2 scenario info:")


def _format_prompt(instr: Mapping[str, Any], *, prompt_fields: Sequence[str]) -> str:
    """
    Format the output dataframe's `prompt` field from TAU2 scenario instructions.

    Args:
        instr: Task instructions mapping (e.g., task["user_scenario"]["instructions"]).
        prompt_fields: Ordered keys to include in the prompt.
            - If exactly one key is provided, the prompt is that raw string (no labels),
              preserving the common "prompt == known_info" behavior.
            - If multiple keys are provided, the prompt is a labeled, concatenated string.

    Returns:
        Prompt string for the output dataframe.
    """

    if len(prompt_fields) == 1:
        only_key = prompt_fields[0]
        return str(instr[only_key]).strip()

    return _format_keyed_sections(instr, fields=prompt_fields, title=None)


def _format_keyed_sections(
    instr: Mapping[str, Any],
    *,
    fields: Sequence[str],
    title: str | None,
) -> str:
    """
    Render selected instruction fields into a single text block.

    Args:
        instr: Task instructions mapping (e.g., task["user_scenario"]["instructions"]).
        fields: Ordered keys to include.
        title: Optional title line prepended to the block.

    Returns:
        String with optional title and [key] sections.
    """

    lines: List[str] = []
    if title is not None:
        lines.append(title)
    for key in fields:
        value = instr[key]
        lines.append(f"\n[{key}]\n{value}")
    return "\n".join(lines).strip()


def _tau2_messages_to_oai(messages: List[Mapping[str, Any]]) -> List[OAIMessage]:
    """
    Convert TAU2 recorded messages into a minimal conversation format.

    TAU2 message shape (observed):
    - role: str
    - content: str
    - tool_calls: optional list of {"id": str, "name": str, "arguments": dict, ...} for assistant turns
    - id: str (for tool messages; corresponds to the tool call id)

    Returns:
        List of messages compatible with StringSight's conversation renderer.

        Tool calls:
            For assistant messages, tool calls are emitted as a list of dicts:
            - id: str
            - name: str
            - arguments: dict
    """

    out: List[OAIMessage] = []
    for m in messages:
        role = m["role"]

        if role == "tool":
            tool_call_id = m["id"]
            content = m.get("content", "")
            out.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
            continue

        msg: OAIMessage = {"role": role, "content": m.get("content", "")}

        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                for tc in tool_calls
            ]
            # For tool-calling messages, OpenAI often uses `content: None`. StringSight accepts either.
            if msg.get("content", "") == "":
                msg.pop("content", None)

        out.append(msg)

    return out
