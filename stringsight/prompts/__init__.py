"""
Prompts module for StringSight.

This module contains system prompts and prompt utilities for property extraction.
"""

from .extractor_prompts import (
    sbs_system_prompt,
    sbs_system_prompt_custom,
    single_model_system_prompt,
    single_model_system_prompt_custom,
    # Default task descriptions for extractor prompts
    sbs_default_task_description,
    single_model_default_task_description,
    single_model_system_prompt_custom_revised,
)

# Import agent-specific prompts for agentic environments
from .agents import (
    agent_system_prompt,
    taubench_comparison_system_prompt,
    agentic_swe_system_prompt,
    agentic_tool_focused_prompt,
    agentic_reasoning_focused_prompt,
    agentic_reward_hacking_focused_prompt,
    # Agent custom templates
    agent_system_prompt_custom,
    agent_sbs_system_prompt_custom,
    # Default task descriptions for agent prompts
    agent_system_prompt_custom_task_description,
    agent_sbs_system_prompt_custom_task_description,
    agent_system_prompt_custom_revised
)


# Import fixed-axis prompts
from .fixed_axes import (
    fixed_axis_prompt,
)

# ------------------------------------------------------------------
# Prompt dictionaries (aliases)
# ------------------------------------------------------------------

DEFAULT_PROMPTS = {
    "single_model": {
        "template": single_model_system_prompt_custom_revised,
        "default_task_description": single_model_default_task_description,
    },
    "side_by_side": {
        "template": sbs_system_prompt_custom,
        "default_task_description": sbs_default_task_description,
    },
}

AGENT_PROMPTS = {
    "single_model": {
        "template": agent_system_prompt_custom_revised,
        "default_task_description": agent_system_prompt_custom_task_description,
    },
    "side_by_side": {
        "template": agent_sbs_system_prompt_custom,
        "default_task_description": agent_sbs_system_prompt_custom_task_description,
    },
}

PROMPTS = {
    "default": DEFAULT_PROMPTS,
    "agent": AGENT_PROMPTS,
}

def _format_task_aware(template: str, task_description: str) -> str:
    """Safely format only the {task_description} placeholder without interpreting other braces.

    We temporarily replace the {task_description} token, escape all other braces, then
    restore the placeholder and format. This prevents KeyError on JSON braces.
    """
    if "{task_description}" not in template:
        return template
    token = "___TASK_DESC_PLACEHOLDER___"
    temp = template.replace("{task_description}", token)
    temp = temp.replace("{", "{{").replace("}", "}}")
    temp = temp.replace(token, "{task_description}")
    return temp.format(task_description=task_description)

def get_default_system_prompt(method: str) -> str:
    """Return the fully formatted default prompt for the given method."""
    if method not in ("single_model", "side_by_side"):
        raise ValueError(f"Unknown method: {method}. Supported methods: 'side_by_side', 'single_model'")
    entry = PROMPTS["default"][method]
    template = entry["template"]
    default_desc = entry["default_task_description"]
    return _format_task_aware(template, default_desc)


def get_system_prompt(method: str, system_prompt: str | None = None, task_description: str | None = None) -> str:
    """Resolve and return the final system prompt string.

    Supported values for system_prompt: None, "default", "agent", a prompt name (e.g., "agent_system_prompt"),
    or a literal prompt string.
    """
    if method not in ("single_model", "side_by_side"):
        raise ValueError(f"Unknown method: {method}. Supported methods: 'side_by_side', 'single_model'")

    # No explicit prompt → use default alias
    if system_prompt is None:
        entry = PROMPTS["default"][method]
        template = entry["template"]
        default_desc = entry["default_task_description"]
        desc = task_description if task_description is not None else default_desc
        return _format_task_aware(template, desc)

    # Alias: "default" or "agent"
    if system_prompt in PROMPTS:
        entry = PROMPTS[system_prompt][method]
        template = entry["template"]
        default_desc = entry["default_task_description"]
        desc = task_description if task_description is not None else default_desc
        return _format_task_aware(template, desc)

    # Try to resolve as a prompt name from the prompts module
    # This allows names like "agent_system_prompt" to be resolved
    import sys
    current_module = sys.modules[__name__]
    if hasattr(current_module, system_prompt):
        template = getattr(current_module, system_prompt)
        # If the template has {task_description}, format it
        if isinstance(template, str) and "{task_description}" in template:
            default_desc = PROMPTS["default"][method]["default_task_description"]
            desc = task_description if task_description is not None else default_desc
            return _format_task_aware(template, desc)
        # Otherwise return as-is (no task description support)
        if isinstance(template, str):
            if task_description is not None:
                # Warn that task_description was provided but won't be used
                import warnings
                warnings.warn(
                    f"task_description was provided but prompt '{system_prompt}' does not support it. "
                    "The task_description will be ignored."
                )
            return template

    # Literal string
    template = system_prompt
    if "{task_description}" in template:
        default_desc = PROMPTS["default"][method]["default_task_description"]
        desc = task_description if task_description is not None else default_desc
        return _format_task_aware(template, desc)
    if task_description is not None:
        raise ValueError(
            "A task_description was provided, but the given system_prompt string does not "
            "contain {task_description}. Please include the placeholder or use an alias ('default'|'agent')."
        )
    return template


__all__ = [
    "get_default_system_prompt",
    "get_system_prompt",
    "PROMPTS",
    # Supported prompt templates (limited set)
    "single_model_system_prompt_custom",
    "single_model_system_prompt_custom_revised",
    "sbs_system_prompt_custom",
    "agent_system_prompt_custom",
    "agent_sbs_system_prompt_custom",
    "agent_system_prompt_custom_revised",
    # Fixed-axis prompts
    "fixed_axis_prompt",
]