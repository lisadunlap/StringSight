"""
Prompts module for StringSight.

This module contains system prompts and prompt utilities for property extraction.
"""

from .task_descriptions import (
    sbs_default_task_description,
    single_model_default_task_description,
    agent_system_prompt_custom_task_description,
    agent_sbs_system_prompt_custom_task_description,
)

# Import clustering prompts
from .clustering.prompts import (
    clustering_systems_prompt,
    deduplication_clustering_systems_prompt,
    outlier_clustering_systems_prompt,
    coarse_clustering_systems_prompt,
)

# Import fixed-axis prompts
from .fixed_axes import (
    fixed_axis_prompt,
)

# Import universal prompt system
from .extraction.universal import (
    format_universal_prompt,
    single_model_config,
    sbs_config,
    agent_single_model_config,
    agent_sbs_config,
    get_single_model_prompt,
    get_sbs_prompt,
    get_agent_single_model_prompt,
    get_agent_sbs_prompt,
)

# ------------------------------------------------------------------
# Prompt dictionaries (aliases)
# ------------------------------------------------------------------

DEFAULT_PROMPTS = {
    "single_model": {
        "config": single_model_config,
        "default_task_description": single_model_default_task_description,
    },
    "side_by_side": {
        "config": sbs_config,
        "default_task_description": sbs_default_task_description,
    },
}

AGENT_PROMPTS = {
    "single_model": {
        "config": agent_single_model_config,
        "default_task_description": agent_system_prompt_custom_task_description,
    },
    "side_by_side": {
        "config": agent_sbs_config,
        "default_task_description": agent_sbs_system_prompt_custom_task_description,
    },
}

# Universal prompt configurations
UNIVERSAL_PROMPTS = {
    "single_model": {
        "config": single_model_config,
        "default_task_description": single_model_default_task_description,
    },
    "side_by_side": {
        "config": sbs_config,
        "default_task_description": sbs_default_task_description,
    },
}

AGENT_UNIVERSAL_PROMPTS = {
    "single_model": {
        "config": agent_single_model_config,
        "default_task_description": agent_system_prompt_custom_task_description,
    },
    "side_by_side": {
        "config": agent_sbs_config,
        "default_task_description": agent_sbs_system_prompt_custom_task_description,
    },
}

PROMPTS = {
    "default": DEFAULT_PROMPTS,
    "agent": AGENT_PROMPTS,
    "universal": UNIVERSAL_PROMPTS,
    "agent_universal": AGENT_UNIVERSAL_PROMPTS,
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
    default_desc = entry["default_task_description"]
    
    # Handle config-based prompts (universal)
    if "config" in entry:
        return format_universal_prompt(default_desc, entry["config"])
    
    # Handle template-based prompts (legacy)
    template = entry["template"]
    return _format_task_aware(template, default_desc)


def get_system_prompt(method: str, system_prompt: str | None = None, task_description: str | None = None) -> str:
    """Resolve and return the final system prompt string.

    Supported values for system_prompt: None, "default", "agent", "universal", "agent_universal",
    a prompt name (e.g., "agent_system_prompt"), or a literal prompt string.
    
    When using "universal" or "agent_universal", the universal prompt template is used with
    the appropriate configuration dictionary.
    """
    if method not in ("single_model", "side_by_side"):
        raise ValueError(f"Unknown method: {method}. Supported methods: 'side_by_side', 'single_model'")

    try:
        # No explicit prompt → use default alias
        if system_prompt is None:
            entry = PROMPTS["default"][method]
            default_desc = entry.get("default_task_description")
            if default_desc is None:
                raise ValueError(f"No default task description found for method '{method}'")
            desc = task_description if task_description is not None else default_desc
            
            # Handle config-based prompts (universal)
            if "config" in entry:
                config = entry.get("config")
                if config is None:
                    raise ValueError(f"No config found for default prompt with method '{method}'")
                result = format_universal_prompt(desc, config)
                if result is None:
                    raise ValueError(f"format_universal_prompt returned None for method '{method}'")
                return result
            
            # Handle template-based prompts (legacy)
            template = entry.get("template")
            if template is None:
                raise ValueError(f"No template found for default prompt with method '{method}'")
            return _format_task_aware(template, desc)

        # Alias: "default", "agent", "universal", or "agent_universal"
        if system_prompt in PROMPTS:
            entry = PROMPTS[system_prompt][method]
            default_desc = entry.get("default_task_description")
            if default_desc is None:
                raise ValueError(f"No default task description found for prompt '{system_prompt}' with method '{method}'")
            desc = task_description if task_description is not None else default_desc
            
            # Handle config-based prompts (universal)
            if "config" in entry:
                config = entry.get("config")
                if config is None:
                    raise ValueError(f"No config found for prompt '{system_prompt}' with method '{method}'")
                result = format_universal_prompt(desc, config)
                if result is None:
                    raise ValueError(f"format_universal_prompt returned None for prompt '{system_prompt}' with method '{method}'")
                return result
            
            # Handle template-based prompts (legacy)
            template = entry.get("template")
            if template is None:
                raise ValueError(f"No template found for prompt '{system_prompt}' with method '{method}'")
            return _format_task_aware(template, desc)

        # Try to resolve as a prompt name from the prompts module
        # This allows names like "agent_system_prompt" to be resolved
        import sys
        current_module = sys.modules[__name__]
        if hasattr(current_module, system_prompt):
            template = getattr(current_module, system_prompt)
            # If the template has {task_description}, format it
            if isinstance(template, str) and "{task_description}" in template:
                default_desc = PROMPTS["default"][method].get("default_task_description")
                if default_desc is None:
                    raise ValueError(f"No default task description found for method '{method}'")
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
            default_desc = PROMPTS["default"][method].get("default_task_description")
            if default_desc is None:
                raise ValueError(f"No default task description found for method '{method}'")
            desc = task_description if task_description is not None else default_desc
            return _format_task_aware(template, desc)
        if task_description is not None:
            raise ValueError(
                "A task_description was provided, but the given system_prompt string does not "
                "contain {task_description}. Please include the placeholder or use an alias ('default'|'agent')."
            )
        return template
    except Exception as e:
        # Add more context to any error
        raise ValueError(
            f"Failed to generate system prompt for method='{method}', "
            f"system_prompt={system_prompt!r}, task_description={task_description!r}. "
            f"Error: {str(e)}"
        ) from e


__all__ = [
    "get_default_system_prompt",
    "get_system_prompt",
    "PROMPTS",
    # Universal prompt system
    "format_universal_prompt",
    "get_single_model_prompt",
    "get_sbs_prompt",
    "get_agent_single_model_prompt",
    "get_agent_sbs_prompt",
    "single_model_config",
    "sbs_config",
    "agent_single_model_config",
    "agent_sbs_config",
    # Fixed-axis prompts
    "fixed_axis_prompt",
    # Clustering prompts
    "clustering_systems_prompt",
    "deduplication_clustering_systems_prompt",
    "outlier_clustering_systems_prompt",
    "coarse_clustering_systems_prompt",
]
