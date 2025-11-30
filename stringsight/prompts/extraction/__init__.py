"""
Extraction prompts module.

Contains prompts for property extraction from model responses.
"""

from .standard import (
    single_model_system_prompt_custom_revised,
    sbs_system_prompt_custom_revised,
)

from .agent import (
    agent_system_prompt_custom_revised,
    agent_sbs_system_prompt_custom_revised,
)

__all__ = [
    # Standard prompts (revised only)
    "single_model_system_prompt_custom_revised",
    "sbs_system_prompt_custom_revised",
    # Agent prompts (revised only)
    "agent_system_prompt_custom_revised",
    "agent_sbs_system_prompt_custom_revised",
]

