"""
Validation utilities for StringSight.

This module provides validation functions for configuration and environment setup.
"""

import os
from typing import List, Set


def _is_gpt_model(model_name: str) -> bool:
    """Check if a model name corresponds to a GPT model that requires OpenAI API key."""
    if not model_name:
        return False
    return model_name.lower().startswith("gpt") or model_name.lower().startswith("text-embedding")


def _collect_gpt_models(**kwargs) -> Set[str]:
    """Collect all GPT model names from function arguments."""
    gpt_models = set()
    
    # Check all potential model parameters
    model_params = [
        'model_name', 'embedding_model', 'summary_model', 
        'cluster_assignment_model', 'model'
    ]
    
    for param in model_params:
        if param in kwargs:
            model_value = kwargs[param]
            if isinstance(model_value, str) and _is_gpt_model(model_value):
                gpt_models.add(model_value)
    
    return gpt_models


def validate_openai_api_key(**kwargs) -> None:
    """
    Validate that OPENAI_API_KEY is set when GPT models are configured.
    
    This function checks if any of the model parameters are GPT models and validates
    that the OPENAI_API_KEY environment variable is set.
    
    Args:
        **kwargs: Function parameters that may contain model names
        
    Raises:
        ValueError: If GPT models are configured but OPENAI_API_KEY is not set
    """
    gpt_models = _collect_gpt_models(**kwargs)
    
    if gpt_models:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key or not openai_api_key.strip():
            model_list = ", ".join(sorted(gpt_models))
            raise ValueError(
                f"OpenAI API key is required when using GPT models: {model_list}\n\n"
                f"Please set your OpenAI API key:\n"
                f"  export OPENAI_API_KEY='your-api-key-here'\n\n"
                f"Or create a .env file in your project root:\n"
                f"  echo 'OPENAI_API_KEY=your-api-key-here' > .env"
            )
