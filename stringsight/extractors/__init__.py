"""
Property extraction stages for StringSight.

This module contains stages that extract behavioral properties from model responses.
"""

from typing import Callable, Optional
from ..core.stage import PipelineStage


def get_extractor(
    model_name: str = "gpt-4.1-mini",
    system_prompt: str = "one_sided_system_prompt",
    prompt_builder: Optional[Callable] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = 64,
    include_scores_in_prompt: bool = False,
    **kwargs
) -> PipelineStage:
    """
    Factory function to get the appropriate extractor based on model name.
    
    Args:
        model_name: Name of the LLM to use for extraction
        system_prompt: System prompt for property extraction
        prompt_builder: Optional custom prompt builder function
        temperature: Temperature for LLM
        top_p: Top-p for LLM  
        max_tokens: Max tokens for LLM
        max_workers: Max parallel workers for API calls
        **kwargs: Additional configuration
        
    Returns:
        Configured extractor stage
    """
    
    # Route common hosted providers through the LiteLLM-backed extractor.
    lower_name = model_name.lower().strip()
    litellm_prefixes = (
        "gpt",           # OpenAI (bare)
        "openai/",      # OpenAI (provider-prefixed)
        "claude",       # Anthropic (bare)
        "anthropic/",   # Anthropic (provider-prefixed)
        "gemini",       # Google Gemini (bare)
        "google/",      # Google (provider-prefixed)
        "vertex",       # Vertex AI (provider-prefixed is usually vertex/..., allow bare prefix)
        "azure/",       # Azure OpenAI
        "cohere/",      # Cohere
        "mistral/",     # Mistral hosted
        "bedrock/",     # AWS Bedrock
    )

    # All models go through OpenAIExtractor which uses LiteLLM
    # LiteLLM handles routing to OpenAI, Anthropic, vLLM, etc.
    from .openai import OpenAIExtractor
    return OpenAIExtractor(
        model=model_name,
        system_prompt=system_prompt,
        prompt_builder=prompt_builder,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        include_scores_in_prompt=include_scores_in_prompt,
        **kwargs
    )


# Re-export key classes
from .openai import OpenAIExtractor
from .batch import BatchExtractor

__all__ = [
    "get_extractor",
    "OpenAIExtractor", 
    "BatchExtractor"
] 