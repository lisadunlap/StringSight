"""
Pricing data for different model providers.

This module contains up-to-date pricing information for OpenAI, Anthropic, 
and Google models that can be used with LiteLLM.
"""

from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a model."""
    input_price_per_1m_tokens: float
    output_price_per_1m_tokens: float
    context_window: int
    provider: str
    notes: str = ""


# Pricing data as of December 2024
# Sources: Official provider pricing pages
PRICING_TABLE: Dict[str, ModelPricing] = {
    # OpenAI Models
    "gpt-4": ModelPricing(
        input_price_per_1m_tokens=30.00,
        output_price_per_1m_tokens=60.00,
        context_window=128000,
        provider="openai",
        notes="High quality, slower"
    ),
    "gpt-4.1": ModelPricing(  # Assuming this maps to GPT-4 Turbo
        input_price_per_1m_tokens=10.00,
        output_price_per_1m_tokens=30.00,
        context_window=128000,
        provider="openai",
    ),
    "gpt-4.1": ModelPricing(
        input_price_per_1m_tokens=5.00,
        output_price_per_1m_tokens=15.00,
        context_window=128000,
        provider="openai",
        notes="Fast and high quality"
    ),
    "gpt-4.1-mini": ModelPricing(
        input_price_per_1m_tokens=0.60,
        output_price_per_1m_tokens=1.80,
        context_window=128000,
        provider="openai",
        notes="Most cost-effective"
    ),
    "gpt-3.5-turbo": ModelPricing(
        input_price_per_1m_tokens=0.50,
        output_price_per_1m_tokens=1.50,
        context_window=4000,
        provider="openai",
        notes="Legacy model"
    ),
    
    # OpenAI Embedding Models
    "text-embedding-3-large": ModelPricing(
        input_price_per_1m_tokens=0.13,
        output_price_per_1m_tokens=0.0,  # Embeddings don't have output costs
        context_window=8191,
        provider="openai",
        notes="High quality embeddings"
    ),
    "text-embedding-3-large": ModelPricing(
        input_price_per_1m_tokens=0.02,
        output_price_per_1m_tokens=0.0,
        context_window=8191,
        provider="openai",
        notes="Cost-effective embeddings"
    ),
    
    # Anthropic Models
    "claude-3-5-haiku": ModelPricing(
        input_price_per_1m_tokens=0.80,
        output_price_per_1m_tokens=4.00,
        context_window=200000,
        provider="anthropic",
        notes="Fast and cost-effective"
    ),
    "claude-3-5-sonnet": ModelPricing(
        input_price_per_1m_tokens=3.00,
        output_price_per_1m_tokens=15.00,
        context_window=200000,
        provider="anthropic",
        notes="Balanced performance"
    ),
    "claude-3-opus": ModelPricing(
        input_price_per_1m_tokens=15.00,
        output_price_per_1m_tokens=75.00,
        context_window=200000,
        provider="anthropic",
        notes="Highest quality"
    ),
    
    # Google Gemini Models
    "gemini-2.0-flash": ModelPricing(
        input_price_per_1m_tokens=0.10,
        output_price_per_1m_tokens=0.40,
        context_window=1000000,
        provider="google",
        notes="Very fast, large context"
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        input_price_per_1m_tokens=0.075,
        output_price_per_1m_tokens=0.30,
        context_window=1000000,
        provider="google",
        notes="Lightweight version"
    ),
    "gemini-1.5-flash-8b": ModelPricing(
        input_price_per_1m_tokens=0.0375,
        output_price_per_1m_tokens=0.15,
        context_window=1000000,
        provider="google",
        notes="Very cost-effective"
    ),
    "gemini-1.5-pro": ModelPricing(
        input_price_per_1m_tokens=1.25,
        output_price_per_1m_tokens=5.00,
        context_window=2000000,
        provider="google",
        notes="High quality, massive context"
    ),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """
    Get pricing information for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelPricing object if found, None otherwise
    """
    # Try exact match first
    if model_name in PRICING_TABLE:
        return PRICING_TABLE[model_name]
    
    # Try to normalize model name (handle common variations)
    normalized_name = model_name.lower().replace("_", "-")
    
    for key, pricing in PRICING_TABLE.items():
        if key.lower().replace("_", "-") == normalized_name:
            return pricing
    
    return None


def estimate_tokens_cost(
    input_tokens: int, 
    output_tokens: int, 
    model_name: str
) -> Optional[float]:
    """
    Estimate cost for a given number of tokens.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens  
        model_name: Name of the model
        
    Returns:
        Estimated cost in USD, None if model not found
    """
    pricing = get_model_pricing(model_name)
    if not pricing:
        return None
    
    input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m_tokens
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m_tokens
    
    return input_cost + output_cost


def get_supported_providers() -> Dict[str, list]:
    """
    Get all supported providers and their models.
    
    Returns:
        Dictionary mapping provider names to lists of model names
    """
    providers = {}
    for model_name, pricing in PRICING_TABLE.items():
        provider = pricing.provider
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model_name)
    
    return providers
