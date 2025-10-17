"""
Cost calculation and estimation for StringSight pipeline operations.

This module provides functions to estimate costs for extraction and clustering
operations based on model usage and token counts.
"""

import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict

from .pricing import get_model_pricing, estimate_tokens_cost


@dataclass
class CostEstimate:
    """Cost estimation breakdown for a pipeline operation."""
    
    # Core estimates
    extraction_cost: float = 0.0
    clustering_embedding_cost: float = 0.0
    clustering_llm_cost: float = 0.0
    total_cost: float = 0.0
    
    # Token estimates
    extraction_input_tokens: int = 0
    extraction_output_tokens: int = 0
    embedding_tokens: int = 0
    clustering_llm_input_tokens: int = 0
    clustering_llm_output_tokens: int = 0
    
    # Model information
    extraction_model: str = ""
    embedding_model: str = ""
    summary_model: str = ""
    cluster_assignment_model: str = ""
    
    # Configuration
    num_conversations: int = 0
    estimated_properties: int = 0
    estimated_clusters: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CostCalculator:
    """Calculator for estimating API costs across the StringSight pipeline."""
    
    def __init__(self):
        # Average token counts based on typical usage patterns
        self.avg_conversation_tokens = 500  # Average tokens per conversation
        self.avg_system_prompt_tokens = 200  # System prompt overhead
        self.avg_property_description_tokens = 50  # Property description length
        self.avg_cluster_summary_tokens = 100  # Generated cluster summary length
        self.avg_cluster_assignment_tokens = 150  # Cluster assignment context
        
        # Estimation factors
        self.properties_per_conversation = 2.5  # Avg properties extracted per conversation
        self.clusters_per_100_properties = 15  # Avg clusters per 100 properties
        self.completion_rate = 0.8  # Rate of successful completions
        
    def estimate_extraction_cost(
        self,
        num_conversations: int,
        model_name: str = "gpt-4.1",
        max_tokens: int = 16000,
        avg_conversation_tokens: Optional[int] = None,
        system_prompt_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate cost for property extraction stage.
        
        Args:
            num_conversations: Number of conversations to process
            model_name: LLM model for extraction  
            max_tokens: Maximum tokens per response
            avg_conversation_tokens: Override default conversation length
            system_prompt_tokens: Override default system prompt length
            
        Returns:
            Dictionary with cost breakdown
        """
        pricing = get_model_pricing(model_name)
        if not pricing:
            return {"error": f"Pricing not found for model: {model_name}", "total_cost": 0.0}
        
        # Use provided values or defaults
        conv_tokens = avg_conversation_tokens or self.avg_conversation_tokens
        prompt_tokens = system_prompt_tokens or self.avg_system_prompt_tokens
        
        # Calculate token usage
        input_tokens_per_call = conv_tokens + prompt_tokens
        total_input_tokens = input_tokens_per_call * num_conversations
        
        # Estimate output tokens (conservative estimate based on max_tokens and completion rate)
        estimated_output_tokens_per_call = min(max_tokens * self.completion_rate, max_tokens)
        total_output_tokens = int(estimated_output_tokens_per_call * num_conversations)
        
        # Calculate costs
        total_cost = estimate_tokens_cost(total_input_tokens, total_output_tokens, model_name)
        
        return {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "input_cost": (total_input_tokens / 1_000_000) * pricing.input_price_per_1m_tokens,
            "output_cost": (total_output_tokens / 1_000_000) * pricing.output_price_per_1m_tokens,
            "total_cost": total_cost,
            "model": model_name
        }
    
    def estimate_clustering_cost(
        self,
        num_properties: int,
        embedding_model: str = "openai",
        summary_model: str = "gpt-4.1", 
        avg_property_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate cost for clustering stage.
        
        Args:
            num_properties: Number of properties to cluster
            embedding_model: Embedding model ("openai" or local model name)
            summary_model: LLM for generating cluster summaries
            avg_property_tokens: Override default property description length
            
        Returns:
            Dictionary with cost breakdown
        """
        prop_tokens = avg_property_tokens or self.avg_property_description_tokens
        
        # Estimate number of clusters
        estimated_clusters = max(1, int(num_properties * self.clusters_per_100_properties / 100))
        
        costs = {
            "embedding_cost": 0.0,
            "summary_cost": 0.0,
            "total_cost": 0.0,
            "estimated_clusters": estimated_clusters,
            "embedding_tokens": 0,
            "summary_input_tokens": 0,
            "summary_output_tokens": 0
        }
        
        # 1. Embedding costs (only for OpenAI embeddings)
        if embedding_model == "openai":
            embedding_tokens = num_properties * prop_tokens
            embedding_cost = estimate_tokens_cost(embedding_tokens, 0, "text-embedding-3-large")
            costs["embedding_cost"] = embedding_cost or 0.0
            costs["embedding_tokens"] = embedding_tokens
        
        # 2. Cluster summary generation costs
        summary_pricing = get_model_pricing(summary_model)
        if summary_pricing:
            # Each cluster needs a summary based on its properties
            summary_input_tokens = estimated_clusters * prop_tokens * 5  # Context from multiple properties
            summary_output_tokens = estimated_clusters * self.avg_cluster_summary_tokens
            
            summary_cost = estimate_tokens_cost(summary_input_tokens, summary_output_tokens, summary_model)
            costs["summary_cost"] = summary_cost or 0.0
            costs["summary_input_tokens"] = summary_input_tokens
            costs["summary_output_tokens"] = summary_output_tokens
        
        costs["total_cost"] = costs["embedding_cost"] + costs["summary_cost"]
        
        return costs
    
    def estimate_total_pipeline_cost(
        self,
        num_conversations: int,
        # Extraction parameters
        extraction_model: str = "gpt-4.1",
        max_tokens: int = 16000,
        # Clustering parameters  
        embedding_model: str = "openai",
        summary_model: str = "gpt-4.1",
        # Optional overrides
        avg_conversation_tokens: Optional[int] = None,
        properties_per_conversation: Optional[float] = None
    ) -> CostEstimate:
        """
        Estimate total cost for the entire pipeline.
        
        Args:
            num_conversations: Number of conversations to process
            extraction_model: Model for property extraction
            max_tokens: Max tokens for extraction
            embedding_model: Model for embeddings
            summary_model: Model for cluster summaries
            avg_conversation_tokens: Override conversation token count
            properties_per_conversation: Override properties per conversation ratio
            
        Returns:
            Comprehensive cost estimate
        """
        # Estimate number of properties
        prop_ratio = properties_per_conversation or self.properties_per_conversation
        estimated_properties = int(num_conversations * prop_ratio)
        
        # Get extraction costs
        extraction_costs = self.estimate_extraction_cost(
            num_conversations,
            extraction_model,
            max_tokens,
            avg_conversation_tokens
        )
        
        # Get clustering costs
        clustering_costs = self.estimate_clustering_cost(
            estimated_properties,
            embedding_model,
            summary_model
        )
        
        # Build comprehensive estimate
        estimate = CostEstimate(
            # Costs
            extraction_cost=extraction_costs.get("total_cost", 0.0),
            clustering_embedding_cost=clustering_costs.get("embedding_cost", 0.0),
            clustering_llm_cost=clustering_costs.get("summary_cost", 0.0),
            total_cost=extraction_costs.get("total_cost", 0.0) + clustering_costs.get("total_cost", 0.0),
            
            # Tokens
            extraction_input_tokens=extraction_costs.get("input_tokens", 0),
            extraction_output_tokens=extraction_costs.get("output_tokens", 0),
            embedding_tokens=clustering_costs.get("embedding_tokens", 0),
            clustering_llm_input_tokens=clustering_costs.get("summary_input_tokens", 0),
            clustering_llm_output_tokens=clustering_costs.get("summary_output_tokens", 0),
            
            # Models
            extraction_model=extraction_model,
            embedding_model=embedding_model,
            summary_model=summary_model,
            
            # Configuration
            num_conversations=num_conversations,
            estimated_properties=estimated_properties,
            estimated_clusters=clustering_costs.get("estimated_clusters", 0)
        )
        
        return estimate


# Convenience functions for quick estimates
def estimate_extraction_cost(num_conversations: int, model_name: str = "gpt-4.1", **kwargs) -> float:
    """Quick extraction cost estimate."""
    calculator = CostCalculator()
    result = calculator.estimate_extraction_cost(num_conversations, model_name, **kwargs)
    return result.get("total_cost", 0.0)


def estimate_clustering_cost(num_properties: int, embedding_model: str = "openai", **kwargs) -> float:
    """Quick clustering cost estimate."""
    calculator = CostCalculator()
    result = calculator.estimate_clustering_cost(num_properties, embedding_model, **kwargs)
    return result.get("total_cost", 0.0)


