"""
Cost tracking for actual API usage during pipeline execution.

This module tracks real API costs and compares them with estimates.
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .pricing import estimate_tokens_cost
from ..storage.adapter import StorageAdapter, get_storage_adapter


@dataclass 
class APICall:
    """Record of a single API call and its cost."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    stage: str  # "extraction", "embedding", "clustering_summary", "clustering_assignment"
    call_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CostSummary:
    """Summary of costs by stage and model."""
    total_cost: float = 0.0
    extraction_cost: float = 0.0
    embedding_cost: float = 0.0
    clustering_cost: float = 0.0
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    calls_by_stage: Dict[str, int] = None
    costs_by_model: Dict[str, float] = None
    
    def __post_init__(self):
        if self.calls_by_stage is None:
            self.calls_by_stage = {}
        if self.costs_by_model is None:
            self.costs_by_model = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CostTracker:
    """Tracks API costs throughout pipeline execution."""
    
    def __init__(self, output_dir: Optional[str] = None, storage: Optional[StorageAdapter] = None):
        self.calls: List[APICall] = []
        self.output_dir = Path(output_dir) if output_dir else None
        self.storage = storage or get_storage_adapter()
        self.session_start = time.time()
        
    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        stage: str,
        call_id: str = ""
    ) -> float:
        """
        Record an API call and calculate its cost.
        
        Args:
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            stage: Pipeline stage ("extraction", "embedding", etc.)
            call_id: Optional identifier for the call
            
        Returns:
            Cost of the call in USD
        """
        cost = estimate_tokens_cost(input_tokens, output_tokens, model) or 0.0
        
        call = APICall(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            stage=stage,
            call_id=call_id
        )
        
        self.calls.append(call)
        return cost
    
    def record_extraction_call(
        self,
        model: str,
        input_tokens: int, 
        output_tokens: int,
        call_id: str = ""
    ) -> float:
        """Record a property extraction API call."""
        return self.record_call(model, input_tokens, output_tokens, "extraction", call_id)
    
    def record_embedding_call(
        self,
        model: str,
        input_tokens: int,
        call_id: str = ""
    ) -> float:
        """Record an embedding API call."""
        return self.record_call(model, input_tokens, 0, "embedding", call_id)
    
    def record_clustering_summary_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_id: str = ""
    ) -> float:
        """Record a cluster summary generation call."""
        return self.record_call(model, input_tokens, output_tokens, "clustering_summary", call_id)
    
    def record_clustering_assignment_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        call_id: str = ""
    ) -> float:
        """Record a cluster assignment call."""
        return self.record_call(model, input_tokens, output_tokens, "clustering_assignment", call_id)
    
    def get_summary(self) -> CostSummary:
        """Get a summary of all tracked costs."""
        summary = CostSummary()
        
        for call in self.calls:
            # Update totals
            summary.total_cost += call.cost
            summary.total_input_tokens += call.input_tokens
            summary.total_output_tokens += call.output_tokens
            
            # Update stage costs
            if call.stage == "extraction":
                summary.extraction_cost += call.cost
            elif call.stage == "embedding":
                summary.embedding_cost += call.cost
            elif call.stage in ["clustering_summary", "clustering_assignment"]:
                summary.clustering_cost += call.cost
            
            # Update calls by stage
            summary.calls_by_stage[call.stage] = summary.calls_by_stage.get(call.stage, 0) + 1
            
            # Update costs by model
            summary.costs_by_model[call.model] = summary.costs_by_model.get(call.model, 0.0) + call.cost
        
        return summary
    
    def get_costs_by_stage(self) -> Dict[str, float]:
        """Get total costs broken down by pipeline stage."""
        costs = {}
        for call in self.calls:
            costs[call.stage] = costs.get(call.stage, 0.0) + call.cost
        return costs
    
    def get_costs_by_model(self) -> Dict[str, float]:
        """Get total costs broken down by model."""
        costs = {}
        for call in self.calls:
            costs[call.model] = costs.get(call.model, 0.0) + call.cost
        return costs
    
    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return sum(call.cost for call in self.calls)
    
    def save_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save cost tracking data to a JSON file.
        
        Args:
            filename: Optional filename override
            
        Returns:
            Path to saved file
        """
        if not self.output_dir:
            raise ValueError("No output directory specified")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"cost_tracking_{int(self.session_start)}.json"
        
        filepath = str(self.output_dir / filename)

        data = {
            "session_start": self.session_start,
            "session_duration": time.time() - self.session_start,
            "summary": self.get_summary().to_dict(),
            "calls": [call.to_dict() for call in self.calls]
        }

        self.storage.write_json(filepath, data)

        return filepath

    def load_from_file(self, filepath: str) -> None:
        """Load cost tracking data from a JSON file."""
        data = self.storage.read_json(filepath)
        
        self.session_start = data.get("session_start", time.time())
        
        self.calls = []
        for call_data in data.get("calls", []):
            call = APICall(**call_data)
            self.calls.append(call)
    
    def compare_with_estimate(self, estimate_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare actual costs with initial estimates.
        
        Args:
            estimate_dict: Dictionary from CostEstimate.to_dict()
            
        Returns:
            Comparison showing actual vs estimated costs
        """
        actual = self.get_summary()
        
        comparison = {
            "estimated_total": estimate_dict.get("total_cost", 0.0),
            "actual_total": actual.total_cost,
            "difference": actual.total_cost - estimate_dict.get("total_cost", 0.0),
            "accuracy_pct": 0.0,
            
            "by_stage": {
                "extraction": {
                    "estimated": estimate_dict.get("extraction_cost", 0.0),
                    "actual": actual.extraction_cost,
                    "difference": actual.extraction_cost - estimate_dict.get("extraction_cost", 0.0)
                },
                "embedding": {
                    "estimated": estimate_dict.get("clustering_embedding_cost", 0.0), 
                    "actual": actual.embedding_cost,
                    "difference": actual.embedding_cost - estimate_dict.get("clustering_embedding_cost", 0.0)
                },
                "clustering": {
                    "estimated": estimate_dict.get("clustering_llm_cost", 0.0),
                    "actual": actual.clustering_cost,
                    "difference": actual.clustering_cost - estimate_dict.get("clustering_llm_cost", 0.0)
                }
            }
        }
        
        # Calculate accuracy percentage
        estimated_total = estimate_dict.get("total_cost", 0.0)
        if estimated_total > 0:
            comparison["accuracy_pct"] = (1 - abs(comparison["difference"]) / estimated_total) * 100
        
        return comparison
    
    def clear(self) -> None:
        """Clear all tracked calls."""
        self.calls = []
        self.session_start = time.time()
