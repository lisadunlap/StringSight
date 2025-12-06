"""
Base class for prompt expansion.

This module provides the abstract interface for different prompt expansion methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class PromptExpander(ABC):
    """Abstract base class for prompt expansion methods.
    
    This class defines the interface for expanding task descriptions using
    different techniques (e.g., trace-based, few-shot, retrieval-based).
    """
    
    @abstractmethod
    def expand(
        self,
        task_description: str,
        traces: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Expand a task description using provided traces.
        
        Args:
            task_description: The original task description to expand.
            traces: List of trace dictionaries containing conversation data.
            **kwargs: Additional parameters specific to the expansion method.
        
        Returns:
            Expanded task description string.
        """
        pass





















