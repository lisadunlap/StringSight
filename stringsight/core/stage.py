"""
Pipeline stage interface for StringSight.

All pipeline stages must implement the PipelineStage interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .data_objects import PropertyDataset


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Each stage takes a PropertyDataset as input and returns a PropertyDataset as output.
    This allows stages to be composed into pipelines.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the stage with configuration parameters and propagate to mixins."""
        # Store config before passing to mixins (copy to avoid mutating original)
        self.config = dict(kwargs)
        self.name = self.__class__.__name__
        
        # Call next __init__ in MRO – no kwargs so they don't reach object.__init__
        super().__init__()
    
    @abstractmethod
    def run(self, data: PropertyDataset, progress_callback=None) -> PropertyDataset:
        """
        Process the input data and return the modified data.
        
        Args:
            data: Input PropertyDataset
            progress_callback: Optional callback(completed, total) for progress updates
            
        Returns:
            Modified PropertyDataset
        """
        pass
    
    def validate_input(self, data: PropertyDataset) -> None:
        """
        Validate that the input data meets the requirements for this stage.
        
        Args:
            data: Input PropertyDataset
            
        Raises:
            ValueError: If the input data is invalid
        """
        if not isinstance(data, PropertyDataset):
            raise ValueError(f"Input must be a PropertyDataset, got {type(data)}")
    
    def validate_output(self, data: PropertyDataset) -> None:
        """
        Validate that the output data is valid.
        
        Args:
            data: Output PropertyDataset
            
        Raises:
            ValueError: If the output data is invalid
        """
        if not isinstance(data, PropertyDataset):
            raise ValueError(f"Output must be a PropertyDataset, got {type(data)}")
    
    async def __call__(self, data: PropertyDataset, progress_callback=None) -> PropertyDataset:
        """
        Convenience method to run the stage.
        
        This allows stages to be called directly: stage(data) or await stage(data)
        Handles both sync and async run() methods automatically.
        """
        import inspect
        self.validate_input(data)
        
        # Check if run() is a coroutine function (async)
        if inspect.iscoroutinefunction(self.run):
            # Check if run accepts progress_callback
            sig = inspect.signature(self.run)
            if 'progress_callback' in sig.parameters:
                result = await self.run(data, progress_callback=progress_callback)
            else:
                result = await self.run(data)
        else:
            # Check if run accepts progress_callback
            sig = inspect.signature(self.run)
            if 'progress_callback' in sig.parameters:
                result = self.run(data, progress_callback=progress_callback)
            else:
                result = self.run(data)
            
        self.validate_output(result)
        return result
    
    def __repr__(self) -> str:
        return f"{self.name}({self.config})"


class PassthroughStage(PipelineStage):
    """A stage that passes data through unchanged. Useful for testing."""
    
    def run(self, data: PropertyDataset) -> PropertyDataset:
        return data 