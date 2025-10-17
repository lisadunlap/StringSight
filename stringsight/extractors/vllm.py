"""
VLLM-based property extraction stage.
"""

from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset


class VLLMExtractor(PipelineStage):
    """Stub VLLM extractor."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        # TODO: Implement VLLM extraction
        return data 