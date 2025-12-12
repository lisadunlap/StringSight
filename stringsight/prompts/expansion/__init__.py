"""
Prompt expansion module.

Contains functionality for expanding task descriptions using example traces.

THIS IS NOT DONE YET, SORRY FOR THE INCONVENIENCE
"""

from .base import PromptExpander
from .trace_based import TraceBasedExpander, expand_task_description

__all__ = [
    "PromptExpander",
    "TraceBasedExpander",
    "expand_task_description",
]





















