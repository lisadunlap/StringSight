"""
Trace-based prompt expansion.

This module implements expansion of task descriptions using example traces.
"""

import random
from typing import List, Dict, Any, Optional
import litellm

from .base import PromptExpander
from ...extractors.conv_to_str import conv_to_str
from ..prompt.task_description import single_model_default_task_description


EXPANSION_PROMPT_TEMPLATE = """Below is the given task description and examples of traces of agents solving the task. 
**Original Task Description:**
{task_description}

**Example Traces:**
Below are {num_traces} example traces from the dataset. Each trace shows a conversation between a user and a model.

{traces_text}

**Your Task:**
Based on the original task description and the example traces above, generate a comprehensive and specific list of behaviors to look for when analyzing model responses for this task. 

The expanded description should:
1. Include specific, concrete behaviors types that are relevant to the task
2. Cover a wide range of potential behaviors types (positive, negative, stylistic)
3. Be specific enough that an analyst could identify these behaviors in actual traces
4. Build upon the original task description rather than replacing it

For reference, here is a generic task description and a list of behavior types for a general purpose chatbot (your prompt should expand upon this list and include behaviors specific to the task):
{reference_task_description}

**Output Format:**
Provide an expanded task description that includes:
- The original task context
- A comprehensive list of specific behaviors to look for, organized by category if helpful
- Examples of what each behavior might look like in practice

Return only the expanded task description text, without any additional formatting or explanations."""


class TraceBasedExpander(PromptExpander):
    """Trace-based prompt expander.
    
    Expands task descriptions by analyzing example traces and generating
    a comprehensive list of specific behaviors to look for.
    """
    
    def __init__(
        self,
        model: str = "gpt-4.1",
        num_traces: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """Initialize the trace-based expander.
        
        Args:
            model: LLM model to use for expansion.
            num_traces: Number of traces to sample for expansion (default: 5).
            temperature: Temperature for LLM generation.
            max_tokens: Maximum tokens for expansion response.
        """
        self.model = model
        self.num_traces = num_traces
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _format_trace(self, trace: Dict[str, Any]) -> str:
        """Format a single trace into a readable string.
        
        Args:
            trace: Trace dictionary with 'prompt' and optionally 'messages', 'messages_a', 'messages_b', or 'model_response'.
        
        Returns:
            Formatted trace string.
        """
        prompt = trace.get("prompt", "")
        
        # Try to get response from messages or model_response
        response_text = ""
        if "messages" in trace:
            # Single model format - use conv_to_str for proper formatting
            messages = trace["messages"]
            if messages:
                response_text = conv_to_str(messages)
        elif "messages_a" in trace and "messages_b" in trace:
            # Side-by-side format - use conv_to_str for both responses
            messages_a = trace["messages_a"]
            messages_b = trace["messages_b"]
            response_a = conv_to_str(messages_a) if messages_a else ""
            response_b = conv_to_str(messages_b) if messages_b else ""
            if response_a or response_b:
                model_a = trace.get("model_a", "Model A")
                model_b = trace.get("model_b", "Model B")
                response_text = f"<beginning of {model_a} trace>\n{response_a}\n<end of {model_a} trace>\n\n--------------------------------\n\n<beginning of {model_b} trace>\n{response_b}\n<end of {model_b} trace>"
        elif "model_response" in trace:
            response = trace["model_response"]
            if isinstance(response, str):
                response_text = response
            elif isinstance(response, list):
                # Use conv_to_str for conversation format
                response_text = conv_to_str(response)
            else:
                # Fallback for other types
                response_text = str(response)
        
        formatted = f"**User Prompt:** {prompt}\n"
        if response_text:
            # Truncate very long responses to keep prompt manageable
            if len(response_text) > 2000:
                response_text = response_text[:2000] + "...\n[truncated]"
            formatted += f"**Model Response(s):**\n{response_text}\n"
        formatted += "\n---\n"
        return formatted
    
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
            **kwargs: Additional parameters (model, num_traces, etc. can override defaults).
        
        Returns:
            Expanded task description string.
        """
        # Override defaults with kwargs if provided
        model = kwargs.get("model", self.model)
        num_traces = kwargs.get("num_traces", self.num_traces)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Sample traces
        if len(traces) > num_traces:
            sampled_traces = random.sample(traces, num_traces)
        else:
            sampled_traces = traces
        
        # Format traces
        traces_text = "\n".join([self._format_trace(trace) for trace in sampled_traces])
        
        # Build prompt
        prompt = EXPANSION_PROMPT_TEMPLATE.format(
            task_description=task_description,
            num_traces=len(sampled_traces),
            traces_text=traces_text,
            reference_task_description=single_model_default_task_description,
        )
        
        # Call LLM
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing AI model behavior and creating comprehensive task descriptions. Given a task description and example traces, generate a comprehensive and specific list of behaviors to look for when analyzing model responses for this task. Think about both general categories to look for (e.g. instances of reward hacking, failure to follow instructions, etc) as well as behaviors specific to that task (e.g. which proof styles are used in a math task, choice of character design in a creative writing task, etc). Focus on categories of behaviors that are actionable and can be used to improve the model's performance or user experience or if a user could use the bheavior information from the trace to choose this model over others."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        expanded_description = response.choices[0].message.content.strip()
        return expanded_description


def expand_task_description(
    task_description: str,
    traces: List[Dict[str, Any]],
    model: str = "gpt-4.1",
    num_traces: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """Convenience function to expand a task description using traces.
    
    Args:
        task_description: The original task description to expand.
        traces: List of trace dictionaries containing conversation data.
        model: LLM model to use for expansion (default: "gpt-4.1").
        num_traces: Number of traces to sample for expansion (default: 5).
        temperature: Temperature for LLM generation (default: 0.7).
        max_tokens: Maximum tokens for expansion response (default: 2000).
    
    Returns:
        Expanded task description string.
    """
    expander = TraceBasedExpander(
        model=model,
        num_traces=num_traces,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return expander.expand(task_description, traces)

