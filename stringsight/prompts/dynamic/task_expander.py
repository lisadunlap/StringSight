"""
Task description expansion using sampled conversations.

This module expands user-provided task descriptions by analyzing a sample
of conversations from the dataset.
"""

import random
import tiktoken
from typing import List, Dict, Any, cast
from ...core.data_objects import ConversationRecord
from ...core.caching import UnifiedCache, CacheKeyBuilder
from ..expansion.trace_based import expand_task_description


class TaskExpander:
    """Expands task descriptions using sampled conversation examples."""

    def __init__(self, cache: UnifiedCache | None = None, max_tokens_per_sample: int = 512, seed: int = 42):
        """Initialize the TaskExpander.

        Args:
            cache: Cache instance for storing expansion results.
            max_tokens_per_sample: Maximum tokens per conversation sample (default: 512).
            seed: Random seed for deterministic sampling (default: 42).
        """
        self.cache = cache or UnifiedCache()
        self.max_tokens = max_tokens_per_sample
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.seed = seed

    def expand(
        self,
        task_description: str,
        conversations: List[ConversationRecord],
        num_samples: int = 5,
        model: str = "gpt-4.1"
    ) -> str:
        """Expand task description using sampled conversations.

        Args:
            task_description: Original task description.
            conversations: List of all conversations in dataset.
            num_samples: Number of conversations to sample (default: 5).
            model: LLM model for expansion (default: "gpt-4.1").

        Returns:
            Expanded task description string.
        """
        # Set seed for deterministic sampling
        random.seed(self.seed)

        # Sample conversations once
        sampled = self._sample_conversations(conversations, num_samples)

        # Build cache key from sampled conversations
        sample_ids = sorted([conv.question_id for conv in sampled])
        cache_key = self._build_cache_key(task_description, sample_ids, model)

        # Check cache
        cached = self.cache.get_completion(cache_key)
        if cached is not None:
            return cast(str, cached["expanded_task_description"])

        # Convert to trace format and truncate
        traces = []
        for conv in sampled:
            trace = self._conversation_to_trace(conv)
            truncated = self._truncate_trace(trace)
            traces.append(truncated)

        # Call existing expansion logic
        expanded = expand_task_description(
            task_description=task_description,
            traces=traces,
            model=model,
            num_traces=len(traces),
            seed=self.seed
        )

        # Cache result
        self.cache.set_completion(cache_key, {"expanded_task_description": expanded})
        return expanded

    def _sample_conversations(
        self,
        conversations: List[ConversationRecord],
        num_samples: int
    ) -> List[ConversationRecord]:
        """Randomly sample conversations."""
        if len(conversations) <= num_samples:
            return conversations
        return random.sample(conversations, num_samples)

    def _conversation_to_trace(self, conv: ConversationRecord) -> Dict[str, Any]:
        """Convert ConversationRecord to trace dict format.

        Args:
            conv: ConversationRecord object.

        Returns:
            Trace dictionary compatible with expand_task_description().
        """
        # Determine if single_model or side_by_side
        if isinstance(conv.model, list):
            # Side-by-side format
            return {
                "question_id": conv.question_id,
                "prompt": conv.prompt,
                "model_a": conv.model[0] if len(conv.model) > 0 else "Model A",
                "model_b": conv.model[1] if len(conv.model) > 1 else "Model B",
                "messages_a": conv.responses[0] if isinstance(conv.responses, list) and len(conv.responses) > 0 else [],
                "messages_b": conv.responses[1] if isinstance(conv.responses, list) and len(conv.responses) > 1 else [],
            }
        else:
            # Single model format
            return {
                "question_id": conv.question_id,
                "prompt": conv.prompt,
                "model": conv.model,
                "messages": conv.responses,
            }

    def _truncate_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate trace to max_tokens.

        Args:
            trace: Trace dictionary.

        Returns:
            Truncated trace dictionary.
        """
        # Keep prompt intact, truncate responses
        truncated = trace.copy()

        if "messages" in trace:
            # Single model
            truncated["messages"] = self._truncate_messages(trace["messages"])
        elif "messages_a" in trace and "messages_b" in trace:
            # Side-by-side: allocate half tokens to each
            half_tokens = self.max_tokens // 2
            truncated["messages_a"] = self._truncate_messages(
                trace["messages_a"], max_tokens=half_tokens
            )
            truncated["messages_b"] = self._truncate_messages(
                trace["messages_b"], max_tokens=half_tokens
            )

        return truncated

    def _truncate_messages(
        self,
        messages: Any,
        max_tokens: int | None = None
    ) -> Any:
        """Truncate messages to fit within token limit.

        Args:
            messages: Messages in various formats (string, list of dicts, etc.).
            max_tokens: Maximum tokens (default: self.max_tokens).

        Returns:
            Truncated messages in the same format as input.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        if isinstance(messages, str):
            # Simple string
            tokens = self.tokenizer.encode(messages)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                return self.tokenizer.decode(truncated_tokens) + "...[truncated]"
            return messages

        if isinstance(messages, list):
            # OpenAI message format (list of dicts with role/content)
            total_tokens = 0
            truncated_messages = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                content = msg.get("content", "")
                if not isinstance(content, str):
                    # Handle non-string content (tool calls, etc.)
                    truncated_messages.append(msg)
                    continue

                msg_tokens = self.tokenizer.encode(content)
                remaining = max_tokens - total_tokens

                if remaining <= 0:
                    # No more room
                    break

                if len(msg_tokens) <= remaining:
                    # Message fits entirely
                    truncated_messages.append(msg)
                    total_tokens += len(msg_tokens)
                else:
                    # Truncate this message and stop
                    truncated_tokens = msg_tokens[:remaining]
                    truncated_content = self.tokenizer.decode(truncated_tokens) + "...[truncated]"
                    truncated_msg = msg.copy()
                    truncated_msg["content"] = truncated_content
                    truncated_messages.append(truncated_msg)
                    break

            return truncated_messages

        # For other types, return as-is
        return messages

    def _build_cache_key(
        self,
        task_description: str,
        sample_ids: List[str],
        model: str
    ) -> CacheKeyBuilder:
        """Build cache key for task expansion.

        Args:
            task_description: Original task description.
            sample_ids: Sorted list of sampled conversation IDs.
            model: LLM model used for expansion.

        Returns:
            CacheKeyBuilder for use with UnifiedCache.
        """
        cache_data = {
            "type": "task_expansion",
            "task_description": task_description,
            "sample_ids": sample_ids,  # Already sorted
            "model": model,
            "max_tokens_per_sample": self.max_tokens,
            "version": "1.0",
        }
        return CacheKeyBuilder(cache_data)
