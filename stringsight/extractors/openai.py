"""
OpenAI-based property extraction stage.

This stage migrates the logic from generate_differences.py into the pipeline architecture.
"""

from typing import Callable, Optional, List, Dict, Any, Union
import uuid
import json
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin
from .. import prompts as _extractor_prompts
from ..core.caching import UnifiedCache
from ..core.llm_utils import parallel_completions_async
from .conv_to_str import conv_to_str
from .inp_to_conv import openai_messages_to_conv


class OpenAIExtractor(LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin, PipelineStage):
    """
    Extract behavioral properties using OpenAI models.
    
    This stage takes conversations and extracts structured properties describing
    model behaviors, differences, and characteristics.
    """
    
    def __init__(
        self,
        model: str = "gpt-4.1",
        system_prompt: str = "one_sided_system_prompt_no_examples",
        prompt_builder: Optional[Callable] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 16000,
        max_workers: int = 64,
        include_scores_in_prompt: bool = False,
        **kwargs
    ):
        """
        Initialize the OpenAI extractor.

        Args:
            model: OpenAI model name (e.g., "gpt-4.1-mini")
            system_prompt: System prompt for property extraction
            prompt_builder: Optional custom prompt builder function
            temperature: Temperature for LLM
            top_p: Top-p for LLM
            max_tokens: Max tokens for LLM
            max_workers: Max parallel workers for API calls
            include_scores_in_prompt: Whether to include scores in prompts
            **kwargs: Additional configuration

        Note:
            Caching is handled automatically by UnifiedCache singleton.
            Configure cache via STRINGSIGHT_* environment variables.
        """
        super().__init__(**kwargs)
        self.model = model
        # Allow caller to pass the name of a prompt template or the prompt text itself
        if isinstance(system_prompt, str) and hasattr(_extractor_prompts, system_prompt):
            self.system_prompt = getattr(_extractor_prompts, system_prompt)
        else:
            self.system_prompt = system_prompt

        self.prompt_builder = prompt_builder or self._default_prompt_builder
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        # Control whether to include numeric scores/winner context in prompts
        self.include_scores_in_prompt = include_scores_in_prompt
        # Note: Caching is handled by parallel_completions via UnifiedCache singleton

    async def run(self, data: PropertyDataset, progress_callback=None) -> PropertyDataset:
        """Run OpenAI extraction for all conversations.

        Each conversation is formatted with ``prompt_builder`` and sent to the
        OpenAI model in parallel using async.  The raw LLM response is
        stored inside a *placeholder* ``Property`` object (one per
        conversation).  Down-stream stages (``LLMJsonParser``) will parse these
        raw strings into fully-formed properties.

        Args:
            data: PropertyDataset with conversations to extract from
            progress_callback: Optional callback(completed, total) for progress updates
        """

        n_conv = len(data.conversations)
        if n_conv == 0:
            self.log("No conversations found – skipping extraction")
            return data

        self.log(f"Extracting properties from {n_conv} conversations using {self.model}")


        # ------------------------------------------------------------------
        # 1️⃣  Build user messages for every conversation (in parallel)
        # ------------------------------------------------------------------
        user_messages: List[Union[str, List[Dict[str, Any]]]] = [""] * len(data.conversations)

        def _build_prompt(idx: int, conv):
            return idx, self.prompt_builder(conv)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_build_prompt, idx, conv): idx
                      for idx, conv in enumerate(data.conversations)}
            for future in as_completed(futures):
                idx, prompt = future.result()
                user_messages[idx] = prompt

        # ------------------------------------------------------------------
        # 2️⃣  Call the OpenAI API in parallel batches via shared async LLM utils
        # ------------------------------------------------------------------
        raw_responses = await parallel_completions_async(
            user_messages,
            model=self.model,
            system_prompt=self.system_prompt,
            max_workers=self.max_workers,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            show_progress=True,
            progress_desc="Property extraction",
            progress_callback=progress_callback
        )

        # ------------------------------------------------------------------
        # 3️⃣  Wrap raw responses in placeholder Property objects (filter None)
        # ------------------------------------------------------------------
        properties: List[Property] = []
        skipped_count = 0
        for conv, raw in zip(data.conversations, raw_responses):
            # Skip failed LLM calls (None responses)
            if raw is None:
                skipped_count += 1
                continue

            # We don't yet know which model(s) the individual properties will
            # belong to; parser will figure it out.  Use a placeholder model
            # name so that validation passes.
            prop = Property(
                id=str(uuid.uuid4()),
                question_id=conv.question_id,
                model=conv.model,
                raw_response=raw,
            )
            properties.append(prop)

        if skipped_count > 0:
            self.log(f"Skipped {skipped_count} conversations due to failed LLM calls", level="warning")

        self.log(f"Received {len(properties)} valid LLM responses")


        # Log to wandb if enabled
        if hasattr(self, 'use_wandb') and self.use_wandb:
            self._log_extraction_to_wandb(user_messages, raw_responses, data.conversations)

        # ------------------------------------------------------------------
        # 4️⃣  Return updated dataset
        # ------------------------------------------------------------------
        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=properties,
            clusters=data.clusters,
            model_stats=data.model_stats,
        )

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------

    # Legacy helpers removed in favor of centralized llm_utils
    
    def _default_prompt_builder(self, conversation) -> Union[str, List[Dict[str, Any]]]:
        """
        Default prompt builder for side-by-side comparisons, with multimodal support.
        
        Args:
            conversation: ConversationRecord
            
        Returns:
            - If no images present: a plain string prompt (backwards compatible)
            - If images present: a full OpenAI messages list including a single
              user turn with ordered text/image parts (and a system turn)
        """
        # Check if this is a side-by-side comparison or single model
        if isinstance(conversation.model, list) and len(conversation.model) == 2:
            # Side-by-side format
            model_a, model_b = conversation.model
            try:
                responses_a = conversation.responses[0]
                responses_b = conversation.responses[1]
            except Exception as e:
                raise ValueError(
                    f"Failed to access conversation responses for side-by-side format. "
                    f"Expected two response lists. Error: {str(e)}"
                )

            # Normalize both to our internal segments format
            conv_a = openai_messages_to_conv(responses_a) if isinstance(responses_a, list) else responses_a
            conv_b = openai_messages_to_conv(responses_b) if isinstance(responses_b, list) else responses_b

            has_images = self._conversation_has_images(conv_a) or self._conversation_has_images(conv_b)

            if has_images:
                return self._build_side_by_side_messages(model_a, model_b, conv_a, conv_b)

            # No images: keep string behavior for compatibility
            response_a = conv_to_str(responses_a)
            response_b = conv_to_str(responses_b)

            scores = conversation.scores

            # Handle list format [scores_a, scores_b]
            if isinstance(scores, list) and len(scores) == 2:
                scores_a, scores_b = scores[0], scores[1]
                winner = conversation.meta.get("winner")  # Winner stored in meta
                
                # Build the prompt with separate scores for each model
                prompt_parts = [
                    f"<beginning of Model A trace>\n {response_a}\n<end of Model A trace>\n\n--------------------------------\n\n"
                ]
                
                if self.include_scores_in_prompt and scores_a:
                    prompt_parts.append(f"<Quality scores on Model A trace>\n {scores_a}\n</Quality scores on Model A trace>\n\n")
                prompt_parts.append("--------------------------------")
                prompt_parts.append(f"<beginning of Model B trace>\n {response_b}\n<end of Model B trace>\n\n--------------------------------\n\n")

                if self.include_scores_in_prompt and scores_b:
                    prompt_parts.append(f"<Quality scores on Model B trace>\n {scores_b}\n</Quality scores on Model B trace>\n\n")
                
                if self.include_scores_in_prompt and winner:
                    prompt_parts.append(f"<Winner of side-by-side comparison>\n {winner}\n</Winner of side-by-side comparison>\n\n")
                
                return "\n\n".join(prompt_parts)
            else:
                # No scores available
                return (
                    f"<beginning of Model A trace>\n {response_a}\n<end of Model A trace>\n\n--------------------------------\n\n"
                    f"--------------------------------\n"
                    f"<beginning of Model B trace>\n {response_b}\n<end of Model B trace>\n\n--------------------------------\n\n"
                )
        elif isinstance(conversation.model, str):
            # Single model format
            model = conversation.model if isinstance(conversation.model, str) else str(conversation.model)
            responses = conversation.responses

            # Normalize to our internal segments format only to detect images
            conv_norm = openai_messages_to_conv(responses) if isinstance(responses, list) else responses
            if self._conversation_has_images(conv_norm):
                return self._build_single_user_messages(conv_norm)

            # No images: keep string behavior
            try:
                response = conv_to_str(responses)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert conversation response to string format. "
                    f"Expected OpenAI conversation format (list of message dicts with 'role' and 'content' fields). "
                    f"Got: {type(responses)}. "
                    f"Error: {str(e)}"
                )
            scores = conversation.scores

            if not scores or not self.include_scores_in_prompt:
                return response
            return (
                f"{response}\n\n"
                f"<Quality scores on the trace>\n {scores}\n</Quality scores on the trace>\n\n"
            )
        else:
            raise ValueError(f"Invalid conversation format: {conversation}")
    
    def _conversation_has_images(self, conv_msgs: List[Dict[str, Any]]) -> bool:
        """Return True if any message contains an image segment in ordered segments format."""
        for msg in conv_msgs:
            content = msg.get("content", {})
            segs = content.get("segments") if isinstance(content, dict) else None
            if isinstance(segs, list):
                for seg in segs:
                    if isinstance(seg, dict) and seg.get("kind") == "image":
                        return True
        return False

    def _collapse_segments_to_openai_content(self, conv_msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collapse ordered segments into an OpenAI multimodal content list with aggregated text.
        
        Algorithm:
        - Walk through messages in order, accumulating non-image content in a buffer
        - When an image is encountered, flush the buffer (convert to string via conv_to_str),
          then add the image as a separate item
        - Continue until all messages processed, then flush any remaining buffer
        
        This ensures consecutive non-image turns are aggregated into single text items,
        with images interspersed at their proper positions.
        
        Produces items like:
          - {"type": "text", "text": str}  (potentially aggregated from multiple turns)
          - {"type": "image_url", "image_url": {"url": str}}
        """
        content: List[Dict[str, Any]] = []
        message_buffer: List[Dict[str, Any]] = []  # Buffer for messages without images
        
        def flush_buffer():
            """Convert buffered messages to a single text string using conv_to_str."""
            if message_buffer:
                text_str = conv_to_str(message_buffer)
                if text_str and text_str.strip():
                    content.append({"type": "text", "text": text_str})
                message_buffer.clear()
        
        for msg in conv_msgs:
            # Extract segments from this message
            msg_content = msg.get("content", {})
            segs = msg_content.get("segments", []) if isinstance(msg_content, dict) else []
            
            # Check if this message contains any images
            images_in_msg: List[str] = []
            non_image_segments: List[Dict[str, Any]] = []
            
            for seg in segs:
                if not isinstance(seg, dict):
                    non_image_segments.append(seg)
                    continue
                    
                kind = seg.get("kind")
                if kind == "image":
                    # Extract image URL
                    img = seg.get("image")
                    url: Optional[str] = None
                    if isinstance(img, str):
                        url = img
                    elif isinstance(img, dict):
                        if isinstance(img.get("url"), str):
                            url = img.get("url")
                        elif isinstance(img.get("image_url"), dict) and isinstance(img["image_url"].get("url"), str):
                            url = img["image_url"].get("url")
                        elif isinstance(img.get("source"), str):
                            url = img.get("source")
                    if url:
                        images_in_msg.append(url)
                else:
                    # Keep non-image segments (text, tool, etc.)
                    non_image_segments.append(seg)
            
            # Build a message dict with only non-image content for the buffer
            if non_image_segments:
                msg_for_buffer = dict(msg)  # Copy message structure
                msg_for_buffer["content"] = {
                    "segments": non_image_segments
                }
                message_buffer.append(msg_for_buffer)
            
            # If we encountered images, flush buffer then add images
            if images_in_msg:
                flush_buffer()
                for img_url in images_in_msg:
                    content.append({"type": "image_url", "image_url": {"url": img_url}})
        
        # Flush any remaining buffered messages at the end
        flush_buffer()
        
        return content

    def _build_single_user_messages(self, conv_msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a full messages list with system + single multimodal user turn."""
        content = self._collapse_segments_to_openai_content(conv_msgs)
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": f"<beginning of Model trace>\n {content}\n<end of Model trace>\n\n"})
        return messages

    def _build_side_by_side_messages(
        self,
        model_a: str,
        model_b: str,
        conv_a: List[Dict[str, Any]],
        conv_b: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build a full messages list with system + single user turn containing A/B sections."""
        content: List[Dict[str, Any]] = []
        content += (
            [{"type": "text", "text": "<beginning of Model A trace>"}]
            + self._collapse_segments_to_openai_content(conv_a)
            + [{"type": "text", "text": "<end of Model A trace>\n\n--------------------------------\n\n<beginning of Model B trace>"}]
            + self._collapse_segments_to_openai_content(conv_b)
            + [{"type": "text", "text": "<end of Model B trace>"}]
        )

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages
    
    def _log_extraction_to_wandb(self, user_messages: List[str], raw_responses: List[str], conversations):
        """Log extraction inputs/outputs to wandb."""
        try:
            import wandb
            # import weave

            # Create a table of inputs and outputs
            extraction_data = []
            for i, (msg, response, conv) in enumerate(zip(user_messages, raw_responses, conversations)):
                # Handle None responses (failed LLM calls)
                if response is None:
                    extraction_data.append({
                        "question_id": conv.question_id,
                        "system_prompt": self.system_prompt,
                        "input_message": msg,
                        "raw_response": "FAILED: None",
                        "response_length": 0,
                        "has_error": True,
                    })
                else:
                    extraction_data.append({
                        "question_id": conv.question_id,
                        "system_prompt": self.system_prompt,
                        "input_message": msg,
                        "raw_response": response,
                        "response_length": len(response),
                        "has_error": False,
                    })

            # Log extraction table (as table, not summary)
            self.log_wandb({
                "Property_Extraction/extraction_inputs_outputs": wandb.Table(
                    columns=["question_id", "system_prompt", "input_message", "raw_response", "response_length", "has_error"],
                    data=[[row[col] for col in ["question_id", "system_prompt", "input_message", "raw_response", "response_length", "has_error"]]
                          for row in extraction_data]
                )
            })

            # Log extraction metrics as summary metrics (not regular metrics)
            error_count = sum(1 for r in raw_responses if r is None)
            valid_responses = [r for r in raw_responses if r is not None]
            extraction_metrics = {
                "extraction_total_requests": len(raw_responses),
                "extraction_error_count": error_count,
                "extraction_success_rate": (len(raw_responses) - error_count) / len(raw_responses) if raw_responses else 0,
                "extraction_avg_response_length": sum(len(r) for r in valid_responses) / len(valid_responses) if valid_responses else 0,
            }
            self.log_wandb(extraction_metrics, is_summary=True)

        except Exception as e:
            self.log(f"Failed to log extraction to wandb: {e}", level="warning")        