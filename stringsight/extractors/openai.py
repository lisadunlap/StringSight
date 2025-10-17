"""
OpenAI-based property extraction stage.

This stage migrates the logic from generate_differences.py into the pipeline architecture.
"""

from typing import Callable, Optional, List
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Property
from ..core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin
from ..prompts import extractor_prompts as _extractor_prompts
from ..core.caching import Cache
from ..core.llm_utils import parallel_completions
from .conv_to_str import conv_to_str


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
        max_workers: int = 16,
        cache_dir: str = ".cache/stringsight",
        include_scores_in_prompt: bool = False,
        **kwargs
    ):
        """
        Initialize the OpenAI extractor.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4o-mini")
            system_prompt: System prompt for property extraction
            prompt_builder: Optional custom prompt builder function
            temperature: Temperature for LLM
            top_p: Top-p for LLM
            max_tokens: Max tokens for LLM
            max_workers: Max parallel workers for API calls
            cache_dir: Directory for on-disk cache
            **kwargs: Additional configuration
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
        # Keep cache instance for other potential uses, but LLM calls will go through llm_utils
        self.cache = Cache(cache_dir=cache_dir)
        # Control whether to include numeric scores/winner context in prompts
        self.include_scores_in_prompt = include_scores_in_prompt

    def __del__(self):
        """Cleanup cache on deletion."""
        if hasattr(self, 'cache'):
            self.cache.close()

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Run OpenAI extraction for all conversations.

        Each conversation is formatted with ``prompt_builder`` and sent to the
        OpenAI model in parallel using a thread pool.  The raw LLM response is
        stored inside a *placeholder* ``Property`` object (one per
        conversation).  Down-stream stages (``LLMJsonParser``) will parse these
        raw strings into fully-formed properties.
        """

        n_conv = len(data.conversations)
        if n_conv == 0:
            self.log("No conversations found – skipping extraction")
            return data

        self.log(f"Extracting properties from {n_conv} conversations using {self.model}")

        # Print diagnostic information about models before extraction
        print(f"\n🔍 Property extraction diagnostic:")
        print(f"   • Input dataset has {len(data.all_models)} models: {sorted(data.all_models)}")
        print(f"   • Input conversations: {len(data.conversations)}")
        
        # Count conversations per model before extraction
        model_conversation_counts = {}
        for conv in data.conversations:
            if isinstance(conv.model, str):
                model_conversation_counts[conv.model] = model_conversation_counts.get(conv.model, 0) + 1
            elif isinstance(conv.model, list):
                for model in conv.model:
                    model_conversation_counts[model] = model_conversation_counts.get(model, 0) + 1
        
        print(f"   • Conversations per model before extraction:")
        for model in sorted(data.all_models):
            count = model_conversation_counts.get(model, 0)
            print(f"     - {model}: {count} conversations")
        print()

        # ------------------------------------------------------------------
        # 1️⃣  Build user messages for every conversation
        # ------------------------------------------------------------------
        user_messages: List[str] = []
        for conv in data.conversations:
            user_messages.append(self.prompt_builder(conv))

        # ------------------------------------------------------------------
        # 2️⃣  Call the OpenAI API in parallel batches via shared LLM utils
        # ------------------------------------------------------------------
        raw_responses = parallel_completions(
            user_messages,
            model=self.model,
            system_prompt=self.system_prompt,
            max_workers=self.max_workers,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            show_progress=True,
            progress_desc="Property extraction"
        )

        # ------------------------------------------------------------------
        # 3️⃣  Wrap raw responses in placeholder Property objects
        # ------------------------------------------------------------------
        properties: List[Property] = []
        for conv, raw in zip(data.conversations, raw_responses):
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

        self.log(f"Received {len(properties)} LLM responses")

        # Print diagnostic information about properties after extraction
        print(f"   • Properties created after extraction: {len(properties)}")
        
        # Count properties per model after extraction (these are placeholder properties)
        model_property_counts = {}
        for prop in properties:
            if isinstance(prop.model, str):
                model_property_counts[prop.model] = model_property_counts.get(prop.model, 0) + 1
            elif isinstance(prop.model, list):
                for model in prop.model:
                    model_property_counts[model] = model_property_counts.get(model, 0) + 1
        
        print(f"   • Properties per model after extraction (placeholder):")
        for model in sorted(data.all_models):
            count = model_property_counts.get(model, 0)
            print(f"     - {model}: {count} properties")
            
            # Show if there's a mismatch between conversations and properties
            conv_count = model_conversation_counts.get(model, 0)
            if count != conv_count:
                print(f"       ⚠️  Mismatch: {conv_count} conversations but {count} properties")

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
    
    def _default_prompt_builder(self, conversation) -> str:
        """
        Default prompt builder for side-by-side comparisons.
        
        Args:
            conversation: ConversationRecord
            
        Returns:
            Formatted prompt string
        """
        # Check if this is a side-by-side comparison or single model
        if isinstance(conversation.model, list) and len(conversation.model) == 2:
            # Side-by-side format
            model_a, model_b = conversation.model
            try:
                response_a = conv_to_str(conversation.responses[0])
                response_b = conv_to_str(conversation.responses[1])
            except Exception as e:
                raise ValueError(
                    f"Failed to convert conversation responses to string format. "
                    f"Expected OpenAI conversation format (list of message dicts with 'role' and 'content' fields). "
                    f"Got: {type(conversation.responses[0])}, {type(conversation.responses[1])}. "
                    f"Error: {str(e)}"
                )
            scores = conversation.scores

            # Handle list format [scores_a, scores_b]
            if isinstance(scores, list) and len(scores) == 2:
                scores_a, scores_b = scores[0], scores[1]
                winner = conversation.meta.get("winner")  # Winner stored in meta
                
                # Build the prompt with separate scores for each model
                prompt_parts = [
                    f"# Model A (Name: \"{model_a}\") conversation:\n {response_a}"
                ]
                
                if self.include_scores_in_prompt and scores_a:
                    prompt_parts.append(f"# Model A Scores:\n {scores_a}")
                
                prompt_parts.append("--------------------------------")
                prompt_parts.append(f"# Model B (Name: \"{model_b}\") conversation:\n {response_b}")
                
                if self.include_scores_in_prompt and scores_b:
                    prompt_parts.append(f"# Model B Scores:\n {scores_b}")
                
                if self.include_scores_in_prompt and winner:
                    prompt_parts.append(f"# Winner: {winner}")
                
                return "\n\n".join(prompt_parts)
            else:
                # No scores available
                return (
                    f"# Model A (Name: \"{model_a}\") conversation:\n {response_a}\n\n"
                    f"--------------------------------\n"
                    f"# Model B (Name: \"{model_b}\") conversation:\n {response_b}"
                )
        elif isinstance(conversation.model, str):
            # Single model format
            model = conversation.model if isinstance(conversation.model, str) else str(conversation.model)
            try:
                response = conv_to_str(conversation.responses)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert conversation response to string format. "
                    f"Expected OpenAI conversation format (list of message dicts with 'role' and 'content' fields). "
                    f"Got: {type(conversation.responses)}. "
                    f"Error: {str(e)}"
                )
            scores = conversation.scores

            if not scores or not self.include_scores_in_prompt:
                return response
            return (
                f"{response}\n\n"
                f"### Scores:\n {scores}"
            )
        else:
            raise ValueError(f"Invalid conversation format: {conversation}")
    
    def _log_extraction_to_wandb(self, user_messages: List[str], raw_responses: List[str], conversations):
        """Log extraction inputs/outputs to wandb."""
        try:
            import wandb
            # import weave
            
            # Create a table of inputs and outputs
            extraction_data = []
            for i, (msg, response, conv) in enumerate(zip(user_messages, raw_responses, conversations)):
                extraction_data.append({
                    "question_id": conv.question_id,
                    "system_prompt": self.system_prompt,
                    "input_message": msg,
                    "raw_response": response,
                    "response_length": len(response),
                    "has_error": response.startswith("ERROR:"),
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
            error_count = sum(1 for r in raw_responses if r.startswith("ERROR:"))
            extraction_metrics = {
                "extraction_total_requests": len(raw_responses),
                "extraction_error_count": error_count,
                "extraction_success_rate": (len(raw_responses) - error_count) / len(raw_responses) if raw_responses else 0,
                "extraction_avg_response_length": sum(len(r) for r in raw_responses) / len(raw_responses) if raw_responses else 0,
            }
            self.log_wandb(extraction_metrics, is_summary=True)
            
        except Exception as e:
            self.log(f"Failed to log extraction to wandb: {e}", level="warning")        