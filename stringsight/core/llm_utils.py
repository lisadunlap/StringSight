"""
General utilities for parallel LLM and embedding calls with caching support.

This module provides reusable functions that eliminate code duplication across 
the codebase for parallel LLM operations while maintaining order preservation
and robust error handling.
"""

import time
import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import litellm
import numpy as np
from tqdm import tqdm
import logging
# Weave removed

from .caching import UnifiedCache
from ..utils.validation import validate_openai_api_key

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global LiteLLM configuration
# ------------------------------------------------------------------
# Some providers (e.g., Bedrock) do not support certain parameters such as
# `max_completion_tokens`. Enabling `drop_params` tells LiteLLM to silently
# drop any unsupported parameters instead of raising UnsupportedParamsError.
# This keeps StringSight's higher-level LLM plumbing provider-agnostic while
# still allowing us to set sensible defaults for providers that support them.
litellm.drop_params = True


# ==================== LiteLLM Timing Callback ====================
try:
    from litellm.integrations.custom_logger import CustomLogger as LiteLLMCustomLogger
except ImportError:
    LiteLLMCustomLogger = object


class LLMTimingCallback(LiteLLMCustomLogger):
    """Custom LiteLLM callback to log API call timing information."""

    def log_pre_api_call(self, model, messages, kwargs):
        """Called before API call - not used for timing."""
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        """Log timing for every API call."""
        try:
            duration = (end_time - start_time).total_seconds()
            model = kwargs.get("model", "unknown")
            print(f"[LLM Timing] {model}: {duration:.3f}s", flush=True)
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful API calls with detailed timing."""
        try:
            duration = (end_time - start_time).total_seconds()
            model = kwargs.get("model", "unknown")

            # Extract token usage if available
            usage = getattr(response_obj, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                print(
                    f"[LLM Success] {model}: {duration:.3f}s | "
                    f"tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})",
                    flush=True
                )
            else:
                print(f"[LLM Success] {model}: {duration:.3f}s", flush=True)
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed API calls."""
        try:
            duration = (end_time - start_time).total_seconds()
            model = kwargs.get("model", "unknown")
            error = kwargs.get("exception", "unknown error")
            print(f"[LLM Failure] {model}: {duration:.3f}s | error: {error}", flush=True)
        except Exception:
            pass


# Global timing callback instance
_timing_callback = None


def enable_timing_logs():
    """Enable detailed timing logs for all LLM API calls."""
    global _timing_callback
    if _timing_callback is None:
        _timing_callback = LLMTimingCallback()
        if not hasattr(litellm, 'callbacks') or litellm.callbacks is None:
            litellm.callbacks = []
        litellm.callbacks.append(_timing_callback)
        logger.info("✓ LLM timing logs enabled")


def disable_timing_logs():
    """Disable detailed timing logs for LLM API calls."""
    global _timing_callback
    if _timing_callback is not None and hasattr(litellm, 'callbacks'):
        if litellm.callbacks and _timing_callback in litellm.callbacks:
            litellm.callbacks.remove(_timing_callback)
        _timing_callback = None
        logger.info("✓ LLM timing logs disabled")


# Enable timing logs by default
# enable_timing_logs()  # Disabled by default - call manually to enable
# ================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    model: str = "gpt-4.1-mini"
    max_workers: int = 64
    max_retries: int = 3
    base_sleep_time: float = 2.0
    timeout: Optional[float] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    

@dataclass
class EmbeddingConfig:
    """Configuration for embedding calls."""
    model: str = "text-embedding-3-large"
    batch_size: int = 100
    max_workers: int = 64
    max_retries: int = 3
    base_sleep_time: float = 2.0


class LLMUtils:
    """Utility class for parallel LLM operations with caching."""

    def __init__(self, cache: Optional[UnifiedCache] = None):
        """Initialize with optional cache instance."""
        self.cache = cache
        self._lock = threading.Lock()
    
    def parallel_completions(
        self,
        messages: List[Union[str, List[Dict[str, Any]]]],
        config: LLMConfig,
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        progress_desc: str = "LLM calls",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Execute LLM completions in parallel with order preservation.
        
        Args:
            messages: List of user messages (strings) or full message lists
            config: LLM configuration
            system_prompt: Optional system prompt (if messages are strings)
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            progress_callback: Optional callback(completed, total) called after each completion

        Returns:
            List of completion responses in the same order as input
        """
        # Fail fast with a clear message if the configured model requires OpenAI credentials.
        # This avoids cryptic provider errors deep in LiteLLM/OpenAI client code.
        validate_openai_api_key(model=config.model)

        if not messages:
            return []
            
        # Pre-allocate results to preserve order
        results: List[str] = [""] * len(messages)
        
        # Weave removed: directly define function
        def _single_completion(idx: int, message: Union[str, List[Dict[str, Any]]]) -> Tuple[int, str]:
            """Process a single completion with retries."""
            for attempt in range(config.max_retries):
                try:
                    # Build request data
                    if isinstance(message, str):
                        request_messages = []
                        if system_prompt:
                            request_messages.append({"role": "system", "content": system_prompt})
                        request_messages.append({"role": "user", "content": message})
                    else:
                        request_messages = message
                    
                    request_data = {
                        "model": config.model,
                        "messages": request_messages,
                    }
                    
                    # GPT-5 models only support temperature=1, so skip temperature/top_p for them
                    if "gpt-5" not in config.model.lower():
                        request_data["temperature"] = config.temperature
                        request_data["top_p"] = config.top_p
                    
                    if config.max_tokens:
                        request_data["max_completion_tokens"] = config.max_tokens
                    
                    # Check cache first
                    cached_response = None
                    if self.cache:
                        cached_response = self.cache.get_completion(request_data)
                        if cached_response is not None:
                            cached_content = cached_response["choices"][0]["message"]["content"]
                            # Validate cached content - bypass cache if empty/invalid
                            if not cached_content:
                                logger.warning(f"⚠️ Invalid cached response (empty content)! Bypassing cache and retrying...")
                                logger.warning(f"  Model: {config.model}")
                                cached_response = None  # Force bypass cache
                            else:
                                logger.debug("CACHE HIT")
                                return idx, cached_content
                    
                    # Make API call with graceful fallback for provider-specific unsupported params
                    try:
                        logger.debug("CACHE MISS")
                        response = litellm.completion(
                            **request_data,
                            caching=False,  # Use our own caching
                            timeout=config.timeout
                        )
                    except Exception as api_err:
                        # Check if this is an unsupported params error
                        err_text = str(api_err).lower()
                        err_type = type(api_err).__name__.lower()
                        
                        # Handle unsupported parameter errors (BadRequestError, UnsupportedParamsError, etc.)
                        if ("unsupported" in err_text or "don't support" in err_text or 
                            "badrequest" in err_type or "unsupportedparams" in err_type):
                            safe_request = dict(request_data)
                            retried = False
                            
                            if "temperature" in err_text:
                                safe_request.pop("temperature", None)
                                retried = True
                                logger.info(f"Dropping temperature parameter for {config.model}")
                            if "top_p" in err_text:
                                safe_request.pop("top_p", None)
                                retried = True
                                logger.info(f"Dropping top_p parameter for {config.model}")
                            
                            if retried:
                                response = litellm.completion(
                                    **safe_request,
                                    caching=False,
                                    timeout=config.timeout
                                )
                            else:
                                raise
                        else:
                            # Not an unsupported params error, re-raise to outer handler
                            raise
                    
                    # Extract content with validation
                    if not response or not response.choices or not response.choices[0].message:
                        logger.warning(f"⚠️ LLM returned malformed response (skipping)")
                        logger.warning(f"  Model: {config.model}")
                        # Don't cache, return None to be filtered out later
                        return idx, None

                    content = response.choices[0].message.content

                    # Validate content - if empty/None, return None (don't cache)
                    if content is None or not content:
                        logger.warning(f"⚠️ LLM returned empty content (skipping)")
                        logger.warning(f"  Model: {config.model}")
                        logger.warning(f"  This usually means invalid model name or API error")
                        # Don't cache empty responses, return None to be filtered out later
                        return idx, None

                    # Only cache valid (non-empty) responses
                    if self.cache:
                        response_dict = {
                            "choices": [{
                                "message": {"content": content}
                            }]
                        }
                        self.cache.set_completion(request_data, response_dict)
                        logger.debug("CACHE STORE (completion)")

                    return idx, content
                    
                except Exception as e:
                    if attempt == config.max_retries - 1:
                        logger.warning(f"⚠️ LLM call failed after {config.max_retries} attempts (skipping): {e}")
                        logger.warning(f"  Model: {config.model}")
                        logger.warning(f"  Message: {str(request_messages)[:200]}")
                        # Don't cache, return None to be filtered out later
                        return idx, None
                    else:
                        # Check if this is a rate limit error
                        is_rate_limit = False
                        retry_after = None
                        
                        # Check for litellm rate limit errors
                        if hasattr(litellm, 'RateLimitError') and isinstance(e, litellm.RateLimitError):
                            is_rate_limit = True
                        elif hasattr(e, 'status_code') and e.status_code == 429:
                            is_rate_limit = True
                        elif "rate limit" in str(e).lower() or "429" in str(e):
                            is_rate_limit = True
                        
                        # Try to extract Retry-After header from exception
                        if is_rate_limit:
                            try:
                                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                                    retry_after_str = e.response.headers.get('Retry-After', '')
                                    if retry_after_str:
                                        retry_after = float(retry_after_str)
                            except Exception:
                                pass
                        
                        # Use Retry-After if available, otherwise use exponential backoff
                        if is_rate_limit and retry_after:
                            sleep_time = retry_after
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{config.max_retries}), waiting {sleep_time}s per Retry-After header: {e}")
                        elif is_rate_limit:
                            # For rate limits, use shorter initial backoff but still exponential
                            sleep_time = max(1.0, config.base_sleep_time * (1.5 ** attempt))
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{config.max_retries}), retrying in {sleep_time}s: {e}")
                        else:
                            # For other errors, use standard exponential backoff
                            sleep_time = config.base_sleep_time * (2 ** attempt)
                            logger.warning(f"LLM call attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
                        
                        time.sleep(sleep_time)

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(_single_completion, idx, msg): idx
                for idx, msg in enumerate(messages)
            }

            # Process results with optional progress bar
            # Use manual progress updates for smoother per-item tracking
            pbar = tqdm(total=len(messages), desc=progress_desc, disable=not show_progress)
            completed = 0
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                pbar.update(1)
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, len(messages))
            pbar.close()

        # Log summary of failed calls but keep None values to maintain index alignment
        failed_count = sum(1 for r in results if r is None)
        if failed_count > 0:
            logger.warning(f"⚠️ {failed_count}/{len(results)} LLM calls failed (returned None)")

        # Return results with None values preserved - caller must filter
        return results
    
    def parallel_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig,
        show_progress: bool = True,
        progress_desc: str = "Generating embeddings"
    ) -> List[List[float]]:
        """
        Generate embeddings in parallel with batching and order preservation.
        
        Args:
            texts: List of texts to embed
            config: Embedding configuration
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            
        Returns:
            List of embeddings in the same order as input texts
        """
        if not texts:
            return []
        
        # Pre-allocate results to preserve order
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        
        # Normalize model identifier for provider-prefixed LiteLLM routing
        model_name = _normalize_embedding_model_name(config.model)

        # Create batches with their positions
        batches = []
        for start in range(0, len(texts), config.batch_size):
            end = min(start + config.batch_size, len(texts))
            batch_texts = texts[start:end]
            batches.append((start, batch_texts))
        
        def _single_embedding_batch(start_idx: int, batch_texts: List[str]) -> Tuple[int, List[List[float]]]:
            """Process a single batch of embeddings with retries."""
            for attempt in range(config.max_retries):
                try:
                    # Check cache for each text in batch
                    batch_embeddings = []
                    uncached_indices = []
                    uncached_texts = []
                    
                    for i, text in enumerate(batch_texts):
                        if self.cache:
                            # Namespace by model to avoid mixing dims across providers/models
                            cached_embedding = self.cache.get_embedding((model_name, text))
                            if cached_embedding is not None:
                                logger.debug("CACHE HIT")
                                batch_embeddings.append(cached_embedding.tolist())
                                continue
                        
                        # Mark as needing API call
                        batch_embeddings.append(None)
                        uncached_indices.append(i)
                        uncached_texts.append(text)
                    
                    # Make API call for uncached texts
                    if uncached_texts:
                        logger.debug("CACHE MISS")
                        response = litellm.embedding(
                            model=model_name,
                            input=uncached_texts
                        )
                        
                        # Fill in uncached results
                        for i, embedding_data in enumerate(response.data):
                            batch_idx = uncached_indices[i]
                            embedding = embedding_data.embedding
                            batch_embeddings[batch_idx] = embedding
                            
                            # Cache the embedding
                            if self.cache:
                                self.cache.set_embedding((model_name, uncached_texts[i]), np.array(embedding))
                                logger.debug("CACHE STORE (embedding)")
                    
                    return start_idx, batch_embeddings
                    
                except Exception as e:
                    if attempt == config.max_retries - 1:
                        logger.error(f"Embedding batch failed after {config.max_retries} attempts: {e}")
                        # Return zero embeddings as fallback
                        fallback_embeddings = [[0.0] * 1536] * len(batch_texts)  # Default to 1536 dims
                        return start_idx, fallback_embeddings
                    else:
                        # Check if this is a rate limit error
                        is_rate_limit = False
                        retry_after = None
                        
                        # Check for litellm rate limit errors
                        if hasattr(litellm, 'RateLimitError') and isinstance(e, litellm.RateLimitError):
                            is_rate_limit = True
                        elif hasattr(e, 'status_code') and e.status_code == 429:
                            is_rate_limit = True
                        elif "rate limit" in str(e).lower() or "429" in str(e):
                            is_rate_limit = True
                        
                        # Try to extract Retry-After header from exception
                        if is_rate_limit:
                            try:
                                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                                    retry_after_str = e.response.headers.get('Retry-After', '')
                                    if retry_after_str:
                                        retry_after = float(retry_after_str)
                            except Exception:
                                pass
                        
                        # Use Retry-After if available, otherwise use exponential backoff
                        if is_rate_limit and retry_after:
                            sleep_time = retry_after
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{config.max_retries}), waiting {sleep_time}s per Retry-After header: {e}")
                        elif is_rate_limit:
                            # For rate limits, use shorter initial backoff but still exponential
                            sleep_time = max(1.0, config.base_sleep_time * (1.5 ** attempt))
                            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{config.max_retries}), retrying in {sleep_time}s: {e}")
                        else:
                            # For other errors, use standard exponential backoff
                            sleep_time = config.base_sleep_time * (2 ** attempt)
                            logger.warning(f"Embedding batch attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
                        
                        time.sleep(sleep_time)
            
            # Fallback
            fallback_embeddings = [[0.0] * 1536] * len(batch_texts)
            return start_idx, fallback_embeddings
        
        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(_single_embedding_batch, start, batch_texts): (start, len(batch_texts))
                for start, batch_texts in batches
            }

            # Process results with optional progress bar
            # Track by number of texts embedded, not batches, for better granularity
            pbar = tqdm(total=len(texts), desc=progress_desc, disable=not show_progress)
            for future in as_completed(futures):
                start_idx, batch_embeddings = future.result()
                batch_size = len(batch_embeddings)
                embeddings[start_idx:start_idx + batch_size] = batch_embeddings
                pbar.update(batch_size)
            pbar.close()
        
        # Verify all embeddings were filled
        if any(e is None for e in embeddings):
            raise RuntimeError("Some embeddings are missing - check logs for errors.")
        
        return embeddings
    
    def single_completion(
        self,
        message: Union[str, List[Dict[str, Any]]],
        config: LLMConfig,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Single completion call with caching (convenience method).
        
        Args:
            message: User message (string) or full message list
            config: LLM configuration
            system_prompt: Optional system prompt (if message is string)
            
        Returns:
            Completion response
        """
        results = self.parallel_completions([message], config, system_prompt, show_progress=False)
        return results[0]


# Global instance with default cache (singleton)
_default_llm_utils = None

def get_default_llm_utils() -> LLMUtils:
    """Get default LLMUtils instance with shared cache.

    Uses UnifiedCache singleton which handles all configuration via
    environment variables.
    """
    global _default_llm_utils

    if _default_llm_utils is None:
        # UnifiedCache is a singleton - same instance across all modules
        # It handles STRINGSIGHT_DISABLE_CACHE internally
        cache = UnifiedCache()
        _default_llm_utils = LLMUtils(cache)
        logger.info("LLM utilities initialized with UnifiedCache")

    return _default_llm_utils


# Convenience functions for common use cases
def parallel_completions(
    messages: List[Union[str, List[Dict[str, Any]]]],
    model: str = "gpt-4.1-mini",
    system_prompt: Optional[str] = None,
    max_workers: int = 64,
    show_progress: bool = True,
    progress_desc: str = "LLM calls",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs
) -> List[str]:
    """Convenience function for parallel completions with default settings."""
    # Separate function-specific parameters from config parameters
    config_kwargs = {k: v for k, v in kwargs.items()
                    if k in ['max_retries', 'base_sleep_time', 'timeout', 'temperature', 'top_p', 'max_tokens']}

    config = LLMConfig(model=model, max_workers=max_workers, **config_kwargs)
    utils = get_default_llm_utils()
    return utils.parallel_completions(messages, config, system_prompt, show_progress, progress_desc, progress_callback)


async def parallel_completions_async(
    messages: List[Union[str, List[Dict[str, Any]]]],
    model: str = "gpt-4.1-mini",
    system_prompt: Optional[str] = None,
    max_workers: int = 64,
    show_progress: bool = True,
    progress_desc: str = "LLM calls",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs
) -> List[str]:
    """Async wrapper for parallel completions with default settings."""
    # Run the synchronous function in an executor to avoid blocking
    loop = asyncio.get_event_loop()
    config_kwargs = {k: v for k, v in kwargs.items()
                    if k in ['max_retries', 'base_sleep_time', 'timeout', 'temperature', 'top_p', 'max_tokens']}

    config = LLMConfig(model=model, max_workers=max_workers, **config_kwargs)
    utils = get_default_llm_utils()
    
    # Use run_in_executor to run the sync function without blocking the event loop
    return await loop.run_in_executor(
        None, 
        lambda: utils.parallel_completions(messages, config, system_prompt, show_progress, progress_desc, progress_callback)
    )


def parallel_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 100,
    max_workers: int = 64,
    show_progress: bool = True,
    progress_desc: str = "Generating embeddings",
    **kwargs
) -> List[List[float]]:
    """Convenience function for parallel embeddings with default settings."""
    # Separate function-specific parameters from config parameters
    config_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['max_retries', 'base_sleep_time']}
    
    config = EmbeddingConfig(model=model, batch_size=batch_size, max_workers=max_workers, **config_kwargs)
    utils = get_default_llm_utils()
    return utils.parallel_embeddings(texts, config, show_progress, progress_desc)


# -----------------------------
# Provider/model normalization
# -----------------------------
def _normalize_embedding_model_name(model: str) -> str:
    """Return a LiteLLM-compatible embedding model name.

    Simply passes through the model name to litellm, which handles all provider
    routing and validation. This allows any embedding model supported by litellm
    to be used without hardcoding model names.

    For OpenAI models, litellm accepts both:
    - 'text-embedding-3-small' (litellm will auto-detect as OpenAI)
    - 'openai/text-embedding-3-small' (explicit provider prefix)

    For other providers, use the litellm provider prefix format:
    - 'bedrock/amazon.titan-embed-text-v1'
    - 'azure/text-embedding-ada-002'
    - 'cohere/embed-english-v3.0'
    - etc.
    """
    if not model:
        return "text-embedding-3-large"

    return str(model).strip()


def single_completion(
    message: Union[str, List[Dict[str, Any]]],
    model: str = "gpt-4.1-mini", 
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function for single completion with caching."""
    # Separate function-specific parameters from config parameters (no function-specific params for single)
    config_kwargs = {k: v for k, v in kwargs.items() 
                    if k in ['max_workers', 'max_retries', 'base_sleep_time', 'timeout', 'temperature', 'top_p', 'max_tokens']}
    
    config = LLMConfig(model=model, **config_kwargs)
    utils = get_default_llm_utils()
    return utils.single_completion(message, config, system_prompt)
