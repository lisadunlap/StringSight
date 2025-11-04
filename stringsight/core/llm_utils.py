"""
General utilities for parallel LLM and embedding calls with caching support.

This module provides reusable functions that eliminate code duplication across 
the codebase for parallel LLM operations while maintaining order preservation
and robust error handling.
"""

import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import litellm
import numpy as np
from tqdm import tqdm
import logging
# Weave removed

from .caching import Cache

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    model: str = "gpt-4o-mini"
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
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_workers: int = 64
    max_retries: int = 3
    base_sleep_time: float = 2.0


class LLMUtils:
    """Utility class for parallel LLM operations with caching."""
    
    def __init__(self, cache: Optional[Cache] = None):
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
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                    }
                    if config.max_tokens:
                        request_data["max_completion_tokens"] = config.max_tokens
                    
                    # Check cache first
                    if self.cache:
                        cached_response = self.cache.get_completion(request_data)
                        if cached_response is not None:
                            logger.debug("CACHE HIT")
                            return idx, cached_response["choices"][0]["message"]["content"]
                    
                    # Make API call with graceful fallback for provider-specific unsupported params
                    try:
                        logger.debug("CACHE MISS")
                        response = litellm.completion(
                            **request_data,
                            caching=False,  # Use our own caching
                            timeout=config.timeout
                        )
                    except litellm.BadRequestError as api_err:
                        # Some models only accept default temperature/top_p; drop unsupported args and retry once
                        err_text = str(api_err).lower()
                        safe_request = dict(request_data)
                        retried = False
                        if "temperature" in err_text:
                            safe_request.pop("temperature", None)
                            retried = True
                        if "top_p" in err_text:
                            safe_request.pop("top_p", None)
                            retried = True
                        if retried:
                            response = litellm.completion(
                                **safe_request,
                                caching=False,
                                timeout=config.timeout
                            )
                        else:
                            raise
                    
                    content = response.choices[0].message.content
                    
                    # Cache the response
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
                        logger.error(f"LLM call failed after {config.max_retries} attempts: {e}")
                        return idx, f"ERROR: {e}"
                    else:
                        sleep_time = config.base_sleep_time * (2 ** attempt)
                        logger.warning(f"LLM call attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}")
                        time.sleep(sleep_time)
            
            return idx, "ERROR: Max retries exceeded"
        
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
        
        return results
    
    def parallel_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig,
        show_progress: bool = True,
        progress_desc: str = "Embedding calls"
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


# Global instance with default cache
_default_cache = None
_default_llm_utils = None

def get_default_llm_utils() -> LLMUtils:
    """Get default LLMUtils instance with shared cache."""
    global _default_cache, _default_llm_utils
    
    if _default_llm_utils is None:
        if _default_cache is None:
            # Always use DiskCache-based Cache by default unless explicitly disabled
            import os
            if os.environ.get("STRINGSIGHT_DISABLE_CACHE", "0") in ("1", "true", "True"):
                _default_cache = None
                logger.info("LLM caching disabled (STRINGSIGHT_DISABLE_CACHE=1)")
            else:
                _default_cache = Cache()
                embedding_status = "disabled" if os.environ.get("STRINGSIGHT_DISABLE_EMBEDDING_CACHE", "0") in ("1", "true", "True") else "enabled"
                logger.info(f"LLM caching enabled (embeddings={embedding_status})")
        _default_llm_utils = LLMUtils(_default_cache)
    
    return _default_llm_utils


# Convenience functions for common use cases
def parallel_completions(
    messages: List[Union[str, List[Dict[str, Any]]]],
    model: str = "gpt-4o-mini",
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


def parallel_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_workers: int = 64,
    show_progress: bool = True,
    progress_desc: str = "Embedding calls",
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
_OPENAI_EMBED_SYNONYMS = {
    "text-embedding-3-large": "openai/text-embedding-3-large",
    "text-embedding-3-small": "openai/text-embedding-3-small",
    "text-embedding-ada-002": "openai/text-embedding-ada-002",
    "openai/text-embedding-3-large": "openai/text-embedding-3-large",
    "openai/text-embedding-3-small": "openai/text-embedding-3-small",
    "openai/text-embedding-ada-002": "openai/text-embedding-ada-002",
    "e3-large": "openai/text-embedding-3-large",
    "e3-small": "openai/text-embedding-3-small",
}

# Known valid OpenAI embedding models
_VALID_OPENAI_MODELS = {
    "text-embedding-3-large",
    "text-embedding-3-small", 
    "text-embedding-ada-002",
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-small",
    "openai/text-embedding-ada-002",
    "e3-large",
    "e3-small"
}

def _normalize_embedding_model_name(model: str) -> str:
    """Return a LiteLLM-compatible provider-prefixed embedding model name.

    Validates that OpenAI models are known/valid and throws clear errors for invalid names.
    Keeps non-OpenAI identifiers as-is (e.g., sentence-transformers, bge-*).
    
    Raises:
        ValueError: If an invalid OpenAI model name is provided
    """
    if not model:
        return "openai/text-embedding-3-small"
    m = str(model).strip()
    
    # Handle the common case where someone just specifies "openai"
    if m.lower() == "openai":
        raise ValueError(
            f"Invalid embedding model '{model}'. Please specify a complete model name like:\n"
            f"  - 'text-embedding-3-small' (recommended)\n"
            f"  - 'text-embedding-3-large'\n" 
            f"  - 'text-embedding-ada-002'\n"
            f"  - or any sentence-transformers model name"
        )
    
    # Check if it looks like an OpenAI model but is invalid
    if (m.startswith("openai/") or m.startswith("text-embedding-") or 
        m.startswith("e3-") or "embedding" in m.lower()):
        if m not in _VALID_OPENAI_MODELS:
            valid_models = sorted([name for name in _VALID_OPENAI_MODELS if not name.startswith("openai/")])
            raise ValueError(
                f"Invalid OpenAI embedding model '{model}'. Valid options are:\n" +
                "\n".join(f"  - '{name}'" for name in valid_models)
            )
    
    # If caller already provided a provider prefix, keep unless we know a canonical mapping
    if m in _OPENAI_EMBED_SYNONYMS:
        return _OPENAI_EMBED_SYNONYMS[m]
    # Heuristic: bare OpenAI E3 names without provider
    if m.startswith("text-embedding-3-") or m in ("e3-large", "e3-small"):
        return _OPENAI_EMBED_SYNONYMS.get(m, f"openai/{m}")
    # Otherwise pass through unchanged (e.g., sentence-transformers models)
    return m


def single_completion(
    message: Union[str, List[Dict[str, Any]]],
    model: str = "gpt-4o-mini", 
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
