"""
embeddings.py

Embedding generation utilities for clustering, supporting both OpenAI and local models.
"""

from __future__ import annotations

import concurrent.futures
import time
import logging
from typing import List
from tqdm import tqdm

import numpy as np
import litellm  # type: ignore

from stringsight.logging_config import get_logger
from stringsight.constants import DEFAULT_MAX_WORKERS
from ..utils.validation import validate_openai_api_key
from ..core.llm_utils import _normalize_embedding_model_name
from ..core.caching import UnifiedCache

logger = get_logger(__name__)

# Shared cache - UnifiedCache singleton used across all modules
_cache = UnifiedCache()


def _get_openai_embeddings_batch(batch: List[str], model: str, retries: int = 3, sleep_time: float = 2.0):
    """Fetch embeddings for one batch with simple exponential back-off."""
    # Fail fast with a clear message if OpenAI embeddings are requested without credentials.
    validate_openai_api_key(embedding_model=model)

    # Check cache first for each text in batch
    cached_embeddings = []
    texts_to_embed = []
    indices_to_embed = []

    for i, text in enumerate(batch):
        # Namespace cache by model to prevent mixing different embedding dimensions
        cache_key = (model, text)
        cached_emb = _cache.get_embedding(cache_key) if _cache else None
        if cached_emb is not None:
            cached_embeddings.append((i, cached_emb))
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    # If all embeddings were cached, return them in order
    if not texts_to_embed:
        # Sort by original index and return only embeddings
        # Convert numpy arrays to lists for consistency
        return [emb.tolist() if isinstance(emb, np.ndarray) else emb
                for _, emb in sorted(cached_embeddings, key=lambda x: x[0])]

    # Get embeddings for texts not in cache
    for attempt in range(retries):
        try:
            norm_model = _normalize_embedding_model_name(model)
            logger.debug(f"[emb-debug] requesting embeddings: model={norm_model} batch_size={len(texts_to_embed)}")
            resp = litellm.embedding(
                model=norm_model,
                input=texts_to_embed,
                caching=False,  # Disable litellm caching since we're using our own
            )
            new_embeddings = [item["embedding"] for item in resp["data"]]
            try:
                dim = len(new_embeddings[0]) if new_embeddings else 0
            except Exception:
                dim = -1
            logger.debug(f"[emb-debug] received embeddings: count={len(new_embeddings)} dim={dim}")

            # Cache the new embeddings
            if _cache:
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    cache_key = (model, text)
                    _cache.set_embedding(cache_key, embedding)

            # Combine cached and new embeddings
            all_embeddings = [None] * len(batch)
            # Convert numpy arrays to lists for consistency
            for i, emb in cached_embeddings:
                all_embeddings[i] = emb.tolist() if isinstance(emb, np.ndarray) else emb
            for i, emb in zip(indices_to_embed, new_embeddings):
                all_embeddings[i] = emb

            return all_embeddings

        except Exception as exc:
            if attempt == retries - 1:
                raise

            # Check if this is a rate limit error
            is_rate_limit = False
            retry_after = None

            # Check for litellm rate limit errors
            if hasattr(litellm, 'RateLimitError') and isinstance(exc, litellm.RateLimitError):
                is_rate_limit = True
            elif hasattr(exc, 'status_code') and exc.status_code == 429:
                is_rate_limit = True
            elif "rate limit" in str(exc).lower() or "429" in str(exc):
                is_rate_limit = True

            # Try to extract Retry-After header from exception
            if is_rate_limit:
                try:
                    if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
                        retry_after_str = exc.response.headers.get('Retry-After', '')
                        if retry_after_str:
                            retry_after = float(retry_after_str)
                except Exception:
                    pass

            # Use Retry-After if available, otherwise use exponential backoff
            if is_rate_limit and retry_after:
                actual_sleep = retry_after
                logger.warning(f"[retry {attempt + 1}/{retries}] Rate limit hit, waiting {actual_sleep}s per Retry-After header: {exc}")
            elif is_rate_limit:
                # For rate limits, use shorter initial backoff
                actual_sleep = max(1.0, sleep_time * (1.5 ** attempt))
                logger.warning(f"[retry {attempt + 1}/{retries}] Rate limit hit, retrying in {actual_sleep}s: {exc}")
            else:
                actual_sleep = sleep_time
                logger.warning(f"[retry {attempt + 1}/{retries}] {exc}. Sleeping {actual_sleep}s.")

            time.sleep(actual_sleep)


def _get_openai_embeddings(texts: List[str], *, model: str = "openai/text-embedding-3-large", batch_size: int = 100, max_workers: int = DEFAULT_MAX_WORKERS) -> List[List[float]]:
    """Get embeddings for *texts* from the OpenAI API whilst preserving order."""

    if not texts:
        return []

    embeddings: List[List[float] | None] = [None] * len(texts)
    batches = [(start, texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_span = {
            executor.submit(_get_openai_embeddings_batch, batch_texts, model): (start, len(batch_texts))
            for start, batch_texts in batches
        }
        iterator = concurrent.futures.as_completed(future_to_span)
        # Always show progress for embedding computation, regardless of verbosity
        iterator = tqdm(iterator, total=len(batches), desc="Embedding calls")
        for fut in iterator:
            start, length = future_to_span[fut]
            batch_embeddings = fut.result()
            embeddings[start : start + length] = batch_embeddings

    if any(e is None for e in embeddings):
        raise RuntimeError("Some embeddings are missing – check logs for errors.")

    # mypy: we just checked there are no Nones
    return embeddings  # type: ignore[return-value]


def _get_embeddings(texts: List[str], embedding_model: str, verbose: bool = False, use_gpu: bool | None = None) -> List[List[float]]:
    """Return embeddings for *texts* using either OpenAI or a SentenceTransformer.

    Args:
        texts: List of strings to embed
        embedding_model: Model name (OpenAI or HuggingFace)
        verbose: Enable verbose logging
        use_gpu: Use GPU acceleration for sentence transformers (default: False)

    Returns:
        List of embeddings as lists of floats
    """

    # Treat OpenAI models either as "openai" keyword or provider-prefixed names
    if embedding_model == "openai" or str(embedding_model).startswith("openai/") or embedding_model in {"text-embedding-3-large", "text-embedding-3-large", "e3-large", "e3-small"}:
        return _get_openai_embeddings(texts, model=_normalize_embedding_model_name(embedding_model))

    # Lazy import of sentence-transformers (optional dependency)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for local embedding models. "
            "Install it with: pip install stringsight[local-embeddings] "
            "or: pip install sentence-transformers"
        )

    if verbose:
        device = "cuda" if use_gpu else "cpu"
        logger.info(f"Computing embeddings with {embedding_model} on {device}…")

    # Set device for sentence transformer
    device = "cuda" if use_gpu else "cpu"
    model = SentenceTransformer(embedding_model, device=device)
    # Always show a progress bar for local embedding computation
    return model.encode(texts, show_progress_bar=True).tolist()


def _setup_embeddings(texts, embedding_model, verbose=False, use_gpu=False):
    """Setup embeddings based on model type. Uses LMDB-based caching.

    Args:
        texts: List of strings to embed
        embedding_model: Model name (OpenAI or HuggingFace)
        verbose: Enable verbose logging
        use_gpu: Use GPU acceleration for sentence transformers (default: False)

    Returns:
        Tuple of (embeddings array or None, model or None)
    """
    if embedding_model == "openai" or str(embedding_model).startswith("openai/") or embedding_model in {"text-embedding-3-large", "text-embedding-3-large", "e3-large", "e3-small"}:
        if verbose:
            logger.info("Using OpenAI embeddings (with disk caching)...")
        embeddings = _get_openai_embeddings(texts, model=_normalize_embedding_model_name(embedding_model))
        embeddings = np.array(embeddings, dtype=float)
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            nan_cnt = int(np.isnan(embeddings).sum())
            inf_cnt = int(np.isinf(embeddings).sum())
            logger.debug(f"[emb-debug] raw embeddings contain NaN/Inf: nan={nan_cnt} inf={inf_cnt}")
        # Normalize embeddings (numerically stable)
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        std = np.where(std == 0, 1e-8, std)
        embeddings = (embeddings - mean) / std
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            nan_cnt = int(np.isnan(embeddings).sum())
            inf_cnt = int(np.isinf(embeddings).sum())
            logger.debug(f"[emb-debug] normalized embeddings contain NaN/Inf: nan={nan_cnt} inf={inf_cnt}")
        return embeddings, None
    else:
        # Lazy import of sentence-transformers (optional dependency)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding models. "
                "Install it with: pip install stringsight[local-embeddings] "
                "or: pip install sentence-transformers"
            )
        if verbose:
            device = "cuda" if use_gpu else "cpu"
            logger.info(f"Using sentence transformer: {embedding_model} on {device}")
        device = "cuda" if use_gpu else "cpu"
        model = SentenceTransformer(embedding_model, device=device)
        return None, model
