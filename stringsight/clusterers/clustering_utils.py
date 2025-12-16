"""
clustering_utils.py

Utility functions shared across clustering modules, particularly for 
LLM‐based summarisation and embedding handling.

These were originally defined in hierarchical_clustering.py but were 
extracted to make them re-usable across different scripts and to keep the
main algorithm file focussed on clustering logic.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import random
import time
import logging
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pickle
import pandas as pd
# wandb is optional - imported lazily when needed
# import weave

import numpy as np
import litellm  # type: ignore
# sentence-transformers is optional - imported lazily when needed
from stringsight.prompts.clustering.prompts import clustering_systems_prompt, coarse_clustering_systems_prompt, deduplication_clustering_systems_prompt
from stringsight.logging_config import get_logger
from ..utils.validation import validate_openai_api_key

logger = get_logger(__name__)
from ..core.llm_utils import parallel_completions_async
from ..core.caching import UnifiedCache

# ----------------------------------------------------------
# Shared cache - UnifiedCache singleton used across all modules
# ----------------------------------------------------------
# UnifiedCache is now shared with llm_utils to avoid duplicate API calls.
# Configuration is handled via STRINGSIGHT_* environment variables.
# No separate STRINGSIGHT_ENABLE_CLUSTERING_CACHE needed - uses same cache.
_cache = UnifiedCache()
logger.info("Clustering utilities initialized with shared UnifiedCache")

# -----------------------------------------------------------------------------
# Ensure deterministic behaviour in sampling helpers and downstream clustering
# -----------------------------------------------------------------------------
# A single global seed keeps `random.sample` calls reproducible across runs.
random.seed(42)

# -----------------------------------------------------------------------------
# OpenAI embeddings helpers
# -----------------------------------------------------------------------------

# Import the unified normalization function
from ..core.llm_utils import _normalize_embedding_model_name


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


def _get_openai_embeddings(texts: List[str], *, model: str = "openai/text-embedding-3-large", batch_size: int = 100, max_workers: int = 64) -> List[List[float]]:
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


# -----------------------------------------------------------------------------
# Generic embedding helper
# -----------------------------------------------------------------------------

def _get_embeddings(texts: List[str], embedding_model: str, verbose: bool = False, use_gpu: Optional[bool] = None) -> List[List[float]]:
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

# -----------------------------------------------------------------------------
# LLM-based cluster summarisation helpers
# -----------------------------------------------------------------------------

# _get_llm_cluster_summary function removed - replaced with parallel generate_cluster_summaries


def _clean_list_item(text: str) -> str:
    """
    Clean up numbered or bulleted list items to extract just the content.
    
    Handles formats like:
    - "1. Item name" -> "Item name"
    - "• Item name" -> "Item name" 
    - "- Item name" -> "Item name"
    - "* Item name" -> "Item name"
    - "a) Item name" -> "Item name"
    - "i. Item name" -> "Item name"
    """
    import re
    
    # Remove common list prefixes
    patterns = [
        r'^\s*\d+\.\s*',           # "1. ", "10. ", etc.
        r'^\s*\d+\)\s*',           # "1) ", "10) ", etc.
        r'^\s*[a-zA-Z]\.\s*',      # "a. ", "A. ", etc.
        r'^\s*[a-zA-Z]\)\s*',      # "a) ", "A) ", etc.
        r'^\s*[ivxlc]+\.\s*',      # Roman numerals "i. ", "iv. ", etc.
        r'^\s*[IVXLC]+\.\s*',      # "I. ", "IV. ", etc.
        r'^\s*[•·◦▪▫‣⁃]\s*',       # Bullet characters
        r'^\s*[-+]\s*',            # Dash, plus bullets
        r'^\s*\*(?!\*)\s*',        # Single asterisk bullet (not double ** for bold)
        r'^\s*[→⟶]\s*',            # Arrow bullets
    ]
    
    cleaned = text.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned.strip()


def generate_coarse_labels(
    cluster_names: List[str],
    max_coarse_clusters: int,
    *,
    systems_prompt: str = deduplication_clustering_systems_prompt,
    model: str = "gpt-4.1",
    verbose: bool = True,
) -> List[str]:
    """Return a cleaned list of coarse-grained labels created by an LLM.

    This function is *pure* w.r.t. its inputs: it never mutates global
    state other than consulting / writing to the on-disk cache.
    """
    if verbose:
        logger.debug(f"cluster_names = {cluster_names}")
    
    # Handle non-string cluster names safely
    valid_fine_names = []
    for n in cluster_names:
        if not isinstance(n, str):
            logger.warning(f"Non-string cluster name found: {n} (type: {type(n)})")
            # Convert to string if it's not already
            n = str(n)
        
        if not (n == "Outliers" or n.startswith("Outliers - ")):
            valid_fine_names.append(n)
    if max_coarse_clusters and len(valid_fine_names) > max_coarse_clusters:
        systems_prompt = systems_prompt.format(max_coarse_clusters=max_coarse_clusters)

    if not valid_fine_names:
        return ["Outliers"]

    # If the list is already small, just return it unchanged.
    if max_coarse_clusters and len(valid_fine_names) <= max_coarse_clusters:
        return valid_fine_names

    user_prompt = f"Fine-grained properties:\n\n" + "\n".join(valid_fine_names)

    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": systems_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 2000,
    }

    if verbose:
        logger.debug(f"🔍 Calling LLM for cluster label generation:")
        logger.debug(f"  Model: {model}")
        logger.debug(f"  System prompt length: {len(systems_prompt)} chars")
        logger.debug(f"  User prompt length: {len(user_prompt)} chars")
        logger.debug(f"  Valid fine names count: {len(valid_fine_names)}")
        logger.debug(f"  First 3 cluster names: {valid_fine_names[:3]}")

    # Try cache first
    cached = _cache.get_completion(request_data) if _cache else None
    if cached is not None:
        content = cached["choices"][0]["message"]["content"]
        # Validate cached content - bypass cache if empty/invalid
        if not content:
            logger.warning(f"⚠️ Invalid cached response (empty content)! Bypassing cache and retrying...")
            logger.warning(f"  Model: {model}")
            cached = None  # Force bypass cache and make fresh call
        else:
            if verbose:
                logger.debug(f"📦 Cache hit for cluster label generation! Content length: {len(content)}")

    if cached is None:
        try:
            resp = litellm.completion(**request_data, caching=False)

            # Extract content with validation
            if not resp or not resp.choices or not resp.choices[0].message:
                logger.error(f"❌ LLM returned malformed response!")
                logger.error(f"  Model: {model}")
                logger.error(f"  Response: {resp}")
                raise ValueError(f"LLM returned malformed response structure for model {model}")

            content = resp.choices[0].message.content

            # Validate response immediately - if empty/None, return fallback
            if content is None or not content:
                logger.warning(f"⚠️ LLM returned empty content!")
                logger.warning(f"  Model: {model}")
                logger.warning(f"  This usually means invalid model name or API error")
                logger.warning(f"  Falling back to input cluster names")
                # Don't cache empty responses, return input as fallback
                return valid_fine_names if valid_fine_names else ["Outliers"]

            if verbose:
                logger.debug(f"✅ LLM call succeeded, response length: {len(content)} chars")

            # Only cache valid (non-empty) responses
            if _cache:
                _cache.set_completion(request_data, {
                    "choices": [{"message": {"content": content}}]
                })
        except Exception as e:
            logger.warning(f"⚠️ LLM call failed with error: {e}")
            logger.warning(f"  Model: {model}")
            logger.warning(f"  Falling back to input cluster names")
            # Don't cache errors, return fallback
            return valid_fine_names if valid_fine_names else ["Outliers"]

    # Clean and split response into individual labels
    raw_names = [line.strip() for line in content.split("\n") if line.strip()]
    coarse_labels = [_clean_list_item(name) for name in raw_names if _clean_list_item(name)]

    if verbose:
        logger.info("Generated cluster labels:")
        for i, lbl in enumerate(coarse_labels):
            logger.info(f"  {i}: {lbl}")

    # Validate that we got results after cleaning
    if not coarse_labels:
        logger.warning(f"⚠️ generate_coarse_labels produced empty list after cleaning!")
        logger.warning(f"  LLM raw response: {content[:200]}")
        logger.warning(f"  Raw names after split: {raw_names[:5]}")
        logger.warning(f"  Falling back to input cluster names")
        # Fallback to input cluster names if parsing failed
        return valid_fine_names if valid_fine_names else ["Outliers"]

    return coarse_labels


async def assign_fine_to_coarse(
    cluster_names: List[str],
    coarse_cluster_names: List[str],
    *,
    model: str = "gpt-4.1-mini",
    strategy: str = "llm",
    verbose: bool = True,
    max_workers: int = 64,
) -> Dict[str, str]:
    """Assign each fine cluster name to one of the coarse cluster names.

    Parameters
    ----------
    strategy : "llm" | "embedding"
        • "llm" – use chat-based matching (async, relies on litellm).
        • "embedding" – cosine-similarity in embedding space (fast, no chat calls).
    """
    # Debug: Check for empty coarse_cluster_names
    if not coarse_cluster_names:
        logger.warning(f"⚠️ assign_fine_to_coarse called with empty coarse_cluster_names!")
        logger.warning(f"  cluster_names count: {len(cluster_names)}")
        logger.warning(f"  cluster_names: {cluster_names[:10]}")  # Show first 10
        # Return all as Outliers since there are no coarse clusters to assign to
        return {name: "Outliers" for name in cluster_names}

    if strategy == "embedding":
        return embedding_match(cluster_names, coarse_cluster_names)
    elif strategy == "llm":
        return await llm_match(cluster_names, coarse_cluster_names, max_workers=max_workers, model=model)
    else:
        raise ValueError(f"Unknown assignment strategy: {strategy}")


async def llm_coarse_cluster_with_centers(
    cluster_names: List[str],
    max_coarse_clusters: int,
    verbose: bool = True,
    model: str = "gpt-4.1",
    cluster_assignment_model: str = "gpt-4.1-mini",
    systems_prompt: str = deduplication_clustering_systems_prompt,
) -> Tuple[Dict[str, str], List[str]]:
    """High-level convenience wrapper that returns both mapping and centres."""
    valid_fine_names = [n for n in cluster_names if not (n == "Outliers" or n.startswith("Outliers - "))]
    if not valid_fine_names:
        return {}, ["Outliers"]

    coarse_labels = generate_coarse_labels(
        valid_fine_names,
        max_coarse_clusters=max_coarse_clusters,
        systems_prompt=systems_prompt,
        model=model,
        verbose=verbose,
    )

    fine_to_coarse = await assign_fine_to_coarse(
        valid_fine_names,
        coarse_labels,
        model=cluster_assignment_model,
        strategy="llm",
        verbose=verbose,
    )

    return fine_to_coarse, coarse_labels

def embedding_match(cluster_names, coarse_cluster_names):
    """Match fine-grained cluster names to coarse-grained cluster names using embeddings."""
    fine_emb = _get_embeddings(cluster_names, "openai", verbose=False)
    coarse_emb = _get_embeddings(coarse_cluster_names, "openai", verbose=False)
    fine_emb = np.array(fine_emb) / np.linalg.norm(fine_emb, axis=1, keepdims=True)
    coarse_emb = np.array(coarse_emb) / np.linalg.norm(coarse_emb, axis=1, keepdims=True)
    sim = fine_emb @ coarse_emb.T
    fine_to_coarse = np.argmax(sim, axis=1)
    # turn into dictionary of {fine_name: coarse_name}
    return {cluster_names[i]: coarse_cluster_names[j] for i, j in enumerate(fine_to_coarse)}

# Duplicate imports removed - already imported at top of file

def match_label_names(label_name, label_options):
    """See if label_name is in label_options, not taking into account capitalization or whitespace or punctuation. Return original option if found, otherwise return None"""
    if "outliers" in label_name.lower():
        # Check if it's a group-specific outlier label
        if label_name.startswith("Outliers - "):
            return label_name  # Return the full group-specific label
        return "Outliers"
    label_name_clean = label_name.lower().strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("'", "").replace("\"", "").replace("`", "").replace("~", "").replace("*", "").replace("+", "").replace("-", "").replace("_", "").replace("=", "").replace("|", "").replace("\\", "").replace("/", "").replace("<", "").replace(">", "").replace(" ", "")
    for option in label_options:
        option_clean = option.lower().strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("'", "").replace("\"", "").replace("`", "").replace("~", "").replace("*", "").replace("+", "").replace("-", "").replace("_", "").replace("=", "").replace("|", "").replace("\\", "").replace("/", "").replace("<", "").replace(">", "").replace(" ", "")
        if label_name_clean in option_clean:
            return option
    return None

async def llm_match(cluster_names, coarse_cluster_names, max_workers=16, model="gpt-4.1-mini"):
    """Match fine-grained cluster names to coarse-grained cluster names using an LLM with parallel processing."""
    coarse_names_text = "\n".join(coarse_cluster_names)
    
    system_prompt = "You are a machine learning expert specializing in the behavior of large language models. Given the following coarse grained properties of model behavior, match the given fine grained property to the coarse grained property that it most closely resembles. Respond with the name of the coarse grained property that the fine grained property most resembles. If it is okay if the match is not perfect, just respond with the property that is most similar. If the fine grained property has absolutely no relation to any of the coarse grained properties, respond with 'Outliers'. Do NOT include anything but the name of the coarse grained property in your response."
    
    # Build user messages for each fine cluster name
    messages = []
    for fine_name in cluster_names:
        bullet_points = "\n".join([f"- {name}" for name in coarse_cluster_names])
        user_prompt = (
            f"Coarse grained properties:\n\n{bullet_points}\n\n"
            f"Fine grained property: {fine_name}\n\n"
            "Closest coarse grained property to the fine grained property:"
        )
        messages.append(user_prompt)
    
    # Use parallel processing with built-in caching and retries
    responses = await parallel_completions_async(
        messages,
        model=model,
        system_prompt=system_prompt,
        max_workers=max_workers,
        show_progress=True,
        progress_desc="Matching fine to coarse clusters"
    )
    
    # Build result mapping with validation
    fine_to_coarse = {}
    for fine_name, response in zip(cluster_names, responses):
        # Handle None responses (failed LLM calls)
        if response is None:
            logger.warning(f"LLM call failed for fine cluster '{fine_name}', assigning to 'Outliers'")
            fine_to_coarse[fine_name] = "Outliers"
            continue

        coarse_label = response.strip()
        coarse_label = match_label_names(coarse_label, coarse_cluster_names)

        # Validate the response
        if (coarse_label in coarse_cluster_names) or (coarse_label == "Outliers"):
            fine_to_coarse[fine_name] = coarse_label
        else:
            if coarse_label is None:
                logger.warning(f"Could not match label '{response.strip()}' for fine name '{fine_name}', assigning to 'Outliers'")
            else:
                logger.warning(f"Invalid coarse label '{coarse_label}' for fine name '{fine_name}', assigning to 'Outliers'")
            fine_to_coarse[fine_name] = "Outliers"

    return fine_to_coarse

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


# -------------------------------------------------------------------------
# Legacy Litellm-based helpers (kept for reference – not used by pipeline)
# -------------------------------------------------------------------------
def _get_openai_embeddings_batch_litellm(batch, retries=3, sleep_time=2.0):
    """Fetch embeddings for one batch with retry logic (uses LiteLLM cache)."""
    for attempt in range(retries):
        try:
            resp = litellm.embedding(
                model="text-embedding-3-large",
                input=batch,
                caching=True
            )
            embeddings = [item["embedding"] for item in resp["data"]]
            return embeddings
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"[retry {attempt + 1}/{retries}] {e}. Sleeping {sleep_time}s.")
            time.sleep(sleep_time)


# NOTE: renamed to avoid overriding the DiskCache-cached version defined earlier
def _get_openai_embeddings_litellm(texts, batch_size=100, max_workers=16):
    """Get embeddings using OpenAI API (LiteLLM cache)."""

    if not texts:
        return []

    # Pre-allocate output list to preserve order
    embeddings = [None] * len(texts)

    # Prepare batches with their position in the output list
    batches = [
        (start, texts[start:start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_span = {
            executor.submit(_get_openai_embeddings_batch_litellm, batch_texts): (start, len(batch_texts))
            for start, batch_texts in batches
        }

        for fut in concurrent.futures.as_completed(future_to_span):
            start, length = future_to_span[fut]
            batch_embeddings = fut.result()   # exceptions propagate – fail fast
            embeddings[start:start + length] = batch_embeddings

    # Final sanity-check
    if any(e is None for e in embeddings):
        raise RuntimeError("Some embeddings are missing – check logs for errors.")

    return embeddings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_clustered_results(parquet_path):
    """Load previously clustered results from parquet file."""
    df = pd.read_parquet(parquet_path)
    
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower() or 'topic' in col.lower()]
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    
    if cluster_cols:
        logger.info(f"Cluster columns: {cluster_cols}")
    if embedding_cols:
        logger.info(f"Embedding columns: {embedding_cols}")
    
    return df


def save_clustered_results(df, base_filename, include_embeddings=True, config=None, output_dir=None):
    """Save clustered results in multiple formats and optionally log to wandb.
    
    Args:
        df: DataFrame with clustered results
        base_filename: Base name for output files
        include_embeddings: Whether to include embeddings in output
        config: ClusterConfig object for wandb logging
        output_dir: Output directory (if None, uses cluster_results/{base_filename})
    """

    # Determine output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_dir = output_dir
    else:
        os.makedirs(f"cluster_results/{base_filename}", exist_ok=True)
        save_dir = f"cluster_results/{base_filename}"
    
    # Handle JSON serialization - preserve dictionaries for score columns
    df = df.copy()
    score_columns = [col for col in df.columns if 'score' in col.lower()]
    
    # Debug print to see what score columns we're finding
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Available columns: {list(df.columns)}")
        logger.debug(f"Detected score columns: {score_columns}")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Preserve dictionary columns (like scores) - don't convert to string
            if col in score_columns:
                # Keep as-is for JSON serialization - pandas.to_json can handle dicts
                continue
            else:
                # Convert other object columns to strings for safety
                df[col] = df[col].astype(str)
    
    # 1. Save clustered results as JSON (preserves all data structures)
    df.to_json(f"{save_dir}/clustered_results.jsonl", orient='records', lines=True)
    logger.info(f"Saved clustered results (JSON): {save_dir}/clustered_results.jsonl")
    
    # 2. Save embeddings separately if they exist
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    if embedding_cols and include_embeddings:
        # Create embeddings-only DataFrame
        embedding_df = df[embedding_cols].copy()
        
        # Add key columns for reference
        key_cols = ['property_description', 'question_id', 'model', 'property_description_cluster_label']
        for col in key_cols:
            if col in df.columns:
                embedding_df[col] = df[col]
        
        # Save embeddings as parquet (more efficient for large arrays)
        embeddings_path = os.path.join(save_dir, "embeddings.parquet")
        embedding_df.to_parquet(embeddings_path, compression='snappy')
        logger.info(f"Saved embeddings: {embeddings_path}")
        
        # Also save as JSON for compatibility
        embedding_df.to_json(f"{save_dir}/embeddings.jsonl", orient='records', lines=True, force_ascii=False)
        logger.info(f"Saved embeddings (JSON): {save_dir}/embeddings.jsonl")
    
    # 3. Save lightweight version without embeddings
    df_light = df.drop(columns=embedding_cols) if embedding_cols else df
    
    # Save lightweight as json
    df_light.to_json(f"{save_dir}/clustered_results_lightweight.jsonl", orient='records', lines=True)
    logger.info(f"Saved lightweight results (JSON): {save_dir}/clustered_results_lightweight.jsonl")
    
    # 4. Create and save summary table
    summary_table = create_summary_table(df_light, config)
    summary_table.to_json(f"{save_dir}/summary_table.jsonl", orient='records', lines=True)
    logger.info(f"Saved summary table: {save_dir}/summary_table.jsonl")

    # 6. Log to wandb if enabled
    if config and config.use_wandb:
        log_results_to_wandb(df_light, f"{save_dir}/clustered_results_lightweight.jsonl", base_filename, config)
        logger.info(f"Logged results to wandb")
    
    return {
        'clustered_json': f"{save_dir}/clustered_results.jsonl",
        'embeddings_parquet': f"{save_dir}/embeddings.parquet" if embedding_cols and include_embeddings else None,
        'summary_table': f"{save_dir}/summary_table.jsonl"
    }


def log_results_to_wandb(df_light, light_json_path, base_filename, config):
    """Log clustering results to wandb."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb is required for logging but is not installed. Install with: pip install wandb")
        return
    
    if not wandb.run:
        logger.warning("⚠️ wandb not initialized, skipping logging")
        return
    
    logger.info("📊 Logging results to wandb...")
    
    # Log the lightweight CSV file
    artifact = wandb.Artifact(
        name=f"{base_filename}_clustered_data",
        type="clustered_dataset",
        description=f"Clustered dataset without embeddings - {base_filename}"
    )
    artifact.add_file(light_json_path)
    wandb.log_artifact(artifact)
    
    # Log the actual clustering results as a table
    # Find the original column that was clustered
    original_col = None
    for col in df_light.columns:
        if not any(suffix in col for suffix in ['_cluster', '_embedding']):
            # This is likely the original column
            original_col = col
            break
    
    if original_col:
        # Create a table with the key clustering results
        cluster_cols = [col for col in df_light.columns if 'cluster' in col.lower()]
        table_cols = [original_col] + cluster_cols
        
        # Sample the data if it's too large (wandb has limits)
        sample_size = min(100, len(df_light))
        if len(df_light) > sample_size:
            df_sample = df_light[table_cols].sample(n=sample_size, random_state=42)
            logger.info(f"📋 Logging sample of {sample_size} rows (out of {len(df_light)} total)")
        else:
            df_sample = df_light[table_cols]
            logger.info(f"📋 Logging all {len(df_sample)} rows")
        
        # Convert to string to handle any non-serializable data
        df_sample_str = df_sample.astype(str)
        wandb.log({f"{base_filename}_clustering_results": wandb.Table(dataframe=df_sample_str)})
    
    # Calculate clustering metrics
    cluster_cols = [col for col in df_light.columns if 'cluster_id' in col.lower()]
    metrics = {"clustering_dataset_size": len(df_light)}
    
    for col in cluster_cols:
        cluster_ids = df_light[col].values
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_outliers = list(cluster_ids).count(-1)
        
        level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
        metrics[f"clustering_{level}_clusters"] = n_clusters
        metrics[f"clustering_{level}_outliers"] = n_outliers
        metrics[f"clustering_{level}_outlier_rate"] = n_outliers / len(cluster_ids) if len(cluster_ids) > 0 else 0
        
        # Calculate cluster size distribution
        cluster_sizes = [list(cluster_ids).count(cid) for cid in set(cluster_ids) if cid >= 0]
        if cluster_sizes:
            metrics[f"clustering_{level}_avg_cluster_size"] = np.mean(cluster_sizes)
            metrics[f"clustering_{level}_min_cluster_size"] = min(cluster_sizes)
            metrics[f"clustering_{level}_max_cluster_size"] = max(cluster_sizes)
    
    # Log clustering configuration
    config_dict = {
        "clustering_min_cluster_size": config.min_cluster_size,
        "clustering_embedding_model": config.embedding_model,
        "clustering_assign_outliers": config.assign_outliers,
        "clustering_disable_dim_reduction": config.disable_dim_reduction,
        "clustering_min_samples": config.min_samples,
        "clustering_cluster_selection_epsilon": config.cluster_selection_epsilon
    }
    
    # Log all metrics as summary metrics (not regular metrics)
    # Note: This function doesn't have access to WandbMixin, so we'll log directly to wandb.run.summary
    all_metrics = {**metrics, **config_dict}
    for key, value in all_metrics.items():
        wandb.run.summary[key] = value
    
    logger.info(f"✅ Logged clustering results to wandb")
    logger.info(f"   - Dataset artifact: {base_filename}_clustered_data")
    logger.info(f"   - Clustering results table: {base_filename}_clustering_results")
    logger.info(f"   - Summary metrics: {list(all_metrics.keys())}")


def initialize_wandb(config, method_name, input_file):
    """Initialize wandb logging if enabled."""
    if not config.use_wandb:
        return
    
    try:
        import wandb
    except ImportError:
        logger.error("wandb is required for logging but is not installed. Install with: pip install wandb")
        return
    
    logger.info("🔧 Initializing wandb...")
    
    # Create run name if not provided
    run_name = config.wandb_run_name
    if not run_name:
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        run_name = f"{input_basename}_{method_name}_clustering"
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project or "hierarchical_clustering",
        entity=config.wandb_entity,
        name=run_name,
        config={
            "method": method_name,
            "input_file": input_file,
            "min_cluster_size": config.min_cluster_size,
            "embedding_model": config.embedding_model,
            "assign_outliers": config.assign_outliers,
            "disable_dim_reduction": config.disable_dim_reduction,
            "min_samples": config.min_samples,
            "cluster_selection_epsilon": config.cluster_selection_epsilon
        }
    )
    
    logger.info(f"✅ Initialized wandb run: {run_name}")


def load_precomputed_embeddings(embeddings_path, verbose=True):
    """Load precomputed embeddings from various file formats."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    if verbose:
        logger.info(f"Loading precomputed embeddings from {embeddings_path}...")
    
    if embeddings_path.endswith('.pkl'):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                # Check if it's a cache file with 'embeddings' key
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                    if verbose:
                        logger.info(f"Loaded {len(embeddings)} embeddings from cache file")
                else:
                    # Assume it's a direct mapping of values to embeddings
                    embeddings = data
                    if verbose:
                        logger.info(f"Loaded {len(embeddings)} embeddings from mapping file")
            else:
                # Assume it's a direct array/list of embeddings
                embeddings = data
                if verbose:
                    logger.info(f"Loaded {len(embeddings)} embeddings from array file")
    
    elif embeddings_path.endswith('.npy'):
        embeddings = np.load(embeddings_path)
        if verbose:
            logger.info(f"Loaded {len(embeddings)} embeddings from numpy file")
    
    elif embeddings_path.endswith('.parquet'):
        # Load from parquet file with embedding column
        if verbose:
            logger.info("Loading parquet file...")
        df = pd.read_parquet(embeddings_path)
        
        embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
        if not embedding_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")
        
        embedding_col = embedding_cols[0]  # Use first embedding column
        
        # Find the column that was clustered (should be the base name of embedding column)
        base_col = embedding_col.replace('_embedding', '')
        if base_col not in df.columns:
            # Try to find any text column that might be the source
            text_cols = [col for col in df.columns if col not in embedding_cols and 
                        df[col].dtype == 'object']
            if text_cols:
                base_col = text_cols[0]
                if verbose:
                    logger.info(f"Using column '{base_col}' as source column")
            else:
                raise ValueError(f"Cannot find source text column in {embeddings_path}")
        
        if verbose:
            logger.info(f"Creating value-to-embedding mapping from column '{base_col}'...")
        
        # Create mapping from values to embedding
        embeddings = {}
        for _, row in df.iterrows():
            value = str(row[base_col])
            embedding = row[embedding_col]
            embeddings[value] = embedding
        
        if verbose:
            logger.info(f"Loaded {len(embeddings)} embeddings from parquet file (column: {embedding_col})")
    
    else:
        raise ValueError(f"Unsupported file format: {embeddings_path}. Supported: .pkl, .npy, .parquet")
    
    return embeddings

def create_summary_table(df, config=None, **kwargs):
    labels = df.property_description_cluster_label.value_counts()
    cols = [
        'property_description',
        ]
    existing_cols = [c for c in cols if c in df.columns]
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results
    results = []
    for label in labels.index:
        df_label = df[df.property_description_cluster_label == label].drop_duplicates(subset=['question_id', 'model'])
        global_model_counts = df.model.value_counts()
        examples = {}
        for model in df_label.model.unique():
            examples[model] = df_label[df_label.model == model].head(3)[existing_cols].to_dict(orient='records')
        model_percent_global = {
            k: v / global_model_counts[k] for k, v in df_label.model.value_counts().to_dict().items()
        }
        res = {
            "label": label,
            "count": df_label.shape[0],
            "percent": df_label.shape[0] / len(df),
            "model_counts": df_label.model.value_counts().to_dict(),
            "model_percent_global": model_percent_global,
            "model_local_proportions": {
                k: v / np.median(list(model_percent_global.values())) for k, v in model_percent_global.items()
            },
            "examples": examples,
        }
        results.append(res)

    results = pd.DataFrame(results)
    return results