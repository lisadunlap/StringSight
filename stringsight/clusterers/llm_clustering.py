"""
llm_clustering.py

LLM-based clustering utilities for generating and matching cluster labels.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple
import numpy as np
import litellm  # type: ignore

from stringsight.prompts.clustering.prompts import deduplication_clustering_systems_prompt
from stringsight.logging_config import get_logger
from stringsight.constants import DEFAULT_MAX_WORKERS
from ..core.llm_utils import parallel_completions_async
from ..core.caching import UnifiedCache
from .embeddings import _get_embeddings

logger = get_logger(__name__)

# Shared cache
_cache = UnifiedCache()


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
    max_coarse_clusters: int | None,
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
    max_workers: int = DEFAULT_MAX_WORKERS,
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


async def llm_match(cluster_names, coarse_cluster_names, max_workers=DEFAULT_MAX_WORKERS, model="gpt-4.1-mini"):
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
