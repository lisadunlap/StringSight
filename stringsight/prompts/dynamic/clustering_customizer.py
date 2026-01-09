"""
Clustering prompt customization with task-specific context.

This module customizes clustering prompts by adding task-specific context.
"""

import litellm
import logging
from typing import Optional

from ...core.caching import UnifiedCache, CacheKeyBuilder

logger = logging.getLogger(__name__)


class ClusteringPromptCustomizer:
    """Customizes clustering prompts with task-specific context using LLM."""

    def __init__(self, cache: UnifiedCache | None = None):
        """Initialize the ClusteringPromptCustomizer.

        Args:
            cache: Optional cache instance for storing customized prompts.
        """
        self.cache = cache or UnifiedCache()

    def customize_clustering_prompt(
        self,
        task_description: str,
        model: str = "gpt-4.1"
    ) -> str:
        """Generate task-aware clustering prompt using LLM.

        Args:
            task_description: Expanded task description.
            model: LLM model to use for customization.

        Returns:
            Customized clustering prompt.
        """
        from ..clustering.prompts import clustering_systems_prompt
        return self._generate_task_aware_prompt(
            base_prompt=clustering_systems_prompt,
            task_description=task_description,
            prompt_type="clustering",
            model=model
        )

    def customize_deduplication_prompt(
        self,
        task_description: str,
        model: str = "gpt-4.1"
    ) -> str:
        """Generate task-aware deduplication prompt using LLM.

        Args:
            task_description: Expanded task description.
            model: LLM model to use for customization.

        Returns:
            Deduplication prompt with task-specific context.
        """
        from ..clustering.prompts import deduplication_clustering_systems_prompt
        return self._generate_task_aware_prompt(
            base_prompt=deduplication_clustering_systems_prompt,
            task_description=task_description,
            prompt_type="deduplication",
            model=model
        )

    def customize_outlier_prompt(
        self,
        task_description: str,
        model: str = "gpt-4.1"
    ) -> str:
        """Generate task-aware outlier clustering prompt using LLM.

        Args:
            task_description: Expanded task description.
            model: LLM model to use for customization.

        Returns:
            Outlier clustering prompt with task-specific context.
        """
        from ..clustering.prompts import outlier_clustering_systems_prompt
        return self._generate_task_aware_prompt(
            base_prompt=outlier_clustering_systems_prompt,
            task_description=task_description,
            prompt_type="outlier_clustering",
            model=model
        )

    def _generate_task_aware_prompt(
        self,
        base_prompt: str,
        task_description: str,
        prompt_type: str,
        model: str
    ) -> str:
        """Use LLM to naturally integrate task description into base prompt.

        Args:
            base_prompt: The base clustering/deduplication/outlier prompt.
            task_description: Expanded task description.
            prompt_type: Type of prompt ("clustering", "deduplication", "outlier_clustering").
            model: LLM model to use.

        Returns:
            Task-aware prompt with naturally integrated context.
        """
        from .meta_prompts import CLUSTERING_CUSTOMIZATION_SYSTEM_PROMPT, CLUSTERING_CUSTOMIZATION_TEMPLATE

        logger.info(f"Generating task-aware {prompt_type} prompt using LLM")

        # Deterministic cache key: same inputs -> same cached output.
        meta_prompt_hash = CacheKeyBuilder({
            "system": CLUSTERING_CUSTOMIZATION_SYSTEM_PROMPT,
            "template": CLUSTERING_CUSTOMIZATION_TEMPLATE,
        }).get_key()
        cache_key = CacheKeyBuilder({
            "type": "clustering_prompt_customization",
            "prompt_type": prompt_type,
            "model": model,
            "base_prompt": base_prompt,
            "task_description": task_description,
            "meta_prompt_hash": meta_prompt_hash,
            "version": "1.0",
        })
        cached = self.cache.get_completion(cache_key)
        if cached is not None:
            # Cache may return a raw string (we store strings for these completions).
            return str(cached)

        customization_prompt = CLUSTERING_CUSTOMIZATION_TEMPLATE.format(
            base_prompt=base_prompt,
            task_description=task_description,
            prompt_type=prompt_type
        )

        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": CLUSTERING_CUSTOMIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": customization_prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )

            customized_prompt = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated task-aware {prompt_type} prompt")
            self.cache.set_completion(cache_key, customized_prompt)
            return customized_prompt
        except Exception as e:
            logger.error(f"Failed to customize {prompt_type} prompt: {e}. Falling back to base prompt.")
            return base_prompt
