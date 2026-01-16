"""
Discovery prompt generation using meta-prompting.

This module generates custom discovery prompt sections tailored to specific tasks.
"""

import litellm
import logging
from typing import Dict, Any, cast
from concurrent.futures import ThreadPoolExecutor
from ...core.caching import UnifiedCache, CacheKeyBuilder

logger = logging.getLogger(__name__)


class DiscoveryPromptGenerator:
    """Generates custom discovery prompt sections using LLM."""

    def __init__(self, cache: UnifiedCache | None = None):
        """Initialize the DiscoveryPromptGenerator.

        Args:
            cache: Cache instance for storing generated prompts.
        """
        self.cache = cache or UnifiedCache()

    def generate(
        self,
        expanded_task_description: str,
        method: str,
        base_config: Dict[str, str],
        model: str = "gpt-4.1"
    ) -> Dict[str, str]:
        """Generate custom config dict with task-specific sections.

        Args:
            expanded_task_description: Expanded task description from TaskExpander.
            method: "single_model", "side_by_side", "agent_single_model", or "agent_sbs".
            base_config: Base config dict (e.g., single_model_config).
            model: LLM model for generation.

        Returns:
            Custom config dict with generated sections.
        """
        # Build cache key
        cache_key = self._build_cache_key(expanded_task_description, method, model)

        # Check cache
        cached = self.cache.get_completion(cache_key)
        if cached is not None:
            return cast(Dict[str, str], cached)

        # Generate custom sections in parallel
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all three sections in parallel
                future_intro = executor.submit(
                    self._generate_intro_task,
                    expanded_task_description, method, model
                )
                future_goal = executor.submit(
                    self._generate_goal_instructions,
                    expanded_task_description, method, model
                )
                future_process = executor.submit(
                    self._generate_analysis_process,
                    expanded_task_description, method, model
                )

                # Collect results
                custom_intro = future_intro.result()
                custom_goal = future_goal.result()
                custom_process = future_process.result()
        except Exception as e:
            # Fall back to base config on error
            logger.warning(f"Failed to generate custom prompt sections: {e}. Using base config.")
            return base_config

        # Merge with base config (preserve critical sections)
        custom_config = base_config.copy()
        custom_config["intro_task"] = custom_intro
        custom_config["goal_instructions"] = custom_goal
        custom_config["analysis_process"] = custom_process
        # Keep: json_schema, model_naming_rule, reasoning_suffix from base

        # Cache result
        self.cache.set_completion(cache_key, custom_config)
        return custom_config

    def _generate_intro_task(
        self,
        expanded_description: str,
        method: str,
        model: str
    ) -> str:
        """Generate custom intro_task section.

        Args:
            expanded_description: Expanded task description.
            method: Analysis method.
            model: LLM model.

        Returns:
            Generated intro_task text.
        """
        from .meta_prompts import INTRO_TASK_GENERATION_TEMPLATE

        prompt = INTRO_TASK_GENERATION_TEMPLATE.format(
            task_description=expanded_description,
            method=method
        )

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating LLM analysis prompts. Generate concise, task-specific prompt sections."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def _generate_goal_instructions(
        self,
        expanded_description: str,
        method: str,
        model: str
    ) -> str:
        """Generate custom goal_instructions section.

        Args:
            expanded_description: Expanded task description.
            method: Analysis method.
            model: LLM model.

        Returns:
            Generated goal_instructions text.
        """
        from .meta_prompts import GOAL_INSTRUCTIONS_GENERATION_TEMPLATE

        prompt = GOAL_INSTRUCTIONS_GENERATION_TEMPLATE.format(
            task_description=expanded_description,
            method=method
        )

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating LLM analysis prompts."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def _generate_analysis_process(
        self,
        expanded_description: str,
        method: str,
        model: str
    ) -> str:
        """Generate custom analysis_process section.

        Args:
            expanded_description: Expanded task description.
            method: Analysis method.
            model: LLM model.

        Returns:
            Generated analysis_process text.
        """
        from .meta_prompts import ANALYSIS_PROCESS_GENERATION_TEMPLATE

        prompt = ANALYSIS_PROCESS_GENERATION_TEMPLATE.format(
            task_description=expanded_description,
            method=method
        )

        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating LLM analysis prompts."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    def _build_cache_key(
        self,
        expanded_description: str,
        method: str,
        model: str
    ) -> CacheKeyBuilder:
        """Build cache key for discovery prompt generation.

        Args:
            expanded_description: Expanded task description.
            method: Analysis method.
            model: LLM model.

        Returns:
            CacheKeyBuilder for use with UnifiedCache.
        """
        from .meta_prompts import (
            INTRO_TASK_GENERATION_TEMPLATE,
            GOAL_INSTRUCTIONS_GENERATION_TEMPLATE,
            ANALYSIS_PROCESS_GENERATION_TEMPLATE
        )

        cache_data = {
            "type": "discovery_prompt_generation",
            "expanded_description": expanded_description,
            "method": method,
            "model": model,
            "version": "1.1",  # Bumped to 1.1 for deduplication step
            # Include a deterministic meta-prompt hash for cache invalidation.
            #
            # NOTE: Do NOT use Python's built-in `hash()` here; it is salted per
            # process (PYTHONHASHSEED) and will change across runs, breaking cache
            # stability and making prompt generation less reproducible.
            "meta_prompt_hash": CacheKeyBuilder({
                "intro_task": INTRO_TASK_GENERATION_TEMPLATE,
                "goal_instructions": GOAL_INSTRUCTIONS_GENERATION_TEMPLATE,
                "analysis_process": ANALYSIS_PROCESS_GENERATION_TEMPLATE,
            }).get_key(),
        }
        return CacheKeyBuilder(cache_data)
