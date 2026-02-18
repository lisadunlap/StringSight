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
            max_tokens=2048
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
            max_tokens=2048
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
            max_tokens=2048
        )

        return response.choices[0].message.content.strip()
    
    def generate_role_specific(
        self,
        role: str,
        expanded_task_description: str,
        sample_conversations: list[dict] | None = None,
        model: str = "gpt-4o-mini"
    ) -> str:
        """Generate role-specific extraction prompt using dynamic discovery.
        
        This method analyzes sample conversations to discover patterns specific
        to a particular role and generates a tailored extraction prompt.
        
        Args:
            role: Role to analyze (e.g., "user", "assistant", "tool", etc.)
            expanded_task_description: Expanded task description
            sample_conversations: Optional sample conversations to analyze
            model: LLM model for generation
            
        Returns:
            Generated role-specific system prompt
        """
        # Build cache key
        cache_key = self._build_role_cache_key(
            role, expanded_task_description, model
        )
        
        # Check cache
        cached = self.cache.get_completion(cache_key)
        if cached is not None:
            return str(cached)
        
        # Import role templates for fallback
        from ...extraction.role_analysis import get_role_analysis_prompt
        
        # If we have sample conversations, analyze them for patterns
        pattern_analysis = ""
        if sample_conversations:
            pattern_analysis = self._analyze_role_patterns(
                role, sample_conversations, model
            )
        
        # Generate role-specific prompt
        prompt = self._generate_role_prompt(
            role, expanded_task_description, pattern_analysis, model
        )
        
        # Cache and return
        self.cache.set_completion(cache_key, prompt)
        return prompt
    
    def _analyze_role_patterns(
        self,
        role: str,
        sample_conversations: list[dict],
        model: str
    ) -> str:
        """Analyze sample conversations to identify role-specific patterns.
        
        Args:
            role: Role to analyze
            sample_conversations: Sample conversations in OAI format
            model: LLM model
            
        Returns:
            Analysis of patterns specific to this role
        """
        # Extract messages from the specified role
        role_messages = []
        for conv in sample_conversations[:10]:  # Limit to 10 samples
            if isinstance(conv, list):
                for msg in conv:
                    if isinstance(msg, dict) and msg.get("role") == role:
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            role_messages.append(content)
        
        if not role_messages:
            return f"No sample messages found for role '{role}'."
        
        # Limit total text for analysis
        sample_text = "\n---\n".join(role_messages[:20])
        if len(sample_text) > 10000:
            sample_text = sample_text[:10000] + "..."
        
        # Ask LLM to identify patterns
        analysis_prompt = f"""Analyze these sample messages from the "{role}" role and identify key patterns, characteristics, and behaviors that would be useful to extract as properties.

Sample messages:
{sample_text}

Provide a concise analysis covering:
1. Common patterns or behaviors
2. Distinctive characteristics
3. Areas of potential interest for quality analysis
4. Specific aspects that differ from other roles

Keep your analysis under 300 words."""
        
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert conversation analyst."
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to analyze role patterns: {e}")
            return ""
    
    def _generate_role_prompt(
        self,
        role: str,
        task_description: str,
        pattern_analysis: str,
        model: str
    ) -> str:
        """Generate a custom extraction prompt for a specific role.
        
        Args:
            role: Role to analyze
            task_description: Task description
            pattern_analysis: Analysis of role-specific patterns
            model: LLM model
            
        Returns:
            Generated role-specific extraction prompt
        """
        pattern_section = ""
        if pattern_analysis:
            pattern_section = f"Patterns identified in sample data:\n{pattern_analysis}\n\n"
        
        generation_prompt = f"""Create a system prompt for extracting behavioral properties from messages with role="{role}" in conversations.

Task context:
{task_description}

{pattern_section}The prompt should:
1. Focus on extracting properties specific to the "{role}" role
2. Use the JSON format with fields: behavior_type, property_description, category, evidence, reason, contains_errors, unexpected_behavior, role
3. Include clear instructions for what to look for in "{role}" messages
4. Emphasize that evidence must come ONLY from messages with role="{role}"
5. Define appropriate behavior_type categories for this role
6. Set "role": "{role}" in all extracted properties

Generate a complete, ready-to-use system prompt (no placeholders)."""
        
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating LLM analysis prompts. Generate clear, specific prompts."
                    },
                    {"role": "user", "content": generation_prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate role prompt: {e}")
            # Fall back to template
            from ...extraction.role_analysis import get_role_analysis_prompt
            return get_role_analysis_prompt(role, task_description)
    
    def _build_role_cache_key(
        self,
        role: str,
        task_description: str,
        model: str
    ) -> CacheKeyBuilder:
        """Build cache key for role-specific prompt generation.
        
        Args:
            role: Role being analyzed
            task_description: Task description
            model: LLM model
            
        Returns:
            CacheKeyBuilder for use with UnifiedCache
        """
        cache_data = {
            "type": "role_specific_prompt_generation",
            "role": role,
            "task_description": task_description,
            "model": model,
            "version": "1.0"
        }
        return CacheKeyBuilder(cache_data)

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
                "analysis_process": ANALYSIS_PROCESS_TEMPLATE,
            }).get_key(),
        }
        return CacheKeyBuilder(cache_data)
