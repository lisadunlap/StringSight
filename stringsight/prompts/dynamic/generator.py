"""
Dynamic prompt generation orchestrator.

This module orchestrates the entire dynamic prompt generation pipeline,
coordinating task expansion, discovery prompt generation, verification,
and clustering prompt customization.
"""

import random
import logging
from dataclasses import dataclass
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...core.data_objects import ConversationRecord
from ...core.caching import UnifiedCache
from .task_expander import TaskExpander
from .discovery_generator import DiscoveryPromptGenerator
from .prompt_verifier import PromptVerifier
from .prompt_reflector import PromptReflector
from .clustering_customizer import ClusteringPromptCustomizer

logger = logging.getLogger(__name__)


@dataclass
class DynamicPromptResult:
    """Result of dynamic prompt generation."""
    discovery_prompt: str
    clustering_prompt: str
    dedup_prompt: str
    outlier_prompt: str
    expanded_task_description: str
    verification_passed: bool
    reflection_attempts: int = 0  # Number of reflection attempts made


class DynamicPromptGenerator:
    """Orchestrates dynamic prompt generation pipeline."""

    def __init__(self, cache: UnifiedCache | None = None, seed: int = 42):
        """Initialize the DynamicPromptGenerator.

        Args:
            cache: Optional cache instance (will create one if not provided).
            seed: Random seed for deterministic behavior (default: 42).
        """
        cache = cache or UnifiedCache()
        self.seed = seed
        self.task_expander = TaskExpander(cache, seed=seed)
        self.discovery_generator = DiscoveryPromptGenerator(cache)
        self.prompt_verifier = PromptVerifier()
        self.prompt_reflector = PromptReflector()
        self.clustering_customizer = ClusteringPromptCustomizer()

    def generate_all_prompts(
        self,
        task_description: str,
        conversations: List[ConversationRecord],
        method: str,
        num_samples: int = 5,
        model: str = "gpt-4.1",
        max_reflection_attempts: int = 3,
        skip_verification: bool = True,
        customize_clustering: bool = True
    ) -> DynamicPromptResult:
        """Generate all custom prompts for the pipeline.

        Args:
            task_description: Original task description.
            conversations: All conversations in dataset.
            method: Analysis method ("single_model", "side_by_side", etc.).
            num_samples: Number of conversations to sample for expansion.
            model: LLM model for meta-prompting.
            max_reflection_attempts: Maximum number of reflection attempts to fix failed prompts (default: 3).
            skip_verification: If True, skip verification and use expanded task with base template (default: True).
            customize_clustering: If True, customize clustering prompts for the task (default: True).

        Returns:
            DynamicPromptResult with all generated prompts.
        """
        # Set random seed for deterministic behavior
        random.seed(self.seed)

        # Step 1: Expand task description
        logger.info("Step 1: Expanding task description...")
        expanded_description = self.task_expander.expand(
            task_description=task_description,
            conversations=conversations,
            num_samples=num_samples,
            model=model
        )
        logger.info(f"Task description expanded ({len(expanded_description)} chars)")

        # Get base config for the method
        from ..extraction.universal import (
            single_model_config,
            sbs_config,
            agent_single_model_config,
            agent_sbs_config,
            format_universal_prompt
        )

        config_map = {
            "single_model": single_model_config,
            "side_by_side": sbs_config,
            "agent_single_model": agent_single_model_config,
            "agent_sbs": agent_sbs_config,
        }

        base_config = config_map.get(method, single_model_config)

        if skip_verification:
            # Simple path: Just use expanded task with base template (no verification needed)
            logger.info("Step 2: Using base template with expanded task (skipping custom generation and verification)...")
            discovery_prompt = format_universal_prompt(
                task_description=expanded_description,
                config=base_config
            )
            verification_passed = True  # No verification performed, assume valid
            reflection_attempts = 0
            logger.info(f"Discovery prompt created ({len(discovery_prompt)} chars)")
        else:
            # Complex path: Generate custom prompt and verify (old behavior)
            logger.info("Step 2: Generating custom discovery prompt...")
            custom_config = self.discovery_generator.generate(
                expanded_task_description=expanded_description,
                method=method,
                base_config=base_config,
                model=model
            )

            # Format into full prompt
            discovery_prompt = format_universal_prompt(
                task_description=expanded_description,
                config=custom_config
            )
            logger.info(f"Discovery prompt generated ({len(discovery_prompt)} chars)")

            # Step 3: Verify prompt with reflection loop
            logger.info("Step 3: Verifying prompt produces correct JSON...")
            sample_conv = self._select_verification_sample(conversations)

            verification_passed = False
            current_prompt = discovery_prompt
            reflection_attempts = 0

            for attempt in range(max_reflection_attempts + 1):
                logger.info(f"Verification attempt {attempt + 1}/{max_reflection_attempts + 1}")

                # Verify current prompt
                result = self.prompt_verifier.verify(
                    custom_prompt=current_prompt,
                    sample_conversation=sample_conv,
                    method=method,
                    model=model
                )

                if result.passed:
                    logger.info(f"Verification passed on attempt {attempt + 1}!")
                    verification_passed = True
                    discovery_prompt = current_prompt
                    break
                else:
                    logger.warning(
                        f"Verification failed on attempt {attempt + 1}: "
                        f"{result.error_type} - {result.error_details}"
                    )

                    # If this was the last attempt, fall back to static
                    if attempt >= max_reflection_attempts:
                        logger.warning(
                            f"All {max_reflection_attempts} reflection attempts exhausted. "
                            "Falling back to static prompt."
                        )
                        discovery_prompt = format_universal_prompt(
                            task_description=expanded_description,
                            config=base_config
                        )
                        break

                    # Reflect and fix the prompt
                    logger.info("Running LLM reflection to fix prompt...")
                    reflection_attempts += 1
                    current_prompt = self.prompt_reflector.reflect_and_fix(
                        original_prompt=current_prompt,
                        verification_result=result,
                        expanded_task_description=expanded_description,
                        method=method,
                        model=model
                    )

        # Clustering customization (optional)
        if customize_clustering:
            step_num = "Step 3" if skip_verification else "Step 4"
            logger.info(f"{step_num}: Customizing clustering prompts in parallel...")

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all three clustering prompts in parallel
                future_clustering = executor.submit(
                    self.clustering_customizer.customize_clustering_prompt,
                    task_description=expanded_description,
                    model=model
                )
                future_dedup = executor.submit(
                    self.clustering_customizer.customize_deduplication_prompt,
                    task_description=expanded_description,
                    model=model
                )
                future_outlier = executor.submit(
                    self.clustering_customizer.customize_outlier_prompt,
                    task_description=expanded_description,
                    model=model
                )

                # Collect results
                clustering_prompt = future_clustering.result()
                dedup_prompt = future_dedup.result()
                outlier_prompt = future_outlier.result()

            logger.info("Clustering prompts customized (parallel execution)")
        else:
            # Use default clustering prompts
            logger.info("Using default clustering prompts (customization skipped)")
            from ..clustering.prompts import (
                clustering_systems_prompt,
                deduplication_clustering_systems_prompt,
                outlier_clustering_systems_prompt
            )
            clustering_prompt = clustering_systems_prompt
            dedup_prompt = deduplication_clustering_systems_prompt
            outlier_prompt = outlier_clustering_systems_prompt

        return DynamicPromptResult(
            discovery_prompt=discovery_prompt,
            clustering_prompt=clustering_prompt,
            dedup_prompt=dedup_prompt,
            outlier_prompt=outlier_prompt,
            expanded_task_description=expanded_description,
            verification_passed=verification_passed,
            reflection_attempts=reflection_attempts
        )

    def _select_verification_sample(
        self,
        conversations: List[ConversationRecord]
    ) -> ConversationRecord:
        """Select a deterministic conversation for verification.

        Args:
            conversations: List of all conversations.

        Returns:
            A deterministically selected conversation (first one for consistency).
        """
        if not conversations:
            raise ValueError("No conversations available for verification")

        # Use first conversation for deterministic verification
        return conversations[0]
