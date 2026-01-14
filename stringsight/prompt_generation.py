"""
Prompt generation utilities for StringSight.

This module handles both static and dynamic prompt generation, keeping the main
public.py file clean and focused.
"""

from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
from .core.data_objects import PropertyDataset
from .prompts import get_system_prompt
from .prompts.dynamic import DynamicPromptGenerator
from .schemas import PromptsMetadata

logger = logging.getLogger(__name__)


def generate_prompts(
    task_description: str | None,
    dataset: PropertyDataset,
    method: str,
    use_dynamic_prompts: bool = True,
    dynamic_prompt_samples: int = 5,
    model: str = "gpt-4.1",
    system_prompt_override: str | None = None,
    output_dir: str | None = None,
    seed: int = 42,
) -> Tuple[str, Optional[Dict[str, str]], Optional[PromptsMetadata]]:
    """Generate discovery and clustering prompts.

    This function handles both static and dynamic prompt generation, with
    automatic fallback to static prompts if dynamic generation fails.

    Args:
        task_description: User's task description.
        dataset: PropertyDataset with conversations.
        method: "single_model" or "side_by_side".
        use_dynamic_prompts: Whether to use dynamic prompt generation.
        dynamic_prompt_samples: Number of samples for dynamic generation.
        model: LLM model for meta-prompting.
        system_prompt_override: Optional override for system prompt.
        output_dir: Optional directory to save generated prompts.
        seed: Random seed for deterministic prompt generation (default: 42).

    Returns:
        Tuple of (discovery_prompt, custom_clustering_prompts, prompts_metadata)
    """
    # Normalize task_description once:
    # - strip only leading/trailing whitespace (preserve internal formatting/newlines)
    # - treat empty/whitespace-only as None
    task_description_clean = task_description.strip() if task_description is not None else None
    if task_description_clean == "":
        task_description_clean = None

    # Dynamic prompt generation
    # Can generate prompts with or without task_description (will infer from conversations if not provided)
    logger.info(f"Prompt generation config: use_dynamic_prompts={use_dynamic_prompts}, system_prompt_override={system_prompt_override is not None}")

    # Check if system_prompt_override is a known template alias (not a custom literal prompt)
    KNOWN_PROMPT_ALIASES = {"default", "agent", "universal", "agent_universal"}
    is_custom_literal_prompt = system_prompt_override is not None and system_prompt_override not in KNOWN_PROMPT_ALIASES

    # Only skip dynamic generation if there's a custom literal prompt
    # Template aliases like "default" should still allow dynamic generation
    if use_dynamic_prompts and not is_custom_literal_prompt:
        logger.info(f"Generating dynamic prompts (system_prompt_override={system_prompt_override})...")

        # Use a default task description if none provided
        task_desc_for_generation = task_description_clean or "Analyze the behavioral patterns and characteristics in these AI model conversations."
        logger.info(f"Using task description: {task_desc_for_generation[:100]}...")

        try:
            logger.info(f"Attempting dynamic prompt generation with {len(dataset.conversations)} total conversations, sampling {min(dynamic_prompt_samples, len(dataset.conversations))} for prompt generation")
            generator = DynamicPromptGenerator(seed=seed)
            result = generator.generate_all_prompts(
                task_description=task_desc_for_generation,
                conversations=dataset.conversations,
                method=method,
                num_samples=dynamic_prompt_samples,
                model=model
            )
            logger.info(f"Dynamic prompt generation completed. Verification passed: {result.verification_passed}")

            if result.verification_passed:
                logger.info("Dynamic prompt verification passed.")
                discovery_prompt = result.discovery_prompt
                custom_clustering_prompts = {
                    "clustering": result.clustering_prompt,
                    "deduplication": result.dedup_prompt,
                    "outlier": result.outlier_prompt,
                }

                # Create metadata object for successful dynamic generation
                metadata = PromptsMetadata(
                    discovery_prompt=result.discovery_prompt,
                    clustering_prompt=result.clustering_prompt,
                    dedup_prompt=result.dedup_prompt,
                    outlier_prompt=result.outlier_prompt,
                    expanded_task_description=result.expanded_task_description,
                    task_description_original=task_description_clean,
                    dynamic_prompts_used=True,
                    verification_passed=result.verification_passed,
                    reflection_attempts=result.reflection_attempts
                )
            else:
                logger.warning("Dynamic prompt verification failed, but using generated prompts anyway.")
                discovery_prompt = result.discovery_prompt
                custom_clustering_prompts = {
                    "clustering": result.clustering_prompt,
                    "deduplication": result.dedup_prompt,
                    "outlier": result.outlier_prompt,
                }

                # Create metadata for dynamic generation with failed verification
                # Still report as dynamic_prompts_used=True since we ARE using the generated prompts
                metadata = PromptsMetadata(
                    discovery_prompt=result.discovery_prompt,
                    clustering_prompt=result.clustering_prompt,
                    dedup_prompt=result.dedup_prompt,
                    outlier_prompt=result.outlier_prompt,
                    expanded_task_description=result.expanded_task_description,
                    task_description_original=task_description_clean,
                    dynamic_prompts_used=True,
                    verification_passed=False,
                    reflection_attempts=result.reflection_attempts
                )

            # Save prompts to file if output_dir provided (regardless of verification)
            if output_dir:
                _save_prompts_to_file(
                    output_dir=output_dir,
                    discovery_prompt=discovery_prompt,
                    clustering_prompts=custom_clustering_prompts,
                    expanded_task_description=result.expanded_task_description
                )
                _save_metadata_to_file(output_dir=output_dir, metadata=metadata)
        except Exception as e:
            logger.error(f"Dynamic prompt generation failed: {e}. Using static prompts.")
            discovery_prompt = get_system_prompt(method, system_prompt_override, task_description_clean)
            custom_clustering_prompts = None

            # Create metadata for error case
            metadata = PromptsMetadata(
                discovery_prompt=discovery_prompt,
                task_description_original=task_description_clean,
                dynamic_prompts_used=False
            )
    else:
        # Static prompts (original behavior)
        discovery_prompt = get_system_prompt(method, system_prompt_override, task_description_clean)
        custom_clustering_prompts = None

        # Create metadata for static prompts
        metadata = PromptsMetadata(
            discovery_prompt=discovery_prompt,
            task_description_original=task_description_clean,
            dynamic_prompts_used=False
        )

        # Still save static prompt if output_dir provided
        if output_dir and discovery_prompt:
            _save_prompts_to_file(
                output_dir=output_dir,
                discovery_prompt=discovery_prompt,
                clustering_prompts=None,
                expanded_task_description=task_description_clean
            )
            _save_metadata_to_file(output_dir=output_dir, metadata=metadata)

    return discovery_prompt, custom_clustering_prompts, metadata


def _save_metadata_to_file(
    output_dir: str,
    metadata: PromptsMetadata
) -> None:
    """Save prompts metadata to JSON file in output directory.

    Args:
        output_dir: Directory to save metadata to (relative paths resolved relative to results dir).
        metadata: PromptsMetadata object to save.
    """
    import json
    from stringsight.utils.paths import _get_results_dir

    # Resolve output_dir relative to results directory if it's not absolute
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        results_base = _get_results_dir()
        output_path = results_base / output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    metadata_file = output_path / "prompts_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata.dict(), f, indent=2)
    logger.info(f"Saved prompts metadata to {metadata_file}")


def _save_prompts_to_file(
    output_dir: str,
    discovery_prompt: str,
    clustering_prompts: Optional[Dict[str, str]],
    expanded_task_description: str | None
) -> None:
    """Save generated prompts to text files in output directory.

    Args:
        output_dir: Directory to save prompts to (relative paths resolved relative to results dir).
        discovery_prompt: The discovery/extraction prompt.
        clustering_prompts: Optional dict of clustering prompts.
        expanded_task_description: Optional expanded task description.
    """
    from stringsight.utils.paths import _get_results_dir

    # Resolve output_dir relative to results directory if it's not absolute
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        results_base = _get_results_dir()
        output_path = results_base / output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    # Save discovery prompt
    discovery_file = output_path / "discovery_prompt.txt"
    with open(discovery_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DISCOVERY PROMPT\n")
        f.write("=" * 80 + "\n\n")
        f.write(discovery_prompt)
    logger.info(f"Saved discovery prompt to {discovery_file}")

    # Save clustering prompts if available
    if clustering_prompts:
        clustering_file = output_path / "clustering_prompts.txt"
        with open(clustering_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTERING PROMPTS\n")
            f.write("=" * 80 + "\n\n")

            for name, prompt in clustering_prompts.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{name.upper()}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(prompt)
                f.write("\n\n")
        logger.info(f"Saved clustering prompts to {clustering_file}")

    # Save expanded task description if available
    if expanded_task_description:
        task_file = output_path / "expanded_task_description.txt"
        with open(task_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EXPANDED TASK DESCRIPTION\n")
            f.write("=" * 80 + "\n\n")
            f.write(expanded_task_description)
        logger.info(f"Saved expanded task description to {task_file}")
