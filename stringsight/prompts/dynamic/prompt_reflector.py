"""
Prompt reflection for fixing failed prompts.

This module implements LLM-based reflection to analyze and fix prompts
that fail verification.
"""

import json
import litellm
import logging
from .prompt_verifier import VerificationResult
from .meta_prompts import REFLECTION_SYSTEM_PROMPT, REFLECTION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class PromptReflector:
    """Reflects on verification failures and generates corrected prompts."""

    def reflect_and_fix(
        self,
        original_prompt: str,
        verification_result: VerificationResult,
        expanded_task_description: str,
        method: str,
        model: str = "gpt-4.1"
    ) -> str:
        """Analyze verification failure and generate corrected prompt.

        Args:
            original_prompt: The prompt that failed verification.
            verification_result: Detailed verification failure info.
            expanded_task_description: Task context.
            method: "single_model" or "side_by_side".
            model: LLM model for reflection.

        Returns:
            Corrected prompt string.
        """
        logger.info(
            f"Running reflection to fix prompt. Error type: {verification_result.error_type}"
        )

        # Build reflection prompt
        reflection_prompt = self._build_reflection_prompt(
            original_prompt=original_prompt,
            verification_result=verification_result,
            expanded_task_description=expanded_task_description,
            method=method
        )

        # Call LLM to reflect and fix
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0.0,  # Zero temp for deterministic fixes
                max_tokens=4000
            )

            corrected_prompt = response.choices[0].message.content.strip()
            logger.info("Reflection completed. Generated corrected prompt.")
            return corrected_prompt
        except Exception as e:
            logger.error(f"Reflection failed with exception: {e}. Returning original prompt.")
            # If reflection itself fails, return original prompt
            return original_prompt

    def _build_reflection_prompt(
        self,
        original_prompt: str,
        verification_result: VerificationResult,
        expanded_task_description: str,
        method: str
    ) -> str:
        """Build prompt for LLM reflection.

        Args:
            original_prompt: Prompt that failed.
            verification_result: Verification failure details.
            expanded_task_description: Task context.
            method: Analysis method.

        Returns:
            Reflection prompt string.
        """
        # Truncate LLM output if too long (keep first 1000 chars for context)
        llm_output = verification_result.llm_output or "N/A"
        if len(llm_output) > 1000:
            llm_output = llm_output[:1000] + "\n...[truncated]"

        return REFLECTION_PROMPT_TEMPLATE.format(
            original_prompt=original_prompt,
            error_type=verification_result.error_type,
            error_details=verification_result.error_details,
            llm_output=llm_output,
            expected_schema=json.dumps(verification_result.expected_schema, indent=2),
            task_description=expanded_task_description,
            method=method
        )
