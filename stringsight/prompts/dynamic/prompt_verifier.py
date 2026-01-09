"""
Prompt verification to ensure generated prompts produce correct JSON output.

This module validates that dynamically generated prompts will produce the expected
JSON schema when used for property extraction.
"""

import json
import re
import litellm
import logging
from typing import Tuple
from dataclasses import dataclass
from ...core.data_objects import ConversationRecord

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of prompt verification with detailed error information."""
    passed: bool
    error_type: str | None  # "invalid_json", "missing_fields", "wrong_types", "not_array", "exception"
    error_details: str | None  # Detailed error message
    llm_output: str | None  # The actual LLM output (for reflection)
    expected_schema: dict | None  # What we expected


class PromptVerifier:
    """Verifies that generated prompts produce correct JSON output."""

    def verify(
        self,
        custom_prompt: str,
        sample_conversation: ConversationRecord,
        method: str,
        model: str = "gpt-4.1"
    ) -> VerificationResult:
        """Verify prompt on sample conversation.

        Args:
            custom_prompt: The generated discovery prompt to test.
            sample_conversation: A sample conversation to test the prompt on.
            method: Analysis method ("single_model", "side_by_side", etc.).
            model: LLM model to use for verification.

        Returns:
            VerificationResult with detailed error information.
        """
        try:
            # Test prompt on sample
            output = self._test_prompt(custom_prompt, sample_conversation, model)

            # Validate JSON schema with detailed error info
            validation_result = self._validate_output_detailed(output, method)

            return validation_result
        except Exception as e:
            error_msg = f"Verification failed with exception: {str(e)}"
            logger.error(error_msg)
            return VerificationResult(
                passed=False,
                error_type="exception",
                error_details=error_msg,
                llm_output=None,
                expected_schema=self._get_expected_schema(method)
            )

    def _test_prompt(
        self,
        prompt: str,
        conversation: ConversationRecord,
        model: str
    ) -> str:
        """Call LLM with custom prompt on sample conversation.

        Args:
            prompt: System prompt to test.
            conversation: Sample conversation.
            model: LLM model.

        Returns:
            LLM response content.
        """
        # Format conversation into user message
        from ...extractors.conv_to_str import conv_to_str

        # Build user message based on conversation format
        if isinstance(conversation.model, list):
            # Side-by-side format
            model_a = conversation.model[0] if len(conversation.model) > 0 else "Model A"
            model_b = conversation.model[1] if len(conversation.model) > 1 else "Model B"

            messages_a = conversation.responses[0] if isinstance(conversation.responses, list) and len(conversation.responses) > 0 else []
            messages_b = conversation.responses[1] if isinstance(conversation.responses, list) and len(conversation.responses) > 1 else []

            response_a = conv_to_str(messages_a) if messages_a else ""
            response_b = conv_to_str(messages_b) if messages_b else ""

            user_message = f"**Prompt:** {conversation.prompt}\n\n"
            user_message += f"**{model_a} Response:**\n{response_a}\n\n"
            user_message += f"**{model_b} Response:**\n{response_b}\n"
        else:
            # Single model format
            messages = conversation.responses
            response_text = conv_to_str(messages) if messages else ""

            user_message = f"**Prompt:** {conversation.prompt}\n\n"
            user_message += f"**Model Response:**\n{response_text}\n"

        # Call LLM
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=4000
        )

        return response.choices[0].message.content

    def _validate_output_detailed(
        self,
        output: str,
        method: str
    ) -> VerificationResult:
        """Validate JSON schema of output with detailed error reporting.

        Args:
            output: LLM output to validate.
            method: Analysis method.

        Returns:
            VerificationResult with detailed error information.
        """
        expected_schema = self._get_expected_schema(method)

        # Extract JSON from response (may be wrapped in ```json blocks)
        json_str = self._extract_json(output)

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return VerificationResult(
                passed=False,
                error_type="invalid_json",
                error_details=f"Invalid JSON: {str(e)}",
                llm_output=output,
                expected_schema=expected_schema
            )

        # Check is list
        if not isinstance(data, list):
            return VerificationResult(
                passed=False,
                error_type="not_array",
                error_details="Output is not a JSON array",
                llm_output=output,
                expected_schema=expected_schema
            )

        # If empty list, that's actually OK (no properties found)
        if len(data) == 0:
            return VerificationResult(
                passed=True,
                error_type=None,
                error_details=None,
                llm_output=output,
                expected_schema=expected_schema
            )

        # Check each object has required fields
        required_fields = {
            "behavior_type",
            "property_description",
            "category",
            "evidence",
            "reason",
            "contains_errors",
            "unexpected_behavior"
        }

        # For side-by-side, also require "model" field
        if "side_by_side" in method or "sbs" in method:
            required_fields.add("model")

        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i} is not an object",
                    llm_output=output,
                    expected_schema=expected_schema
                )

            missing = required_fields - set(obj.keys())
            if missing:
                return VerificationResult(
                    passed=False,
                    error_type="missing_fields",
                    error_details=f"Item {i} missing fields: {missing}",
                    llm_output=output,
                    expected_schema=expected_schema
                )

            # Validate field types
            if not isinstance(obj.get("behavior_type"), str):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i}: behavior_type must be string",
                    llm_output=output,
                    expected_schema=expected_schema
                )
            if not isinstance(obj.get("property_description"), str):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i}: property_description must be string",
                    llm_output=output,
                    expected_schema=expected_schema
                )
            if not isinstance(obj.get("category"), str):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i}: category must be string",
                    llm_output=output,
                    expected_schema=expected_schema
                )
            if not isinstance(obj.get("reason"), str):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i}: reason must be string",
                    llm_output=output,
                    expected_schema=expected_schema
                )

            # Validate boolean fields (can be bool or string "True"/"False")
            contains_errors = obj.get("contains_errors")
            if not isinstance(contains_errors, (bool, str)):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i}: contains_errors must be boolean or string",
                    llm_output=output,
                    expected_schema=expected_schema
                )

            unexpected_behavior = obj.get("unexpected_behavior")
            if not isinstance(unexpected_behavior, (bool, str)):
                return VerificationResult(
                    passed=False,
                    error_type="wrong_types",
                    error_details=f"Item {i}: unexpected_behavior must be boolean or string",
                    llm_output=output,
                    expected_schema=expected_schema
                )

        return VerificationResult(
            passed=True,
            error_type=None,
            error_details=None,
            llm_output=output,
            expected_schema=expected_schema
        )

    def _get_expected_schema(self, method: str) -> dict:
        """Get the expected JSON schema for the given method.

        Args:
            method: Analysis method.

        Returns:
            Dictionary describing expected schema.
        """
        base_fields = {
            "behavior_type": "string",
            "property_description": "string",
            "category": "string",
            "evidence": "string",
            "reason": "string",
            "contains_errors": "boolean or string",
            "unexpected_behavior": "boolean or string"
        }

        # For side-by-side, also require "model" field
        if "side_by_side" in method or "sbs" in method:
            base_fields["model"] = "string"

        return {
            "type": "array",
            "items": base_fields
        }

    def _extract_json(self, output: str) -> str:
        """Extract JSON from output (may be in ```json blocks).

        Args:
            output: LLM output that may contain JSON.

        Returns:
            Extracted JSON string.
        """
        # Try to find JSON in code blocks first
        match = re.search(r'```json\s*\n(.*?)\n```', output, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find JSON array directly
        match = re.search(r'\[.*\]', output, re.DOTALL)
        if match:
            return match.group(0)

        # Return as-is and let parser fail
        return output
