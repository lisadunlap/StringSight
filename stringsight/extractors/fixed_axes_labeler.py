from __future__ import annotations

from typing import Dict, Callable, Optional, Any, List

from .openai import OpenAIExtractor
from ..prompts.fixed_axes import fixed_axis_prompt


class FixedAxesLabeler(OpenAIExtractor):
    """Extractor that asks an LLM to *label* a conversation using a fixed taxonomy.

    It reuses the OpenAIExtractor infrastructure (parallel requests, caching,
    wandb logging) but fills in a special *system prompt* that enumerates the
    user-supplied taxonomy (a mapping ``{label: description}``).

    Only the ``property_description``, ``reason`` and ``evidence`` keys are
    expected in the JSON response; missing optional keys are tolerated by the
    existing :pyclass:`stringsight.postprocess.parser.LLMJsonParser`.
    """

    def __init__(
        self,
        taxonomy: Dict[str, str],
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        max_workers: int = 64,
        cache_dir: str = ".cache/stringsight",
        prompt_builder: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        self.taxonomy = taxonomy

        # ------------------------------------------------------------------
        # Build system prompt from template
        # ------------------------------------------------------------------
        fixed_axes = "\n".join(f"- **{name}**: {desc}" for name, desc in taxonomy.items())
        fixed_axes_names = ", ".join(taxonomy.keys())

        # Str.format() would treat all {{…}} in the JSON example as placeholders.
        # Avoid KeyError by simple placeholder replacement instead.
        system_prompt = (
            fixed_axis_prompt
            .replace("{fixed_axes}", fixed_axes)
            .replace("{fixed_axes_names}", fixed_axes_names)
        )

        # Default prompt builder can be re-used from OpenAIExtractor for single-model
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            prompt_builder=prompt_builder,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_workers=max_workers,
            cache_dir=cache_dir,
            **kwargs,
        )

    # We only support *single-model* conversations – override validation
    def _default_prompt_builder(self, conversation) -> str:  # type: ignore[override]
        """Much simpler than the parent implementation – no side-by-side logic."""

        if not isinstance(conversation.model, str):
            raise ValueError(
                "FixedAxesLabeler supports only single-model data (method='single_model')."
            )

        response = (
            conversation.responses
            if isinstance(conversation.responses, str)
            else str(conversation.responses)
        )
        return f"# Model response:\n {response}" 