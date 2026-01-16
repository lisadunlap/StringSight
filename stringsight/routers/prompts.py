"""
Prompt management endpoints.

Endpoints for listing, retrieving, and configuring system prompts.
"""

from typing import Dict, List, Any
import time
import pandas as pd

from fastapi import APIRouter, HTTPException

from stringsight.schemas import LabelPromptRequest, GeneratePromptsRequest
from stringsight.formatters import detect_method
from stringsight.core.data_objects import PropertyDataset
from stringsight.prompt_generation import generate_prompts as _generate_prompts
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["prompts"])


@router.get("/prompts")
def list_prompts() -> Dict[str, Any]:
    """Return only 'default' and 'agent' prompt choices with metadata and defaults."""
    from stringsight import prompts as _prompts
    from stringsight.prompts import get_system_prompt as _get

    # Build entries for aliases; provide defaults for both methods so UI can prefill
    default_single = getattr(_prompts, "single_model_default_task_description", None)
    default_sbs = getattr(_prompts, "sbs_default_task_description", None)
    agent_single = getattr(_prompts, "agent_system_prompt_custom_task_description", None)
    agent_sbs = getattr(_prompts, "agent_sbs_system_prompt_custom_task_description", None)

    out: List[Dict[str, Any]] = []
    out.append({
        "name": "default",
        "label": "Default",
        "has_task_description": True,
        "default_task_description_single": default_single,
        "default_task_description_sbs": default_sbs,
        "preview": (_get("single_model", "default") or "")[:180],
    })
    out.append({
        "name": "agent",
        "label": "Agent",
        "has_task_description": True,
        "default_task_description_single": agent_single,
        "default_task_description_sbs": agent_sbs,
        "preview": (_get("single_model", "agent") or "")[:180],
    })
    return {"prompts": out}


@router.get("/prompt-text")
def prompt_text(name: str, task_description: str | None = None, method: str | None = None) -> Dict[str, Any]:
    """Return full text of a prompt by name or alias (default/agent), formatted.

    If 'name' is an alias, 'method' determines the template ('single_model' or 'side_by_side').
    Defaults to 'single_model' when omitted.
    """
    from stringsight.prompts import get_system_prompt as _get
    m = method or "single_model"
    try:
        value = _get(m, name, task_description)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"name": name, "text": value}


@router.post("/label/prompt")
def label_prompt(req: LabelPromptRequest) -> Dict[str, Any]:
    """Return the system prompt that will be used for fixed-taxonomy labeling.

    This endpoint generates the same system prompt used by the label() function,
    allowing users to preview the exact prompt before running labeling.

    Args:
        req: LabelPromptRequest containing taxonomy dictionary

    Returns:
        Dictionary with 'text' key containing the full system prompt
    """
    from stringsight.prompts.fixed_axes import fixed_axis_prompt

    if not req.taxonomy or len(req.taxonomy) == 0:
        raise HTTPException(status_code=400, detail="Taxonomy must contain at least one label")

    fixed_axes = "\n".join(f"- **{name}**: {desc}" for name, desc in req.taxonomy.items())
    fixed_axes_names = ", ".join(req.taxonomy.keys())

    system_prompt = (
        fixed_axis_prompt
        .replace("{fixed_axes}", fixed_axes)
        .replace("{fixed_axes_names}", fixed_axes_names)
    )

    return {"text": system_prompt}


@router.post("/generate")
async def generate_prompts_endpoint(req: GeneratePromptsRequest) -> Dict[str, Any]:
    """
    Generate dynamic prompts without running extraction.

    This endpoint:
    1. Samples conversations from provided rows
    2. Expands task description using samples
    3. Generates discovery + clustering prompts
    4. Verifies prompts via test extraction
    5. Returns all prompts + metadata

    Returns:
        Dictionary containing:
        - prompts: PromptsMetadata with all generated prompts
        - generation_time_seconds: Time taken to generate prompts
    """
    t_start = time.perf_counter()

    # Build DataFrame from rows
    df = pd.DataFrame(req.rows)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No rows provided for prompt generation")

    # Detect method
    method = req.method or detect_method(list(df.columns))
    if method is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to detect dataset method from columns. Please specify method explicitly."
        )

    logger.info(f"Generating prompts for {len(df)} rows using method: {method}")

    try:
        # Create dataset from dataframe
        dataset = PropertyDataset.from_dataframe(df, method=method)

        # Generate prompts (discovery + clustering)
        discovery_prompt, custom_clustering_prompts, prompts_metadata = _generate_prompts(
            task_description=req.task_description,
            dataset=dataset,
            method=method,
            use_dynamic_prompts=True,
            dynamic_prompt_samples=req.num_samples or 5,
            model=req.model or "gpt-4.1",
            system_prompt_override=None,
            output_dir=req.output_dir,
            seed=req.seed or 42
        )

        generation_time = time.perf_counter() - t_start
        logger.info(f"Prompt generation completed in {generation_time:.2f}s")

        # Return metadata
        if prompts_metadata is None:
            raise HTTPException(
                status_code=500,
                detail="Prompt generation succeeded but returned no metadata."
            )
        return {
            "prompts": prompts_metadata.dict(),
            "generation_time_seconds": generation_time
        }

    except ValueError as e:
        logger.error(f"Validation error during prompt generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during prompt generation")
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {e}")
