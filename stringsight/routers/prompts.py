"""
Prompt management endpoints.

Endpoints for listing, retrieving, and configuring system prompts.
"""

from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException

from stringsight.schemas import LabelPromptRequest
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
