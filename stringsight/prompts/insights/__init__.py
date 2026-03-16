"""
Insight prompt templates.
"""

from .metrics_summary import (
    METRICS_INSIGHTS_SYSTEM_PROMPT,
    build_metrics_insights_user_prompt,
)

__all__ = [
    "METRICS_INSIGHTS_SYSTEM_PROMPT",
    "build_metrics_insights_user_prompt",
]

