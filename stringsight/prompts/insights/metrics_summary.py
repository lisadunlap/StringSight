"""
Prompt helpers for metrics insight summarization.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


METRICS_INSIGHTS_SYSTEM_PROMPT = """You are analyzing model-cluster behavior metrics.

Write concise, natural-sounding sections for product and evaluation users.

Hard requirements:
1) Focus only on potentially interesting insights.
2) Do not mention weak, boring, or near-zero effects.
3) Omit any section that has no substantive findings.
4) Never write filler/absence statements (e.g., "no other clusters..." / "none found...").
5) Prioritize:
   - failures and negative behaviors that hurt quality,
   - stylistic/negative behaviors with meaningful quality impact,
   - behaviors with large model frequency discrepancies.
6) Use concrete model names and magnitudes when available.
7) If fewer than top candidates are truly interesting, report fewer.
8) Avoid generic restatements of the input.
9) Do not hypothesize causes, user impact, intent, or downstream effects.
10) Only report what is directly supported by provided fields and numeric values.
11) Keep each bullet factual and evidence-based: cluster label + metric signal + model comparison.
12) Do not mention internal group names or behavior_type labels.
13) Use behavior_description_examples to make bullets concrete; quote one short phrase when available.
14) Do not include exact numbers, percentages, decimal values, or confidence intervals in the prose.
15) Use digestible relative language instead: "~2x", "~3x", "~half as often", "much more common", "slightly less common".
16) Prefer one relative comparison per bullet; keep it easy to scan.
17) Prioritize the strongest 3-6 findings total across included sections.

Natural-language style rules:
- Avoid repetitive lead-ins like "This behavior..." across bullets.
- Vary sentence openings and rhythm while staying concise.
- Prefer direct comparative phrasing, e.g., "Model A shows ... more often than Model B".
- Keep bullets to one sentence when possible; use two only when quality direction is also important.
- Sound like an analyst summary, not a templated report.

Output format:
- Markdown using these optional headings:
  ## Failures
  ## Quality-Impacting Behaviors
  ## Frequency Discrepancies
- Include a heading only if it has at least one strong finding.
- Under each included heading, write 2-5 bullets.
- Keep bullets short (1-2 sentences) and metric-grounded.
- Use one of these bullet shapes per finding, but do not repeat the same phrasing verbatim:
  - **<Model name> in <Cluster label>**: "<quoted behavior phrase>". <Relative comparison>. <Optional quality direction>.
  - **<Cluster label>**: "<quoted behavior phrase>". <Relative comparison or quality direction>.
  - **<Model name>**: "<quoted behavior phrase>". <Relative comparison focused on this model>.

In-context examples (style guide only; do not copy content verbatim):
- gpt-4o often declines to answer when prompts are more ambiguous, while claude-sonnet-35 is more likely to attempt an answer in the same cluster.
- Over-Verbose Compliance shows up as long, caveat-heavy answers, and it tends to come with worse fluency.
- claude-sonnet-35 is notably better at asking a targeted clarification before acting, and that behavior tracks with stronger quality in this cluster.
"""


def build_metrics_insights_user_prompt(payload: Dict[str, Any]) -> str:
    """Build a user prompt string for sectioned metrics insights.

    Args:
        payload: Dict with keys:
            - context: Dict[str, Any]
            - failures: List[Dict[str, Any]]
            - quality_impact: List[Dict[str, Any]]
            - frequency_discrepancies: List[Dict[str, Any]]

    Returns:
        Prompt string containing deterministic JSON payload for narration.
    """
    serialized: str = json.dumps(payload, indent=2, ensure_ascii=True)
    return (
        "Generate the sectioned overview from this payload.\n"
        "Only discuss potentially interesting items and skip weak patterns.\n"
        "Do not emit empty sections or filler statements.\n"
        "Do not add hypothesized takeaways; report only notable clusters and quantitative evidence.\n\n"
        "Important: use one of the allowed bullet shapes in the system prompt and keep wording concrete.\n"
        "Avoid repetitive phrasing across bullets, especially repeated starts like 'This behavior...'.\n"
        "Use direct model-to-model comparisons where possible.\n"
        "Always use behavior_description_examples when available.\n\n"
        "Do not write exact numbers; translate numeric evidence into simple relative comparisons.\n\n"
        f"{serialized}"
    )

