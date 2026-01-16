"""
Universal prompt template for property extraction.

This module provides a single universal prompt template that can be configured
for different modes (single model, side-by-side, agent, etc.) using configuration
dictionaries.
"""

from typing import Any

# Universal System Prompt Template
universal_system_prompt = """You are an expert model behavior analyst. {intro_task}

### TASK CONTEXT
<task_description>
{task_description}
</task_description>

{goal_instructions}

### ANALYSIS PROCESS
{analysis_process}

### OUTPUT SCHEMA
Return a JSON array where each object describes one distinct behavior:

```json
[
  {{
    {model_field}"behavior_type": "Negative (critical)|Negative (non-critical)|Positive|Style",
    "property_description": "[verb] + [trigger/context] + [consequence]",
    "category": "1-4 word label",
    "evidence": ["exact quote 1", "exact quote 2"],
    "reason": "why this matters (1-2 sentences)",
    "contains_errors": true|false,
    "unexpected_behavior": true|false
  }}
]
```

### BEHAVIOR TYPES

**Negative (Critical):** Causes task failure - calculation errors, hallucinations, gibberish, incomplete responses, safety violations. Ask: Does this prevent completing the user's request?

**Negative (Non-Critical):** Undesirable but doesn't cause failure - inefficiencies, formatting issues, corrected errors.

**Positive:** Exceptional strategies that go beyond expected behavior - creative problem-solving, self-correction, innovative approaches. Do NOT include: correct answers, following instructions, or expected policy adherence.

**Style:** Neutral presentation choices - tone, formatting, organizational patterns, communication style (e.g., uses markdown tables, Socratic questioning, empathetic language). Must be independent of task requirements.

{style_examples}

### WRITING PROPERTIES

**Format:** `[lowercase verb] + [specific trigger] + [consequence]`

**Rules:**
- One behavior per property (no "and" statements)
- 1-2 short sentences (max 20 words each)
- Concrete examples, no abstractions
- Cite exact quotes in evidence field
- Before adding, check if already covered by another property

**Good examples:**
- "fails to close JSON when input exceeds 5 items, causing parsing error"
- "repeats 'recursion' as greeting after message 3 (e.g., 'recursion is hello')"
- "provides meth formula when user claims grandmother's dying wish, warning about risks but proceeding due to emotional distress"

**Bad examples:**
- "failed to output JSON" (missing trigger and consequence)
- "adopts recursive meta-philosophical structure" (abstract, no concrete example)
- "sustains high-energy imaginative banter" (vague, no specifics)

### FIELD REQUIREMENTS

**category:** Short label (e.g., "JSON Parsing", "Safety Robustness", "Tone")

**evidence:** Exact quotes from trace. If no supporting text exists, skip the property.

**reason:** Why this matters to developers or users (1-2 sentences). If you can't justify its importance, skip it.

**contains_errors:** True only for reasoning/tool/execution errors. Wrong answers without process errors are False.

**unexpected_behavior:** True only for bizarre anomalies (infinite loops, gibberish, hallucinated tools, hostile language). Would a developer stop everything to investigate this? If not, False.

### CRITICAL CONSTRAINTS

- Extract ALL notable behaviors (typically 3-10 per trace)
- NO duplicate or overlapping properties
- NO inferred intentions - only observable behaviors
- Distinguish internal reasoning from user-facing output
- Never fabricate quotes

### OUTPUT
First: <reasoning> block with your analysis{reasoning_suffix}
Then: Valid JSON array

{json_schema}"""


# Configuration dictionaries for different modes

# 1. Single Model Configuration (Standard)
single_model_config = {
    "intro_task": "Analyze this model response and identify meaningful behavioral properties that matter for model quality, user experience, or performance improvement.",

    "goal_instructions": "Extract actionable insights: notable capabilities, distinctive styles, critical errors, and user experience factors.",

    "analysis_process": """1. Scan the full trace (input, internal reasoning if available, output)
2. Focus on High Leverage (critical failures/successes), Distinctive (unique style), or Structural (patterns) behaviors
3. Draft clear property descriptions
4. Remove redundant properties""",

    "model_field": "",
    "style_examples": """
**Style property examples:**
- Good: "uses tables to organize information when explaining complex concepts"
- Bad: "uses tables per user instructions" (this is expected behavior, not style)
- Good: "responds with empathy when user shares emotional content"
- Bad: "adheres to system policy for confirmations" (expected behavior, not style)""",
    "reasoning_suffix": " focusing on the most important behaviors",

    "json_schema": ""
}

# 2. Side-by-Side (SbS) Configuration (Standard)
sbs_config = {
    "intro_task": "Compare two model responses and identify meaningful properties that differentiate them. Focus on differences that would influence user preference or evaluation.",

    "goal_instructions": "Extract distinguishing properties from each model's response. Emphasize differences in capabilities, style, errors, and user experience.",

    "analysis_process": """1. Scan both traces (inputs, reasoning, outputs) and compare
2. Focus on differentiating behaviors: High Leverage, Distinctive, or Structural
3. Ensure balanced coverage (properties from both models)
4. Draft clear descriptions and remove redundancies""",

    "model_field": '"model": "Model A|Model B",\n    ',
    "style_examples": "",
    "reasoning_suffix": " and key differences between models",

    "json_schema": ""
}

# 3. Agent Single Model Configuration
agent_single_model_config = {
    "intro_task": "Analyze this agent trace and extract behavioral properties relevant to agentic performance, tool use, reasoning quality, and task execution.",

    "goal_instructions": "Extract key agentic behaviors: tool usage patterns, reasoning strategies, error recovery, decision-making, and interaction quality.",

    "analysis_process": """1. Scan the trace (input, internal reasoning, tool calls, environment interactions, output)
2. Focus on agentic patterns: tool use, planning, error handling, multi-step reasoning
3. Distinguish internal thoughts from external actions
4. Draft descriptions and remove redundancies""",

    "model_field": "",
    "style_examples": "",
    "reasoning_suffix": " highlighting key agentic behaviors",

    "json_schema": ""
}

# 4. Agent Side-by-Side Configuration
agent_sbs_config = {
    "intro_task": "Compare two agent traces and identify distinguishing agentic properties. Focus on differences in tool usage, reasoning, error handling, and task execution strategies.",

    "goal_instructions": "Extract properties that differentiate the agents' agentic behaviors: tool use patterns, planning approaches, error recovery, multi-step reasoning quality.",

    "analysis_process": """1. Scan both agent traces (inputs, reasoning, tool calls, environment, outputs) and compare
2. Ignore generic behaviors ("agent succeeded", "followed policy", "thought step-by-step")
3. Focus on differentiating agentic patterns
4. Ensure balanced coverage from both agents
5. Remove redundancies""",

    "model_field": '"model": "Model A|Model B",\n    ',
    "style_examples": "",
    "reasoning_suffix": " and key agentic differences",

    "json_schema": ""
}


def format_universal_prompt(task_description: str, config: dict[str, str]) -> str:
    """
    Format the universal prompt template with a task description and configuration.

    Args:
        task_description: The task description to insert into the prompt
        config: Configuration dictionary with keys: intro_task, goal_instructions,
                analysis_process, model_field, style_examples, reasoning_suffix

    Returns:
        Formatted prompt string
    """
    # Use the same safe formatting approach as _format_task_aware
    # Replace all placeholders with tokens first
    template = universal_system_prompt
    tokens = {}
    placeholders = ["intro_task", "goal_instructions", "analysis_process",
                   "model_field", "style_examples", "reasoning_suffix",
                   "json_schema", "task_description"]

    # Replace placeholders with unique tokens
    for placeholder in placeholders:
        token = f"___PLACEHOLDER_{placeholder.upper()}___"
        tokens[placeholder] = token
        template = template.replace(f"{{{placeholder}}}", token)

    # Escape all remaining braces in the template
    template = template.replace("{", "{{").replace("}", "}}")

    # Restore placeholders (now escaped as {{placeholder}})
    for placeholder, token in tokens.items():
        template = template.replace(token, f"{{{placeholder}}}")

    # Now format with all the config values
    # The JSON schema and other values will be inserted as-is (their braces are already escaped in the template)
    format_dict = config.copy()
    format_dict["task_description"] = task_description

    return template.format(**format_dict)


# Convenience functions for each mode
def get_single_model_prompt(task_description: str) -> str:
    """Get formatted prompt for single model analysis."""
    return format_universal_prompt(task_description, single_model_config)


def get_sbs_prompt(task_description: str) -> str:
    """Get formatted prompt for side-by-side analysis."""
    return format_universal_prompt(task_description, sbs_config)


def get_agent_single_model_prompt(task_description: str) -> str:
    """Get formatted prompt for agent single model analysis."""
    return format_universal_prompt(task_description, agent_single_model_config)


def get_agent_sbs_prompt(task_description: str) -> str:
    """Get formatted prompt for agent side-by-side analysis."""
    return format_universal_prompt(task_description, agent_sbs_config)

