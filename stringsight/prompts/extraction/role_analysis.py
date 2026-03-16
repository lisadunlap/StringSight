"""
Role-specific extraction prompts for analyzing different components of conversations.

This module provides both template-based and dynamic prompt generation for
analyzing properties of specific roles (e.g., user, assistant, tool, custom roles).
"""

from typing import Dict, Any
import pandas as pd


# Template-based prompts for common roles
ROLE_TEMPLATES: Dict[str, str] = {
    "user": """You are an expert conversation analyst specializing in user input analysis. Your task is to meticulously analyze user messages from conversations and identify unique, meaningful properties of how users communicate, request information, and frame their needs.

### INPUT CONTEXT
You are analyzing user inputs for the following task:
<task_description>
{task_description}
</task_description>

**Your Goal:**
Produce a JSON list of objects. Each object should represent a single, distinct property found in the user's messages. Identify MULTIPLE properties per conversation, especially Skills Required (most user requests demand several capabilities). Focus on:
- **Clarity & Specificity:** How clear, specific, or ambiguous are the user's requests?
- **Context Provision:** How much relevant context does the user provide?
- **Communication Style:** Tone, formality, emotional content
- **Request Complexity:** Simple vs complex, single vs multi-part requests
- **Domain Knowledge:** Evidence of user expertise or lack thereof
- **Skills Required:** For each distinct capability the request demands, produce one property (typically 2-5 per conversation)

```json
[
  {{
    "behavior_type": "Phrasing|Problem Domain|Skills Required",
    "property_description": "lowercase verb + specific pattern + context + consequence (1-3 sentences)",
    "category": "1-4 word category (e.g., 'Ambiguous Request', 'Context Rich', 'Technical Jargon')",
    "evidence": "exact quote from user messages",
    "reason": "1-2 sentence explanation of why this pattern is notable",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False",
    "role": "user"
  }},
  ...
]
```

### ANALYSIS PROCESS
1. **Read User Messages:** Focus only on messages with role="user"
2. **Identify Patterns:** Look for recurring patterns in how the user communicates
3. **Assess Impact:** Consider how these patterns might affect response quality
4. **Draft Properties:** Write clear descriptions following the format

### BEHAVIOR TYPES (use each applicable to the user's messages)
* **Phrasing:** How the prompt is presented (wording, clarity, structure, tone, constraints, and framing choices)
* **Problem Domain:** The specific problem area or subject matter the user is asking about (for example, abstract algebra, tax law, customer support policy)
* **Skills Required:** The concrete capabilities an assistant would need to fulfill the request. **Produce 2-5 Skills Required properties per conversation**—most requests require multiple capabilities (e.g., multi-hop reasoning, handling ambiguity, tool use, adapting to feedback, empathetic communication). Each skill gets its own property.

### CRITICAL CONSTRAINTS
* **EVIDENCE FROM USER ONLY:** All quotes must come from user messages, never from assistant responses
* **OBSERVABLE PATTERNS:** Describe what the user actually wrote, not what you infer they meant
* **NO HALLUCINATIONS:** Never fabricate quotes or exaggerate patterns
* Set "role": "user" for all properties

### OUTPUT FORMAT
First, output a brief **<reasoning>** block summarizing patterns in user input.
Then, output a valid **JSON Array**.""",

    "assistant": """You are an expert model behavior analyst. Your task is to meticulously analyze assistant/model responses and identify unique, meaningful qualitative properties, failure modes, and interesting behaviors. Focus only on properties that genuinely matter to users, evaluators, or developers when judging model quality.

### INPUT CONTEXT
You are analyzing assistant responses for the following task:
<task_description>
{task_description}
</task_description>

**Your Goal:**
Produce a JSON list of objects. Each object should represent a single, distinct property found in the assistant's responses. Focus on capabilities, style, errors, and user experience factors.

```json
[
  {{
    "behavior_type": "Negative (non-critical)|Negative (critical)|Style|Positive",
    "property_description": "lowercase verb + exact action + trigger + consequence (1-3 sentences)",
    "category": "1-4 word category (e.g., 'Reasoning Error', 'Safety Robustness', 'Formatting Style')",
    "evidence": "exact quote from assistant messages",
    "reason": "1-2 sentence explanation of why this property is notable",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False",
    "role": "assistant"
  }},
  ...
]
```

### ANALYSIS PROCESS
1. **Read Assistant Messages:** Focus only on messages with role="assistant"
2. **Identify Behaviors:** Look for unique capabilities, failure modes, style choices
3. **Assess Significance:** Focus on properties that impact user experience or model quality
4. **Draft Properties:** Write clear descriptions following the format

### BEHAVIOR TYPES
* **Negative (Critical):** Task failures, hallucinations, safety violations
* **Negative (Non-Critical):** Inefficiencies, formatting issues, partial errors
* **Style:** Distinctive persona, tone, formatting choices
* **Positive:** Uncommon but effective strategies (max 1 per trace)

### CRITICAL CONSTRAINTS
* **EVIDENCE FROM ASSISTANT ONLY:** All quotes must come from assistant messages
* **NO HALLUCINATIONS:** Never fabricate or alter quotes
* **OBSERVABLE BEHAVIORS:** Describe what the assistant actually wrote
* Set "role": "assistant" for all properties

### OUTPUT FORMAT
First, output a brief **<reasoning>** block summarizing key assistant behaviors.
Then, output a valid **JSON Array**.""",

    "non_user": """You are an expert conversation analyst. Your task is to analyze all messages where role!="user" and identify unique, meaningful qualitative properties and recurring patterns.

### INPUT CONTEXT
You are analyzing non-user messages for the following task:
<task_description>
{task_description}
</task_description>

**Your Goal:**
Produce a JSON list of objects. Each object should represent one distinct property found in messages where role is not "user" (for example assistant, tool, or system messages).

```json
[
  {{
    "behavior_type": "Negative (non-critical)|Negative (critical)|Style|Positive|Tool Pattern|System Pattern",
    "property_description": "lowercase verb + specific behavior + context + consequence (1-3 sentences)",
    "category": "1-4 word category",
    "evidence": "exact quote from non-user messages",
    "reason": "1-2 sentence explanation of why this pattern is notable",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False",
    "role": "non_user"
  }},
  ...
]
```

### ANALYSIS PROCESS
1. **Read Non-User Messages:** Focus only on messages where role is not "user"
2. **Identify Behaviors:** Look for failures, style patterns, tool/system effects, and notable strengths
3. **Assess Significance:** Focus on patterns that materially affect quality or behavior
4. **Draft Properties:** Write clear descriptions following the format

### CRITICAL CONSTRAINTS
* **EVIDENCE FROM NON-USER ONLY:** All quotes must come from messages where role!="user"
* **NO HALLUCINATIONS:** Never fabricate or alter quotes
* **OBSERVABLE BEHAVIORS:** Describe what is actually present in the messages
* Set "role": "non_user" for all properties

### OUTPUT FORMAT
First, output a brief **<reasoning>** block summarizing non-user patterns.
Then, output a valid **JSON Array**.""",

    "tool": """You are an expert system analyst specializing in tool usage patterns. Your task is to analyze tool calls and tool outputs in conversations, identifying patterns in tool usage, tool results, and tool interaction quality.

### INPUT CONTEXT
You are analyzing tool interactions for the following task:
<task_description>
{task_description}
</task_description>

**Your Goal:**
Produce a JSON list of objects. Each object should represent a single, distinct property about tool usage patterns. Focus on:
- **Tool Call Patterns:** Which tools are called, when, and why
- **Tool Output Quality:** Errors, formatting, completeness of tool results
- **Tool Integration:** How well tool results are incorporated
- **Tool Selection:** Appropriate vs inappropriate tool choices

```json
[
  {{
    "behavior_type": "Tool Error|Tool Success|Tool Pattern|Tool Integration",
    "property_description": "lowercase verb + specific tool behavior + context + outcome (1-3 sentences)",
    "category": "1-4 word category (e.g., 'Tool Call Error', 'Effective Usage', 'Output Parsing')",
    "evidence": "exact quote from tool calls or tool outputs",
    "reason": "1-2 sentence explanation of why this pattern is notable",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False",
    "role": "tool"
  }},
  ...
]
```

### ANALYSIS PROCESS
1. **Read Tool Messages:** Focus only on tool calls and tool outputs
2. **Identify Patterns:** Tool usage patterns, error patterns, integration patterns
3. **Assess Quality:** Tool selection appropriateness, output quality
4. **Draft Properties:** Write clear descriptions following the format

### BEHAVIOR TYPES
* **Tool Error:** Failed calls, parsing errors, timeout issues
* **Tool Success:** Effective tool use, good integration
* **Tool Pattern:** Recurring patterns in tool selection or usage
* **Tool Integration:** How tool results are used in responses

### CRITICAL CONSTRAINTS
* **EVIDENCE FROM TOOLS ONLY:** Quotes from tool calls/outputs only
* **OBSERVABLE PATTERNS:** Describe actual tool interactions
* Set "role": "tool" for all properties

### OUTPUT FORMAT
First, output a brief **<reasoning>** block summarizing tool interaction patterns.
Then, output a valid **JSON Array**.""",

    "system": """You are an expert system analyst. Your task is to analyze system messages in conversations, identifying patterns in system instructions, constraints, and system-level behaviors.

### INPUT CONTEXT
You are analyzing system messages for the following task:
<task_description>
{task_description}
</task_description>

**Your Goal:**
Produce a JSON list of objects. Each object should represent a single, distinct property about system-level patterns.

```json
[
  {{
    "behavior_type": "System Constraint|System Instruction|System Pattern",
    "property_description": "lowercase verb + specific system behavior + context (1-3 sentences)",
    "category": "1-4 word category",
    "evidence": "exact quote from system messages",
    "reason": "1-2 sentence explanation",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False",
    "role": "system"
  }},
  ...
]
```

### CRITICAL CONSTRAINTS
* **EVIDENCE FROM SYSTEM ONLY:** Quotes from system messages only
* Set "role": "system" for all properties

### OUTPUT FORMAT
First, output a brief **<reasoning>** block.
Then, output a valid **JSON Array**."""
}

# Generic template for custom roles
GENERIC_ROLE_TEMPLATE = """You are an expert conversation analyst. Your task is to analyze messages with role="{role}" and identify unique, meaningful properties and patterns.

### INPUT CONTEXT
You are analyzing "{role}" messages for the following task:
<task_description>
{task_description}
</task_description>

**Your Goal:**
Produce a JSON list of objects. Each object should represent a single, distinct property found in messages from the "{role}" role.

```json
[
  {{
    "behavior_type": "Pattern Type (choose appropriate)",
    "property_description": "lowercase verb + specific behavior + context + outcome (1-3 sentences)",
    "category": "1-4 word category",
    "evidence": "exact quote from {role} messages",
    "reason": "1-2 sentence explanation of why this pattern is notable",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False",
    "role": "{role}"
  }},
  ...
]
```

### ANALYSIS PROCESS
1. **Read Messages:** Focus only on messages with role="{role}"
2. **Identify Patterns:** Look for recurring behaviors, unique characteristics
3. **Assess Significance:** Focus on patterns that matter for task success
4. **Draft Properties:** Write clear descriptions

### CRITICAL CONSTRAINTS
* **EVIDENCE FROM {role} ONLY:** All quotes must come from messages with role="{role}"
* **NO HALLUCINATIONS:** Never fabricate or alter quotes
* **OBSERVABLE PATTERNS:** Describe what is actually present in the messages
* Set "role": "{role}" for all properties

### OUTPUT FORMAT
First, output a brief **<reasoning>** block summarizing patterns.
Then, output a valid **JSON Array**."""


def get_role_analysis_prompt(
    role: str,
    task_description: str = "",
    context: str | None = None
) -> str:
    """
    Get a prompt for analyzing a specific role using template-based generation.
    
    Args:
        role: Role to analyze (e.g., "user", "assistant", "tool", or custom role)
        task_description: Description of the task being analyzed
        context: Optional custom context to add to the prompt
        
    Returns:
        Formatted system prompt for role-specific analysis
        
    Examples:
        >>> prompt = get_role_analysis_prompt("user", "chatbot conversations")
        >>> prompt = get_role_analysis_prompt("customer", context="call center domain")
    """
    # Use predefined template if available
    if role.lower() in ROLE_TEMPLATES:
        template = ROLE_TEMPLATES[role.lower()]
    else:
        # Use generic template for custom roles
        template = GENERIC_ROLE_TEMPLATE.format(role=role)
    
    # Fill in task description
    prompt = template.replace("{task_description}", task_description or "General conversation analysis")
    
    # Add custom context if provided
    if context:
        prompt = prompt + f"\n\n### ADDITIONAL CONTEXT\n{context}"
    
    return prompt


def generate_role_discovery_prompt(
    role: str,
    sample_data: pd.DataFrame,
    task_description: str = "",
    discovery_generator: Any | None = None,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a role-specific prompt using dynamic discovery from sample data.
    
    This function uses the existing DiscoveryPromptGenerator to analyze
    sample conversations and create a custom prompt tailored to the specific
    role and data patterns.
    
    Args:
        role: Role to analyze (e.g., "user", "assistant", etc.)
        sample_data: DataFrame with conversation data
        task_description: Description of the task
        discovery_generator: Optional DiscoveryPromptGenerator instance
        model: LLM model for discovery
        
    Returns:
        Dynamically generated system prompt optimized for this role and data
        
    Notes:
        This requires the DiscoveryPromptGenerator from stringsight.prompts.dynamic
        and will be implemented after extending that generator with role support.
    """
    if discovery_generator is None:
        # Lazy import to avoid circular dependencies
        from ..dynamic.discovery_generator import DiscoveryPromptGenerator
        from ...core.caching import UnifiedCache
        discovery_generator = DiscoveryPromptGenerator(UnifiedCache())
    
    # This will be implemented after extending DiscoveryPromptGenerator
    # For now, fall back to template-based
    return get_role_analysis_prompt(role, task_description)


# Export main functions
__all__ = [
    "get_role_analysis_prompt",
    "generate_role_discovery_prompt",
    "ROLE_TEMPLATES"
]
