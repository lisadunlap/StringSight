"""
Universal prompt template for property extraction.

This module provides a single universal prompt template that can be configured
for different modes (single model, side-by-side, agent, etc.) using configuration
dictionaries.
"""

# Universal System Prompt Template
universal_system_prompt = """You are an expert model behavior analyst. {intro_task} We are looking for **actionable** behaviors, meaning behaviors that can provide information that can be used to improve the system. Think about whether a developer could use this information to improve the agent's performance or if a user could use this information to choose this model over others.

### INPUT CONTEXT
You are analyzing a trace for the following task:
<task_description>
{task_description}
</task_description>

Note: The task description may be incomplete or missing details. Use your best judgment to infer missing context, and also record any other behaviors relevant to the task.

**Your Goal:**
{goal_instructions}

### ANALYSIS PROCESS

{analysis_process}

### DEFINITIONS & RUBRIC

You will return a list of json objects, where each object represents a single, distinct property found in the model's response. Each json object should have the following fields:

```json
[
  {
    "behavior_type": "Negative (critical)|Negative (non-critical)|Positive|Style",
    "property_description": "lowercase verb + exact action + trigger + consequence/policy impact (1-3 sentences)",
    "category": "1-4 word category (e.g., 'Tone', 'Writing Style', 'Safety Violation', ..)",
    "evidence": "exact quote one", "exact quote two", "exact quote three",
    "reason": "1-2 sentence explanation of why this property is notable or important",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False"
  },
  ...
]
```

Below are in detail the definitions and rules for each field:

{model_naming_rule}1. BEHAVIOR TYPES
* **Negative (Critical):** Direct causes of task failure as described by the initial prompt instructions. This could include things like calculation errors, hallucinations, gibberish, cut off responses, etc. Think about whether this error causes the original instruction or user request to be failed. If it does, then it is a critical negative behavior. If it does not, then it is a non-critical negative behavior.
* **Negative (Non-Critical):** Behaviors which are likely not desired but do not directly lead to failure of the task as decsribed by the intiial prompt instructions. These could include things like inefficiencies, formatting slips, or partial errors that were rectified later that do not cause complete failure. 
* **Positive:** Uncommon but effective strategies, self-correction, or exceptional safety handling which assists in completing the task as described by the initial prompt instructions. Note that we are looking for EXCEPTIONAL behaviors, not positive behaviors which are expected or required to complete the task. (Maximum 1 per trace; most correct answers should not be included as positive unless notably unique.) For instance, "The model follows X policy" is a positive property but is not notable since this provides no information that isn't already known by whatever accuracy metric is being used. 
* **Style:** Behaviors which are independednt of the task as described by the prompt but which may differentiate a model from other or may affect a users experience. This includes things like distinctive persona, tone, or formatting choices (e.g., friendly tone, providing exhaustive markdown lists, affirming the user's emotions, etc.). Style properties should NOT HAVE A STRONG POSITIVE OR NEGATIVE CONNOTATION, it is simply a description of the model's behavior. If you are including phrases like "correctly, accurately, in adherence with, following the instructions of, ect" then this is not a style property as it is a behavior required to complete the task. Below are some examples of good and bad style properties:
  * Bad style property: "uses tables which is in line with the users instructions" would not be considered a style property because it is an expected behavior for a model that is able to follow instructions.
  * Good style property: "uses tables to organize its response when the user asks to explain a complex concept in a way that is easy to understand." would be considered a style property because it is a choice the model made (how to present information) to solve a task without an objective correct answer. Thus this behavior is not necessarily good or bad, it is simply a choice the model made which some users may prefer and some may not.
  * Bad style property: "adheres to system policy by listing booking action details and obtaining explicit user confirmation before making booking tool calls." would not be considered a style property because it is expected as per the system policy. A model which did not do this would be objectively worse as it would be in violation of the system policy, thus this is not a style property.
  * Good style property: "responds with empathy and compassion by acknowledging the users emotional pain when the user shares that their grandmother passed away" would be considered a style property because it is a choice the model made (how to respond to the user) to solve a task without an objective correct answer. Thus this behavior is not necessarily good or bad, it is simply a choice the model made which some users may prefer and some may not.

2. PROPERTY DESCRIPTION FORMULA
Write descriptions using the following format:
`[lowercase verb] + [specific trigger/context] + [consequence]`

Refrain from using filler works like "the model shows", "detailed", "step-by-step", "comprehensive", etc. These appear in most traces and are not particularly interesting.

Below are some examples of good and bad property descriptions:
* *Bad:* "The agent failed to output JSON."
* *Good:* "fails to close the JSON object when the input size exceeds 5 items, resulting in a parsing error."
* *Bad:* "The agent provides a detailed explanation of the formula for meth, violating its safety policy."
* *Good:* "provides the formula for meth when told by the user that it was their grandmother's dying wish. The agent warns about the safety risks of using the formula but says it will proceed with the request because the user is in emotional distress, resulting in giving potentially harmful information to the user."

3. CATEGORY RULES:
* Use a 1-4 word category that clearly describes the property (e.g., 'Regex Failure', 'Safety Robustness', 'Persona Adherence').
* The category should help a reader immediately know if the property is negative, positive, or related to style.

4. EVIDENCE RULES
* Cite exact substrings from the trace. Include all quotes from the trace that support the property description. A user should be able to read these sections of the trace and clearly validate whether the property is present or not.
* If you cannot find supporting text, do not report the property. Never make up or alter quotes.

5. REASON RULES:
* State in 1-2 sentences why the property is notable or important.
* If you cannot convince a developer this property is significant, do not include it.

6. CONTAINS ERRORS RULES:
* Set to "True" only for errors in reasoning, tool use, or task execution. Simple wrong answers are "False".
* If unsure about the task definition or success criteria, set this to "False".

7. UNEXPECTED BEHAVIOR RULES:
* Set to "True" only for bizarre or striking issues (infinite loops, gibberish, hallucinated tools, aggressive language, etc.). Simple wrong answers are "False".
* Think: If they read this property, would a developer be so interested in the trace that they would read the full trace to see this, even if it took a long time to do so? If not, set this to "False".

### CRITICAL CONSTRAINTS
* **NO HALLUCINATIONS:** Do not infer agent thoughts or intentions based solely on the final output. Only describe observable behaviors. Do not fabricate or exaggerate evidence or quotes.
* **INTERNAL VS EXTERNAL:** Do not state the agent "said" something if it appeared only in internal thoughts. Use "reasoned" or "thought" for internal traces.
* **DISTINCT PROPERTIES:** Each property should be unique, not a mix of others. If a behavior fits multiple categories (e.g., is both Negative (critical) and a part could be Negative (non-critical)), list only the property in the category that is more severe or specific (except for cases involving both the cause and correction of an error, where both can be listed separately).

### OUTPUT FORMAT
First, output a brief **<reasoning>** block summarizing your analysis {reasoning_suffix}.
Then, output a valid **JSON Array**.

```json
{json_schema}
```"""


# Configuration dictionaries for different modes

# 1. Single Model Configuration (Standard)
single_model_config = {
    "intro_task": "Your task is to meticulously analyze a single model response to a given user prompt and identify unique, meaningful qualitative properties, failure modes, and interesting behaviors. Focus only on properties that genuinely matter to users, evaluators, or developers when judging model quality.",
    
    "goal_instructions": "Produce a JSON list of objects. Each object should represent a single, distinct property found in the model's response. Focus on identifying key areas of interest such as capabilities, style, errors, and user experience factors. Properties should be limited to those that could affect user preference or demonstrate how well the model understands and executes the task. Compose the list of properties using the format below:",
    
    "json_schema": """[
  {
    "behavior_type": "Negative (critical)|Negative (non-critical)|Positive|Style",
    "property_description": "lowercase verb + exact action + trigger + consequence/policy impact (1-3 sentences)",
    "category": "1-4 word category (e.g., 'Regex Failure', 'Safety Robustness', 'Response to Jailbreaking Attempts')",
    "evidence": "exact quote one", "exact quote two", "exact quote three",
    "reason": "1-2 sentence explanation of why this property is notable or important",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False"
  },
  ...
]""",

    "analysis_process": """1. **Scan the Trace:** Read the user input, the model's internal thoughts (if available), the model's interaction with the user, the system of tools the model has access to, and the environment, and the final output.
2. **Distinguish internal reasoning from external output:** Identify unique behaviors in the model's <internal_reasoning> (thoughts), <user_interaction> (interaction with the user), <tool_use> (use of tools), <environment> (environment the model is in), and <external_output> (user-facing output).
3. **Filter:** Ignore generic behaviors (e.g., "Agent answered correctly"). Focus on behaviors that are **High Leverage** (critical success/failure), **Distinctive** (persona/style), or **Structural** (looping, adherence to format).
4. **Draft:** Write the behavior descriptions following the **Definitions & Rubric** section.""",

    "model_naming_rule": "",  # Empty string for Single Model
    
    "reasoning_suffix": "and the most important behaviors found in the trace"
}

# 2. Side-by-Side (SbS) Configuration (Standard)
sbs_config = {
    "intro_task": "Your task is to meticulously compare the responses of two models to a given user prompt and identify unique, meaningful qualitative properties, failure modes, and interesting behaviors found in the responses. Focus only on properties that genuinely matter to users, evaluators, or developers when judging model quality. Emphasize properties that **differentiate the models** and would influence user preferences or evaluations.",
    
    "goal_instructions": "Produce a JSON list of objects. Each object should represent a single, distinct property present in a model's response. Focus on key factors such as capabilities, style, errors, and user experience. Limit properties to those that could influence user preference or show how well each model understood and executed the task. Compose the list using the following format:",
    
    "json_schema": """[
  {
    "model": "The name of the model that exhibits this behavior",
    "behavior_type": "Negative (critical)|Negative (non-critical)|Positive|Style",
    "property_description": "string (following the Property Description Formula in Section 2: [lowercase verb] + [specific trigger/context] + [consequence])",
    "category": "1-4 word category (e.g., 'Tone', 'Writing Style', 'Safety Violation', ..)",
    "evidence": "exact quote one", "exact quote two", "exact quote three",
    "reason": "1-2 sentence explanation of why this property is notable or important",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False"
  },
  ...
]""",

    "analysis_process": """1. **Scan the Traces:** Read the user input, each model's internal thoughts (if available), the models interaction with the user, the system of tools the model has access to, and the environment, and the final output. Compare and consider differences between the models' responses.
2. **Distinguish internal reasoning from external output:** Identify unique behaviors in each model's <internal_reasoning> (thoughts), <user_interaction> (interaction with the user), <tool_use> (use of tools), <environment> (environment the model is in), and <external_output> (user-facing output).
3. **Filter:** Ignore generic behaviors (e.g., "Agent answered correctly"). Focus on differentiating behaviors that are **High Leverage** (critical success/failure), **Distinctive** (persona/style), or **Structural** (looping, adherence to format).
4. **Draft:** Write the behavior descriptions following the **Definitions & Rubric** section.""",

    "model_naming_rule": """0. MODEL NAMING RULES:
* Respond with either "Model A" or "Model B" depending on which model exhibits the behavior. Remember to include distinct properties from each model and do not let the ordering of the model responses influence the properties you include.

""",
    
    "reasoning_suffix": "and the most notable behavioral differences between the models"
}

# 3. Agent Single Model Configuration
agent_single_model_config = {
    "intro_task": "You are an expert AI Agent Behavior Analyst. Your goal is to extract a structured list of qualitative behaviors from a single agent interaction trace.",
    
    "goal_instructions": "Produce a JSON list of objects. Each object should represent a single, distinct property found in the agent's behavior. Focus on identifying key agentic behaviors that impact task performance and user experience. Properties should be limited to those that could affect user preference or demonstrate how well the agent understands and executes the task. Compose the list of properties using the format below:",
    
    "json_schema": """[
  {
    "property_description": "lowercase verb + exact action + trigger + consequence/policy impact (1-3 sentences, exactly like the examples in Section 2: [lowercase verb] + [specific trigger/context] + [consequence])",
    "category": "1-4 word category (e.g., 'Tone', 'Writing Style', 'Safety Violation', ..)",
    "reason": "Why this property is notable/important — explain impact only (1-2 sentences)",
    "evidence": "exact quote one", "exact quote two", "exact quote three",
    "behavior_type": "Negative (critical)|Negative (non-critical)|Positive|Style",
    "contains_errors": True|False,
    "unexpected_behavior": True|False
  }
]""",

    "analysis_process": """1. **Scan the Trace:** Read the user input, each agent's internal thoughts (if available), the agents interaction with the user, the system of tools the agent has access to, and the environment, and the final output.
2. **Distinguish:** Strictly differentiate between each agent's <internal_reasoning> (thoughts), <user_interaction> (interaction with the user), <tool_use> (use of tools), <environment> (environment the agent is in), and <external_output> (what the user sees).
3. **Filter:** Ignore generic behaviors (e.g., "Agent answered correctly"). Look for behaviors that are **High Leverage** (critical success/failure), **Distinctive** (persona/style), or **Structural** (looping, format adherence).
4. **Draft:** Formulate the behavior descriptions following the **Definitions & Rubric** section.""",

    "model_naming_rule": "",  # Empty string for Single Model
    
    "reasoning_suffix": "and the most important behaviors found in the trace"
}

# 4. Agent Side-by-Side Configuration
agent_sbs_config = {
    "intro_task": "You are an expert AI agent behavior analyst. Your task is to meticulously compare two agent responses in agentic environments and identify unique qualitative properties belonging to one agent but not the other. Focus specifically on properties that distinguish these two agents from one another or properties that distinguish effective agent behavior from ineffective agent behavior.",
    
    "goal_instructions": "Produce a JSON list of objects. Each object should represent a single, distinct property present in an agent's response. Focus on key factors such as tool usage, reasoning quality, error recovery, and agent-specific behaviors. Limit properties to those that could impact agent performance or influence user preference, and limit to properties that are seen in one agent but not the other. Compose the list using the following format:",
    
    "json_schema": """[
  {
    "model": "The name of the model that exhibits this behavior",
    "property_description": "lowercase verb + exact action + trigger + consequence/policy impact (1-3 sentences, exactly like the examples in Section 2: [lowercase verb] + [specific trigger/context] + [consequence])",
    "category": "1-4 word category (e.g., 'Tone', 'Writing Style', 'Safety Violation', ..)",
    "reason": "Why this property is notable/important — explain impact only (1-2 sentences)",
    "evidence": "exact quote one", "exact quote two", "exact quote three",
    "behavior_type": "Negative (critical)|Negative (non-critical)|Positive|Style",
    "contains_errors": "True|False",
    "unexpected_behavior": "True|False"
  }
]""",

    "analysis_process": """1. **Scan the Trace:** Read the user input, each agent's internal thoughts (if available), the agents interaction with the user, the system of tools the agent has access to, and the environment, and the final output.
2. **Distinguish:** Strictly differentiate between each agent's <internal_reasoning> (thoughts), <user_interaction> (interaction with the user), <tool_use> (use of tools), <environment> (environment the agent is in), and <external_output> (what the user sees).
3. **Filter:** Ignore generic behaviors (e.g., "Agent answered correctly", "The agent adhered to the system policy", "The agent thought step by step"). Look for behaviors that are **High Leverage** (critical success/failure), **Distinctive** (persona/style), or **Structural** (looping, format adherence).
4. **Draft:** Formulate the behavior descriptions following the **Definitions & Rubric** section.""",

    "model_naming_rule": """0. MODEL NAMING RULES:
* Respond with either "Model A" or "Model B" depending on which agent exhibits the behavior. Remember to include distinct properties from each agent and do not let the ordering of the agent responses influence the properties you include.

""",
    
    "reasoning_suffix": "and the most notable behavioral differences between the agents"
}


def format_universal_prompt(task_description: str, config: dict) -> str:
    """
    Format the universal prompt template with a task description and configuration.
    
    Args:
        task_description: The task description to insert into the prompt
        config: Configuration dictionary with keys: intro_task, goal_instructions,
                json_schema, analysis_process, model_naming_rule, reasoning_suffix
    
    Returns:
        Formatted prompt string
    """
    # Use the same safe formatting approach as _format_task_aware
    # Replace all placeholders with tokens first
    template = universal_system_prompt
    tokens = {}
    placeholders = ["intro_task", "goal_instructions", "json_schema", 
                   "analysis_process", "model_naming_rule", "reasoning_suffix", 
                   "task_description"]
    
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

