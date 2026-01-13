"""
Meta-prompt templates for dynamic prompt generation.

These templates are used to generate custom discovery prompt sections
that are tailored to specific tasks.
"""

INTRO_TASK_GENERATION_TEMPLATE = """You are creating a custom system prompt section for analyzing AI model behavior on a specific task.

**Task Description:**
{task_description}

**Analysis Method:** {method}

**Your Goal:**
Generate a concise "intro_task" section (2-3 sentences) that:
1. Describes the analyst's role
2. References the specific task context
3. Sets expectations for what behaviors to look for
4. Emphasizes actionable behaviors (can improve system or inform model choice)
5. **CRITICAL**: Emphasizes extracting ONE distinct behavior per property (not multiple behaviors combined)
6. **CRITICAL**: Warns against creating redundant properties that describe the same behavior with different wording

**Format Requirements:**
- Return ONLY the intro_task text
- No additional formatting, explanations, or quotes
- 2-3 sentences maximum
- Start with "You are..." or "Your task is..."
- Must mention: "Each property should describe ONE distinct behavior" and "avoid redundant properties"

**Example for a general chatbot:**
"You are an expert model behavior analyst. Your task is to meticulously analyze model responses and identify unique, meaningful qualitative properties, failure modes, and interesting behaviors. Each property should describe ONE distinct, concrete behavior with specific examples from the trace—avoid creating multiple properties that describe the same underlying behavior with different wording. Focus only on properties that genuinely matter to users, evaluators, or developers when judging model quality."

**Now generate an intro_task specifically for the task described above:**"""


GOAL_INSTRUCTIONS_GENERATION_TEMPLATE = """You are creating a custom system prompt section for analyzing AI model behavior on a specific task.

**Task Description:**
{task_description}

**Analysis Method:** {method}

**Your Goal:**
Generate a "goal_instructions" section (2-4 sentences) that:
1. States what output format to produce (JSON list of objects)
2. Mentions specific behavior categories relevant to THIS task
3. Emphasizes extracting ALL notable behaviors (typically 3-10 per trace) that affect user preference or task performance
4. **CRITICAL**: Emphasizes that each property should be distinct and non-redundant
5. References the JSON format template

**Format Requirements:**
- Return ONLY the goal_instructions text
- No additional formatting or explanations
- 2-4 sentences maximum
- Must mention "JSON list of objects", "behavior categories", and "distinct"

**Example for a general task:**
"Produce a JSON list of objects. Each object should represent a single, distinct property found in the model's response—ensure properties are not redundant or overlapping. Focus on identifying key areas of interest such as capabilities, style, errors, and user experience factors. Properties should be limited to those that could affect user preference or demonstrate how well the model understands and executes the task."

**Now generate goal_instructions specifically for the task described above:**"""


ANALYSIS_PROCESS_GENERATION_TEMPLATE = """You are creating a custom system prompt section for analyzing AI model behavior on a specific task.

**Task Description:**
{task_description}

**Analysis Method:** {method}

**Your Goal:**
Generate an "analysis_process" section (4-step process) that:
1. **Step 1 (Scan)**: Describes what to read in the trace, mentioning task-specific elements
2. **Step 2 (Filter)**: Describes what to focus on (high leverage, distinctive, structural behaviors)
3. **Step 3 (Draft)**: Instructs to write descriptions following the rubric **with emphasis on concrete, concise descriptions with specific examples from the trace**
4. **Step 4 (Deduplicate)**: Instructs to review the list and merge any redundant properties that describe the same behavior

**Format Requirements:**
- Return ONLY the analysis_process text
- Use numbered list format: "1. **Step Name:** Description"
- 4 steps exactly: Scan, Filter, Draft, Deduplicate
- Each step 1-2 sentences
- Mention task-specific behaviors in Step 1
- **CRITICAL**: Step 3 MUST emphasize: "Use 1-2 short sentences (max 20 words each). Include specific examples. Avoid run-on sentences and abstract/philosophical language."
- **CRITICAL**: Step 4 MUST emphasize: "Review your list for redundant properties. Merge any properties that describe the same underlying behavior with different wording."

**Example for a general task:**
"1. **Scan the Trace:** Read the user input, the model's internal thoughts (if available), the model's interaction with the user, the system of tools the model has access to, the environment, and the final output.
2. **Filter:** Ignore generic behaviors (e.g., 'Agent answered correctly'). Focus on behaviors that are **High Leverage** (critical success/failure), **Distinctive** (persona/style), or **Structural** (looping, adherence to format).
3. **Draft:** Write behavior descriptions in 1-2 short sentences (max 20 words each) following the **Definitions & Rubric** section. Include specific examples from the trace. Avoid run-on sentences with multiple clauses, abstract characterizations, and philosophical language.
4. **Deduplicate:** Review your list for redundant properties. Merge any properties that describe the same underlying behavior with different wording (e.g., 'uses friendly tone' and 'maintains warm language' should be one property)."

**Now generate an analysis_process specifically for the task described above:**"""


# Reflection templates for prompt fixing
REFLECTION_SYSTEM_PROMPT = """You are an expert at debugging and fixing prompts for LLMs. Your task is to analyze why a prompt failed to produce correct JSON output and generate a corrected version that will work."""


REFLECTION_PROMPT_TEMPLATE = """A prompt designed to extract structured properties from conversations failed verification. Analyze the failure and generate a corrected prompt.

**Task Context:**
{task_description}

**Method:** {method}

**Original Prompt:**
{original_prompt}

**Verification Failure:**
- Error Type: {error_type}
- Error Details: {error_details}

**What the LLM Actually Produced:**
{llm_output}

**Expected JSON Schema:**
{expected_schema}

**Your Task:**
1. Analyze why the original prompt failed:
   - Did it not emphasize the JSON format strongly enough?
   - Were the field names unclear?
   - Did it encourage extra commentary outside the JSON?
   - Were the instructions ambiguous?
   - Did it use unclear language about what counts as "behavior" or "property"?

2. Generate a corrected prompt that:
   - Fixes the identified issues
   - Maintains the same task goals and analysis objectives
   - Produces output matching the expected schema exactly
   - Is clear and unambiguous about JSON formatting requirements
   - Emphasizes returning ONLY the JSON array, no other text
   - Clearly defines what each field means and what format it should be in

**Output:**
Return ONLY the corrected prompt text (no explanations or meta-commentary before or after the prompt)."""


# Clustering customization templates
CLUSTERING_CUSTOMIZATION_SYSTEM_PROMPT = """You are an expert at adapting prompts to specific tasks. Your task is to naturally integrate task-specific context into clustering prompts without changing the core instructions or making the integration feel forced."""


CLUSTERING_CUSTOMIZATION_TEMPLATE = """You are given a base prompt used for {prompt_type} in an LLM behavior analysis pipeline. Your task is to adapt this prompt to be more relevant for a specific task by naturally integrating the task context into the instructions.

**Base Prompt:**
{base_prompt}

**Task Description:**
{task_description}

**Your Goal:**
Rewrite the base prompt to naturally incorporate awareness of this specific task. The adapted prompt should:

1. **Maintain all core instructions**: Keep the same requirements about output format, quality guidelines, and structural rules
2. **Natural integration**: Weave task awareness into the existing sentences rather than adding separate "Task Context" sections
3. **Specific guidance**: Where the base prompt talks about "behaviors" or "properties" generically, make references more specific to this task where appropriate
4. **Keep it concise**: Don't make the prompt significantly longer; the goal is natural integration, not addition

**Examples of Natural Integration:**
- Instead of: "Given a list of behaviors..." → "Given a list of behaviors from responses to this airline booking task..."
- Instead of: "Focus on meaningful behaviors" → "Focus on behaviors that reveal how models handle complex booking scenarios"
- Instead of: "...that could affect user preference..." → "...that could affect a user's satisfaction with the booking experience..."

**Important:**
- DO NOT just append task context at the end with a "Task Context:" header
- DO integrate task awareness naturally throughout the prompt
- DO maintain all formatting requirements, quality guidelines, and output specifications from the base prompt
- DO keep the same level of detail and instruction quality

**Output:**
Return ONLY the adapted prompt text (no explanations, no meta-commentary, just the rewritten prompt)."""
