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

**Method-Specific Requirements:**
- If method is "single_model":
  - Focus on analyzing individual model behaviors and characteristics
  - Language: "analyze", "identify", "examine", "evaluate"
  - Emphasis: What behaviors does THIS model exhibit?

- If method is "side_by_side":
  - Focus on COMPARISON and DIFFERENTIATION between two models
  - Language: "compare", "contrast", "differentiate", "distinguish"
  - Emphasis: What behaviors DIFFER between Model A and Model B?
  - **CRITICAL**: Must explicitly mention comparing two responses
  - Must emphasize extracting distinguishing properties from EACH model

**Format Requirements:**
- Return ONLY the intro_task text
- No additional formatting, explanations, or quotes
- 2-3 sentences maximum
- Start with "You are..." or "Your task is..."
- Must mention: "Each property should describe ONE distinct behavior" and "avoid redundant properties"

**Example for a general chatbot (single_model):**
"You are an expert model behavior analyst. Your task is to meticulously analyze model responses and identify unique, meaningful qualitative properties, failure modes, and interesting behaviors. Each property should describe ONE distinct, concrete behavior with specific examples from the trace—avoid creating multiple properties that describe the same underlying behavior with different wording. Focus only on properties that genuinely matter to users, evaluators, or developers when judging model quality."

**Example for side_by_side analysis:**
"You are an expert model behavior analyst. Your task is to **compare** two model responses (Model A and Model B) and identify unique, meaningful properties that **differentiate** them. Each property should describe ONE distinct, concrete difference with specific examples from the traces—avoid creating multiple properties that describe the same underlying difference with different wording. Focus only on distinguishing behaviors that genuinely matter when choosing between models."

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

**Method-Specific Requirements:**
- If method is "single_model":
  - Instruct to extract properties from the single model response
  - Focus on capabilities, errors, style, and task performance

- If method is "side_by_side":
  - Instruct to extract DISTINGUISHING properties from BOTH models
  - Explicitly mention "Extract properties from both Model A and Model B"
  - Emphasize "differences that affect user preference or model comparison"
  - Must ensure balanced coverage (properties from both models)

**Format Requirements:**
- Return ONLY the goal_instructions text
- No additional formatting or explanations
- 2-4 sentences maximum
- Must mention "JSON list of objects", "behavior categories", and "distinct"

**Example for a general task (single_model):**
"Produce a JSON list of objects. Each object should represent a single, distinct property found in the model's response—ensure properties are not redundant or overlapping. Focus on identifying key areas of interest such as capabilities, style, errors, and user experience factors. Properties should be limited to those that could affect user preference or demonstrate how well the model understands and executes the task."

**Example for side_by_side:**
"Produce a JSON list of objects. Each object should represent a single, distinct property that differentiates Model A from Model B—ensure properties are not redundant or overlapping. Extract distinguishing properties from BOTH models, ensuring balanced coverage. Focus on differences in capabilities, style, errors, and user experience factors that could affect preference when comparing the models. Each property must include a 'model' field indicating which model exhibits the behavior."

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

**Method-Specific Requirements:**
- If method is "single_model":
  - Step 1 should mention reading "the trace" (singular)
  - Step 2 should focus on behaviors that are distinctive or impactful
  - No need to mention model comparison

- If method is "side_by_side":
  - Step 1 should mention reading "BOTH traces" and comparing them
  - Step 2 should focus on DIFFERENTIATING behaviors between models
  - Must explicitly say "Focus on differences between Model A and Model B"
  - Step 3 should mention ensuring balanced coverage from both models

**Format Requirements:**
- Return ONLY the analysis_process text
- Use numbered list format: "1. **Step Name:** Description"
- 4 steps exactly: Scan, Filter, Draft, Deduplicate
- Each step 1-2 sentences
- Mention task-specific behaviors in Step 1
- **CRITICAL**: Step 3 MUST emphasize: "Use 1-2 short sentences (max 20 words each). Include specific examples. Avoid run-on sentences and abstract/philosophical language."
- **CRITICAL**: Step 4 MUST emphasize: "Review your list for redundant properties. Merge any properties that describe the same underlying behavior with different wording."

**Example for single_model:**
"1. **Scan the Trace:** Read the user input, the model's internal thoughts (if available), and the final output.
2. **Filter:** Focus on behaviors that are **High Leverage** (critical success/failure), **Distinctive** (unique style), or **Structural** (patterns).
3. **Draft:** Write behavior descriptions in 1-2 short sentences (max 20 words each). Include specific examples from the trace. Avoid run-on sentences and abstract language.
4. **Deduplicate:** Review your list for redundant properties. Merge properties describing the same underlying behavior."

**Example for side_by_side:**
"1. **Scan Both Traces:** Read the user input and compare Model A's response with Model B's response, including internal thoughts if available.
2. **Filter:** Focus on DIFFERENTIATING behaviors between Model A and Model B. Ignore generic similarities. Look for **High Leverage** differences (critical success/failure), **Distinctive** differences (style, approach), or **Structural** differences (format, patterns).
3. **Draft:** Write behavior descriptions in 1-2 short sentences (max 20 words each) that clearly identify WHICH model exhibits the behavior. Include specific examples from the traces. Ensure balanced coverage of both models.
4. **Deduplicate:** Review your list for redundant properties. Merge properties describing the same underlying difference."

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
   - If method is "side_by_side": Did it fail to emphasize comparison and differentiation between models?

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
