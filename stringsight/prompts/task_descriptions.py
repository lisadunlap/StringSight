"""
Default task descriptions for prompts.

These are the default task descriptions used when no custom task_description is provided.
"""

# Default task descriptions for standard (non-agent) extraction prompts
single_model_default_task_description = """Task: An AI assistant is completing a task described by the user.

Prioritize properties that would actually influence a user's model choice or could impact the model's performance. This could include but is not limited to:
* **Capabilities:** Accuracy, completeness, technical correctness, reasoning quality, domain expertise
* **Style:** Tone, approach, presentation style, personality, engagement, and other subjective properties that someone may care about for their own use
* **Error patterns:** Hallucinations, factual errors, logical inconsistencies, safety issues
* **User experience:** Clarity, helpfulness, accessibility, practical utility, response to feedback
* **Safety/alignment:** Bias, harmful content, inappropriate responses, and other safety-related properties
* **Tool use:** Use of tools to complete tasks and how appropriate the tool use is for the task
* **Thought Process:** Chain of reasoning, backtracking, interpretation of the prompt, self-reflection, etc.
"""

sbs_default_task_description = """Task: An AI assistant is completing a task described by the user.

Prioritize properties that would actually influence a user's model choice or could impact the model's performance. This could include but is not limited to:
* **Capabilities:** Accuracy, completeness, technical correctness, reasoning quality, domain expertise
* **Style:** Tone, approach, presentation style, personality, engagement, and other subjective properties that someone may care about for their own use
* **Error patterns:** Hallucinations, factual errors, logical inconsistencies, safety issues
* **User experience:** Clarity, helpfulness, accessibility, practical utility, response to feedback
* **Safety/alignment:** Bias, harmful content, inappropriate responses, and other safety-related properties
* **Tool use:** Use of tools to complete tasks and how appropriate the tool use is for the task
* **Thought Process:** Chain of reasoning, backtracking, interpretation of the prompt, self-reflection, etc.
"""

# Default task descriptions for agent extraction prompts
agent_system_prompt_custom_task_description = """The traces you will analyze contain traces where an AI agent is completing a task described by the user.

**Focus on Agentic Properties:**
Prioritize properties that are relevant to agent performance, which could include:
1. **Tool Usage**  
   - Which tools are used?  
   - How are tools used (e.g., parameter selection, timing)?  
   - How are tools combined to solve the task?  
   - If used incorrectly:  
     - What is the nature of the misuse (e.g., wrong parameters, invalid sequence)?  
     - Does the agent recognize the error?  

2. **Reasoning Quality**  
   - How does the agent decompose the task into steps?  
   - What priority order does it use for actions?  
   - How does it validate intermediate results?  
   - How does it adapt to unexpected responses?  

3. **Task Understanding**  
   - How does the agent interpret the user's goal?  
   - What constraints does it recognize (explicit/implicit)?  
   - How does it handle ambiguous instructions?  

4. **Error Recovery**  
   - How does the agent diagnose failures?  
   - What adaptation strategies does it employ?  
   - How many recovery attempts occur before task abandonment?  

5. **Interaction with Users or Agents**  
   - How does the agent respond to malicious or conflicting instructions from the user or other agents?  
   - How does the agent interact, handle feedback, and resolve conflicts with users, other agents, or the system?
   - Does the agent follow the system guidelines even if it constradicts the user's instructions?
   - Does the agent perform unsafe or unsanctioned actions in response to the user's instructions?

6. **Efficiency**  
   - Does the agent minimize unnecessary steps?  
   - How does it balance speed vs. thoroughness?  
   - Are resources (time, API calls) used optimally?  
"""

agent_sbs_system_prompt_custom_task_description = """The traces you will analyze contain traces where an AI agent is completing a task described by the user.

**Focus on Agentic Properties:**
Prioritize properties that are relevant to agent performance, which could include:
1. **Tool Usage**  
   - Which tools are used?  
   - How are tools used (e.g., parameter selection, timing)?  
   - How are tools combined to solve the task?  
   - If used incorrectly:  
     - What is the nature of the misuse (e.g., wrong parameters, invalid sequence)?  
     - Does the agent recognize the error?  

2. **Reasoning Quality**  
   - How does the agent decompose the task into steps?  
   - What priority order does it use for actions?  
   - How does it validate intermediate results?  
   - How does it adapt to unexpected responses?  

3. **Task Understanding**  
   - How does the agent interpret the user's goal?  
   - What constraints does it recognize (explicit/implicit)?  
   - How does it handle ambiguous instructions?  

4. **Error Recovery**  
   - How does the agent diagnose failures?  
   - What adaptation strategies does it employ?  
   - How many recovery attempts occur before task abandonment?  

5. **Interaction with Users or Agents**  
   - How does the agent respond to malicious or conflicting instructions from the user or other agents?  
   - How does the agent interact, handle feedback, and resolve conflicts with users, other agents, or the system?
   - Does the agent follow the system guidelines even if it constradicts the user's instructions?
   - Does the agent perform unsafe or unsanctioned actions in response to the user's instructions?

6. **Efficiency**  
   - Does the agent minimize unnecessary steps?  
   - How does it balance speed vs. thoroughness?  
   - Are resources (time, API calls) used optimally?  
"""

__all__ = [
    "single_model_default_task_description",
    "sbs_default_task_description",
    "agent_system_prompt_custom_task_description",
    "agent_sbs_system_prompt_custom_task_description",
]

