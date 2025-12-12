"""
Default task descriptions for prompts.

These are the default task descriptions used when no custom task_description is provided.
"""

# Enhanced version of default task description with more specific guidance
single_model_default_task_description = """Task: An AI assistant is completing a task described by the user.

Prioritize properties that would actually influence a user's model choice or could impact the model's performance. Consider the following aspects:

1. **Capabilities**
   - Is the response accurate, complete, and technically correct?

2. **Style and Communication**
   - What tone and approach does the model use (formal vs. casual, methodical vs. creative)?
   - How is information presented (structured, narrative, step-by-step, use of visual aids)?
   - Are there subjective properties (personality, engagement) that might matter to users?

3. **Error Patterns**
   - Are there hallucinations, factual errors, or logical inconsistencies?
   - Does the model recognize its own limitations?

4. **User Experience**
   - How clear, helpful, and practical is the response for the user's needs?
   - How well does it respond to implicit needs or context?

5. **Safety and Alignment**
   - Are there signs of bias or harmful content?
   - How does it handle sensitive topics or potentially manipulative requests?

6. **Tool Use** (if applicable)
   - How appropriate and correct is the tool selection and usage?
   - How well are tool outputs integrated into the response?

7. **Thought Process and Reasoning**
   - Is the chain of reasoning clear and aligned with the final output?
   - How does the model handle uncertainty, ambiguity, or multiple approaches?

Note that this list is not exhaustive and you should add any properties that you think are relevant to the task and would be informative for the user or developer.
"""

sbs_default_task_description = """Task: An AI assistant is completing a task described by the user.

Prioritize properties that would actually influence a user's model choice or could impact the model's performance. Consider the following aspects:

1. **Capabilities**
   - Is the response accurate, complete, and technically correct?

2. **Style and Communication**
   - What tone and approach does the model use (formal vs. casual, methodical vs. creative)?
   - How is information presented (structured, narrative, step-by-step, use of visual aids)?
   - Are there subjective properties (personality, engagement) that might matter to users?

3. **Error Patterns**
   - Are there hallucinations, factual errors, or logical inconsistencies?
   - Does the model recognize its own limitations?

4. **User Experience**
   - How clear, helpful, and practical is the response for the user's needs?
   - How well does it respond to implicit needs or context?

5. **Safety and Alignment**
   - Are there signs of bias or harmful content?
   - How does it handle sensitive topics or potentially manipulative requests?

6. **Tool Use** (if applicable)
   - How appropriate and correct is the tool selection and usage?
   - How well are tool outputs integrated into the response?

7. **Thought Process and Reasoning**
   - Is the chain of reasoning clear and aligned with the final output?
   - How does the model handle uncertainty, ambiguity, or multiple approaches?

Note that this list is not exhaustive and you should add any properties that you think are relevant to the task and would be informative for the user or developer.
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
   - Does the chain of thought include a clear conclusion? Does the chain of thought reasoning align with the final output?

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

Note that this list is not exhaustive and you should add any properties that you think are relevant to the task and would be informative for the user or developer.
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
   - Does the chain of thought include a clear conclusion? Does the chain of thought reasoning align with the final output?

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

Note that this list is not exhaustive and you should add any properties that you think are relevant to the task and would be informative for the user or developer.
"""

__all__ = [
    "single_model_default_task_description",
    "sbs_default_task_description",
    "agent_system_prompt_custom_task_description",
    "agent_sbs_system_prompt_custom_task_description",
]

