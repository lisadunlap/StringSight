"""Predefined demo example configurations for the Gradio launcher.

Each demo contains:
- data_path: absolute path to the dataset file
- explain: parameters for the Explain pipeline (aligned with exposed UI controls)
- label: parameters for the Label pipeline (aligned with exposed UI controls)
- advanced: shared advanced parameters (optional)
"""

from __future__ import annotations

from typing import Dict, Any, List


# Single initial example using the existing project demo file and default params
EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Summarizing IT Support Calls": {
        "data_path": "data/demo_data/call_center.jsonl",
        "explain": {
            "method": "single_model",
            "system_prompt": "single_model_system_prompt",
            "clusterer": "hdbscan",
            "min_cluster_size": 8,
            "assign_outliers": False,
            "groupby_column": "behavior_type",
        },
        "label": {
            "taxonomy": {
                "incorrectly states resolution": "The model incorrectly says how/if the issue was resolved.",
                "fabricates information": "The model fabricates information about the transcript.",
                "missing important information": "The model does not include important details about the user, the problem, or the resolution in its summary.",
            },
            "label_model_name": "gpt-5-mini",
        },
        "advanced": {
            "sample_size": None,
            "max_workers": 64,
            "use_wandb": False,
            "verbose": True,
        },
    },
    "Airline Agent Customer Service": {
        "data_path": "data/demo_data/taubench_airline.jsonl",
        "explain": {
            "method": "single_model",
            "system_prompt": "agent_system_prompt",
            "clusterer": "hdbscan",
            "min_cluster_size": 8,
            "assign_outliers": False,
            "groupby_column": "behavior_type",
        },
        "label": {
            "taxonomy": {
                "Disobey Task Specification": "This error occurs when an agent or system fails to adhere to specified constraints, guidelines, or requirements associated with a particular task. Non-compliance can result from unclear, incomplete, or ambiguous instructions provided by the user, system prompts, or task descriptions. It may also arise from an agent's inadequate ability to interpret or apply constraints effectively. Consequences of poor task constraint compliance include incorrect, suboptimal, or irrelevant outputs, reduced system performance and increased resource consumption.",
                "Disobey Role Specification": "Failure to adhere to the defined responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another.",
                "Step Repetition": "Step repetition occurs when an agent or system unnecessarily repeats a phase, a task, a stage that have already been completed. Such redundancy can arise from inadequate state or context tracking, inefficient workflow management, unclear or ambiguous instructions, or failure to recognize completed tasks.\n\nAn example of step repetition is in the following Hyperagent trace where the Planner repeated exactly the same thought twice. \n\n[...]\nHyperAgent_matplotlib__matplotlib-23563 - INFO - Planner's Response: Thought: To address this issue, we need to understand the root cause of the 'Line3D' object not having the '_verts3d' attribute. This error typically occurs when the data passed to the plot function is invalid or when there's an issue with the 3D rendering. Let's start by examining the code and reproducing the issue.\n[...]\nHyperAgent_matplotlib__matplotlib-23563 - INFO - Planner's Response: Thought: To address this issue, we need to understand the root cause of the 'Line3D' object not having the '_verts3d' attribute. This error typically occurs when the data passed to the plot function is invalid or when there's an issue with the 3D rendering. Let's start by examining the code and reproducing the issue.\n[...]\nIntern Name: Codebase Editor\nSubgoal: Please prepare a response for the user with the following content:\n1. Explain the identified issue and recommend updating matplotlib to version 3.5.1 or later.\n2. Provide a modified version of their code that attempts to work around the issue.\n3. Suggest an alternative plotting method using scatter plots instead of lines.\n[...]",
                "Loss of Conversation History": "Unexpected context truncation, disregarding recent interaction history and reverting to an antecedent conversational state.",
                "Unaware of Termination Conditions": "This error occurs when an agent or system fails to adhere to criteria designed to trigger the termination of an interaction, conversation, phase, or task. Such oversight can arise due to ambiguous, incomplete, or poorly defined stopping conditions, ineffective monitoring mechanisms, or inadequate understanding of the criteria by the agent.\n\nConsequences of being unaware of stopping conditions include unnecessary conversation turns, inefficient use of resources, or potential harm to the correctness of the system due to performing actions beyond what is necessary.\n\nAn example of \"Unaware of Stopping Conditions\" involving two agents attempting to collaboratively solve a math problem is the following:\n{\n    \"content\": [\n        \"In the current problem, I don't have enough information to solve it. If I were to ask you for the relevant missing information now, it would be:\",\n        \"1. What is the total length of the ribbon that Monica has?\",\n        \"or\",\n        \"2. How many inches of ribbon does Monica use for each gift bow?\",\n        \"\",\n        \"Could you please provide either piece of information so that I can solve the problem? Without either piece, the problem remains unsolvable.\"\n    ],\n    \"role\": \"user\",\n    \"name\": \"assistant\"\n},\n{\n    \"content\": [\n        \"Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\\\boxed{}.)\"\n    ],\n    \"role\": \"assistant\",\n    \"name\": \"mathproxyagent\"\n},\n{\n    \"content\": [\n        \"I'm sorry, but I really need more information to solve this problem. It can't be solved with the current data.\"\n    ],\n    \"role\": \"user\",\n    \"name\": \"assistant\"\n},\n{\n    \"content\": [\n        \"Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\\\boxed{}.)\"\n    ],\n    \"role\": \"assistant\",\n    \"name\": \"mathproxyagent\"\n}\n\nIn this example, the mathproxyagent is poorly designed as it does not recognize that the assistant agent requires additional information to continue. The mathproxyagent either needs to provide the necessary information or acknowledge that it does not have it, thereby appropriately terminating the interaction rather than repeating instructions unnecessarily.",
                "Conversation Reset": "Unexpected or unwarranted restarting of a dialogue, potentially losing context and progress made in the interaction.",
                "Fail to Ask for Clarification": "Inability to request additional information between agent when faced with unclear or incomplete data, potentially resulting in incorrect actions.",
                "Task Derailment": "Deviation from the intended objective or focus of a given task, potentially resulting in irrelevant or unproductive actions.",
                "Information Withholding": "This error occurs when an agent or group of agents possesses critical information but fails to share it promptly or effectively with other agents or system components that rely upon this information for their operations. The failure to disseminate relevant information may arise from ineffective or insufficient communication protocols, erroneous assumptions regarding the relevance or priority of the information, inadequate system coordination mechanisms, or deliberate withholding stemming from overly restrictive privacy policies or security constraints. Consequences of withholding relevant information can be severe, potentially leading to reduced operational efficiency, increased latency in task completion, unnecessary redundant processing, incorrect or suboptimal decision-making, and even complete system failures. Additionally, this error can significantly impair collaborative effectiveness, leading to misunderstandings, mistrust, or inefficiencies within the multi-agent environment. Furthermore, initial failures due to withheld information can trigger cascading errors, amplifying the negative impact on overall system performance and reliability. For instance, consider a scenario where a bug localization agent identifies a software defect, accurately determining the affected file and specific line number. The intended process requires this agent to immediately report such detailed bug information to a coding or repair agent responsible for addressing and resolving the issue. However, if the bug localization agent instead attempts to fix the bug independently without sharing the vital bug identification details with the coding agent, this withholding of relevant information could lead to duplicated effort, delayed resolution, incorrect fixes, or further system instability.",
                "Ignored Other Agent's Input": "Not properly considering input or recommendations provided by other agents in the system (ignore their suggestions), potentially leading to bad decisions, stalled progress, or missed opportunities for solving the task.",
                "Action-Reasoning Mismatch": "This error occurs when there is a discrepancy or mismatch between agents' logical discussion conclusion or a single agent's internal decision-making processes and the actual actions or outputs the system produces. Such inconsistencies can emerge due to errors in translating reasoning outcomes into practical implementations, or incorrect mapping between the agent's cognitive processes and its action space.\n\nThe consequences of this inconsistency can include unexpected, unintended, or counterproductive behaviors, reduced reliability, and diminished user trust. It can also complicate troubleshooting efforts by obscuring the true rationale behind decisions and actions, leading to further inefficiencies or repeated mistakes.\n\nFor example, in the trace below the agent states that `_add_prefix_for_feature_names_out` method is not explicitly shown in the code snippet, but only mentioned in the context of the `_iter` method. This conclusion is not coherent with the previosu step where the agent showed the implementation of this method. \n\n[...]\n ```python\ndef _add_prefix_for_feature_names_out(self, feature_names_out):\n   \"\"\"\n   Add prefix to feature names.\",\n   \"\"\"\n  if self.verbose_feature_names_out:\n[...]\nNote that the `_add_prefix_for_feature_names_out` method is not explicitly shown in the code snippet, but it is mentioned in the context of the `_iter` method.\n[...]",
                "Premature Termination": "Ending a dialogue, interaction or task before all necessary information has been exchanged or objectives have been met. Necessary information constitutes verification of outputs, key data (e.g. api tokens) etc. that are necessary for the success of the task, and agents could have obtained if they tried more or already obtained but failed to communicate to other agents before termination.",
                "Weak Verification": "Weak verification refers to situations where verification mechanisms (agent or step) exist within the system but fail to comprehensively cover all essential aspects of the design necessary for generating robust and reliable outputs. While verification steps are present, they may be incomplete, superficial, or insufficiently rigorous, thereby overlooking critical system attributes or interactions.\n\nConsequences of weak verification include partial validation that allows subtle errors, inconsistencies, or vulnerabilities to remain undetected, potentially compromising overall system reliability and effectiveness. This inadequacy can result in suboptimal system performance, unforeseen failures, cascade to final output if occur during substeps.\n\n\"You are a Code Reviewer. We are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You can help programmers assess source code for software troubleshooting, fix bugs to enhance code quality and robustness, and propose improvements to the source code. Here is a new customer's task: {task}. To complete the task, you must write a response that appropriately solves the requested instruction based on your expertise and the customer's needs.\"\n\nHowever, when asked to review generated code for a Sudoku game, the reviewer failed to recognize that standard Sudoku puzzles typically come pre-filled with numbers for the player to solve, an element absent in the generated implementation. Numerous Sudoku implementations and specifications are readily available online, which the verification agent could easily consult to ensure robustness and completeness.\n\nAnother example occurred with a TicTacToe implementation. While the game was functional and playable, the system incorrectly announced the winning player at the game's conclusion, despite employing the same ChatDev code reviewer prompt.",
                "No or Incorrect Verification": "Omission of proper checking or confirmation of task outcomes or system outputs, potentially allowing errors or inconsistencies to propagate undetected. So, either no verification or verification is designed to exist in MAS, but verifier fail to complete what was exactly prompted to do. Eg: make sure the code compiles, but the code doesn't even compile.\nVerification is particularly critical in cases where tasks or outputs are readily verifiable by the system itself without human intervention.\n\nConsequences of inadequate or absent verification include the propagation of undetected errors, system inconsistencies, reduced reliability, and failure in the generated output.\n\nA few examples are as follows:\n1. In ChatDev, when prompted by a user to generate a game (e.g., \"textBasedSpaceInvaders\"), verification steps failed despite multiple review stages. Although the code was reportedly verified, compilation errors persisted, leading to runtime failures:\nyes Error: The file 'ship.bmp' was not found in the directory /Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911.\nTraceback (most recent call last):\n  File \"/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/main.py\", line 31, in <module>\n    run_game()\n  File \"/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/main.py\", line 22, in run_game\n    gf.create_fleet(ai_settings, screen, aliens)\n  File \"/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/game_functions.py\", line 64, in create_fleet\n    alien = Alien(ai_settings, screen)\n  File \"/Users/user/Documents/*/ChatDev/WareHouse/TextBasedSpaceInvaders_DefaultOrganization_20250117121911/alien.py\", line 13, in __init__\n    self.image = pygame.image.load('alien.bmp')\nFileNotFoundError: No file 'alien.bmp' found in working directory '/Users/*/Documents/*/ChatDev'."
              },
            "label_model_name": "gpt-5",
        },
    }
}


def get_demo_names() -> List[str]:
    return list(EXAMPLES.keys())


def get_demo_config(name: str) -> Dict[str, Any] | None:
    return EXAMPLES.get(name)


