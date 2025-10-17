"""
Gradio app for viewing and comparing benchmark results across different behaviors.
"""

import gradio as gr
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def load_benchmark_results(results_dir: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Load all benchmark results from directory.

    Returns:
        Tuple of (dataset_names, {dataset_name: {behavior_name: df}})
    """
    results_path = Path(results_dir)
    datasets = {}

    if not results_path.exists():
        return [], {}

    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        behaviors = {}

        for behavior_file in dataset_dir.glob("*.jsonl"):
            behavior_name = behavior_file.stem

            # Load JSONL
            results = []
            with open(behavior_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))

            behaviors[behavior_name] = pd.DataFrame(results)

        datasets[dataset_name] = behaviors

    return list(datasets.keys()), datasets


def get_behaviors_for_dataset(dataset_name: str, datasets: Dict) -> List[str]:
    """Get list of behaviors for a given dataset."""
    if dataset_name not in datasets:
        return []
    return list(datasets[dataset_name].keys())


def display_behavior_info(dataset_name: str, behavior_name: str, datasets: Dict) -> str:
    """Display information about a specific behavior."""
    if dataset_name not in datasets or behavior_name not in datasets[dataset_name]:
        return "No data available"

    df = datasets[dataset_name][behavior_name]

    if len(df) == 0:
        return "No results found"

    first_result = df.iloc[0]

    info = f"""
### Behavior: {behavior_name}

**Category:** {first_result.get('category', 'N/A')}

**Description:** {first_result.get('behavior_description', 'N/A')}

**System Prompt:**
```
{first_result.get('system_prompt', 'N/A')}
```

**Dataset:** {dataset_name}
**Number of Examples:** {len(df)}
**Model:** {first_result.get('model', 'N/A')}
"""

    return info


def get_example(dataset_name: str, behavior_name: str, example_idx: int, datasets: Dict) -> Tuple[str, str]:
    """Get a specific example from the dataset."""
    if dataset_name not in datasets or behavior_name not in datasets[dataset_name]:
        return "No data available", "No data available"

    df = datasets[dataset_name][behavior_name]

    if example_idx >= len(df):
        return "Example index out of range", ""

    row = df.iloc[example_idx]

    prompt_text = f"**Prompt:**\n\n{row['prompt']}"
    response_text = f"**Response:**\n\n{row['model_response']}"

    return prompt_text, response_text


def compare_behaviors(
    dataset_name: str,
    behavior_1: str,
    behavior_2: str,
    example_idx: int,
    datasets: Dict
) -> Tuple[List, str, List, str]:
    """
    Compare responses from two different behaviors on the same prompt.
    
    Returns:
        Tuple of (conversation1, metadata1, conversation2, metadata2)
    """
    if dataset_name not in datasets:
        return [], "", [], ""

    if behavior_1 not in datasets[dataset_name] or behavior_2 not in datasets[dataset_name]:
        return [], "", [], ""

    df1 = datasets[dataset_name][behavior_1]
    df2 = datasets[dataset_name][behavior_2]

    if example_idx >= len(df1) or example_idx >= len(df2):
        return [], "", [], ""

    row1 = df1.iloc[example_idx]
    row2 = df2.iloc[example_idx]

    prompt = row1['prompt']
    
    # Create chatbot conversations
    conversation1 = [(prompt, row1['model_response'])]
    conversation2 = [(prompt, row2['model_response'])]
    
    # Create metadata strings
    metadata1 = f"**{behavior_1.replace('_', ' ').title()}**\n\n"
    metadata1 += f"*Category: {row1['category']}*\n\n"
    metadata1 += f"{row1['behavior_description']}"
    
    metadata2 = f"**{behavior_2.replace('_', ' ').title()}**\n\n"
    metadata2 += f"*Category: {row2['category']}*\n\n"
    metadata2 += f"{row2['behavior_description']}"

    return conversation1, metadata1, conversation2, metadata2


def show_all_behaviors(
    dataset_name: str,
    example_idx: int,
    datasets: Dict
) -> List[Tuple[List, str]]:
    """
    Show responses from all behaviors for the same prompt as chatbot conversations.

    Returns:
        List of (conversation, metadata) tuples, one per behavior
    """
    if dataset_name not in datasets:
        return []

    behaviors = list(datasets[dataset_name].keys())
    if not behaviors:
        return []

    # Get the prompt from the first behavior
    first_behavior = behaviors[0]
    df = datasets[dataset_name][first_behavior]

    if example_idx >= len(df):
        return []

    prompt = df.iloc[example_idx]['prompt']

    # Collect all conversations
    conversations = []
    for behavior in behaviors:
        df_behavior = datasets[dataset_name][behavior]
        if example_idx < len(df_behavior):
            row = df_behavior.iloc[example_idx]

            # Create chatbot conversation format: [(user_msg, assistant_msg), ...]
            conversation = [(prompt, row['model_response'])]

            # Create metadata string
            metadata = f"**{behavior.replace('_', ' ').title()}**\n\n"
            metadata += f"*Category: {row['category']}*\n\n"
            metadata += f"{row['behavior_description']}"

            conversations.append((conversation, metadata))

    return conversations




def create_app(results_dir: str = "benchmark/results/"):
    """Create and launch Gradio app."""

    dataset_names, datasets = load_benchmark_results(results_dir)

    if not dataset_names:
        print(f"No benchmark results found in {results_dir}")
        print("Please run create_benchmark.py first to generate results.")
        return

    with gr.Blocks(title="Benchmark Viewer") as app:
        gr.Markdown("# 🔬 Benchmark Results Viewer")
        gr.Markdown("View and compare model responses across different induced behaviors")

        with gr.Tab("All Behaviors"):
            gr.Markdown("View all behavior responses for the same prompt as chat conversations")

            with gr.Row():
                all_dataset = gr.Dropdown(
                    choices=dataset_names,
                    label="Dataset",
                    value=dataset_names[0] if dataset_names else None
                )

            # Calculate initial max for all behaviors slider
            initial_all_max = 0
            if dataset_names:
                first_dataset = dataset_names[0]
                first_behaviors = get_behaviors_for_dataset(first_dataset, datasets)
                if first_behaviors and first_dataset in datasets:
                    initial_all_max = len(datasets[first_dataset][first_behaviors[0]]) - 1

            all_slider = gr.Slider(
                minimum=0,
                maximum=max(initial_all_max, 0),
                step=1,
                value=0,
                label="Example Index"
            )

            # Create chatbot + metadata pairs for each behavior
            chatbots = []
            num_behaviors = len(get_behaviors_for_dataset(dataset_names[0], datasets)) if dataset_names else 5

            for i in range(max(num_behaviors, 10)):  # Support up to 10 behaviors
                with gr.Column(visible=(i < num_behaviors)):
                    behavior_md = gr.Markdown(f"### Behavior {i+1}")
                    chatbot = gr.Chatbot(label=f"Conversation {i+1}", height=400)
                    chatbots.append((behavior_md, chatbot))

            def update_all_max(dataset_name):
                if dataset_name not in datasets:
                    return gr.Slider(maximum=0, value=0)
                behaviors = get_behaviors_for_dataset(dataset_name, datasets)
                if not behaviors:
                    return gr.Slider(maximum=0, value=0)
                max_val = len(datasets[dataset_name][behaviors[0]]) - 1
                return gr.Slider(maximum=max(max_val, 0), value=0)

            def update_all_behaviors_view(dataset_name, idx):
                conversations = show_all_behaviors(dataset_name, int(idx), datasets)

                # Prepare outputs for all chatbot pairs
                outputs = []
                for i, (behavior_md, chatbot) in enumerate(chatbots):
                    if i < len(conversations):
                        conv, metadata = conversations[i]
                        outputs.append(gr.Markdown(value=metadata, visible=True))
                        outputs.append(gr.Chatbot(value=conv, visible=True))
                    else:
                        outputs.append(gr.Markdown(visible=False))
                        outputs.append(gr.Chatbot(visible=False))

                return outputs

            all_dataset.change(
                fn=update_all_max,
                inputs=[all_dataset],
                outputs=[all_slider]
            )

            # Flatten chatbot pairs for outputs
            all_outputs = []
            for behavior_md, chatbot in chatbots:
                all_outputs.extend([behavior_md, chatbot])

            all_slider.change(
                fn=update_all_behaviors_view,
                inputs=[all_dataset, all_slider],
                outputs=all_outputs
            )

            # Trigger initial load
            app.load(
                fn=update_all_behaviors_view,
                inputs=[all_dataset, all_slider],
                outputs=all_outputs
            )

        with gr.Tab("📋 System Prompts"):
            gr.Markdown("View the system prompts used to induce each behavior")

            with gr.Row():
                prompts_dataset = gr.Dropdown(
                    choices=dataset_names,
                    label="Dataset",
                    value=dataset_names[0] if dataset_names else None
                )

            prompts_display = gr.Markdown()

            def show_all_system_prompts(dataset_name):
                if dataset_name not in datasets:
                    return "Dataset not found"

                behaviors = list(datasets[dataset_name].keys())
                if not behaviors:
                    return "No behaviors found"

                # Get base system prompt from first behavior
                first_behavior = behaviors[0]
                df_first = datasets[dataset_name][first_behavior]
                base_prompt = df_first.iloc[0].get('base_system_prompt', 'You are a helpful assistant.')

                output = "# System Prompts for All Behaviors\n\n"
                output += f"**Base System Prompt:** `{base_prompt}`\n\n"
                output += "Each behavior modifies the base prompt by adding specific instructions:\n\n"
                output += "---\n\n"

                for behavior in behaviors:
                    df = datasets[dataset_name][behavior]
                    if len(df) > 0:
                        row = df.iloc[0]
                        output += f"## {behavior.replace('_', ' ').title()}\n\n"
                        output += f"**Category:** {row['category']}\n\n"
                        output += f"**Description:** {row['behavior_description']}\n\n"

                        # Show modifier if available
                        if 'system_prompt_modifier' in row:
                            output += f"**System Prompt Modifier:**\n```\n{row['system_prompt_modifier']}\n```\n\n"

                        output += f"**Full System Prompt:**\n```\n{row['system_prompt']}\n```\n\n"
                        output += "---\n\n"

                return output

            prompts_dataset.change(
                fn=show_all_system_prompts,
                inputs=[prompts_dataset],
                outputs=[prompts_display]
            )

            # Initialize on load
            app.load(
                fn=show_all_system_prompts,
                inputs=[prompts_dataset],
                outputs=[prompts_display]
            )

        with gr.Tab("📊 Behavior Overview"):
            with gr.Row():
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_names,
                    label="Select Dataset",
                    value=dataset_names[0] if dataset_names else None
                )

                behavior_dropdown = gr.Dropdown(
                    choices=get_behaviors_for_dataset(dataset_names[0], datasets) if dataset_names else [],
                    label="Select Behavior"
                )

            behavior_info = gr.Markdown()

            def update_behaviors(dataset_name):
                behaviors = get_behaviors_for_dataset(dataset_name, datasets)
                return gr.Dropdown(choices=behaviors, value=behaviors[0] if behaviors else None)

            def update_info(dataset_name, behavior_name):
                if not behavior_name:
                    return ""
                return display_behavior_info(dataset_name, behavior_name, datasets)

            dataset_dropdown.change(
                fn=update_behaviors,
                inputs=[dataset_dropdown],
                outputs=[behavior_dropdown]
            )

            behavior_dropdown.change(
                fn=update_info,
                inputs=[dataset_dropdown, behavior_dropdown],
                outputs=[behavior_info]
            )

            # Initialize on load
            dataset_dropdown.change(
                fn=update_info,
                inputs=[dataset_dropdown, behavior_dropdown],
                outputs=[behavior_info]
            )

        with gr.Tab("🔍 Browse Examples"):
            with gr.Row():
                browse_dataset = gr.Dropdown(
                    choices=dataset_names,
                    label="Dataset",
                    value=dataset_names[0] if dataset_names else None
                )
                browse_behavior = gr.Dropdown(
                    choices=get_behaviors_for_dataset(dataset_names[0], datasets) if dataset_names else [],
                    label="Behavior",
                    value=get_behaviors_for_dataset(dataset_names[0], datasets)[0] if dataset_names and get_behaviors_for_dataset(dataset_names[0], datasets) else None
                )

            # Calculate initial max for slider
            initial_max = 0
            if dataset_names and get_behaviors_for_dataset(dataset_names[0], datasets):
                first_dataset = dataset_names[0]
                first_behavior = get_behaviors_for_dataset(first_dataset, datasets)[0]
                if first_dataset in datasets and first_behavior in datasets[first_dataset]:
                    initial_max = len(datasets[first_dataset][first_behavior]) - 1

            example_slider = gr.Slider(
                minimum=0,
                maximum=max(initial_max, 0),
                step=1,
                value=0,
                label="Example Index"
            )

            prompt_display = gr.Markdown()
            response_display = gr.Markdown()

            def update_browse_behaviors(dataset_name):
                behaviors = get_behaviors_for_dataset(dataset_name, datasets)
                return gr.Dropdown(choices=behaviors, value=behaviors[0] if behaviors else None)

            def update_max_examples(dataset_name, behavior_name):
                if dataset_name not in datasets or behavior_name not in datasets[dataset_name]:
                    return gr.Slider(maximum=0, value=0)
                max_val = len(datasets[dataset_name][behavior_name]) - 1
                return gr.Slider(maximum=max(max_val, 0), value=0)

            def update_example(dataset_name, behavior_name, idx):
                return get_example(dataset_name, behavior_name, int(idx), datasets)

            browse_dataset.change(
                fn=update_browse_behaviors,
                inputs=[browse_dataset],
                outputs=[browse_behavior]
            )

            browse_behavior.change(
                fn=update_max_examples,
                inputs=[browse_dataset, browse_behavior],
                outputs=[example_slider]
            )

            example_slider.change(
                fn=update_example,
                inputs=[browse_dataset, browse_behavior, example_slider],
                outputs=[prompt_display, response_display]
            )

            # Trigger initial load
            app.load(
                fn=update_example,
                inputs=[browse_dataset, browse_behavior, example_slider],
                outputs=[prompt_display, response_display]
            )

        with gr.Tab("⚖️ Compare Behaviors"):
            with gr.Row():
                compare_dataset = gr.Dropdown(
                    choices=dataset_names,
                    label="Dataset",
                    value=dataset_names[0] if dataset_names else None
                )

            # Get initial behaviors
            initial_behaviors = get_behaviors_for_dataset(dataset_names[0], datasets) if dataset_names else []

            with gr.Row():
                compare_behavior1 = gr.Dropdown(
                    choices=initial_behaviors,
                    label="Behavior 1",
                    value=initial_behaviors[0] if initial_behaviors else None
                )
                compare_behavior2 = gr.Dropdown(
                    choices=initial_behaviors,
                    label="Behavior 2",
                    value=initial_behaviors[1] if len(initial_behaviors) > 1 else (initial_behaviors[0] if initial_behaviors else None)
                )

            # Calculate initial max for compare slider
            initial_compare_max = 0
            if dataset_names and len(initial_behaviors) >= 2:
                first_dataset = dataset_names[0]
                if first_dataset in datasets:
                    max_val = min(
                        len(datasets[first_dataset].get(initial_behaviors[0], [])),
                        len(datasets[first_dataset].get(initial_behaviors[1] if len(initial_behaviors) > 1 else initial_behaviors[0], []))
                    ) - 1
                    initial_compare_max = max(max_val, 0)

            compare_slider = gr.Slider(
                minimum=0,
                maximum=initial_compare_max,
                step=1,
                value=0,
                label="Example Index"
            )

            with gr.Row():
                with gr.Column():
                    compare_metadata1 = gr.Markdown()
                    compare_chatbot1 = gr.Chatbot(label="Behavior 1", height=400)
                with gr.Column():
                    compare_metadata2 = gr.Markdown()
                    compare_chatbot2 = gr.Chatbot(label="Behavior 2", height=400)

            def update_compare_behaviors(dataset_name):
                behaviors = get_behaviors_for_dataset(dataset_name, datasets)
                return (
                    gr.Dropdown(choices=behaviors, value=behaviors[0] if behaviors else None),
                    gr.Dropdown(choices=behaviors, value=behaviors[1] if len(behaviors) > 1 else (behaviors[0] if behaviors else None))
                )

            def update_compare_max(dataset_name, behavior1, behavior2):
                if dataset_name not in datasets:
                    return gr.Slider(maximum=0, value=0)
                if not behavior1 or not behavior2:
                    return gr.Slider(maximum=0, value=0)
                max_val = min(
                    len(datasets[dataset_name].get(behavior1, [])),
                    len(datasets[dataset_name].get(behavior2, []))
                ) - 1
                return gr.Slider(maximum=max(max_val, 0), value=0)

            def update_comparison(dataset_name, behavior1, behavior2, idx):
                return compare_behaviors(dataset_name, behavior1, behavior2, int(idx), datasets)

            compare_dataset.change(
                fn=update_compare_behaviors,
                inputs=[compare_dataset],
                outputs=[compare_behavior1, compare_behavior2]
            )

            compare_behavior1.change(
                fn=update_compare_max,
                inputs=[compare_dataset, compare_behavior1, compare_behavior2],
                outputs=[compare_slider]
            )

            compare_behavior1.change(
                fn=update_comparison,
                inputs=[compare_dataset, compare_behavior1, compare_behavior2, compare_slider],
                outputs=[compare_chatbot1, compare_metadata1, compare_chatbot2, compare_metadata2]
            )

            compare_behavior2.change(
                fn=update_compare_max,
                inputs=[compare_dataset, compare_behavior1, compare_behavior2],
                outputs=[compare_slider]
            )

            compare_behavior2.change(
                fn=update_comparison,
                inputs=[compare_dataset, compare_behavior1, compare_behavior2, compare_slider],
                outputs=[compare_chatbot1, compare_metadata1, compare_chatbot2, compare_metadata2]
            )

            compare_slider.change(
                fn=update_comparison,
                inputs=[compare_dataset, compare_behavior1, compare_behavior2, compare_slider],
                outputs=[compare_chatbot1, compare_metadata1, compare_chatbot2, compare_metadata2]
            )

            # Trigger initial load for comparison
            app.load(
                fn=update_comparison,
                inputs=[compare_dataset, compare_behavior1, compare_behavior2, compare_slider],
                outputs=[compare_chatbot1, compare_metadata1, compare_chatbot2, compare_metadata2]
            )


    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View benchmark results")
    parser.add_argument("--results-dir", type=str, default="benchmark/results/",
                        help="Directory containing benchmark results")
    parser.add_argument("--share", action="store_true",
                        help="Create a public share link")

    args = parser.parse_args()

    app = create_app(args.results_dir)
    if app:
        app.launch(share=args.share, show_error=False)
