---
title: Benchmark Results Viewer
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Benchmark Results Viewer

This Space allows you to browse and compare model responses across different induced behaviors.

## Features

### All Behaviors
View all behavior responses for the same prompt as chat conversations. Navigate through examples using the slider to see how different system prompts affect model responses.

### System Prompts
View the system prompts used to induce each behavior. See the base prompt and the specific modifications for each behavior category.

### Behavior Overview
Get detailed information about each behavior, including category, description, and the complete system prompt used.

### Browse Examples
Browse individual examples for each behavior. Select a dataset and behavior, then use the slider to navigate through all examples.

### Compare Behaviors
Side-by-side comparison of how two different behaviors respond to the same prompt. Perfect for understanding the differences between behavior modifications.

## Usage

1. **Select a Dataset**: Choose from available benchmark datasets in the dropdown
2. **Choose a Tab**: Navigate to the view that best suits your needs
3. **Browse Examples**: Use sliders to navigate through different examples
4. **Compare**: Use the Compare tab to see side-by-side differences

## Data Format

The app expects benchmark results in the following structure:
```
results/
├── dataset_name/
│   ├── behavior1.jsonl
│   ├── behavior2.jsonl
│   └── ...
```

Each JSONL file should contain entries with:
- `prompt`: The input prompt
- `model_response`: The model's response
- `system_prompt`: The system prompt used
- `category`: Behavior category
- `behavior_description`: Description of the behavior
- `model`: Model identifier

## Setup

To run this locally or deploy your own version:

1. Clone this Space
2. Add your benchmark results in the `results/` directory
3. The app will automatically load and display all available datasets


