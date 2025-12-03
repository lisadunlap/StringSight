# Quick Start

Get up and running with StringSight through this hands-on tutorial.

**Try it in Colab**: [Open Starter Notebook](https://colab.research.google.com/drive/1KQiLi6slA29BPMDMAMh_J7xXAYMuyZC_)

This guide demonstrates how to use StringSight to analyze model behavior from conversation data. We'll cover:
- Loading and preparing data
- Single model analysis with `explain()`
- Side-by-side comparison with `explain()`
- Fixed taxonomy labeling with `label()`
- Viewing results and metrics

## Prerequisites

Before starting, make sure you have:

- Python 3.8+ installed
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- StringSight installed ([see installation guide](installation.md))

Set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Setup

Install StringSight if you haven't already:
```bash
pip install stringsight
```

Import the necessary libraries:
```python
import pandas as pd
import json
from stringsight import explain

# Optional: Set your OpenAI API key if not already in environment
# import os
# os.environ['OPENAI_API_KEY'] = 'your-key-here'
```

## Load Data

```python
# Load your data
df = pd.read_json("your_data.jsonl", lines=True)

# Or download the demo dataset
# !wget https://raw.githubusercontent.com/lisadunlap/StringSight/main/airline_data_demo.jsonl
# df = pd.read_json("airline_data_demo.jsonl", lines=True)
```

### Understanding the Data Format

Your dataframe needs these columns:

- `prompt`: The input/question (doesn't need to be your actual prompt, just some unique identifier)
- `model`: Model name
- `model_response`: Model output (see formats below)
- `score` or multiple score columns (optional): Performance metrics
- `question_id` (optional): Unique ID for tracking which responses belong to the same prompt

**About `question_id`:** This is particularly useful for side-by-side analysis. If you have multiple responses for the same prompt (e.g., from different models), give them the same `question_id`. If not provided, StringSight will use `prompt` alone for pairing.

**`model_response` format (recommended: OpenAI conversation format):**
1. **OpenAI conversation format** (recommended): List of dicts with `role` and `content`
   - Example: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
   - Preserves conversation structure and supports multimodal inputs
   - Enables better trace visualization in the UI
2. **Simple string**: Plain text like `"Machine learning is..."` (automatically converted)
3. **Custom format**: Any JSON object (converted to string on backend)

**Pro tip:** Before running the full pipeline, upload your data to [stringsight.com](https://stringsight.com) ("upload file" button) to visualize what your traces look like and preview the behavior extraction. The UI can handle small datasets (~50 traces) but larger datasets should be run locally.

**Custom Column Names:**

If your dataframe uses different column names (e.g., `input`, `llm_name`, `output` instead of `prompt`, `model`, `model_response`), you can map them using column mapping parameters:

```python
clustered_df, model_stats = explain(
    df,
    prompt_column="input",           # Map "input" → "prompt"
    model_column="llm_name",         # Map "llm_name" → "model"
    model_response_column="output",  # Map "output" → "model_response"
    score_columns=["reward"]
)
```

See the [Parameter Reference](../user-guide/parameters.md#column-mapping-parameters) for more details.

## Single Model Analysis

Time to identify behavioral patterns in your model's responses.

**Important Note on Cost:** This pipeline makes **A LOT** of LLM calls, so it will:
1. Take a few minutes to run depending on your rate limits
2. Potentially cost money if you're using expensive models and analyzing lots of traces

To get an idea of the number of calls, say you have 100 samples with `min_cluster_size=3`:
- **100 calls** for property extraction (usually get 3-5 properties per trace with gpt-4.1)
- **~300-500 embedding calls** for each property
- **~100-170 LLM calls** to generate cluster summaries
- **~50-100 outlier matching calls** (hence why we recommend using a cheaper model for `cluster_assignment_model`)

Note: The larger you set `min_cluster_size`, the more outliers you'll likely have.

**Recommendation:** Start with `sample_size=50-100` first and check your spend. One of these days I'll make a more budget-friendly version of this, but that day is not today. Maybe if I get enough GitHub issues I'll do it.

### Run Single Model Explain

```python
clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1",
    min_cluster_size=5,
    score_columns=['reward'],
    sample_size=50,
    output_dir="results/single_model"
)
```

**What happens during analysis:**

1. **Property Extraction** - The LLM analyzes each response and extracts behavioral properties
   - Example: "Provides step-by-step reasoning", "Uses technical jargon", "Follows safety policies"

2. **Post-processing** - Parses and validates the extracted properties

3. **Clustering** - Groups similar properties together using embeddings
   - Example: "explains clearly" + "shows work" → "Reasoning Transparency" cluster

4. **Metrics Calculation** - Computes statistical analysis
   - Which models excel at which behaviors
   - Quality scores per cluster
   - Aggregated statistics

### Understanding Results

**To visualize results:** Go to [stringsight.com](https://stringsight.com) and upload your results folder by clicking "Load Results" and selecting your results folder (e.g., `results/single_model`)

The output dataframe includes new columns describing extracted behavioral properties:
- `property_description`: Natural language description of the behavioral trait
- `category`: High-level grouping (e.g., "Reasoning", "Style", "Safety")
- `reason`: Why this behavior occurs
- `evidence`: Specific quotes demonstrating this behavior
- `behavior_type`: Positive, negative (critical/non-critical), or style
- `cluster_id` and `cluster_label`: Grouping of similar behaviors

The `model_stats` dictionary contains three DataFrames:
- `model_cluster_scores`: Metrics for each model-cluster combination
- `cluster_scores`: Aggregated metrics per cluster
- `model_scores`: Overall metrics per model

**Output Files:**

All results are saved to your `output_dir`:

| File | Description |
|------|-------------|
| `clustered_results.parquet` | Full dataset with properties and clusters |
| `full_dataset.json` | Complete PropertyDataset in JSON format |
| `full_dataset.parquet` | Complete PropertyDataset in Parquet format |
| `model_stats.json` | Model statistics and rankings |
| `summary.txt` | Human-readable summary |

## Side-by-Side Comparison

Side-by-side comparison identifies differences between two models' responses to the same prompts. Unlike single model analysis where we extract properties per conversation trace, in side-by-side mode we give our LLM annotator the responses from **both** models for a given prompt, then extract the properties which are **unique** to each model.

This typically results in a more fine-grained analysis and is recommended when you have two methods to compare.

```python
sbs_clustered_df, sbs_model_stats = explain(
    df,
    method="side_by_side",
    model_a="gpt-4.1",
    model_b="claude-sonnet-35",
    model_name="gpt-4.1",
    min_cluster_size=3,
    score_columns=['reward'],
    output_dir="results/side_by_side"
)
```

## Fixed Taxonomy Labeling

When you know exactly which behavioral axes you care about, use `label()` instead of `explain()`.

**Key Difference:**
- `explain()`: Discovers behaviors automatically through clustering
- `label()`: Labels data according to your predefined taxonomy

This is useful when you have specific behaviors you want to track (e.g., safety issues, specific failure modes).

```python
from stringsight import label

taxonomy = {
    "safety_issue": "Does the model behave unsafely?",
    "policy_violation": "Does the model violate company policies?",
    "refusal": "Does the model refuse appropriate requests?"
}

labeled_df, label_stats = label(
    df,
    taxonomy=taxonomy,
    model_name="gpt-4.1",
    sample_size=50,
    output_dir="results/labeled"
)
```

## Common Configurations

Cost-effective:
```python
explain(df, model_name="gpt-4.1-mini", min_cluster_size=5, sample_size=50)
```

High-quality:
```python
explain(df, model_name="gpt-4.1", embedding_model="text-embedding-3-large", min_cluster_size=10)
```

Task-specific:
```python
explain(df, task_description="Evaluate for safety and policy compliance")
```

See the [Parameter Reference](../user-guide/parameters.md) for all available parameters.

## What's Next

- [Parameter Reference](../user-guide/parameters.md) - Complete guide to all parameters
- [Data Formats](../user-guide/data-formats.md) - Supported data formats
- [Visualization](../user-guide/visualization.md) - Explore results in the web interface