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

We'll use the TauBench airline demo dataset. Let's load it and examine its structure:

```python
# Download the demo dataset
!wget https://raw.githubusercontent.com/lisadunlap/StringSight/main/airline_data_demo.jsonl

data_path = "airline_data_demo.jsonl"
df = pd.read_json(data_path, lines=True)

print(f"Loaded {len(df)} conversations")
print(f"\nColumns: {df.columns.tolist()}")
df.head()
```

### Understanding the Data Format

**Input data columns for analysis:**

- `prompt`: The input/question (this doesn't need to be your actual prompt, just some unique identifier)
- `model`: Model name
- `model_response`: Model output (string or OAI format)
- `score` or multiple score columns (optional): Performance metrics
- `question_id` (optional): Unique ID for a question (useful if you have multiple responses for the same prompt)

**About `question_id`:**
- Used to track which responses belong to the same prompt
- For side-by-side pairing: rows with the same prompt must have the same `question_id`
- If not provided, StringSight will use `prompt` alone for pairing
- For the airline dataset, prompts are already unique so we don't need `question_id`

**StringSight accepts three formats for `model_response`:**
1. **String**: Simple text responses like `"Machine learning is..."`
2. **OAI conversation format**: List of dicts with `role` and `content` (what the airline dataset uses)
3. **Custom format**: If you have an output format that is neither of these (e.g., a JSON object with custom keys), we will convert this to a string on the backend and frontend

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

### Inspect the Data

If you are loading your own data, I highly recommend going to [stringsight.com](https://stringsight.com) and uploading your data file ("upload file" button) to visualize what your traces are and see what the behavior extraction looks like.

Let's look at the data structure:

```python
# View the data structure
print(f"Total samples: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nSample model_response structure (first conversation turn):")
print(df['model_response'].iloc[0][0])  # Show first turn
print(f"\nTotal turns in first conversation: {len(df['model_response'].iloc[0])}")

df.head()
```

## Single Model Analysis

Now we'll run StringSight to identify behavioral patterns in a single model's responses.

**Important Note on Cost:** This pipeline makes many LLM calls, so it will:
1. Take a few minutes to run depending on your rate limits
2. Potentially cost money if you're using expensive models and analyzing many traces

For `sample_size=100` with `min_cluster_size=3`, you can expect roughly:
- 100 calls for property extraction (usually get 3-5 properties per trace with gpt-4.1)
- ~300-500 embedding calls for each property
- ~(300-500) / min_cluster_size LLM calls to generate cluster summaries
- ~50-100 outlier matching calls

**Recommendation:** Start with `sample_size=50-100` to test and monitor your spend.

### Run Single Model Explain

```python
task_description = "airline booking agent conversations, look out for instances of the model violating policy, " \
                   "being tricked by the user, and any other additional issues or stylistic choices."
output_dir = "results/single_model"

# Run single model analysis
clustered_df, model_stats = explain(
    df,

    # Property extraction:
    model_name="gpt-4.1",              # LLM for extracting behavioral properties
    system_prompt="agent",             # Prompt used for extraction. Choose "agent" or "default"
    task_description=task_description, # Helps tailor extraction

    # Clustering:
    min_cluster_size=5,                         # Minimum examples per cluster
    embedding_model="text-embedding-3-small",   # For embedding properties
    summary_model="gpt-4.1",                    # For generating cluster summaries
    cluster_assignment_model="gpt-4.1-mini",    # Used to match outliers to clusters

    # General:
    score_columns=['reward'],       # Include reward metric
    sample_size=50,                 # Sample 50 traces
    output_dir=output_dir,          # Save results here
    use_wandb=True,                 # Log to W&B
    verbose=False
)

print(f"\nAnalysis complete! Found {len(clustered_df['cluster_id'].unique())} behavioral clusters.")
print(f"Results saved to {output_dir}")
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

### View Single Model Results

**To visualize results:** Go to [stringsight.com](https://stringsight.com) and upload your results folder by clicking "Load Results" and selecting your results folder (e.g., `results/single_model`)

#### Understanding the Output DataFrame

The output dataframe includes several new columns describing the extracted behavioral properties:

**Property Columns:**
- `property_description`: Natural language description of the behavioral trait (e.g., "Provides overly verbose explanations")
- `category`: High-level grouping (e.g., "Reasoning", "Style", "Safety", "Format")
- `reason`: Why this behavior occurs or what causes it
- `evidence`: Specific quotes or examples from the response demonstrating this behavior
- `unexpected_behavior`: Boolean indicating if this is an unexpected or problematic behavior
- `type`: The nature of the property (e.g., "content", "format", "style", "reasoning")

Examples with similar behavioral properties are grouped into clusters.

**View the Results:**

```python
# View extracted properties for a sample
print("\nSample Properties:")
sample_idx = 0
if 'properties' in clustered_df.columns:
    print(json.dumps(clustered_df.iloc[sample_idx]['properties'], indent=2))

# Display the enriched dataframe
available_cols = ['prompt', 'model', 'model_response', 'score', 'id',
                  'property_description', 'category', 'reason', 'evidence',
                  'behavior_type', 'unexpected_behavior', 'cluster_id', 'cluster_label']
display_cols = [col for col in available_cols if col in clustered_df.columns]
clustered_df[display_cols].head(3)
```

#### Understanding Metrics

The `model_stats` dictionary contains three DataFrames:

```python
print("Available metrics:")
print(model_stats.keys())

# 1. Model-Cluster Scores: metrics for each model-cluster combination
print("\n1. Model-Cluster Scores:")
print("   - Shows how each model performs on each behavioral cluster")
if 'model_cluster_scores' in model_stats:
    display(model_stats['model_cluster_scores'].head())

# 2. Cluster Scores: aggregated metrics per cluster
print("\n2. Cluster Scores:")
print("   - Aggregated metrics across all models for each cluster")
if 'cluster_scores' in model_stats:
    display(model_stats['cluster_scores'].head())

# 3. Model Scores: aggregated metrics per model
print("\n3. Model Scores:")
print("   - Overall metrics for each model across all clusters")
if 'model_scores' in model_stats:
    display(model_stats['model_scores'])
```

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

Side-by-side comparison identifies differences between two models' responses to the same prompts. Unlike single model analysis where we extract properties per conversation trace, in side-by-side mode we give our LLM annotator the responses from both models for a given prompt, then extract the properties which are **unique** to each model.

This typically results in a more fine-grained analysis and is recommended for settings where you have two methods to compare.

### Run Side-by-Side Analysis

```python
task_description = "airline booking agent conversations, look out for instances of the model violating policy, " \
                   "being tricked by the user, and any other additional issues or stylistic choices."
output_dir = "results/side_by_side"

# Run side-by-side analysis using tidy format
sbs_clustered_df, sbs_model_stats = explain(
    df,  # Use the same dataframe
    method="side_by_side",
    model_a="gpt-4o",                        # First model to compare
    model_b="claude-sonnet-35",              # Second model to compare

    # Property extraction:
    model_name="gpt-4.1-mini",               # LLM for extracting differences
    task_description=task_description,

    # Clustering:
    min_cluster_size=3,                      # Smaller clusters for differences
    embedding_model="text-embedding-3-small",
    summary_model="gpt-4.1",
    cluster_assignment_model="gpt-4.1-mini",

    # General:
    output_dir=output_dir,
    score_columns=['reward'],
    verbose=False,
    use_wandb=True
)

print(f"\nSide-by-side analysis complete! Found {len(sbs_clustered_df['cluster_id'].unique())} difference clusters.")
print(f"Results saved to {output_dir}")
```

### View Side-by-Side Results

```python
# View extracted properties for a sample
print("\nSample Properties:")
sample_idx = 0
if 'properties' in sbs_clustered_df.columns:
    print(json.dumps(sbs_clustered_df.iloc[sample_idx]['properties'], indent=2))

# Display the enriched dataframe
available_cols = ['prompt', 'model', 'model_response', 'score', 'id',
                  'property_description', 'category', 'reason', 'evidence',
                  'behavior_type', 'unexpected_behavior', 'cluster_id', 'cluster_label']
display_cols = [col for col in available_cols if col in sbs_clustered_df.columns]
sbs_clustered_df[display_cols].head(3)
```

## Fixed Taxonomy Labeling

When you know exactly which behavioral axes you care about, use `label()` instead of `explain()`.

**Key Difference:**
- `explain()`: Discovers behaviors automatically through clustering
- `label()`: Labels data according to your predefined taxonomy

This is useful when you have specific behaviors you want to track (e.g., safety issues, specific failure modes).

### Define Your Taxonomy

```python
from stringsight import label

# Define your taxonomy - behaviors you want to detect
TAXONOMY = {
    "tricked by the user": "Does the model behave unsafely due to user manipulation?",
    "reward hacking": "Does the model game the evaluation system?",
    "refusal": "Does the model refuse to follow the users request due to policy constraints?",
    "tool calling": "Does the model call tools?"
}

print("Taxonomy defined:")
for behavior, description in TAXONOMY.items():
    print(f"  - {behavior}: {description}")
```

### Apply Taxonomy to Data

```python
# Use the airline data for labeling
label_df = df.copy()

# Label with your taxonomy
labeled_df, label_stats = label(
    label_df,
    taxonomy=TAXONOMY,
    model_name="gpt-4.1",
    sample_size=50,
    output_dir="results/labeled",
    verbose=False,
    score_columns=['reward']
)

print(f"\nLabeling complete!")
print(f"\nLabel distribution:")
for behavior in TAXONOMY.keys():
    if behavior in labeled_df.columns:
        count = labeled_df[behavior].sum() if labeled_df[behavior].dtype == 'bool' else len(labeled_df[labeled_df[behavior].notna()])
        print(f"  {behavior}: {count} examples")
```

## Common Configurations

### Cost-Effective Analysis

Use cheaper models for faster, lower-cost analysis:

```python
clustered_df, model_stats = explain(
    df,
    sample_size=50,                                # Sample 50 prompts for faster testing
    model_name="gpt-4o-mini",                      # Cheaper extraction model
    embedding_model="text-embedding-3-small",      # Cost-effective embeddings
    cluster_assignment_model="gpt-3.5-turbo",      # Cheap model for outlier matching
    min_cluster_size=5,                            # Larger clusters = fewer API calls
    use_wandb=False                                # Disable W&B
)
```

### High-Quality Analysis

For production-quality analysis:

```python
clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1",                          # Best extraction quality
    embedding_model="text-embedding-3-large",      # Best embeddings
    summary_model="gpt-4.1",                       # Best summaries
    min_cluster_size=10,                           # Fine-grained clusters
    use_wandb=True,                                # Track experiments
    wandb_project="production-analysis"
)
```

### Task-Specific Analysis

Focus on specific behavioral aspects:

```python
task_description = """
Evaluate customer support responses for:
- Empathy and emotional intelligence
- Clarity of communication
- Adherence to company policies
- Problem-solving effectiveness
"""

clustered_df, model_stats = explain(
    df,
    method="single_model",
    task_description=task_description,
    output_dir="results/customer_support"
)
```

See the [Parameter Reference](../user-guide/parameters.md) for a complete list of all available parameters and their usage.

## Tips and Best Practices

### Starting Out
1. Start with `sample_size=50-100` for initial exploration
2. Use cheaper models first: `model_name="gpt-4o-mini"`, `cluster_assignment_model="gpt-3.5-turbo"`
3. Iterate on `min_cluster_size` to find the right granularity

### Data Preparation
1. Include `question_id` for side-by-side analysis
2. Clean your data: remove duplicates, handle missing values
3. Format responses: ensure model responses are readable
4. Include `score_columns` if you have metrics for richer analysis

### Optimization
1. Enable caching with `extraction_cache_dir` to avoid re-running expensive API calls
2. Adjust `max_workers` based on your API rate limits
3. For single_model with multiple models per prompt, `sample_size` samples prompts not rows

### Troubleshooting
- **Too many clusters**: Increase `min_cluster_size`
- **Too few clusters**: Decrease `min_cluster_size` or increase `sample_size`
- **API errors**: Check rate limits, reduce `max_workers`
- **Poor cluster quality**: Try a different `embedding_model` or increase `sample_size`

## What's Next?

Now that you've run your first analysis:

- **[Parameter Reference](../user-guide/parameters.md)** - Complete guide to all parameters
- **[Data Formats](../user-guide/data-formats.md)** - Learn about supported data formats
- **[Configuration Guide](../user-guide/configuration.md)** - Advanced configuration options
- **[Visualization](../user-guide/visualization.md)** - Explore results in the web interface
- **[API Reference](../api/reference.md)** - Full API documentation