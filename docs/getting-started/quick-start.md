# Quick Start

Get up and running with StringSight in 5 minutes.

## Prerequisites

Before starting, make sure you have:

- Python 3.8+ installed
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- StringSight installed ([see installation guide](installation.md))

Set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Your First Analysis

### Step 1: Import StringSight

```python
import pandas as pd
from stringsight import explain

df = pd.DataFrame({
    "prompt": [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a poem about AI"
    ],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": [
        "Machine learning is a subset of artificial intelligence...",
        "Quantum computing leverages quantum mechanical phenomena...",
        "In circuits of light and code, silicon dreams unfold..."
    ],
    "score": [
        {"accuracy": 1},
        {"accuracy": 0},
        {"accuracy": 1}
    ]
})
```

### Step 2: Prepare Your Data

StringSight analyzes model conversations to extract behavioral properties. You need conversation data in one of two formats:

**Single Model Format** (analyze one model's behavior):

```python
df = pd.DataFrame({
    "prompt": [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a poem about AI"
    ],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": [
        "Machine learning is a subset of artificial intelligence...",
        "Quantum computing leverages quantum mechanical phenomena...",
        "In circuits of light and code, silicon dreams unfold..."
    ],
    "score": [
        {"accuracy": 1, "helpfulness": 4.2},
        {"accuracy": 0, "helpfulness": 3.8},
        {"accuracy": 1, "helpfulness": 4.5}
    ]
})
```

**Side-by-Side Format** (compare two models):

Two options for side-by-side data:

**Option 1: Pre-paired with explicit columns**
```python
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing"],
    "model_a": ["gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3"],
    "model_a_response": ["ML is a subset of AI...", "Quantum computing uses qubits..."],
    "model_b_response": ["Machine learning involves...", "QC leverages quantum..."],
    "score": [{"winner": "model_a"}, {"winner": "model_b"}]
})
```

**Option 2: Tidy format with model selection** (auto-pairing)
```python
df_tidy = pd.DataFrame({
    "prompt": ["What is ML?", "What is ML?", "Explain QC", "Explain QC"],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": ["ML is...", "Machine learning...", "QC uses...", "Quantum..."]
})

# StringSight will automatically pair responses for shared prompts
clustered_df, model_stats = explain(
    df_tidy,
    method="side_by_side",
    model_a="gpt-4",
    model_b="claude-3"
)
```

> **Note**: The `score` column is optional but provides useful context for behavioral analysis.

**Alternative: Using Separate Score Columns**

Instead of creating a score dictionary, you can provide scores in separate columns:

```python
df = pd.DataFrame({
    "prompt": ["What is ML?", "Explain QC", "Write a poem"],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": ["ML is...", "QC uses...", "In circuits..."],
    "accuracy": [1, 0, 1],           # Separate column
    "helpfulness": [4.2, 3.8, 4.5]   # Separate column
})

# StringSight automatically converts these to score dicts
clustered_df, model_stats = explain(
    df,
    method="single_model",
    score_columns=["accuracy", "helpfulness"]  # Specify which columns are scores
)
```

This works with side-by-side format too (use `accuracy_a`, `accuracy_b`, etc.). See [Data Formats](../user-guide/data-formats.md#using-separate-score-columns) for more details.

### Step 3: Run the Analysis

The `explain()` function runs a complete 4-stage pipeline:

```python
# For single model analysis
clustered_df, model_stats = explain(
    df,
    method="single_model",
    min_cluster_size=5,
    output_dir="results/my_first_analysis"
)

# For side-by-side comparison
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=5,
    output_dir="results/comparison"
)
```

**What happens during analysis:**

1. **Property Extraction** - GPT-4 analyzes each response and extracts behavioral properties
   - Example: "Provides step-by-step reasoning", "Uses technical jargon", "Includes creative examples"

2. **Post-processing** - Parses and validates the extracted properties

3. **Clustering** - Groups similar properties together using embeddings
   - Example: "explains clearly" + "shows work" → "Reasoning Transparency" cluster

4. **Metrics Calculation** - Computes statistical analysis
   - Which models excel at which behaviors
   - Quality scores and significance testing
   - Confidence intervals

### Step 4: Explore the Results

**View Behavioral Properties:**

```python
# See what behavioral patterns were found
print(clustered_df[[
    'property_description',
    'property_description_cluster_label'
]].head(10))
```

**Check Model Statistics:**

```python
# See which behaviors each model excels at
for model, stats in model_stats.items():
    print(f"\n{model} top behaviors:")
    for behavior in stats["fine"][:5]:  # top 5
        print(f"  • {behavior.property_description}")
        print(f"    Score: {behavior.score:.2f}")
        if hasattr(behavior, 'quality_scores'):
            print(f"    Quality: {behavior.quality_scores}")
```

**Output Files:**

All results are saved to your `output_dir`:

| File | Description |
|------|-------------|
| `clustered_results.parquet` | Full dataset with properties and clusters |
| `full_dataset.json` | Complete PropertyDataset object |
| `model_cluster_scores_df.jsonl` | Per model-cluster metrics (DataFrame JSONL) |
| `cluster_scores_df.jsonl` | Per cluster aggregated metrics (DataFrame JSONL) |
| `model_scores_df.jsonl` | Per model aggregated metrics (DataFrame JSONL) |
| `summary.txt` | Human-readable summary |

### Step 5: Interactive Visualization (Gradio)

```bash
python gradio_chat_viewer.py
```

## Common Configurations

### Cost-Effective Analysis

Use cheaper models for faster, lower-cost analysis:

```python
clustered_df, model_stats = explain(
    df,
    sample_size=50,                        # Sample 500 prompts for faster testing
    model_name="gpt-4o-mini",              # Cheaper extraction model
    embedding_model="text-embedding-3-small",     # Free local embeddings
    min_cluster_size=5,                    # Larger clusters = fewer API calls
    use_wandb=False                         # Disable W&B (default True)
)
```

### High-Quality Analysis

For production-quality analysis:

```python
clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1",                        # Best extraction quality
    embedding_model="text-embedding-3-large",     # Best embeddings
    min_cluster_size=10,                          # Fine-grained clusters
    use_wandb=True,                               # Track experiments (default True)
    wandb_project="my-analysis"
)
```

### Task-Specific Analysis

Focus on specific behavioral aspects (works for both single_model and side_by_side):

```python
task_description = """
Evaluate customer support responses for:
- Empathy and emotional intelligence
- Clarity of communication
- Adherence to company policies
- Problem-solving effectiveness
"""

# Single model analysis
clustered_df, model_stats = explain(
    df,
    method="single_model",
    task_description=task_description,
    output_dir="results/customer_support"
)

# Or side-by-side comparison
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    task_description=task_description,
    output_dir="results/customer_support_comparison"
)
```

> Tip: Provide `task_description` to guide extraction toward a domain (e.g., customer support, web development). See Task Descriptions in the configuration guide.

## Fixed-Taxonomy Labeling

If you have predefined behavioral categories instead of discovering them:

```python
from stringsight import label

# Define your taxonomy
taxonomy = {
    "technical_accuracy": "Response demonstrates correct technical knowledge",
    "creative_thinking": "Response shows originality and creativity",
    "clear_communication": "Response is easy to understand and well-structured"
}

# Run LLM-as-judge labeling
clustered_df, model_stats = label(
    df,
    taxonomy=taxonomy,
    model_name="gpt-4o-mini",
    output_dir="results/taxonomy_analysis"
)
```

## Command Line Usage

Run analyses from the command line:

```bash
# Side-by-side comparison from tidy single-model data
python scripts/run_full_pipeline.py \
  --data_path data/my_data.jsonl \
  --output_dir results/analysis \
  --method side_by_side \
  --model_a "gpt-4" \
  --model_b "claude-3" \
  --embedding_model "text-embedding-3-small"

# Fixed-taxonomy labeling
python scripts/run_label_pipeline.py \
  --data_path data/my_data.jsonl \
  --output_dir results/labels \
  --model_name "gpt-4o-mini"
```

# Command-line: disable W&B (enabled by default)
```bash
python scripts/run_full_pipeline.py \
  --data_path data/my_data.jsonl \
  --output_dir results/analysis \
  --disable_wandb
```

## What's Next?

Now that you've run your first analysis:

- **[Learn about `