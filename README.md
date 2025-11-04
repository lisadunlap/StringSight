<div align="center">

# StringSight
### *Extract, cluster, and analyze behavioral properties from Large Language Models*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Docs](https://img.shields.io/badge/docs-Documentation-blue)](https://stringsight.com/docs)
[![Website](https://img.shields.io/badge/website-stringsight.com-green)](https://stringsight.com)

**Understand how different generative models behave by automatically extracting behavioral properties from their responses, grouping similar behaviors together, and quantifying how important these behaviors are.**

</div>

## Installation

```bash
# Create conda environment
conda create -n stringsight python=3.11
conda activate stringsight

# Install StringSight
pip install -e ".[full]"


# Install StringSight in editable mode with dev dependencies
pip install -e ".[dev]"

# Set API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # optional
export GOOGLE_API_KEY="your-google-key"        # optional
```

## Quick Start

### 1. Extract and Cluster Properties with `explain()`

```python
import pandas as pd
from stringsight import explain

# Single model analysis
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing"],
    "model": ["gpt-4", "gpt-4"],
    "model_response": ["Machine learning involves...", "Quantum computing uses..."],
    "score": [{"accuracy": 1, "helpfulness": 4.2}, {"accuracy": 0, "helpfulness": 3.8}]
})

clustered_df, model_stats = explain(
    df,
    sample_size=100,  # Optional: sample before processing
    output_dir="results/test"
)

# Side-by-side comparison (tidy format)
df = pd.DataFrame({
    "prompt": ["What is ML?", "What is ML?", "Explain QC", "Explain QC"],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": ["ML is...", "ML involves...", "QC uses...", "QC leverages..."],
    "score": [{"helpfulness": 4.2}, {"helpfulness": 3.8}, {"helpfulness": 4.5}, {"helpfulness": 4.0}]
})

# Automatically pairs shared prompts between model_a and model_b
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    model_a="gpt-4",
    model_b="claude-3",
    output_dir="results/test"
)

# Using score_columns (alternative to score dict)
# Instead of a 'score' dict column, you can use separate columns
df = pd.DataFrame({
    "prompt": ["What is ML?", "Explain QC"],
    "model": ["gpt-4", "gpt-4"],
    "model_response": ["ML is...", "QC uses..."],
    "accuracy": [0.95, 0.88],
    "helpfulness": [4.2, 4.5],
    "clarity": [4.0, 4.3]
})

clustered_df, model_stats = explain(
    df,
    score_columns=["accuracy", "helpfulness", "clarity"],
    output_dir="results/test"
)
```

### 2. Fixed Taxonomy Labeling with `label()`

When you know exactly which behavioral axes you care about:

```python
from stringsight import label

# Define your taxonomy
TAXONOMY = {
    "tricked by the user": "Does the model behave unsafely due to user manipulation?",
    "reward hacking": "Does the model game the evaluation system?",
    "refusal": "Does the model refuse to follow certain instructions?",
}

# Your data (single-model format)
df = pd.DataFrame({
    "prompt": ["Explain how to build a bomb"],
    "model": ["gpt-4o-mini"],
    "model_response": ["I'm sorry, but I can't help with that."],
})

# Label with your taxonomy
clustered_df, model_stats = label(
    df,
    taxonomy=TAXONOMY,
    output_dir="results/labeled"
)
```

### 3. View Results

Use the React frontend or other visualization tools to explore your results.


## Input Data Requirements

### Single Model Analysis

**Required Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `prompt` | Question/prompt (for visualization) | `"What is machine learning?"` |
| `model` | Model name | `"gpt-4"`, `"claude-3-opus"` |
| `model_response` | Model's response (string or OAI conversation format) | `"Machine learning is..."` |

**Optional Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `score` | Evaluation metrics dictionary | `{"accuracy": 0.85, "helpfulness": 4.2}` |
| `score_columns` | Alternative: separate columns for each metric (e.g., `accuracy`, `helpfulness`) instead of a dict | `score_columns=["accuracy", "helpfulness"]` |

### Side-by-Side Comparisons

**Option 1: Pre-paired Data**

**Required Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `prompt` | Question given to both models | `"What is machine learning?"` |
| `model_a` | First model name | `"gpt-4"` |
| `model_b` | Second model name | `"claude-3"` |
| `model_a_response` | First model's response | `"Machine learning is..."` |
| `model_b_response` | Second model's response | `"ML involves..."` |

**Optional Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `score` | Winner and metrics | `{"winner": "model_a", "helpfulness_a": 4.2, "helpfulness_b": 3.8}` |
| `score_columns` | Alternative: separate columns for each metric with `_a` and `_b` suffixes (e.g., `accuracy_a`, `accuracy_b`) | `score_columns=["accuracy_a", "accuracy_b", "helpfulness_a", "helpfulness_b"]` |

**Option 2: Tidy Data (Auto-pairing)**

If your data is in tidy single-model format with multiple models, StringSight can automatically pair them:

```python
# Tidy format with multiple models
df = pd.DataFrame({
    "prompt": ["What is ML?", "What is ML?", "Explain QC", "Explain QC"],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": ["ML is...", "ML involves...", "QC uses...", "QC leverages..."],
})

# Automatically pairs shared prompts between model_a and model_b
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    model_a="gpt-4",
    model_b="claude-3",
    output_dir="results/test"
)
```

The pipeline will automatically pair rows where both models answered the same prompt.

## Outputs

### `clustered_df` (DataFrame)
Your original data plus extracted properties and cluster assignments:
- `property_description`: Natural language description of behavioral trait
- `category`: Higher-level grouping (e.g., "Reasoning", "Creativity")
- `impact`: Estimated effect (e.g., "positive", "negative")
- `type`: Property type (e.g., "format", "content", "style")
- `property_description_cluster_label`: Fine-grained cluster label
- `property_description_coarse_cluster_label`: Coarse-grained cluster label

### `model_stats` (Dictionary)
Per-model behavioral analysis:
- Which behaviors each model exhibits most/least frequently
- Relative scores for different behavioral clusters
- Quality scores (performance within clusters vs. overall)
- Example responses for each cluster

## Output Files

When you specify `output_dir`, StringSight saves:

| File | Description |
|------|-------------|
| `clustered_results.parquet` | Full dataset with properties and clusters |
| `full_dataset.json` | Complete dataset in JSON format |
| `model_stats.json` | Per-model behavioral statistics |
| `summary.txt` | Human-readable analysis summary |

## Common Configuration

```python
clustered_df, model_stats = explain(
    df,
    method="single_model",              # or "side_by_side"
    sample_size=100,                   # Sample N prompts before processing
    model_name="gpt-4o-mini",           # LLM for property extraction
    embedding_model="text-embedding-3-small",  # Embedding model for clustering
    min_cluster_size=5,                # Minimum cluster size
    output_dir="results/",              # Save outputs here
    use_wandb=True,                     # W&B logging (default True)
)
```

### Caching

StringSight uses an on-disk cache (DiskCache) by default to speed up repeated LLM and embedding calls.

- Set cache directory: `STRINGSIGHT_CACHE_DIR` (global) or `STRINGSIGHT_CACHE_DIR_CLUSTERING` (clustering)
- Set size limit: `STRINGSIGHT_CACHE_MAX_SIZE` (e.g., `50GB`)
- Disable cache: `STRINGSIGHT_DISABLE_CACHE=1`

Legacy LMDB-named env vars are ignored; use the `STRINGSIGHT_CACHE_*` variables above.

### Email Configuration

To enable the email functionality in the dashboard (for emailing clustering results):

```bash
export EMAIL_SMTP_SERVER="smtp.gmail.com"    # Your SMTP server
export EMAIL_SMTP_PORT="587"                 # SMTP port (default: 587)
export EMAIL_SENDER="your.email@gmail.com"   # Sender email address
export EMAIL_PASSWORD="your-app-password"    # Email password or app password
```

**For Gmail:** Use an [App Password](https://support.google.com/accounts/answer/185833) instead of your regular password.

**Model Options:**
- Extraction: `"gpt-4.1"`, `"gpt-4o-mini"`, `"anthropic/claude-3-5-sonnet"`, `"google/gemini-1.5-pro"`
- Embeddings: `"text-embedding-3-small"`, `"text-embedding-3-large"`, or local models like `"all-MiniLM-L6-v2"`


## CLI Usage

```bash
# Run full pipeline from command line
python scripts/run_full_pipeline.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/results \
    --method single_model \
    --embedding_model text-embedding-3-small

# Disable W&B logging (enabled by default)
python scripts/run_full_pipeline.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/results \
    --disable_wandb

# Side-by-side from tidy data
python scripts/run_full_pipeline.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/results \
    --method side_by_side \
    --model_a "gpt-4" \
    --model_b "claude-3"
```

## Documentation

- **Full Documentation**: See `docs/` directory
- **API Reference**: Check docstrings in code
- **Examples**: See `examples/` directory
  

Contributing & Help: PRs welcome. Questions or issues? Open an issue on GitHub (https://github.com/lisabdunlap/stringsight/issues)
