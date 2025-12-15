<p align="center">
  <img src="assets/stringsight_github.png" alt="StringSight logo" width="600">
</p>

<h1 align="center">StringSight</h1>

<p align="center">
  <em>Extract, cluster, and analyze behavioral properties from Generative Models</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  </a>
  <a href="https://lisadunlap.github.io/StringSight/">
    <img src="https://img.shields.io/badge/docs-Documentation-blue" alt="Docs">
  </a>
  <a href="https://blog.stringsight.com">
    <img src="https://img.shields.io/badge/blog-blog.stringsight.com-orange" alt="Blog">
  </a>
  <a href="https://stringsight.com">
    <img src="https://img.shields.io/badge/website-stringsight.com-green" alt="Website">
  </a>
</p>

<p align="center">
  <strong>Annoyed at having to look through your long model conversations or agentic traces? Fear not, StringSight has come to ease your woes. Understand and compare model behavior by automatically extracting behavioral properties from their responses, grouping similar behaviors together, and quantifying how important these behaviors are.</strong>
</p>

## Installation & Quick Start

```bash
# Install
pip install stringsight

# Set your API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # optional
export GOOGLE_API_KEY="your-google-key"        # optional

# Launch the web interface
stringsight launch

# Or run in background (survives terminal disconnects)
stringsight launch --daemon

# Check status
stringsight status

# View logs
stringsight logs

# Stop the server
stringsight stop
```

The UI will be available at [http://localhost:5180](http://localhost:5180).

For tutorials and examples, see [starter_notebook.ipynb](starter_notebook.ipynb) or [Google Colab](https://colab.research.google.com/drive/1XBQqDqTK6-9wopqRB51j8cPfnTS5Wjqh?usp=drive_link).

### Deployment Options

**Local Development:**
```bash
stringsight launch              # Foreground mode (stops when terminal closes)
stringsight launch --daemon     # Background mode (persistent)
```

**Docker (for production or multi-user setups):**
```bash
docker compose up -d
```

The Docker setup includes PostgreSQL, Redis, MinIO storage, and Celery workers for handling long-running jobs.

## Usage

### Data Format

**Required columns:**
- `prompt`: Question/prompt text (this doesn't need to be your actual prompt, just some unique identifier of a run)
- `model`: Model name
- `model_response`: Model output in one of three formats:
  - OpenAI conversation format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` (recommended, we also support multimodal inputs in this format)
  - Plain string: `"Model response text..."`
  - Custom format: Any other format will be converted to string and thus will not render all pretty in the ui (if you care about that sort of thing)

**Optional columns:**
- **Scores:** You can provide metrics in separate columns (e.g. `"accuracy"`, `"helpfulness"`, etc.—set them using the `score_columns` parameter) or as a single `score` column containing a dictionary like `{"accuracy": 0.85, "helpfulness": 4.2}`.
- `question_id`: Unique ID for a question (useful if you have multiple responses for the same prompt, especially for side-by-side pairing)
- Custom column names via `prompt_column`, `model_column`, `model_response_column`, `question_id_column` parameters

**For side-by-side:** Use `model_a`, `model_b`, `model_a_response`, `model_b_response` (pre-paired data) or pass tidy data with `method="side_by_side"` to auto-pair by prompt.

### Extract and Cluster Properties

```python
import pandas as pd
from stringsight import explain

# Prepare your data
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing", ...],
    "model": ["gpt-4", "claude-3", ...],
    "model_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning involves..."}, ...],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "Quantum computing uses..."}, ...]
    ],
    "accuracy": [1, 0, ...], 
    "helpfulness": [4.2, 3.8, ...]
})

# Run analysis
clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1-mini",
    output_dir="results/test",
    score_columns=["accuracy", "helpfulness"],
    model_response_column="model_response" # it default checks for a model_response column
)
```

### Side-by-Side Comparison

```python
# Compare two models
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    model_a="gpt-4",
    model_b="claude-3",
    output_dir="results/comparison",
    score_columns=["accuracy", "helpfulness"],
)
```

### Fixed Taxonomy Labeling

```python
from stringsight import label

TAXONOMY = {
    "refusal": "Does the model refuse to follow certain instructions?",
    "hallucination": "Does the model generate false information?",
}

clustered_df, model_stats = label(
    df,
    taxonomy=TAXONOMY,
    output_dir="results/labeled"
)
```

## Output

**Output dataframe columns:**
- `property_description`: Natural language description of behavioral trait
- `category`: High-level grouping (e.g., "Reasoning", "Style", "Safety")
- `reason`: Why this behavior occurs
- `evidence`: Specific quotes demonstrating the behavior
- `unexpected_behavior`: Boolean indicating if this is problematic
- `type`: Nature of the property (e.g., "content", "format", "style")
- `behavior_type`: Classification like "Positive", "Negative (critical)", "Style"
- `cluster_id`: Cluster assignment
- `cluster_label`: Human-readable cluster name

**Output files (when `output_dir` is specified):**
- `clustered_results.parquet`: Main dataframe with cluster assignments
- `clustered_results.jsonl` / `clustered_results_lightweight.jsonl`: JSON formats
- `full_dataset.json` / `full_dataset.parquet`: Complete PropertyDataset
- `model_cluster_scores.json`: Per model-cluster metrics
- `cluster_scores.json`: Aggregated metrics per cluster
- `model_scores.json`: Overall metrics per model
- `summary.txt`: Human-readable summary

**Metrics in model_stats:**

The `model_stats` dictionary contains three DataFrames:
1. `model_cluster_scores`: How each model performs on each behavioral cluster
2. `cluster_scores`: Aggregated metrics across all models for each cluster
3. `model_scores`: Overall metrics for each model across all clusters

## Configuration

```python
explain(
    df,
    model_name="gpt-4.1-mini",                      # LLM for extraction
    embedding_model="text-embedding-3-large",       # Embedding model
    min_cluster_size=5,                             # Min cluster size
    sample_size=100,                                # Sample before processing
    output_dir="results/"
)
```

## Advanced Features

See the [documentation](https://lisadunlap.github.io/StringSight/) for:
- Docker deployment
- Custom column mapping
- Multimodal conversations (text + images)
- Prompt expansion
- Caching configuration
- CLI usage

## Documentation

- **Full Documentation**: [https://lisadunlap.github.io/StringSight/](https://lisadunlap.github.io/StringSight/)
- **DEMO Website**: [https://stringsight.com](https://stringsight.com)
- **Tutorial Notebook**: [starter_notebook.ipynb](starter_notebook.ipynb)
  
## Contributing

PRs very welcome, especially if I forgot to include something important in the readme. Questions or issues? [Open an issue on GitHub](https://github.com/lisadunlap/stringsight/issues)
