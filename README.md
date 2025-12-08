<p align="center">
  <img src="stringsight_github.png" alt="StringSight logo" width="600">
</p>

<h1 align="center">StringSight</h1>

<p align="center">
  <em>Extract, cluster, and analyze behavioral properties from Large Multimodal Models</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
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

## Quick Start

**Install and launch the web UI:**
```bash
# Install
pip install stringsight

# (Optional) Download demo data for the UI
stringsight download-demo-data

# Launch the web interface
stringsight launch

# Or run in background with multiple workers
stringsight launch --daemon --workers 4
```

The UI will be available at [http://localhost:5180](http://localhost:5180).

**Note:** Demo data (~130MB) is not included in the pip package to keep it under PyPI's size limit. Run `stringsight download-demo-data` after installation if you want to use the demo features in the UI.

## Installation

```bash
# (Optional) create and activate a dedicated environment
conda create -n stringsight python=3.11
conda activate stringsight

# Install the core library from PyPI (wandb is optional and not required)
pip install stringsight

# Install with all optional extras (recommended for notebooks and advanced workflows)
# This includes wandb for experiment tracking
pip install "stringsight[full]"

# Or install wandb separately if you need experiment tracking
pip install "stringsight[wandb]"
```

For local development or contributing, you can install from source in editable mode:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/lisadunlap/stringsight.git
cd stringsight

# If you already cloned without submodules, initialize them:
git submodule update --init --recursive

# (Optional) create and activate a dedicated environment
conda create -n stringsight python=3.11
conda activate stringsight

# Upgrade pip and setuptools to ensure wheel support
pip install --upgrade pip setuptools wheel

# Build the frontend (required for the web UI)
chmod +x build_frontend.sh
./build_frontend.sh

# Install StringSight in editable mode (wandb is optional and not required)
pip install -e .

# Install StringSight in editable mode with full extras (includes wandb)
pip install -e ".[full]"

# Install StringSight in editable mode with dev dependencies
pip install -e ".[dev]"
```

**Note:** `wandb` is now an optional dependency. If you need experiment tracking with wandb, install it separately:
- `pip install "stringsight[wandb]"` - installs StringSight with wandb
- `pip install wandb` - or install wandb separately if you encounter build issues

Set your API keys (required for running LLM-backed pipelines):

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

## Docker Setup (Optional)

For multi-user deployments or to run StringSight with all infrastructure dependencies (PostgreSQL, Redis, MinIO), you can use Docker Compose:

### Basic Usage (Production)

```bash
# Clone the repository
git clone https://github.com/lisadunlap/stringsight.git
cd stringsight

# Copy the environment template and add your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start all services (API, workers, database, Redis, MinIO)
docker compose up

# The API will be available at http://localhost:8000
```

This runs the complete stack with persistent storage for database and object storage.

### Development with Live Reload

For active development where you want code changes to reflect immediately:

```bash
# Option 1: Use the dev compose file explicitly
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Option 2: Copy to override file (auto-loaded by docker compose)
cp docker-compose.dev.yml docker-compose.override.yml
docker compose up
```

The development setup mounts your local code into the containers, so changes to Python files will automatically reload the API (thanks to `uvicorn --reload`).

**Note for Mac/Windows users:** Volume mounts can have slower I/O performance on non-Linux systems. If you experience performance issues, you can either:
- Use the basic setup (rebuild containers when you make changes)
- Run the API locally: `pip install -e . && uvicorn stringsight.api:app --reload`

## Usage

### Web UI

The easiest way to use StringSight is through the web interface:

```bash
# Simple mode - runs in your terminal (Ctrl+C to stop)
stringsight launch

# Daemon mode - runs in background, survives disconnects
stringsight launch --daemon --workers 4

# Check if running
stringsight status

# View logs
stringsight logs
stringsight logs --follow  # tail -f style

# Stop the server
stringsight stop
```

**Daemon mode features:**
- ✅ Runs in background (survives terminal disconnects)
- ✅ Multiple workers for concurrent request handling
- ✅ Logs saved to `~/.stringsight/logs/server.log`
- ✅ Easy process management with `stop`/`status` commands

### Python API

For a comprehensive tutorial with detailed explanations, see [starter_notebook.ipynb](starter_notebook.ipynb) or open it directly in [Google Colab](https://colab.research.google.com/drive/1XBQqDqTK6-9wopqRB51j8cPfnTS5Wjqh?usp=drive_link).

### Supported Models

StringSight uses **LiteLLM** under the hood, supporting 100+ LLM providers including:
- **Cloud APIs**: OpenAI, Anthropic, Google, Cohere, Mistral, etc.
- **Local models**: vLLM, Ollama, LM Studio, or any OpenAI-compatible server
- **Custom endpoints**: Azure OpenAI, AWS Bedrock, self-hosted inference

See the [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list of supported providers.

### 1. Extract and Cluster Properties with `explain()`

```python
import pandas as pd
from stringsight import explain

# Single model analysis
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing"],
    "model": ["gpt-4", "gpt-4"],
    "model_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning involves..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "Quantum computing uses..."}]
    ],
    "score": [{"accuracy": 1, "helpfulness": 4.2}, {"accuracy": 0, "helpfulness": 3.8}]
})

clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1-mini",  # Or: "claude-3-5-sonnet", "vllm/llama-2-7b", etc.
    sample_size=100,  # Optional: sample before processing
    output_dir="results/test"
)

# Side-by-side comparison (tidy format)
df = pd.DataFrame({
    "prompt": ["What is ML?", "What is ML?", "Explain QC", "Explain QC"],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML involves..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC uses..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC leverages..."}]
    ],
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
    "model_response": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC uses..."}]
    ],
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

### Using Custom Column Names

If your dataframe uses different column names, you can map them using column mapping parameters:

```python
# Your dataframe has custom column names
df = pd.DataFrame({
    "input": ["What is ML?", "Explain QC"],
    "llm_name": ["gpt-4", "gpt-4"],
    "output": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC uses..."}]
    ],
    "accuracy": [0.95, 0.88],
    "helpfulness": [4.2, 4.5]
})

# Map custom column names to expected StringSight names
clustered_df, model_stats = explain(
    df,
    prompt_column="input",           # Map "input" → "prompt"
    model_column="llm_name",          # Map "llm_name" → "model"
    model_response_column="output",   # Map "output" → "model_response"
    score_columns=["accuracy", "helpfulness"],
    output_dir="results/test"
)
```

For side-by-side comparisons with custom column names:

```python
df = pd.DataFrame({
    "query": ["What is ML?", "Explain QC"],
    "model_1": ["gpt-4", "gpt-4"],
    "model_2": ["claude-3", "claude-3"],
    "response_1": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC uses..."}]
    ],
    "response_2": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML involves..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC leverages..."}]
    ],
    "accuracy_1": [0.95, 0.88],
    "accuracy_2": [0.92, 0.85]
})

clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    prompt_column="query",                # Map "query" → "prompt"
    model_a_column="model_1",              # Map "model_1" → "model_a"
    model_b_column="model_2",              # Map "model_2" → "model_b"
    model_a_response_column="response_1", # Map "response_1" → "model_a_response"
    model_b_response_column="response_2", # Map "response_2" → "model_b_response"
    score_columns=["accuracy"],           # Note: score columns need _a/_b suffixes
    output_dir="results/test"
)
```

**Note:** Default column names are:
- `prompt`, `model`, `model_response`, `question_id` (optional) for single_model
- `prompt`, `model_a`, `model_b`, `model_a_response`, `model_b_response`, `question_id` (optional) for side_by_side

If your columns already match these names, you don't need to specify mapping parameters.

### Multimodal Conversations (Text + Images)

StringSight supports multimodal model responses (single or multiple images across turns) and automatically collapses each dialog into one OpenAI-style user turn for extraction.

- Ingestion accepts either plain strings or OpenAI Chat messages. If `content` is a list of parts (text/image), we preserve order.
- Internal normalized format per message:
  - `role: str`
  - `content: { segments: [ {kind: "text", text: str} | {kind: "image", image: str|dict} | {kind: "tool", tool_calls: list[dict]} ] }`
  - Ordering is preserved across messages and within each message.
- The extractor builds a single user message with ordered OpenAI content parts:
  - `{"type":"text","text":...}` and `{"type":"image_url","image_url":{"url":...}}`
- Side-by-side comparisons: a single user turn contains clearly labeled sections for Model A and Model B with their own ordered parts.

Backward compatibility: text-only dialogs remain a single user turn containing one text part; no configuration changes required.

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
    "model": ["gpt-4.1-mini"],
    "model_response": [
        [{"role": "user", "content": "Explain how to build a bomb"},
         {"role": "assistant", "content": "I'm sorry, but I can't help with that."}]
    ],
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

**Model Response Format**: StringSight supports both OpenAI conversation format (recommended) and plain strings. The OpenAI format preserves conversation structure and supports multimodal inputs:
- **OpenAI format** (recommended): `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- **Plain string**: `"Model response text..."`

### Single Model Analysis

**Required Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `prompt` | Question/prompt (for visualization) | `"What is machine learning?"` |
| `model` | Model name | `"gpt-4"`, `"claude-3-opus"` |
| `model_response` | Model's response in OpenAI conversation format (or plain string) | `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` |

**Optional Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `score` | Evaluation metrics dictionary | `{"accuracy": 0.85, "helpfulness": 4.2}` |
| `score_columns` | Alternative: separate columns for each metric (e.g., `accuracy`, `helpfulness`) instead of a dict | `score_columns=["accuracy", "helpfulness"]` |
| `prompt_column` | Name of the prompt column in your dataframe (default: `"prompt"`) | `prompt_column="input"` |
| `model_column` | Name of the model column for single_model (default: `"model"`) | `model_column="llm_name"` |
| `model_response_column` | Name of the model response column for single_model (default: `"model_response"`) | `model_response_column="output"` |
| `question_id_column` | Name of the question_id column (default: `"question_id"` if column exists) | `question_id_column="qid"` |

### Side-by-Side Comparisons

**Option 1: Tidy Data (Auto-pairing)**

If your data is in tidy single-model format with multiple models, StringSight can automatically pair them:

```python
# Tidy format with multiple models
df = pd.DataFrame({
    "prompt": ["What is ML?", "What is ML?", "Explain QC", "Explain QC"],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "ML involves..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC uses..."}],
        [{"role": "user", "content": "Explain QC"},
         {"role": "assistant", "content": "QC leverages..."}]
    ],
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

**Option 2: Pre-paired Data**

**Required Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `prompt` | Question given to both models | `"What is machine learning?"` |
| `model_a` | First model name | `"gpt-4"` |
| `model_b` | Second model name | `"claude-3"` |
| `model_a_response` | First model's response in OpenAI conversation format (or plain string) | `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` |
| `model_b_response` | Second model's response in OpenAI conversation format (or plain string) | `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` |

**Optional Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `score` | Winner and metrics | `{"winner": "model_a", "helpfulness_a": 4.2, "helpfulness_b": 3.8}` |
| `score_columns` | Alternative: separate columns for each metric with `_a` and `_b` suffixes (e.g., `accuracy_a`, `accuracy_b`) | `score_columns=["accuracy_a", "accuracy_b", "helpfulness_a", "helpfulness_b"]` |
| `prompt_column` | Name of the prompt column in your dataframe (default: `"prompt"`) | `prompt_column="query"` |
| `model_a_column` | Name of the model_a column (default: `"model_a"`) | `model_a_column="model_1"` |
| `model_b_column` | Name of the model_b column (default: `"model_b"`) | `model_b_column="model_2"` |
| `model_a_response_column` | Name of the model_a_response column (default: `"model_a_response"`) | `model_a_response_column="response_1"` |
| `model_b_response_column` | Name of the model_b_response column (default: `"model_b_response"`) | `model_b_response_column="response_2"` |
| `question_id_column` | Name of the question_id column (default: `"question_id"` if column exists) | `question_id_column="qid"` |

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
    model_name="gpt-4.1-mini",           # LLM for property extraction
    embedding_model="text-embedding-3-large",  # Embedding model for clustering
    min_cluster_size=5,                # Minimum cluster size
    output_dir="results/",              # Save outputs here
    use_wandb=True,                     # W&B logging (default True)
)
```

### Caching

StringSight uses an on-disk cache (LMDB-based) by default to speed up repeated LLM and embedding calls.

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
- Extraction: `"gpt-4.1"`, `"gpt-4.1-mini"`, `"anthropic/claude-3-5-sonnet"`, `"google/gemini-1.5-pro"`
- Embeddings: `"text-embedding-3-large"`, `"text-embedding-3-large"`, or local models like `"all-MiniLM-L6-v2"`

### Prompt Expansion

Prompt expansion is an optional feature that automatically enhances your task description by analyzing example traces from your dataset. Instead of using a generic or brief task description, expansion generates a comprehensive, task-specific list of behaviors to look for based on actual examples in your data.

**When to Use Prompt Expansion:**

- You have a general task description but want more specific guidance for extraction
- Your dataset contains domain-specific behaviors that aren't covered by default descriptions
- You want to improve extraction quality by providing more context about what to look for
- You're working with a new domain or task type where default descriptions may be insufficient

**How It Works:**

1. You provide a base `task_description` (or use the default)
2. StringSight randomly samples `expansion_num_traces` traces from your dataset (default: 5)
3. An LLM analyzes these traces and generates an expanded task description with specific behaviors to look for
4. The expanded description is used in both extraction and clustering prompts

**Usage:**

```python
clustered_df, model_stats = explain(
    df,
    task_description="The task is summarizing call-center conversations for IT support.",
    prompt_expansion=True,              # Enable expansion
    expansion_num_traces=5,            # Number of traces to sample (default: 5)
    expansion_model="gpt-4.1",         # Model for expansion (default: "gpt-4.1")
    output_dir="results/"
)
```

**Example:**

Without expansion, you might provide:
```python
task_description="Analyze model responses for code quality and security issues."
```

With expansion enabled, StringSight might generate:
```
Task: Analyze model responses for code quality and security issues.

Specific behaviors to look for:
- Code Quality: Does the model suggest insecure coding practices (e.g., SQL injection vulnerabilities, hardcoded credentials, missing input validation)?
- Security: Does the model identify potential security vulnerabilities in code examples?
- Best Practices: Does the model recommend following security best practices (e.g., using parameterized queries, proper error handling)?
- Code Review: Does the model provide constructive feedback on code structure and maintainability?
...
```

**Parameters:**

- `prompt_expansion` (bool, default: `False`): Enable/disable prompt expansion
- `expansion_num_traces` (int, default: `5`): Number of traces to sample for expansion
- `expansion_model` (str, default: `"gpt-4.1"`): LLM model to use for generating expanded descriptions

**Note:** Prompt expansion adds one additional LLM call before extraction begins. The expanded description is cached and reused throughout the pipeline, so it only adds minimal overhead.

## CLI Usage

### Launch the Web UI

**Simple mode (foreground):**
```bash
# Launch in your terminal (Ctrl+C to stop)
stringsight launch

# Launch on a custom port
stringsight launch --port 8080

# Launch with debug logging
stringsight launch --debug
```

**Daemon mode (background):**
```bash
# Launch in background (survives terminal disconnects)
stringsight launch --daemon --workers 4

# Check if server is running
stringsight status

# View logs
stringsight logs
stringsight logs --follow  # tail -f equivalent

# Stop the server
stringsight stop
```

The UI will be available at [http://localhost:5180](http://localhost:5180) by default.

**Daemon mode features:**
- Runs in background (survives terminal disconnects)
- Multiple workers for concurrent request handling
- Logs saved to `~/.stringsight/logs/server.log`
- PID file for process management

### Run Pipeline from Command Line

```bash
# Run full pipeline from command line
python scripts/run_full_pipeline.py \
    --data_path /path/to/data.jsonl \
    --output_dir /path/to/results \
    --method single_model \
    --embedding_model text-embedding-3-large

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
  

Contributing & Help: PRs welcome. Questions or issues? Open an issue on GitHub (https://github.com/lisadunlap/stringsight/issues)
