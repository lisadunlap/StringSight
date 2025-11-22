# Data Formats

StringSight requires specific data formats for input and produces structured outputs. This guide covers all supported formats and schemas.

## Input Data Formats

StringSight supports two primary analysis methods, each with specific column requirements.

### Single Model Format

Used for analyzing behavioral patterns from individual model responses.

#### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `prompt` | `str` | User question/prompt | `"What is machine learning?"` |
| `model` | `str` | Model identifier | `"gpt-4"`, `"claude-3-opus"` |
| `model_response` | `str` or `list` | Model's response (see [Response Format](#response-format)) | `"Machine learning is..."` |

#### Optional Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `question_id` | `str` | Unique conversation ID | `"q_12345"` (auto-generated if missing) |
| `score` | `dict` | Quality/evaluation scores | `{"accuracy": 0.85, "helpfulness": 4.2}` |

#### Example DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    "question_id": ["q1", "q2", "q3"],
    "prompt": [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a poem about AI"
    ],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning is a subset of AI..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "Quantum computing uses quantum bits..."}],
        [{"role": "user", "content": "Write a poem about AI"},
         {"role": "assistant", "content": "In circuits of light, intelligence grows..."}]
    ],
    "score": [
        {"accuracy": 1, "helpfulness": 4.5},
        {"accuracy": 0, "helpfulness": 3.8},
        {"accuracy": 1, "helpfulness": 4.2}
    ]
})
```

### Side-by-Side Format

Used for head-to-head model comparisons (Arena-style battles).

**StringSight supports TWO ways to provide side-by-side data:**

#### Option 1: Pre-Paired Format (Explicit Columns)

Provide explicit columns for both models with their responses already paired.

**Required Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `prompt` | `str` | Question given to both models | `"What is machine learning?"` |
| `model_a` | `str` | First model identifier | `"gpt-4"` |
| `model_b` | `str` | Second model identifier | `"claude-3"` |
| `model_a_response` | `str` or `list` | First model's response | `"ML is a subset of AI..."` |
| `model_b_response` | `str` or `list` | Second model's response | `"Machine learning involves..."` |

**Optional Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `question_id` | `str` | Unique conversation ID | `"battle_001"` |
| `score` | `dict` | Battle results and scores | `{"winner": "model_a", "helpfulness": 4.2}` |

**Example:**

```python
# Pre-paired side-by-side format
df = pd.DataFrame({
    "question_id": ["b1", "b2", "b3"],
    "prompt": [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a poem about AI"
    ],
    "model_a": ["gpt-4", "gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3", "claude-3"],
    "model_a_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "ML is a subset of AI..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "Quantum computing uses qubits..."}],
        [{"role": "user", "content": "Write a poem about AI"},
         {"role": "assistant", "content": "In circuits of light..."}]
    ],
    "model_b_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning involves..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "QC leverages quantum phenomena..."}],
        [{"role": "user", "content": "Write a poem about AI"},
         {"role": "assistant", "content": "Silicon dreams awaken..."}]
    ],
    "score": [
        {"winner": "model_a", "helpfulness": 4.2},
        {"winner": "model_a", "helpfulness": 3.8},
        {"winner": "model_b", "helpfulness": 4.5}
    ]
})

# Run analysis
from stringsight import explain
clustered_df, model_stats = explain(df, method="side_by_side")
```

#### Option 2: Tidy Format with Model Selection (Auto-Pairing)

Provide data in tidy single-model format and specify which two models to compare using the **`model_a` and `model_b` parameters**. StringSight will automatically pair responses for shared prompts.

**Key Parameters:**
- `model_a="model_name"` - First model to compare
- `model_b="model_name"` - Second model to compare

**Required Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `prompt` | `str` | User question/prompt | `"What is machine learning?"` |
| `model` | `str` | Model identifier | `"gpt-4"`, `"claude-3"` |
| `model_response` | `str` or `list` | Model's response | `"Machine learning is..."` |

**Complete Example:**

```python
# Tidy format with multiple models
df_tidy = pd.DataFrame({
    "prompt": [
        "What is machine learning?",
        "What is machine learning?",  # Same prompt, different model
        "Explain quantum computing",
        "Explain quantum computing",
        "Write a poem about AI",
        "Write a poem about AI"
    ],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "ML is a subset of AI..."}],
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning involves..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "Quantum computing uses qubits..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "QC leverages quantum phenomena..."}],
        [{"role": "user", "content": "Write a poem about AI"},
         {"role": "assistant", "content": "In circuits of light..."}],
        [{"role": "user", "content": "Write a poem about AI"},
         {"role": "assistant", "content": "Silicon dreams awaken..."}]
    ]
})

# Run side-by-side analysis on tidy data
from stringsight import explain

clustered_df, model_stats = explain(
    df_tidy,
    method="side_by_side",
    model_a="gpt-4",      # ← Specify first model
    model_b="claude-3"    # ← Specify second model
)

# Option B: Using CLI script
# python scripts/run_full_pipeline.py \
#   --data_path data/tidy_data.jsonl \
#   --output_dir results/ \
#   --method side_by_side \
#   --model_a "gpt-4" \
#   --model_b "claude-3"
```

**How Auto-Pairing Works:**

1. Filters dataset to only the two specified models
2. Finds all prompts answered by BOTH models
3. Pairs responses for each shared prompt
4. Converts to side-by-side format internally
5. Runs analysis on paired data

**Note:** Only rows where both models answered the same prompt are kept.

## Response Format

**Recommended Format**: Use OpenAI conversation format for all model responses. This format preserves conversation structure, supports multimodal inputs, and enables better trace visualization in the UI.

### Automatic Format Detection

StringSight automatically detects and converts response formats:

1. **OpenAI format** (list of message dicts) → Used as-is (recommended)
2. **Simple strings** → Automatically converted to OpenAI conversation format
3. **Other types** → Converted to string then processed

### OpenAI Conversation Format

For complex conversations involving multiple turns, tool use, or multimodal content, use OpenAI's conversation format.

#### Message Structure

Each message is a dictionary with:

**Required Fields:**
- `role`: `"user"`, `"assistant"`, `"system"`, or `"tool"`
- `content`: Message content (string or dict)

**Optional Fields:**
- `name`: Model/tool identifier
- `id`: Unique identifier for the message

#### Simple Text Conversation

```python
response = [
    {
        "role": "user",
        "content": "What is machine learning?"
    },
    {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
    }
]
```

#### Multi-Turn Conversation

```python
response = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "What about 3+3?"},
    {"role": "assistant", "content": "3+3 equals 6."}
]
```

#### Tool-Augmented Response

```python
response = [
    {
        "role": "user",
        "content": "Search for papers on quantum computing"
    },
    {
        "role": "assistant",
        "content": {
            "tool_calls": [
                {
                    "name": "search_papers",
                    "arguments": {
                        "query": "quantum computing",
                        "year": 2024,
                        "max_results": 5
                    },
                    "tool_call_id": "call_abc123"
                }
            ]
        }
    },
    {
        "role": "tool",
        "name": "search_papers",
        "content": "Found 5 papers: [1] Quantum Error Correction..."
    },
    {
        "role": "assistant",
        "content": "Based on the search results, here are recent developments..."
    }
]
```

#### Multimodal Input

```python
response = [
    {
        "role": "user",
        "content": {
            "text": "What's in this image?",
            "image": "data:image/jpeg;base64,iVBORw0KGgo..."
        }
    },
    {
        "role": "assistant",
        "content": "I can see a diagram showing neural network architecture..."
    }
]
```

## Score Format

### Single Metric

```python
df["score"] = [0.85, 0.92, 0.78]  # Numeric values
```

### Multiple Metrics

```python
df["score"] = [
    {"accuracy": 1, "helpfulness": 4.2, "harmlessness": 4.8},
    {"accuracy": 0, "helpfulness": 3.5, "harmlessness": 4.9},
    {"accuracy": 1, "helpfulness": 4.8, "harmlessness": 4.7}
]
```

### Side-by-Side Winner Format

```python
df["score"] = [
    {"winner": "model_a"},
    {"winner": "model_b"},
    {"winner": "tie"}
]
```

### Combined Format

```python
df["score"] = [
    {"winner": "model_a", "accuracy": 0.9, "helpfulness": 4.5},
    {"winner": "model_b", "accuracy": 0.7, "helpfulness": 3.8},
    {"winner": "tie", "accuracy": 0.8, "helpfulness": 4.0}
]
```

### Using Separate Score Columns

Instead of providing scores as a dictionary in a single column, you can use the `score_columns` parameter to specify separate columns for each metric. StringSight will automatically convert them to the required dictionary format.

#### Single Model with Score Columns

```python
import pandas as pd
from stringsight import explain

# Data with separate score columns
df = pd.DataFrame({
    "prompt": ["What is AI?", "Explain ML", "What is DL?"],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": [
        [{"role": "user", "content": "What is AI?"},
         {"role": "assistant", "content": "AI is..."}],
        [{"role": "user", "content": "Explain ML"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "What is DL?"},
         {"role": "assistant", "content": "DL is..."}]
    ],
    "accuracy": [0.9, 0.85, 0.95],           # Separate column
    "helpfulness": [4.2, 4.0, 4.5],          # Separate column
    "clarity": [4.8, 4.5, 4.7]               # Separate column
})

# Specify which columns contain scores
clustered_df, model_stats = explain(
    df,
    method="single_model",
    score_columns=["accuracy", "helpfulness", "clarity"]  # Automatically converted to score dict
)
```

#### Side-by-Side with Score Columns

For side-by-side comparisons, score columns should have `_a` and `_b` suffixes:

```python
# Data with separate score columns for each model
df = pd.DataFrame({
    "prompt": ["What is AI?", "Explain ML"],
    "model_a": ["gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3"],
    "model_a_response": [
        [{"role": "user", "content": "What is AI?"},
         {"role": "assistant", "content": "AI is..."}],
        [{"role": "user", "content": "Explain ML"},
         {"role": "assistant", "content": "ML is..."}]
    ],
    "model_b_response": [
        [{"role": "user", "content": "What is AI?"},
         {"role": "assistant", "content": "AI involves..."}],
        [{"role": "user", "content": "Explain ML"},
         {"role": "assistant", "content": "Machine learning..."}]
    ],
    "accuracy_a": [0.9, 0.85],               # Model A accuracy
    "accuracy_b": [0.88, 0.90],              # Model B accuracy
    "helpfulness_a": [4.2, 4.0],             # Model A helpfulness
    "helpfulness_b": [4.3, 4.1]              # Model B helpfulness
})

# Specify base metric names (without _a/_b suffixes)
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    score_columns=["accuracy", "helpfulness"]  # Will look for *_a and *_b columns
)
```

#### Tidy Data with Score Columns

When using tidy format with `model_a` and `model_b` parameters, specify the score columns and they'll be pivoted automatically:

```python
# Tidy format with separate score columns
df = pd.DataFrame({
    "prompt": ["What is AI?", "What is AI?", "Explain ML", "Explain ML"],
    "model": ["gpt-4", "claude-3", "gpt-4", "claude-3"],
    "model_response": [
        [{"role": "user", "content": "What is AI?"},
         {"role": "assistant", "content": "AI is..."}],
        [{"role": "user", "content": "What is AI?"},
         {"role": "assistant", "content": "AI involves..."}],
        [{"role": "user", "content": "Explain ML"},
         {"role": "assistant", "content": "ML is..."}],
        [{"role": "user", "content": "Explain ML"},
         {"role": "assistant", "content": "ML involves..."}]
    ],
    "accuracy": [0.9, 0.88, 0.85, 0.90],
    "helpfulness": [4.2, 4.3, 4.0, 4.1]
})

# Convert tidy to side-by-side with score columns
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    model_a="gpt-4",
    model_b="claude-3",
    score_columns=["accuracy", "helpfulness"]  # Automatically pivoted to *_a/*_b format
)
```

**Benefits of using `score_columns`:**
- More natural data format (especially when exporting from databases or spreadsheets)
- No need to manually construct score dictionaries
- Automatic validation that columns contain numeric values
- Works seamlessly with tidy data conversion

## Output Data Formats

When you specify `output_dir`, StringSight saves multiple files in different formats.

### Main Output Files

#### clustered_results.parquet

**Format:** Apache Parquet
**Description:** Full dataset with all original columns plus extracted properties and clusters

**New Columns Added:**

| Column | Type | Description |
|--------|------|-------------|
| `property_id` | `str` | Unique property identifier |
| `property_description` | `str` | Extracted behavioral property |
| `category` | `str` | Property category |
| `reason` | `str` | Why this property was identified |
| `evidence` | `str` | Evidence from the response |
| `behavior_type` | `str` | Type of behavior |
| `property_description_cluster_id` | `int` | Fine-grained cluster ID |
| `property_description_cluster_label` | `str` | Human-readable fine cluster name |
| `embedding` | `list[float]` | Property embedding vector (if `include_embeddings=True`) |

**Loading:**
```python
df = pd.read_parquet("results/clustered_results.parquet")
```

#### full_dataset.json

**Format:** JSON
**Description:** Complete `PropertyDataset` object with all data structures

**Structure:**
```json
{
  "conversations": [...],
  "properties": [...],
  "clusters": [...],
  "model_stats": {...},
  "all_models": [...]
}
```

**Loading:**
```python
from stringsight.core.data_objects import PropertyDataset

dataset = PropertyDataset.load("results/full_dataset.json")
```

#### model_cluster_scores.json

**Format:** JSON
**Description:** Per model-cluster performance metrics

**Structure:**
```json
{
  "gpt-4": {
    "Reasoning Transparency": {
      "size": 150,
      "proportion": 0.25,
      "quality": {"accuracy": 0.92, "helpfulness": 4.5},
      "quality_delta": {"accuracy": 0.15, "helpfulness": 0.3},
      "proportion_delta": 0.12,
      "examples": ["q1", "q2", "q3"],
      "proportion_ci": {"lower": 0.22, "upper": 0.28},
      "quality_ci": {...},
      "quality_delta_ci": {...},
      "proportion_delta_ci": {...}
    }
  }
}
```

#### cluster_scores.json

**Format:** JSON
**Description:** Per-cluster aggregated metrics across all models

#### model_scores.json

**Format:** JSON
**Description:** Per-model aggregated metrics across all clusters

#### summary.txt

**Format:** Plain text
**Description:** Human-readable analysis summary

### Intermediate Files

These are saved during pipeline execution:

- `raw_properties.jsonl` - Raw LLM extraction responses
- `extraction_stats.json` - Extraction stage statistics
- `parsed_properties.jsonl` - Parsed property objects
- `parsing_stats.json` - Parsing success/failure stats
- `validated_properties.jsonl` - Validated properties
- `validation_stats.json` - Validation statistics
- `embeddings.parquet` - Property embeddings data

## Loading Data

### From Files

```python
import pandas as pd
from stringsight.core.data_objects import PropertyDataset

# Load clustered results
df = pd.read_parquet("results/clustered_results.parquet")

# Load full dataset
dataset = PropertyDataset.load("results/full_dataset.json")

# Load metrics
import json
with open("results/model_cluster_scores.json") as f:
    metrics = json.load(f)
```

### From Various Sources

```python
# CSV
df = pd.read_csv("data.csv")

# JSON Lines
df = pd.read_json("data.jsonl", lines=True)

# Parquet
df = pd.read_parquet("data.parquet")

# JSON
df = pd.read_json("data.json")
```

## Data Validation

StringSight automatically validates your data format. You can also validate manually:

```python
from stringsight.formatters import detect_method, validate_required_columns

# Detect single_model vs side_by_side
method = detect_method(df)
print(f"Detected method: {method}")

# Validate required columns
try:
    validate_required_columns(df, method)
    print("✅ Data format is valid")
except ValueError as e:
    print(f"❌ Validation error: {e}")
```

## Converting Between Formats

### Tidy to Side-by-Side

Convert single-model format to side-by-side for comparison:

```bash
python scripts/run_full_pipeline.py \
  --data_path data/tidy_data.jsonl \
  --output_dir results/ \
  --method side_by_side \
  --model_a "gpt-4" \
  --model_b "claude-3"
```

This automatically:
1. Filters to prompts answered by both models
2. Pairs responses for each shared prompt
3. Converts to side-by-side format

### Programmatic Conversion

```python
# Example: Convert tidy to side-by-side
models_a = df[df['model'] == 'gpt-4']
models_b = df[df['model'] == 'claude-3']

side_by_side = pd.merge(
    models_a,
    models_b,
    on='prompt',
    suffixes=('_a', '_b')
).rename(columns={
    'model_a': 'model_a',
    'model_b': 'model_b',
    'model_response_a': 'model_a_response',
    'model_response_b': 'model_b_response'
})
```

## Best Practices

### Data Preparation

1. **Unique IDs**: Use meaningful `question_id` values for easier tracking
2. **Consistent Naming**: Use consistent model names across your dataset
3. **Score Format**: Use dictionaries for multiple evaluation criteria
4. **Response Format**: Use OpenAI format for complex conversations

### Quality Checks

```python
# Check for missing values
print(df.isnull().sum())

# Check model distribution
print(df['model'].value_counts())

# Check response lengths
df['response_length'] = df['model_response'].str.len()
print(df['response_length'].describe())

# Verify score format
if 'score' in df.columns:
    print(df['score'].apply(type).value_counts())
```

### Performance Tips

1. **Use Parquet**: Faster loading/saving than CSV or JSON
2. **Sample Large Datasets**: Use sampling for initial exploration
3. **Cache Results**: Save intermediate results to avoid recomputation

## Troubleshooting

### Common Issues

**"Missing required column"**
```python
# Check your columns
print(df.columns.tolist())

# Rename if needed
df = df.rename(columns={'response': 'model_response'})
```

**"Invalid response format"**
```python
# Convert all responses to strings
df['model_response'] = df['model_response'].astype(str)
```

**"Score column not recognized"**
```python
# Ensure scores are dictionaries
import json
df['score'] = df['score'].apply(json.loads)  # If stored as strings
```

## Next Steps

- **[Configuration Options](configuration.md)** - Learn about all available parameters
- **[Basic Usage](basic-usage.md)** - See how to use these formats with `explain()` and `label()`
- **[Visualization](visualization.md)** - Explore results in the web interface
