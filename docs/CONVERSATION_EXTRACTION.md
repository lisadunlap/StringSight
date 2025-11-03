# Data Extraction

## Overview

The pipeline now automatically saves conversations, properties, and clusters as separate JSONL files in the results directory. This makes it easy to load and work with specific data components without loading the entire dataset.

## Files Created

When running the pipeline with an `output_dir`, these files are created:

1. **`full_dataset.json`** - Complete dataset including conversations, properties, clusters, and model stats
2. **`conversation.jsonl`** - Just the conversations in JSONL format (one conversation per line)
3. **`properties.jsonl`** - Just the properties in JSONL format (one property per line)
4. **`clusters.jsonl`** - Just the clusters in JSONL format (one cluster per line)

## Conversation Format

The `conversation.jsonl` file uses the **input format** expected by the pipeline, making it easy to reuse extracted conversations as new inputs.

### Single Model Format

Each line contains a single conversation:

```json
{
  "question_id": "0",
  "prompt": "User's input prompt",
  "model": "model_name",
  "model_response": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "score": {"Helpfulness": 5.0, "Conciseness": 4.0}
}
```

### Side-by-Side Format

For side-by-side comparisons:

```json
{
  "question_id": "0",
  "prompt": "User's input prompt",
  "model_a": "gpt-4",
  "model_b": "claude-3",
  "model_a_response": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "model_b_response": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "score_a": {"Helpfulness": 5.0},
  "score_b": {"Helpfulness": 4.0},
  "winner": "model_a"
}
```

**Key points:**
- Uses `score` (not `scores`) for single model
- Uses `score_a` and `score_b` (not `scores` list) for side-by-side
- Uses `model_response` (not `responses`) for single model
- Uses `model_a_response` and `model_b_response` for side-by-side
- Metadata fields like `winner` are at the top level

## Properties Format

Each line in `properties.jsonl` contains a single property:

```json
{
  "id": "67891533-db42-45e0-bde2-fe7e1840b4a2",
  "question_id": "352",
  "model": "xai/grok-3-mini-beta",
  "property_description": "Fulfills the user's length constraint",
  "category": "User Experience",
  "reason": "...",
  "evidence": "...",
  "behavior_type": "Positive",
  "raw_response": null,
  "contains_errors": false,
  "unexpected_behavior": false,
  "meta": {}
}
```

## Clusters Format

Each line in `clusters.jsonl` contains a single cluster:

```json
{
  "id": "1",
  "label": "Strictly follows user instructions",
  "size": 120,
  "property_descriptions": ["...", "..."],
  "property_ids": ["...", "..."],
  "question_ids": ["...", "..."],
  "meta": {}
}
```

## Extracting Data from Existing Results

If you have existing results that don't have the separate JSONL files, use the extraction script:

```bash
python scripts/extract_conversations.py <results_directory>
```

Example:
```bash
python scripts/extract_conversations.py results/koala
```

This will:
1. Read `full_dataset.json` from the specified directory
2. Extract the conversations, properties, and clusters fields
3. Save to separate JSONL files in the same directory:
   - `conversation.jsonl`
   - `properties.jsonl` (if properties exist)
   - `clusters.jsonl` (if clusters exist)

## Loading Data

### Python API

```python
import json
import pandas as pd

# Load conversations
conversations = []
with open("results/my_experiment/conversation.jsonl", "r") as f:
    for line in f:
        conversations.append(json.loads(line))

# Or use pandas for any JSONL file
conversations_df = pd.read_json("results/my_experiment/conversation.jsonl", lines=True)
properties_df = pd.read_json("results/my_experiment/properties.jsonl", lines=True)
clusters_df = pd.read_json("results/my_experiment/clusters.jsonl", lines=True)
```

### Frontend

The frontend can now load data components independently for faster initial page loads:

```javascript
// Load conversations
fetch('/path/to/conversation.jsonl')
  .then(response => response.text())
  .then(text => {
    const conversations = text.trim().split('\n').map(line => JSON.parse(line));
    // Use conversations...
  });

// Load properties
fetch('/path/to/properties.jsonl')
  .then(response => response.text())
  .then(text => {
    const properties = text.trim().split('\n').map(line => JSON.parse(line));
    // Use properties...
  });

// Load clusters
fetch('/path/to/clusters.jsonl')
  .then(response => response.text())
  .then(text => {
    const clusters = text.trim().split('\n').map(line => JSON.parse(line));
    // Use clusters...
  });
```

## Benefits

1. **Faster loading** - Load only the data components you need
2. **Streaming** - JSONL format allows streaming line-by-line for large datasets
3. **Smaller files** - Individual components are much smaller than the full dataset
4. **Standard format** - JSONL is widely supported and easy to process
5. **Reusable** - `conversation.jsonl` uses input format, making it easy to reuse as pipeline input
6. **Independent** - Properties and clusters can be loaded without conversations for analysis
