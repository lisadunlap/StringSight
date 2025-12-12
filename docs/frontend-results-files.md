## UI Results Files: Required Inputs and Structures

### Scope
- Only the files the UI actually reads to load results
- Metrics are assumed to use the df.jsonl variants (not the legacy .json files)

### Load Paths
- **Server/API path**: UI calls backend endpoint to read files from a results directory
- **Folder upload path**: UI reads files directly in the browser

### Quick Reference: File Purpose

| File | Purpose | Contains |
|------|---------|----------|
| `conversation.jsonl` | User prompts and model responses | Question ID, prompt, model(s), response(s), scores |
| `properties.jsonl` | Extracted behavioral properties | Property ID, description, category, evidence, behavior type |
| `clusters.jsonl` | Grouped similar properties | Cluster ID, label, size, property/question IDs |
| `full_dataset.json` | Complete archive (fallback) | All conversations, properties, clusters, model stats |
| `model_cluster_scores_df.jsonl` | Per-model-cluster metrics | Model, cluster, size, proportions, quality scores |
| `model_scores_df.jsonl` | Per-model aggregates | Model, total size, average quality scores |
| `cluster_scores_df.jsonl` | Per-cluster aggregates | Cluster, total size, average proportions, quality |

**Recommended**: Load the three JSONL files (`conversation.jsonl`, `properties.jsonl`, `clusters.jsonl`) instead of extracting from `full_dataset.json` for better performance.

### Required files

1) conversation.jsonl
- **Used by**: Server/API path and Folder upload path
- **Purpose**: Contains all conversations/prompts with model responses and scores for the Data tab
- **Format**: JSONL (one JSON object per line)
- **Description**: Each line represents a single conversation with a user prompt and model response(s). Uses the input format with `score` (single model) or `score_a`/`score_b` (side-by-side). Responses are in OpenAI message format with role/content pairs.

**Single model structure**:
```json
{
  "question_id": "352",
  "prompt": "Write a Python function to reverse a string",
  "model": "gpt-4",
  "model_response": [
    {"role": "user", "content": "Write a Python function to reverse a string"},
    {"role": "assistant", "content": "Here's a function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"}
  ],
  "score": {"helpfulness": 5.0, "conciseness": 4.5}
}
```

**Side-by-side structure**:
```json
{
  "question_id": "352",
  "prompt": "Write a Python function to reverse a string",
  "model_a": "gpt-4",
  "model_b": "claude-3-opus",
  "model_a_response": [
    {"role": "user", "content": "Write a Python function to reverse a string"},
    {"role": "assistant", "content": "Here's a function:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"}
  ],
  "model_b_response": [
    {"role": "user", "content": "Write a Python function to reverse a string"},
    {"role": "assistant", "content": "Here's how to reverse a string:\n\n```python\ndef reverse_string(text):\n    return text[::-1]\n```"}
  ],
  "score_a": {"helpfulness": 5.0},
  "score_b": {"helpfulness": 4.5},
  "winner": "model_a"
}
```

---

1b) clustered_results_lightweight.jsonl **(DEPRECATED)**
- **Used by**: Server/API path (legacy support only)
- **Purpose**: Old format for conversations
- **Note**: This file is deprecated. New pipelines generate `conversation.jsonl` instead. Use `conversation.jsonl` for all new implementations.

---

2) properties.jsonl
- **Used by**: Server/API path and Folder upload path
- **Purpose**: Contains extracted behavioral properties from model responses for the Properties and Clusters tabs
- **Format**: JSONL (one JSON object per line)
- **Description**: Each line represents a single extracted property describing a specific behavior exhibited by a model in its response. Properties include categorization, evidence from the response, and behavior type classification. These are standalone objects without cluster joins.

**Sample row**:
```json
{
  "id": "67891533-db42-45e0-bde2-fe7e1840b4a2",
  "question_id": "352",
  "model": "gpt-4",
  "property_description": "Provides code with proper syntax highlighting and formatting",
  "category": "Code Quality",
  "reason": "The model formatted the code in a clear, readable manner with proper Python syntax",
  "evidence": "The response includes properly indented code within markdown code blocks",
  "behavior_type": "Positive",
  "raw_response": null,
  "contains_errors": false,
  "unexpected_behavior": false,
  "meta": {}
}
```

---

2b) parsed_properties.jsonl **(DEPRECATED)**
- **Used by**: Server/API path (legacy support only)
- **Purpose**: Old format for properties with cluster joins
- **Note**: This file is deprecated. New pipelines generate `properties.jsonl` instead (standalone properties without cluster joins). Use `properties.jsonl` for all new implementations.

---

3) clusters.jsonl
- **Used by**: Server/API path and Folder upload path
- **Purpose**: Contains cluster definitions grouping similar properties for the Clusters tab
- **Format**: JSONL (one JSON object per line)
- **Description**: Each line represents a cluster of similar behavioral properties. Clusters group properties by semantic similarity and include lists of property descriptions, property IDs, and question IDs that belong to the cluster. These are standalone cluster definitions that can be joined with properties as needed.

**Sample row**:
```json
{
  "id": "1",
  "label": "Provides well-formatted code examples",
  "size": 45,
  "property_descriptions": [
    "Provides code with proper syntax highlighting and formatting",
    "Includes markdown code blocks for code snippets",
    "Uses consistent indentation in code examples"
  ],
  "property_ids": [
    "67891533-db42-45e0-bde2-fe7e1840b4a2",
    "a1b2c3d4-e5f6-4789-abcd-ef1234567890",
    "12345678-90ab-cdef-1234-567890abcdef"
  ],
  "question_ids": [
    "352",
    "423",
    "501"
  ],
  "meta": {}
}
```

---

4) full_dataset.json
- **Used by**: Folder upload path as fallback; Server/API path (legacy)
- **Purpose**: Complete dataset bundle containing all data (conversations, properties, clusters, model stats)
- **Format**: JSON (single large object)
- **Description**: A single JSON file containing the entire dataset in internal format. This is useful for archiving complete results or as a fallback when individual JSONL files are not available. However, loading individual JSONL files is preferred for better performance.

**Structure**:
```json
{
  "conversations": [
    {
      "question_id": "352",
      "prompt": "Write a Python function to reverse a string",
      "model": "gpt-4",
      "responses": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "scores": {"helpfulness": 5.0},
      "meta": {}
    }
  ],
  "properties": [
    {
      "id": "67891533-db42-45e0-bde2-fe7e1840b4a2",
      "question_id": "352",
      "model": "gpt-4",
      "property_description": "Provides well-formatted code",
      "category": "Code Quality",
      "behavior_type": "Positive",
      "meta": {}
    }
  ],
  "clusters": [
    {
      "id": "1",
      "label": "Provides well-formatted code examples",
      "size": 45,
      "property_descriptions": ["...", "..."],
      "property_ids": ["...", "..."],
      "question_ids": ["...", "..."],
      "meta": {}
    }
  ],
  "model_stats": {
    "gpt-4": {"total_properties": 150, "avg_quality": 4.5}
  },
  "all_models": ["gpt-4", "claude-3-opus", "gemini-pro"]
}
```

**Note**: Prefer loading individual JSONL files (`conversation.jsonl`, `properties.jsonl`, `clusters.jsonl`) for better performance. The `full_dataset.json` uses internal format (e.g., `scores` instead of `score`), while JSONL files use input format.

---

### Metrics Files (df.jsonl variants; load only if metrics are computed)

5) model_cluster_scores_df.jsonl
- **Used by**: Server/API path and Folder upload path
- **Purpose**: Per-model per-cluster metrics showing how each model performs on each behavioral cluster
- **Format**: JSONL (one JSON object per line)
- **Description**: Each line represents metrics for a specific model-cluster combination. Shows the size (number of properties), proportion of that model's total properties in this cluster, attributable impact (weighted delta) compared to baseline, and quality scores for each metric. Used for model comparison charts and cluster breakdowns.

**Sample row**:
```json
{
  "model": "gpt-4",
  "cluster": "Provides well-formatted code examples",
  "size": 25,
  "proportion": 0.35,
  "proportion_delta": 0.05,
  "quality_helpfulness": 0.82,
  "quality_helpfulness_delta": 0.0105,
  "quality_conciseness": 0.78,
  "quality_conciseness_delta": 0.0035
}
```

---

6) model_scores_df.jsonl
- **Used by**: Server/API path and Folder upload path
- **Purpose**: Aggregated metrics for each model across all clusters
- **Format**: JSONL (one JSON object per line)
- **Description**: Each line represents overall metrics for a single model. Includes total size (number of properties), average quality scores across all metrics, and model-level statistics. Used for high-level model comparison summaries.

**Sample row**:
```json
{
  "model": "gpt-4",
  "size": 300,
  "avg_quality_overall": 0.81,
  "avg_quality_helpfulness": 0.83,
  "avg_quality_conciseness": 0.79,
  "avg_quality_harmlessness": 0.95
}
```

---

7) cluster_scores_df.jsonl
- **Used by**: Server/API path and Folder upload path
- **Purpose**: Aggregated metrics for each cluster across all models
- **Format**: JSONL (one JSON object per line)
- **Description**: Each line represents overall metrics for a single behavioral cluster. Shows total size across all models, average proportion (how common this cluster is), and average quality scores. Used for cluster-level analysis and identifying common behavioral patterns.

**Sample row**:
```json
{
  "cluster": "Provides well-formatted code examples",
  "size": 120,
  "proportion": 0.40,
  "avg_quality_overall": 0.79,
  "avg_quality_helpfulness": 0.81,
  "avg_quality_conciseness": 0.77
}
```

### Notes

**Loading order preferences:**
- **Conversations**: `conversation.jsonl` → `clustered_results_lightweight.jsonl` (deprecated) → `full_dataset.json` (fallback)
- **Properties**: `properties.jsonl` → `parsed_properties.jsonl` (deprecated) → `full_dataset.json` (fallback)
- **Clusters**: `clusters.jsonl` → `full_dataset.json` (fallback)

**Key points:**
- The UI should load `conversation.jsonl`, `properties.jsonl`, and `clusters.jsonl` as separate files instead of extracting from `full_dataset.json` for better performance
- `conversation.jsonl` uses the input format (score/score_a/score_b, model_response/model_a_response/model_b_response) which matches the expected pipeline input format
- `properties.jsonl` contains standalone properties without cluster joins (unlike deprecated `parsed_properties.jsonl`)
- `clusters.jsonl` contains standalone cluster metadata
- The UI enriches cluster `meta` at runtime by joining properties and calculating per‑model aggregates
- **Metrics**: prefer df.jsonl files. Do not load legacy `model_cluster_scores.json`, `cluster_scores.json`, or `model_scores.json` if df.jsonl files are present

---

### File Relationships and Joins

The files are designed to be joined on specific keys:

```
conversation.jsonl (question_id, model)
    ↓ join on question_id + model
properties.jsonl (question_id, model, id)
    ↓ join on property_description
clusters.jsonl (id, property_descriptions[], property_ids[], question_ids[])
```

**Example join logic**:
1. Load all three files independently
2. To display properties for a conversation: join `properties.jsonl` with `conversation.jsonl` on `question_id` and `model`
3. To display clusters for properties: join `clusters.jsonl` with `properties.jsonl` where `property.property_description` is in `cluster.property_descriptions[]`
4. To display metrics: load metric files and join on `model` and/or `cluster` fields

**Why separate files?**
- **Performance**: Load only what you need
- **Flexibility**: Different views need different data
- **Streaming**: JSONL can be processed line-by-line
- **Caching**: Cache individual components separately
- **Reusability**: `conversation.jsonl` can be reused as pipeline input


