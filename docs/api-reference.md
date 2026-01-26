# StringSight API Reference

Comprehensive documentation of all API endpoints, their input/output formats, and what they mean.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: Configure via environment variable

---

## Table of Contents

1. [Health & Status](#health--status)
2. [Data Upload & Validation](#data-upload--validation)
3. [File System Operations](#file-system-operations)
4. [Property Extraction](#property-extraction)
5. [Clustering & Metrics](#clustering--metrics)
6. [DataFrame Operations](#dataframe-operations)
7. [Results Management](#results-management)
8. [Configuration](#configuration)

---

## Health & Status

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "ok": true
}
```

### GET `/api/health`

Alternative health check endpoint with `/api` prefix.

**Response:**
```json
{
  "ok": true
}
```

---

## Data Upload & Validation

### POST `/detect-and-validate`

Auto-detect data format and validate required columns.

**Request:** (form-data or JSON)

- **Form-data upload:**
  - `file`: UploadFile (JSONL, JSON, or CSV)

- **JSON payload:**
  ```json
  {
    "rows": [
      {
        "question_id": "q1",
        "prompt": "What is AI?",
        "model": "gpt-4",
        "model_response": "AI is...",
        "score": {"helpfulness": 0.8}
      }
    ],
    "method": "single_model"  // optional: "single_model" | "side_by_side"
  }
  ```

**Response:**
```json
{
  "method": "single_model",  // Detected or provided method
  "valid": true,              // Whether data has all required columns
  "missing": [],              // List of missing column names
  "row_count": 100,           // Total rows in dataset
  "columns": ["question_id", "prompt", "model", "model_response", "score"],
  "preview": [                // First 50 rows
    {
      "question_id": "q1",
      "prompt": "What is AI?",
      "model": "gpt-4",
      "model_response": "AI is...",
      "score": {"helpfulness": 0.8}
    }
  ]
}
```

**What it means:**
- **method**: The analysis format (single_model = one model per row, side_by_side = two models compared)
- **valid**: Whether your data can be processed without errors
- **missing**: Columns you need to add to your data
- **preview**: Sample of your data for verification

---

### POST `/conversations`

Format and validate conversation data.

**Request:** Same as `/detect-and-validate`

**Response:**
```json
{
  "method": "single_model",
  "conversations": [
    {
      "question_id": "q1",
      "prompt": "What is AI?",
      "model": "gpt-4",
      "response": "AI is...",
      "score": {"helpfulness": 0.8}
    }
  ]
}
```

**What it means:**
- Validates and normalizes conversation data
- Expands score columns from dict to individual fields
- Returns formatted conversations ready for processing

---

### POST `/auto-detect-columns`

Automatically detect column mappings from user data.

**Request:**
```json
{
  "rows": [
    {
      "user_query": "What is AI?",
      "gpt4_answer": "AI is...",
      "claude_answer": "Artificial intelligence...",
      "rating": 5
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "suggestions": {
    "prompt_col": "user_query",
    "response_cols": ["gpt4_answer", "claude_answer"],
    "model_cols": [],
    "score_cols": ["rating"],
    "method": "single_model"
  }
}
```

**What it means:**
- Analyzes column names and data to guess the right mappings
- **prompt_col**: Which column contains the user's question/prompt
- **response_cols**: Which columns contain model responses
- **model_cols**: Which columns identify which model generated each response
- **score_cols**: Which columns contain quality ratings/scores

---

### POST `/validate-flexible-mapping`

Validate user-specified column mappings.

**Request:**
```json
{
  "rows": [...],
  "mapping": {
    "prompt_col": "user_query",
    "response_cols": ["gpt4_answer"],
    "model_cols": ["model_name"],
    "score_cols": ["rating"],
    "method": "single_model"
  }
}
```

**Response:**
```json
{
  "valid": true,
  "errors": []  // List of error messages if invalid
}
```

---

### POST `/process-flexible-data`

Convert custom column format to StringSight's operational format.

**Request:**
```json
{
  "rows": [...],
  "mapping": {
    "prompt_col": "user_query",
    "response_cols": ["gpt4_answer"],
    "model_cols": ["model_name"],
    "score_cols": ["rating"],
    "method": "single_model"
  }
}
```

**Response:**
```json
{
  "success": true,
  "rows": [
    {
      "question_id": "0",
      "prompt": "What is AI?",
      "model": "gpt-4",
      "model_response": "AI is...",
      "score": {"rating": 5}
    }
  ],
  "method": "single_model",
  "columns": ["question_id", "prompt", "model", "model_response", "score"]
}
```

**What it means:**
- Transforms your custom format into StringSight's expected format
- Standardizes column names for downstream processing

---

### POST `/flexible-conversations`

Combine flexible mapping and conversation formatting in one step.

**Request:** Same as `/process-flexible-data`

**Response:**
```json
{
  "success": true,
  "method": "single_model",
  "conversations": [...]
}
```

---

## File System Operations

### POST `/read-path`

Read a dataset file from the server filesystem.

**Request:**
```json
{
  "path": "/path/to/data.jsonl",
  "method": "single_model",  // optional
  "limit": 1000              // optional: max rows to return
}
```

**Response:**
```json
{
  "method": "single_model",
  "row_count": 5000,
  "columns": ["question_id", "prompt", "model", "model_response"],
  "preview": [...]  // Up to 'limit' rows
}
```

**What it means:**
- Reads files directly from server storage
- Useful when data is already on the server
- **limit**: Controls how many rows are returned (all if omitted)

---

### POST `/list-path`

List files and directories at a server path.

**Request:**
```json
{
  "path": "/data/experiments/",
  "exts": [".jsonl", ".json", ".csv"]  // optional: filter by extensions
}
```

**Response:**
```json
{
  "entries": [
    {
      "name": "experiment_1.jsonl",
      "path": "/data/experiments/experiment_1.jsonl",
      "type": "file",
      "size": 1048576,          // bytes
      "modified": "2025-10-20T10:30:00"
    },
    {
      "name": "results",
      "path": "/data/experiments/results",
      "type": "dir",
      "modified": "2025-10-21T15:45:00"
    }
  ]
}
```

**What it means:**
- Browse server filesystem (restricted to BASE_BROWSE_DIR)
- **type**: "file" or "dir"
- **size**: File size in bytes (files only)
- **modified**: ISO 8601 timestamp

---

### POST `/results/load`

Load a complete results directory from a previous analysis.

**Request:**
```json
{
  "path": "/results/clustering_20251022_143000",
  "max_conversations": 1000,  // optional: limit conversations loaded
  "max_properties": 5000      // optional: limit properties loaded
}
```

**Response:**
```json
{
  "path": "/results/clustering_20251022_143000",
  "model_cluster_scores": [
    {
      "model": "gpt-4",
      "cluster": "Cluster 0: Detailed explanations",
      "proportion": 0.35,
      "proportion_delta": 0.05,
      "quality_helpfulness": 0.82,
      "quality_helpfulness_delta": 0.03
    }
  ],
  "cluster_scores": [
    {
      "cluster": "Cluster 0: Detailed explanations",
      "size": 150,
      "proportion": 0.30,
      "quality_helpfulness": 0.80
    }
  ],
  "model_scores": [
    {
      "model": "gpt-4",
      "size": 500,
      "quality_helpfulness": 0.79
    }
  ],
  "conversations": [...],  // Operational data rows
  "properties": [...],     // Extracted properties
  "clusters": [...]        // Cluster definitions
}
```

**What it means:**
- Loads complete analysis results from disk
- **model_cluster_scores**: Performance of each model in each cluster
- **cluster_scores**: Aggregate metrics for each behavior cluster
- **model_scores**: Overall model performance across all clusters
- **proportion**: Percentage of conversations showing this behavior
- **quality_***: Performance metrics (e.g., helpfulness, accuracy)
- ***_delta**: Difference from baseline/average

---

### GET `/results/stream/properties`

Stream properties data progressively (for large datasets).

**Query Parameters:**
- `path`: Results directory path
- `offset`: Starting row (default: 0)
- `limit`: Number of rows (default: 1000)

**Response:** JSONL stream (one property per line)
```
{"id": "p1", "question_id": "q1", "model": "gpt-4", "property_description": "..."}
{"id": "p2", "question_id": "q2", "model": "gpt-4", "property_description": "..."}
```

**What it means:**
- Streams data line-by-line instead of loading all at once
- Reduces memory usage for large datasets
- Frontend can render results as they arrive

---

### GET `/results/stream/conversations`

Stream conversations data progressively.

**Query Parameters:** Same as `/results/stream/properties`

**Response:** JSONL stream of conversation objects

---

## Property Extraction

### POST `/extract/single`

Extract properties from a single conversation.

**Request:**
```json
{
  "row": {
    "question_id": "q1",
    "prompt": "What is AI?",
    "model": "gpt-4",
    "model_response": "AI is..."
  },
  "method": "single_model",
  "system_prompt": "default",
  "task_description": "Identify interesting behaviors",
  "model_name": "gpt-4.1",
  "temperature": 0.7,
  "max_tokens": 16000,
  "max_workers": 16,
  "return_debug": false
}
```

**Response:**
```json
{
  "properties": [
    {
      "id": "p1",
      "question_id": "q1",
      "model": "gpt-4",
      "property_description": "Provides structured definition with examples",
      "category": "Response Style",
      "reason": "The response organizes information hierarchically",
      "evidence": "Uses bullet points and sub-categories",
      "behavior_type": "Style",
      "contains_errors": false,
      "unexpected_behavior": false
    }
  ],
  "counts": {
    "properties": 1
  },
  "failures": []
}
```

**What it means:**
- Uses LLM to identify interesting properties/behaviors in the conversation
- **property_description**: What behavior was observed
- **category**: Type of behavior (e.g., "Response Style", "Accuracy")
- **reason**: Why this is notable
- **evidence**: Specific quotes from the response
- **behavior_type**: "Positive" | "Negative (critical)" | "Negative (non-critical)" | "Style"

---

### POST `/extract/batch`

Extract properties from multiple conversations.

**Request:**
```json
{
  "rows": [...],  // Array of conversation objects
  "method": "single_model",
  "system_prompt": "default",
  "model_name": "gpt-4.1",
  "temperature": 0.7,
  "max_workers": 16,
  "sample_size": 100,  // optional: random sample N rows
  "return_debug": false
}
```

**Response:**
```json
{
  "rows": [...],      // Array of property objects
  "columns": [...],   // Property field names
  "counts": {
    "conversations": 100,
    "properties": 247
  },
  "stats": {
    "parse_failures": 3,
    "empty_lists": 1
  },
  "failures": []  // Populated if return_debug=true
}
```

**What it means:**
- Batch processes multiple conversations in parallel
- One conversation can generate multiple properties
- **parse_failures**: How many LLM responses couldn't be parsed
- **empty_lists**: How many conversations had no extractable properties

---

### POST `/extract/jobs/start`

Start async property extraction job (for very large batches).

**Request:** Same as `/extract/batch`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### GET `/extract/jobs/status`

Check extraction job status.

**Query Parameters:**
- `job_id`: Job ID from `/extract/jobs/start`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "running",  // "queued" | "running" | "done" | "error" | "cancelled"
  "progress": 0.45,
  "count_done": 450,
  "count_total": 1000,
  "error": null
}
```

---

### GET `/extract/jobs/result`

Get completed extraction job results.

**Query Parameters:**
- `job_id`: Job ID

**Response:**
```json
{
  "properties": [...],
  "count": 247,
  "cancelled": false
}
```

---

### POST `/extract/jobs/cancel`

Cancel running extraction job.

**Request:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "cancelled",
  "message": "Cancellation requested",
  "properties_count": 123  // Properties extracted before cancellation
}
```

---

## Clustering & Metrics

### POST `/cluster/run`

Run clustering on extracted properties and compute quality metrics.

**Request:**
```json
{
  "operationalRows": [  // Original conversation data
    {
      "question_id": "q1",
      "prompt": "...",
      "model": "gpt-4",
      "model_response": "...",
      "score": {"helpfulness": 0.8}  // Nested dict format (preferred)
    }
  ],
  "properties": [  // Extracted properties
    {
      "id": "p1",
      "question_id": "q1",
      "model": "gpt-4",
      "property_description": "...",
      "category": "Response Style",
      "behavior_type": "neutral"
    }
  ],
  "params": {
    "minClusterSize": 5,
    "embeddingModel": "openai/text-embedding-3-large",
    "groupBy": "none"  // "none" | "category" | "behavior_type"
  },
  "output_dir": "/results/my_analysis",  // optional
  "score_columns": ["helpfulness", "accuracy"],  // optional: see Score Handling below
  "method": "single_model"  // optional: "single_model" | "side_by_side" (default: "single_model")
}
```

#### Score Handling

The API accepts scores in multiple formats and will automatically convert them:

**Option 1: Nested dict format (no conversion needed)**
```json
{
  "operationalRows": [
    {
      "question_id": "q1",
      "model": "gpt-4",
      "model_response": "...",
      "score": {"helpfulness": 0.8, "accuracy": 0.9}  // ✅ Already in correct format
    }
  ]
}
```

**Option 2: Flattened columns with auto-detection (no parameters needed)**
```json
{
  "operationalRows": [
    {
      "question_id": "q1",
      "model": "gpt-4",
      "model_response": "...",
      "helpfulness": 0.8,  // Will be auto-detected
      "accuracy": 0.9      // Will be auto-detected
    }
  ]
  // No score_columns needed - API detects keywords automatically
}
```

Auto-detection looks for numeric columns containing these keywords:
- `score`, `rating`, `quality`, `helpfulness`, `accuracy`, `correctness`, `fluency`, `coherence`, `relevance`

**Option 3: Flattened columns with explicit specification**
```json
{
  "operationalRows": [
    {
      "question_id": "q1",
      "model": "gpt-4",
      "model_response": "...",
      "metric1": 0.8,  // Non-standard name
      "metric2": 0.9
    }
  ],
  "score_columns": ["metric1", "metric2"],  // ✅ Explicitly specify columns
  "method": "single_model"
}
```

**Option 4: Side-by-side with suffix notation**
```json
{
  "operationalRows": [
    {
      "question_id": "q1",
      "model_a": "gpt-4",
      "model_b": "claude-3",
      "model_a_response": "...",
      "model_b_response": "...",
      "helpfulness_a": 0.8,  // Model A scores
      "helpfulness_b": 0.7,  // Model B scores
      "accuracy_a": 0.9,
      "accuracy_b": 0.85
    }
  ],
  "score_columns": ["helpfulness", "accuracy"],  // Base names (without _a/_b)
  "method": "side_by_side"  // ✅ Must specify method
}
```

**When to use `score_columns`:**
- Your score columns have non-standard names (e.g., `metric1`, `custom_score`)
- You want explicit control over which columns are treated as scores
- Auto-detection is picking up wrong columns

**When NOT needed:**
- Scores already in nested dict format (`score: {helpfulness: 0.8}`)
- Column names contain standard keywords and you want auto-detection

**Response:**
```json
{
  "clusters": [
    {
      "id": 0,
      "label": "Cluster 0: Detailed explanations with examples",
      "size": 45,
      "property_descriptions": [
        "Provides structured definition with examples",
        "Uses analogies to explain concepts"
      ],
      "property_ids": ["p1", "p5", "p12"],
      "question_ids": ["q1", "q3", "q8"],
      "meta": {
        "proportion_overall": 0.15,
        "property_items": [
          {
            "property_id": "p1",
            "model": "gpt-4",
            "property_description": "..."
          }
        ]
      }
    }
  ],
  "total_conversations_by_model": {
    "gpt-4": 150,
    "claude-3": 150
  },
  "total_unique_conversations": 300,
  "results_dir": "clustering_20251104_143000",  // Name of saved results directory
  "metrics": {
    "model_cluster_scores": [
      {
        "model": "gpt-4",
        "cluster": "Cluster 0: Detailed explanations with examples",
        "cluster_id": 0,
        "size": 30,  // Number of properties in this model-cluster combination
        "proportion": 0.20,  // Proportion of this model's conversations in this cluster
        "proportion_delta": 0.05,  // Difference from average proportion
        "quality_helpfulness": 0.85,  // ✅ Quality scores (if scores provided)
        "quality_helpfulness_delta": 0.03,
        "quality_accuracy": 0.92,
        "quality_accuracy_delta": 0.04,
        "metadata": {"behavior_type": "neutral", "category": "Response Style"}
      }
    ],
    "cluster_scores": [
      {
        "cluster": "Cluster 0: Detailed explanations with examples",
        "cluster_id": 0,
        "size": 45,  // Total properties across all models
        "proportion": 0.15,  // Proportion of all conversations
        "quality_helpfulness": 0.82,  // Average across all models
        "quality_accuracy": 0.90,
        "metadata": {}
      }
    ],
    "model_scores": []  // Not computed in clustering-only endpoint
  }
}
```

**What it means:**
- Groups similar properties into behavior clusters
- **label**: Human-readable cluster name
- **size**: Number of properties in this cluster
- **property_descriptions**: Sample behaviors in this cluster
- **question_ids**: Which conversations exhibit this behavior
- **proportion_overall**: What % of all conversations show this behavior
- **groupBy**:
  - "none": Cluster all properties together
  - "category": Cluster separately by property category
  - "behavior_type": Cluster separately positive/negative/neutral

---

### POST `/cluster/metrics`

Recompute metrics for a filtered subset without reclustering.

**Request:**
```json
{
  "clusters": [...],
  "properties": [...],
  "operationalRows": [...],
  "included_property_ids": ["p1", "p5", "p12"],  // optional: filter subset
  "score_columns": ["helpfulness", "accuracy"],  // optional: see /cluster/run for details
  "method": "single_model"  // optional: "single_model" | "side_by_side"
}
```

**Score Handling:**
This endpoint accepts the same score formats as `/cluster/run`:
- **Nested dict format**: `score: {helpfulness: 0.8}`
- **Flattened with auto-detection**: `helpfulness: 0.8, accuracy: 0.9`
- **Flattened with explicit columns**: Use `score_columns` parameter
- **Side-by-side format**: Use `_a` and `_b` suffixes with `method: "side_by_side"`

See the [/cluster/run Score Handling](#score-handling) section for detailed examples.

**Response:**
```json
{
  "clusters": [
    {
      "id": 0,
      "label": "Cluster 0: Detailed explanations",
      "size": 35,  // Updated size after filtering
      "property_descriptions": [...],
      "property_ids": [...],
      "question_ids": [...],
      "meta": {
        "proportion_overall": 0.14,  // Updated proportion
        "quality_helpfulness": 0.83,  // ✅ Recomputed quality metrics
        "quality_accuracy": 0.91
      }
    }
  ],
  "total_conversations_by_model": {
    "gpt-4": 120,  // After filtering
    "claude-3": 130
  },
  "total_unique_conversations": 250
}
```

**What it means:**
- Recalculates cluster metrics without re-running clustering
- Useful for filtering by model, time period, or specific properties
- Much faster than full clustering
- **included_property_ids**: Only compute metrics for these properties (filters subset)

---

## DataFrame Operations

### POST `/df/select`

Filter data by including/excluding values.

**Request:**
```json
{
  "rows": [...],
  "include": {
    "model": ["gpt-4", "claude-3"],
    "behavior_type": ["positive"]
  },
  "exclude": {
    "category": ["Errors"]
  }
}
```

**Response:**
```json
{
  "rows": [...]  // Filtered data
}
```

**What it means:**
- **include**: Keep only rows where column matches these values (OR within column, AND across columns)
- **exclude**: Remove rows where column matches these values

---

### POST `/df/groupby/preview`

Preview aggregated statistics grouped by a column.

**Request:**
```json
{
  "rows": [...],
  "by": "model",
  "numeric_cols": ["helpfulness", "accuracy"]  // optional
}
```

**Response:**
```json
{
  "groups": [
    {
      "value": "gpt-4",
      "count": 150,
      "means": {
        "helpfulness": 0.82,
        "accuracy": 0.91
      }
    },
    {
      "value": "claude-3",
      "count": 150,
      "means": {
        "helpfulness": 0.79,
        "accuracy": 0.89
      }
    }
  ]
}
```

**What it means:**
- Groups data by column value
- Calculates count and mean for numeric columns
- Useful for model comparison tables

---

### POST `/df/groupby/rows`

Get individual rows for a specific group value (with pagination).

**Request:**
```json
{
  "rows": [...],
  "by": "model",
  "value": "gpt-4",
  "page": 1,
  "page_size": 10
}
```

**Response:**
```json
{
  "total": 150,
  "rows": [...]  // Page of 10 rows
}
```

---

### POST `/df/custom`

Execute custom pandas expression on data.

**Request:**
```json
{
  "rows": [...],
  "code": "df[df['helpfulness'] > 0.8].sort_values('accuracy', ascending=False)"
}
```

**Response:**
```json
{
  "rows": [...]
}
```

**What it means:**
- Allows arbitrary pandas operations
- Security: Executes in sandboxed environment
- Variable `df` is the DataFrame

---

## Configuration

### GET `/embedding-models`

Get list of available embedding models for clustering.

**Response:**
```json
{
  "models": [
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-large",
    "bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2"
  ]
}
```

---

### GET `/prompts`

Get available system prompts for property extraction.

**Response:**
```json
{
  "prompts": [
    {
      "name": "default",
      "label": "Default",
      "has_task_description": true,
      "default_task_description_single": "Identify notable properties...",
      "default_task_description_sbs": "Compare the two responses...",
      "preview": "You are an expert AI analyst..."
    },
    {
      "name": "agent",
      "label": "Agent",
      "has_task_description": true,
      "default_task_description_single": "...",
      "default_task_description_sbs": "...",
      "preview": "You are an AI agent evaluator..."
    }
  ]
}
```

**What it means:**
- **name**: Identifier to use in extraction requests
- **has_task_description**: Whether this prompt accepts custom task descriptions
- **default_task_description_***: Default task text for this prompt type

---

### GET `/prompt-text`

Get full text of a specific prompt.

**Query Parameters:**
- `name`: Prompt name ("default" or "agent")
- `task_description`: Optional custom task description
- `method`: "single_model" | "side_by_side"

**Response:**
```json
{
  "name": "default",
  "text": "You are an expert AI analyst tasked with identifying notable properties in model responses. Focus on: [task_description]"
}
```

---

### POST `/label/prompt`

Get the system prompt for fixed-taxonomy labeling.

Returns the exact system prompt that will be used by the label() function when labeling with a given taxonomy. This allows users to preview the prompt before running the labeling pipeline.

**Request Body:**
```json
{
  "taxonomy": {
    "refusal": "Does the model refuse to answer the user's request?",
    "helpful_response": "Does the model provide a helpful, appropriate response?",
    "hallucination": "Does the model make up false information?"
  }
}
```

**Response:**
```json
{
  "text": "You are an expert model behavior analyst. Your task is to meticulously analyze the trace of a large language model to identify whether it contains any of the following behaviors:\n\n- **refusal**: Does the model refuse to answer the user's request?\n- **helpful_response**: Does the model provide a helpful, appropriate response?\n- **hallucination**: Does the model make up false information?\n\nIf the trace contains any of the behaviors, return a list of objects with the following structure..."
}
```

**What it means:**
- **taxonomy**: Dictionary mapping label names to their descriptions
- **text**: The complete system prompt that will be used for labeling, with taxonomy labels filled in

**Use case:** Display the prompt in the frontend Advanced Settings accordion so users can see exactly what instructions the LLM will receive before running labeling.

---

## Metrics Endpoints

### GET `/metrics/summary/{results_dir}`

Get summary of available metrics for a results directory.

**Response:**
```json
{
  "source": "jsonl",
  "models": 2,
  "clusters": 15,
  "total_battles": 450,
  "has_confidence_intervals": true,
  "quality_metric_names": ["helpfulness", "accuracy", "coherence"]
}
```

---

### GET `/metrics/model-cluster/{results_dir}`

Get detailed model-cluster metrics.

**Response:**
```json
{
  "source": "jsonl",
  "models": ["gpt-4", "claude-3"],
  "data": [
    {
      "model": "gpt-4",
      "cluster": "Cluster 0: Detailed explanations",
      "proportion": 0.35,
      "proportion_delta": 0.05,
      "quality_helpfulness": 0.82,
      "quality_helpfulness_delta": 0.03
    }
  ]
}
```

---

### GET `/metrics/benchmark/{results_dir}`

Get benchmark metrics (overall model scores).

**Response:**
```json
{
  "source": "jsonl",
  "models": ["gpt-4", "claude-3"],
  "data": [
    {
      "model": "gpt-4",
      "size": 500,
      "quality_helpfulness": 0.79,
      "quality_accuracy": 0.88
    }
  ]
}
```

---

### GET `/metrics/quality-metrics/{results_dir}`

Get list of available quality metrics.

**Response:**
```json
{
  "quality_metrics": ["helpfulness", "accuracy", "coherence"]
}
```

---

## Common Data Formats

### Conversation (Operational Row)

Single-model format:
```json
{
  "question_id": "q1",
  "prompt": "What is AI?",
  "model": "gpt-4",
  "model_response": "AI is...",
  "score": {"helpfulness": 0.8}
}
```

Side-by-side format:
```json
{
  "question_id": "q1",
  "prompt": "What is AI?",
  "model_a": "gpt-4",
  "model_b": "claude-3",
  "model_a_response": "AI is...",
  "model_b_response": "Artificial intelligence...",
  "score": {"winner": "model_a"}
}
```

### Property

```json
{
  "id": "p1",
  "question_id": "q1",
  "model": "gpt-4",
  "property_description": "Provides structured definition with examples",
  "category": "Response Style",
  "reason": "The response organizes information hierarchically",
  "evidence": "Uses bullet points and sub-categories",
  "behavior_type": "Style",
  "contains_errors": false,
  "unexpected_behavior": false,
  "raw_response": "..."  // Original LLM output
}
```

### Cluster

```json
{
  "id": 0,
  "label": "Cluster 0: Detailed explanations",
  "size": 45,
  "property_descriptions": ["...", "..."],
  "property_ids": ["p1", "p5", "p12"],
  "question_ids": ["q1", "q3", "q8"],
  "meta": {
    "proportion_overall": 0.15,
    "helpfulness": 0.82,
    "accuracy": 0.91
  }
}
```

---

## Error Responses

All endpoints return standard error format:

```json
{
  "detail": "Error message or object with details"
}
```

Common status codes:
- **400**: Bad request (invalid parameters)
- **404**: Resource not found
- **422**: Validation error (missing required fields, wrong format)
- **500**: Internal server error

Example validation error:
```json
{
  "detail": {
    "error": "Missing required columns for single_model",
    "missing": ["model", "model_response"],
    "available": ["question_id", "prompt", "score"]
  }
}
```

---

## Environment Variables

**General Configuration:**
- `BASE_BROWSE_DIR`: Base directory for file browsing (defaults to CWD)
- `STRINGSIGHT_DEBUG`: Enable debug logging ("1", "true", or "True")
- `WANDB_DISABLED`: Disable W&B logging ("true")

**Caching Configuration:**
- `STRINGSIGHT_DISABLE_CACHE`: Disable all caching ("1", "true", or "True")
- `STRINGSIGHT_DISABLE_EMBEDDING_CACHE`: Disable only embedding caching ("1", "true", or "True")
- `STRINGSIGHT_CACHE_DIR`: Cache directory path (default: `.cache/stringsight`)
- `STRINGSIGHT_CACHE_MAX_SIZE`: Max cache size (default: `"50GB"`)

---

## Notes

1. **File paths**: All server-side file paths must be within `BASE_BROWSE_DIR` for security
2. **Caching**: JSONL metrics files are cached for 15 minutes to improve performance
3. **Streaming**: Use streaming endpoints for datasets with >10k rows
4. **Async jobs**: Use job endpoints for extraction batches with >1k conversations
5. **CORS**: Configured for development (allows all origins)
6. **Score formats**: The `/cluster/run` and `/cluster/metrics` endpoints accept scores in multiple formats:
   - **Nested dict** (preferred): `score: {"helpfulness": 0.8, "accuracy": 0.9}`
   - **Flattened with auto-detection**: `helpfulness: 0.8, accuracy: 0.9` (no parameters needed)
   - **Flattened with explicit columns**: Add `score_columns: ["helpfulness", "accuracy"]`
   - **Side-by-side**: Use `_a` and `_b` suffixes with `method: "side_by_side"`

   Auto-detection works for columns containing: `score`, `rating`, `quality`, `helpfulness`, `accuracy`, `correctness`, `fluency`, `coherence`, `relevance`

---

## Troubleshooting

### Quality Metrics Not Appearing

If your clustering results don't include `quality_*` fields in the metrics:

1. **Check your score data format:**
   - Open browser DevTools → Network tab
   - Find the `/cluster/run` request
   - Check `operationalRows[0]` in the request payload
   - Verify scores are present (either as `score: {...}` or as separate columns like `helpfulness`, `accuracy`)

2. **Check backend logs:**
   - Look for: `"Converting score columns to dict format: [...]"`
   - If not present, scores may not be detected
   - Look for: `"Auto-detected potential score columns: [...]"`
   - This shows which columns were automatically detected

3. **Try explicit `score_columns`:**
   ```json
   {
     "operationalRows": [...],
     "properties": [...],
     "params": {...},
     "score_columns": ["helpfulness", "accuracy"],  // Add this
     "method": "single_model"  // Add this
   }
   ```

4. **Verify score values are numeric:**
   - Scores must be numbers, not strings
   - Example: `helpfulness: 0.8` ✅, not `helpfulness: "0.8"` ❌

5. **Check for NaN/null values:**
   - If all scores are null/NaN, no quality metrics will be computed
   - Verify your original data has valid score values

### Common Issues

**Issue**: "Score columns contain non-numeric values"
- **Solution**: Ensure all score columns contain numbers, not strings or other data types

**Issue**: Scores are in `score` column but no quality metrics
- **Solution**: Check if `score` is a dict (`{helpfulness: 0.8}`) vs a string (`"0.8"`)

**Issue**: Side-by-side scores not working
- **Solution**: Use `_a` and `_b` suffixes: `helpfulness_a`, `helpfulness_b`, and set `method: "side_by_side"`

**Issue**: Auto-detection picking up wrong columns
- **Solution**: Use explicit `score_columns` parameter to specify exactly which columns are scores
