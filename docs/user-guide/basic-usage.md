# Explain and Label Functions

Learn how to use the two main functions in StringSight for analyzing model behavior.

## Core Functions

StringSight provides two primary functions:

- **`explain()`**: Discovers behavioral patterns through clustering 
- **`label()`**: Classifies behavior using predefined taxonomies

Both functions analyze conversation data and return clustered results with model statistics.

## The `explain()` Function

The `explain()` function automatically discovers behavioral patterns in model responses through property extraction and clustering.

### Basic Usage

```python
import pandas as pd
from stringsight import explain

# Load your conversation data
df = pd.read_csv("model_conversations.csv")

# Single model analysis: Understand what behavioral patterns a model exhibits
clustered_df, model_stats = explain(
    df,
    method="single_model", 
    min_cluster_size=10,        # Minimum conversations per behavior cluster
    output_dir="results/"       # Saves all analysis files here
)
# This will: 1) Extract behavioral properties from each response
#          2) Group similar behaviors into clusters  
#          3) Calculate performance metrics per cluster
#          4) Save comprehensive results

# Side-by-side comparison: Compare two models to find behavioral differences  
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=30,        # Larger datasets need bigger clusters
    output_dir="results/"
)
# This will: 1) Find behavioral differences between model pairs
#          2) Cluster similar difference patterns
#          3) Show which model excels at which behaviors
#          4) Provide statistical significance testing
```

### Parameters

**Core Parameters:**
- `df`: Input DataFrame with conversation data
- `method`: `"side_by_side"` or `"single_model"`
- `system_prompt`: Custom prompt for property extraction (optional)
- `output_dir`: Directory to save results

**Extraction Parameters:**
- `model_name`: LLM for property extraction (default: `"gpt-4o"`) - This model analyzes responses to find behavioral patterns
- `temperature`: Temperature for LLM calls (default: `0.7`) - Higher values = more creative property extraction
- `max_workers`: Parallel workers for API calls (default: `16`) - Speed up analysis with concurrent requests

**Clustering Parameters:**  
- `clusterer`: Clustering method (`"hdbscan"`) - Algorithm to group similar behaviors
- `min_cluster_size`: Minimum cluster size (default: `30`) - Smaller = more granular clusters, larger = broader patterns
- `embedding_model`: `"openai"` or sentence-transformer model - How to convert properties to vectors for clustering

### Examples

**Custom System Prompt:**
```python
# Define what behavioral aspects you want the LLM to focus on
custom_prompt = """
Analyze this conversation and identify behavioral differences.
Focus on: reasoning approach, factual accuracy, response style.
Return a JSON object with 'property_description' and 'property_evidence'.
"""

clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    system_prompt=custom_prompt  # This overrides the default extraction prompt
)
# The LLM will now focus specifically on reasoning, accuracy, and style
# instead of using the general-purpose default prompt
```

## The `label()` Function

The `label()` function classifies model behavior using a predefined taxonomy rather than discovering patterns.

### Basic Usage

```python
from stringsight import label

# Define your evaluation taxonomy
taxonomy = {
    "accuracy": "Is the response factually correct?",
    "helpfulness": "Does the response address the user's needs?", 
    "clarity": "Is the response clear and well-structured?",
    "safety": "Does the response avoid harmful content?"
}

# Classify responses
clustered_df, model_stats = label(
    df,
    taxonomy=taxonomy,
    model_name="gpt-4o-mini",
    output_dir="results/"
)
```

### Parameters

**Core Parameters:**
- `df`: Input DataFrame (must be single-model format)
- `taxonomy`: Dictionary mapping labels to descriptions
- `model_name`: LLM for classification (default: `"gpt-4o-mini"`)
- `output_dir`: Directory to save results

**Other Parameters:**
- `temperature`: Temperature for classification (default: `0.0`)
- `max_workers`: Parallel workers (default: `8`)
- `verbose`: Print progress information (default: `True`)

### Example

**Quality Assessment:**
```python
quality_taxonomy = {
    "excellent": "Response is comprehensive, accurate, and well-structured",
    "good": "Response is mostly accurate with minor issues",
    "fair": "Response has some accuracy or clarity problems", 
    "poor": "Response has significant issues or inaccuracies"
}

clustered_df, model_stats = label(
    df,
    taxonomy=quality_taxonomy,
    temperature=0.0,  # Deterministic classification
    output_dir="quality_results/"
)
```

## Data Formats

### Side-by-side Format (for comparing two models)

**Required columns:**
- `prompt` - The question or prompt given to both models
- `model_a`, `model_b` - Names of the models being compared  
- `model_a_response`, `model_b_response` - Complete responses from each model

**Optional columns:**
- `score` - Dictionary with winner and metrics

```python
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing"],
    "model_a": ["gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3"],
    "model_a_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "ML is a subset of AI..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "Quantum computing uses..."}]
    ],
    "model_b_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning involves..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "QC leverages quantum..."}]
    ],
    "score": [{"winner": "gpt-4", "helpfulness": 4.2}, {"winner": "claude-3", "helpfulness": 3.8}]
})
```

### Single Model Format (for analyzing individual models)

**Required columns:**
- `prompt` - The question given to the model (used for visualization)
- `model` - Name of the model being analyzed
- `model_response` - The model's complete response

**Optional columns:**  
- `score` - Dictionary of evaluation metrics

```python
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing"],
    "model": ["gpt-4", "gpt-4"],
    "model_response": [
        [{"role": "user", "content": "What is machine learning?"},
         {"role": "assistant", "content": "Machine learning involves..."}],
        [{"role": "user", "content": "Explain quantum computing"},
         {"role": "assistant", "content": "QC leverages quantum..."}]
    ],
    "score": [{"accuracy": 1, "helpfulness": 4.2}, {"accuracy": 0, "helpfulness": 3.8}]
})
```

## Response Format Details

StringSight supports flexible response formats to accommodate various data sources and conversation structures.

**Recommended**: Use OpenAI conversation format for all model responses. This preserves conversation structure, supports multimodal inputs, and enables better trace visualization.

### Automatic Format Detection

The system automatically detects and converts response formats:

1. **OpenAI conversation format** (list of message dictionaries) → used as-is (recommended)
2. **Simple string responses** → automatically converted to OpenAI conversation format
3. **Other types** → converted to strings then processed

### OpenAI Conversation Format Specification

The response format follows the standard OpenAI conversation format. Each message dictionary contains:

**Required Fields:**
- `role`: Message sender role (`"user"`, `"assistant"`, `"system"`, `"tool"`)
- `content`: Message content (string or dictionary - see below)

**Optional Fields:**
- `name`: Name of the model/tool (persists for entire conversation)
- `id`: Unique identifier for specific model or tool call
- Additional custom fields are preserved

**Content Field:**
For simple text responses, `content` is a string:
```python
{"role": "assistant", "content": "Machine learning involves training algorithms..."}
```

For multimodal inputs or complex interactions, `content` can be a dictionary following OpenAI's format:
- `text`: Text content
- `image`: Image content (for multimodal models)  
- `tool_calls`: Array of tool call objects (for tool-augmented responses)

### Format Examples

Here are some examples for chatbot conversations, agents, and multimodel models.

*Annoyed with having to convert to yet another data format?* Dude, me too — here are some alternative options:
* Vibe code that bad boy - its decently good at converting formats. One day I aspire to make this a built in feature so if you feel strongly please make a PR
* Make your conversation which is just 1 big string: This will work — you just won't get the nice trace visualization we have in the UI (but it should still localize text). 

**Simple text conversation:**
```python
[
    {
        "role": "user",
        "content": "What is machine learning?"
    },
    {
        "role": "assistant", 
        "content": "Machine learning involves training algorithms..."
    }
]
```

**Tool-augmented response:**
```python
[
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

**Multimodal input (when applicable):**
```python
[
    {
        "role": "user",
        "content": {
            "text": "What's in this image?",
            "image": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        }
    },
    {
        "role": "assistant",
        "content": "I can see a diagram showing neural network architecture..."
    }
]
```

**Format Conversion:**
Simple strings are automatically converted:
```python
# Input: "Machine learning involves..."
# Becomes: [{"role": "assistant", "content": "Machine learning involves..."}]
```

## Understanding Results

### Output DataFrames

Both functions return your original data enriched with extracted behavioral properties:

```python
print(clustered_df.columns)
# Original columns plus new analysis columns:
# 'property_description' - Natural language description of behavior (e.g., "Provides step-by-step reasoning")  
# 'property_evidence' - Evidence from the response supporting this property
# 'category' - Higher-level grouping (e.g., "Reasoning", "Creativity")
# 'impact' - Estimated effect ("positive", "negative", or numeric score)
# 'type' - Kind of property ("format", "content", "style")
# 'property_description_cluster_label' - Human-readable cluster name
```

### Model Statistics  

The `model_stats` contains per-model behavioral analysis:

```python
# For each model, you get statistics about behavioral patterns
for model_name, stats in model_stats.items():
    print(f"{model_name} behavioral analysis:")
    # Which behaviors this model exhibits most/least frequently
    # Relative scores for different behavioral clusters  
    # Example responses for each behavior cluster
    # Quality scores showing how well the model performs within each behavior type
```

### Saved Files

When `output_dir` is specified, both functions save:
- `clustered_results.parquet` - Complete results with clusters
- `model_stats.json` - Model performance statistics
- `full_dataset.json` - Complete dataset for reanalysis  
- `summary.txt` - Human-readable summary

## When to Use Each Function

**Use `explain()` when:**
- You want to discover unknown behavioral patterns
- You're comparing multiple models
- You need flexible, data-driven analysis
- You want to understand what makes models different

**Use `label()` when:**
- You have specific criteria to evaluate
- You need consistent scoring across datasets
- You're building evaluation pipelines  
- You want controlled, taxonomy-based analysis

## Next Steps

- Understand the [output files](configuration.md) in detail
- Explore [configuration options](configuration.md)
- Learn about the [pipeline architecture](../api/core.md) 