# Parameter Reference

This guide provides a comprehensive reference for all parameters available in StringSight's `explain()` and `label()` functions.

Parameters are organized by their role in the analysis pipeline.

## General Parameters

These parameters control data handling, analysis method, and output.

### `method`
- **Purpose**: Type of analysis to perform
- **Options**:
  - `"single_model"`: Extract patterns per trace, recommended if you only have responses from 1 model or if you are comparing 3+ models
  - `"side_by_side"`: Compare two models to find differences, recommended if you are comparing 2 models
- **Default**: `"single_model"`

### `sample_size`
- **Purpose**: Number of samples to process before analysis
- **Type**: `int` (optional)
- **Behavior**:
  - For single_model with balanced datasets (each prompt answered by all models): Samples prompts, keeping all model responses per prompt
  - Otherwise: Samples individual rows
- **Recommended**: Start with 50-100 for testing
- **Default**: None (uses all data)

### `score_columns`
- **Purpose**: Specify which columns contain evaluation metrics
- **Type**: `list[str]` (optional)
- **Format**:
  - Single model: `['accuracy', 'helpfulness']`
  - Side-by-side: `['accuracy_a', 'accuracy_b', 'helpfulness_a', 'helpfulness_b']`
- **Alternative**: Use a `score` dict column instead
- **Default**: None

## Column Mapping Parameters

If your dataframe uses different column names, you can map them using these parameters:

### `prompt_column`
- **Purpose**: Name of your prompt column
- **Type**: `str`
- **Default**: `"prompt"`
- **Example**: `prompt_column="input"`

### `model_column`
- **Purpose**: Name of your model column
- **Type**: `str`
- **Default**: `"model"`
- **Example**: `model_column="llm_name"`

### `model_response_column`
- **Purpose**: Name of your response column
- **Type**: `str`
- **Default**: `"model_response"`
- **Example**: `model_response_column="output"`

### `question_id_column`
- **Purpose**: Name of your question_id column
- **Type**: `str`
- **Default**: `"question_id"`
- **About `question_id`**:
  - Used to track which responses belong to the same prompt
  - Useful for side-by-side pairing: rows with the same prompt must have the same question_id
  - If not provided, StringSight will use `prompt` alone for pairing

**Example with custom column names:**
```python
clustered_df, model_stats = explain(
    df,
    prompt_column="input",           # Map "input" → "prompt"
    model_column="llm_name",         # Map "llm_name" → "model"
    model_response_column="output",  # Map "output" → "model_response"
    score_columns=["reward"]
)
```

## Property Extraction Parameters

These parameters control how behavioral properties are extracted from model responses.

### `model_name`
- **Purpose**: LLM used for extracting behavioral properties
- **Type**: `str`
- **Default**: `"gpt-4.1"`
- **Options**: Any OpenAI model name (`"gpt-4.1"`, `"gpt-4.1-mini"`, etc.)
- **Cost/Quality Tradeoff**:
  - `"gpt-4.1"`: Best quality, higher cost
  - `"gpt-4.1-mini"`: Good balance
  - `"gpt-3.5-turbo"`: Fastest, cheapest

### `system_prompt`
- **Purpose**: Prompt template used for extraction
- **Type**: `str`
- **Options**:
  - `"agent"`: Optimized for agent/conversational data
  - `"default"`: General-purpose extraction
- **Default**: `"agent"`

### `task_description`
- **Purpose**: Helps tailor extraction to your specific domain
- **Type**: `str` (optional)
- **Example**: `"airline booking agent conversations, look out for instances of the model violating policy, being tricked by the user, and any other additional issues or stylistic choices."`
- **Default**: Uses default prompt if not provided
- **Recommendation**: Provide a clear description of what you're analyzing and what behaviors to look for

## Clustering Parameters

These parameters control how properties are grouped into behavioral clusters.

### `min_cluster_size`
- **Purpose**: Minimum number of examples required per cluster
- **Type**: `int`
- **Default**: `5`
- **Effect**:
  - Higher values = fewer, more general clusters
  - Lower values = more, fine-grained clusters
- **Recommendation**:
  - Start with 5 for most analyses
  - Use 3-5 for side-by-side comparisons (smaller datasets)
  - Increase to 10+ for very large datasets

### `embedding_model`
- **Purpose**: Model used for embedding properties during clustering
- **Type**: `str`
- **Default**: `"text-embedding-3-large"`
- **Options**:
  - `"text-embedding-3-large"`: Fast, cost-effective
  - `"text-embedding-3-large"`: Higher quality embeddings
  - Any OpenAI embedding model

### `summary_model`
- **Purpose**: LLM used for generating cluster summaries
- **Type**: `str`
- **Default**: `"gpt-4.1"`
- **Recommendation**: Use a strong model for best summary quality

### `cluster_assignment_model`
- **Purpose**: LLM used to match outliers to clusters
- **Type**: `str`
- **Default**: `"gpt-4.1-mini"`
- **Note**: This makes many calls, so using a cheaper model is recommended
- **Recommendation**: `"gpt-4.1-mini"` or `"gpt-3.5-turbo"` for cost efficiency

## Side-by-Side Specific Parameters

For side-by-side comparison using tidy format (auto-pairing):

### `model_a`
- **Purpose**: Name of first model to compare
- **Type**: `str`
- **Required**: Only when using `method="side_by_side"` with tidy format
- **Example**: `model_a="gpt-4.1"`

### `model_b`
- **Purpose**: Name of second model to compare
- **Type**: `str`
- **Required**: Only when using `method="side_by_side"` with tidy format
- **Example**: `model_b="claude-sonnet-35"`

## Output Parameters

### `output_dir`
- **Purpose**: Directory to save results
- **Type**: `str` (optional)
- **Saves**:
  - `clustered_results.parquet`: Main dataframe with clusters
  - `full_dataset.json`: Complete PropertyDataset (JSON)
  - `full_dataset.parquet`: Complete PropertyDataset (Parquet)
  - `model_stats.json`: Model statistics
  - `summary.txt`: Human-readable summary
- **Default**: None (results not saved to disk)

### `verbose`
- **Purpose**: Whether to print progress messages
- **Type**: `bool`
- **Default**: `True`

### `use_wandb`
- **Purpose**: Whether to log to Weights & Biases
- **Type**: `bool`
- **Default**: `True`
- **Disable**: Set to `False` or set environment variable `WANDB_DISABLED=true`

### `wandb_project`
- **Purpose**: W&B project name for logging
- **Type**: `str` (optional)
- **Default**: `"stringsight"`

## Performance Parameters

### `max_workers`
- **Purpose**: Number of parallel workers for API calls
- **Type**: `int`
- **Default**: `10`
- **Recommendation**: Adjust based on your API rate limits

### `extraction_cache_dir`
- **Purpose**: Directory to cache extraction results to avoid re-running expensive API calls
- **Type**: `str` (optional)
- **Default**: None (no caching)
- **Recommendation**: Enable for iterative development

## Label-Specific Parameters

For the `label()` function:

### `taxonomy`
- **Purpose**: Predefined behavioral categories to detect
- **Type**: `dict[str, str]`
- **Required**: Yes for `label()`
- **Format**: `{"behavior_name": "description of what to detect"}`
- **Example**:
```python
taxonomy = {
    "tricked by the user": "Does the model behave unsafely due to user manipulation?",
    "reward hacking": "Does the model game the evaluation system?",
    "refusal": "Does the model refuse to follow the user's request due to policy constraints?",
    "tool calling": "Does the model call tools?"
}
```

## Cost Estimation

To estimate the number of LLM calls for a given analysis:

For `sample_size=100` with `min_cluster_size=3`:
- **100 calls** for property extraction (usually get 3-5 properties per trace with gpt-4.1)
- **~300-500 embedding calls** for each property
- **~(300-500) / min_cluster_size LLM calls** to generate cluster summaries
- **~50-100 outlier matching calls** (hence why we recommend using a smaller model)

Note: The larger your `min_cluster_size`, the more outliers you will likely have.

## Best Practices

### Starting Out
1. Start with `sample_size=50-100` for initial exploration
2. Use cheaper models first: `model_name="gpt-4.1-mini"`, `cluster_assignment_model="gpt-3.5-turbo"`
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

## Example Configurations

### Cost-Effective Analysis
```python
clustered_df, model_stats = explain(
    df,
    sample_size=50,
    model_name="gpt-4.1-mini",
    embedding_model="text-embedding-3-large",
    cluster_assignment_model="gpt-3.5-turbo",
    min_cluster_size=5,
    use_wandb=False
)
```

### High-Quality Analysis
```python
clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1",
    embedding_model="text-embedding-3-large",
    summary_model="gpt-4.1",
    min_cluster_size=10,
    use_wandb=True,
    wandb_project="production-analysis"
)
```

### Task-Specific Analysis
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
    model_name="gpt-4.1",
    min_cluster_size=5,
    output_dir="results/customer_support"
)
```
