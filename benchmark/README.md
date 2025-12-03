# Benchmark Creation System

## Purpose

This benchmark system creates synthetic datasets with **known ground truth behaviors** to rigorously evaluate behavior extraction methods like StringSight. The core idea: if we deliberately induce specific behaviors in an LLM and then run our behavior extraction tools on the outputs, we can measure how well the tools recover the behaviors we know are present.

## What We're Doing

We generate controlled test data through a 4-stage pipeline:

1. **Behavior Generation**: Use an LLM (GPT-4) to generate diverse, task-specific behavioral patterns tailored to a dataset. These behaviors are designed to be:
   - **Diverse**: Cover different aspects (style, errors, reasoning, safety)
   - **Observable**: Clear, obvious differences in the text of responses
   - **Consistent**: Manifest consistently (not probabilistically)
   - **Realistic**: Behaviors actual LLMs might exhibit
   - **Task-specific**: Informed by actual examples from the target dataset

2. **System Prompt Engineering**: For each behavior, create a complete system prompt that reliably induces that behavior in the model (starts with base prompt, then naturally integrates behavioral instructions)

3. **Response Generation**: Generate responses to all dataset prompts using each behavior-modified system prompt, plus a baseline with no modifications

4. **Result Storage**: Save all responses in structured JSONL format with metadata (behavior name, description, category, system prompts, etc.)

## Our Goals

1. **Quantitative Evaluation**: Create ground truth data to measure precision, recall, and F1 scores for behavior extraction methods

2. **Systematic Testing**: Enable rigorous testing of StringSight's ability to discover and cluster behavioral patterns

3. **Ablation Studies**: Support experiments on different parameters (sample size, cluster size, embedding models, etc.)

4. **Development Validation**: Provide immediate feedback when developing new behavior extraction features

5. **Benchmark Sharing**: Create reusable datasets for comparing different behavior extraction approaches

## How Behaviors Are Generated

The system uses an LLM to generate behaviors tailored to the specific dataset by:
- Analyzing the dataset description and task type
- Sampling random examples from the actual dataset to understand task characteristics
- Generating behaviors that maximize diversity across different categories
- Creating complete system prompts that consistently induce each behavior

**Note**: The LLM generates the entire system prompt for each behavior (not just a modifier appended to the base prompt). This approach produces more natural, cohesive prompts that models adhere to better, avoiding the stilted language that can result from simple concatenation.

### Behavior Categories

Each generated behavior falls into one of these categories:

- **Baseline**: Control conditions with no behavioral modifications (2 automatically added)
  - `baseline_no_modification`: Original system prompt
  - `baseline_paraphrase`: Paraphrased prompt ("You are an assistant who is helpful") - should be functionally identical
- **Stylistic**: How the model expresses itself (tone, verbosity, formality, structure)
- **Error**: Mistakes in following instructions, misunderstanding requirements, violating constraints
- **Reasoning**: How the model approaches the task, what it prioritizes, interpretation patterns
- **Safety**: Overly cautious behaviors, refusals, disclaimers, hedging

The default configuration generates 10 diverse behaviors plus 2 baselines (12 total).

## Inputs and Outputs

### Inputs

**Dataset Description YAML**: Describes the dataset and task
- `name`: Dataset name
- `description`: Task description
- `dataset_path`: Path to JSONL or Parquet file
- `dataset_type`: Type of task (e.g., "qa", "agentic", "conversation")
- `prompt_column`: Column name containing prompts
- `few_shot_examples`: Example prompts to help the LLM understand the task

Example: `input_dataset_descriptions/instructeval.yaml`

**Dataset File**: JSONL or Parquet with prompts
- Must contain the specified `prompt_column`
- Can contain additional metadata columns
- Will be sampled if `sample_size` is specified

### Outputs

**JSONL Files**: One file per behavior containing all responses

Structure: `benchmark/results/{dataset_name}/{behavior_name}.jsonl`

Each line contains:
```json
{
  "prompt": "user prompt",
  "response": "model response with induced behavior",
  "behavior": "behavior_name",
  "behavior_description": "what this behavior does",
  "category": "stylistic|error|reasoning|safety|baseline",
  "model": "gpt-4.1",
  "base_system_prompt": "You are a helpful assistant.",
  "system_prompt": "complete system prompt used for this behavior",
  "metadata": {"row_index": 0}
}
```

This format enables direct comparison with StringSight's output for evaluation.

## Directory Structure

```
benchmark/
├── create_benchmark.py           # Main benchmark generation script
├── test_behavior_generation.py  # Test script for behavior generation only
├── view_benchmark.py            # Gradio app for viewing results
├── input_dataset_descriptions/  # Dataset description YAML files
│   └── instructeval.yaml
└── results/                     # Generated benchmark results
    └── {dataset_name}/
        └── {behavior_name}.jsonl
```

## Quick Start

### End-to-End Workflow

The typical workflow for creating and using a benchmark:

1. **Prepare**: Create a dataset description YAML with task info and examples
2. **Test**: Run `test_behavior_generation.py` to verify generated behaviors
3. **Generate**: Run `create_benchmark.py` to generate benchmark responses (start small!)
4. **View**: Use `view_benchmark.py` to inspect the results
5. **Evaluate**: Run StringSight on the benchmark and compare to ground truth

### Example: Creating a Benchmark for InstructEval

```bash
# Step 1: Test behavior generation (cheap, fast)
python benchmark/test_behavior_generation.py

# Step 2: Generate small sample (10 prompts x 12 behaviors = 120 responses)
python benchmark/create_benchmark.py \
    --dataset-desc input_dataset_descriptions/instructeval.yaml \
    --sample-size 10

# Step 3: View results in browser
python benchmark/view_benchmark.py

# Step 4: If satisfied, generate full benchmark with validation (1000 prompts x 12 behaviors = 12,000 responses + validation)
python benchmark/create_benchmark.py \
    --dataset-desc input_dataset_descriptions/instructeval.yaml \
    --sample-size 1000 \
    --enable-validation
```

## Usage

### 1. Create Dataset Description

Create a YAML file in `input_dataset_descriptions/`:

```yaml
name: "dataset_name"
description: "Detailed description of the dataset and task"
dataset_path: "path/to/dataset.jsonl"
dataset_type: "qa"  # or "agentic", "conversation"
prompt_column: "prompt"

few_shot_examples:
  - "Example prompt 1"
  - "Example prompt 2"
  - "Example prompt 3"

# Optional: for agentic datasets
tools: ["tool1", "tool2"]
```

### 2. Test Behavior Generation

Before generating full results, test the behavior generation:

```bash
python benchmark/test_behavior_generation.py
```

This will:
- Generate 10 task-specific behaviors tailored to the dataset
- Display behavior descriptions and system prompt modifiers
- Save behaviors to `benchmark/generated_behaviors.json`

Review the generated behaviors to ensure quality before proceeding.

### 3. Generate Benchmark with Small Sample

Test the full pipeline with a small sample:

```bash
python benchmark/create_benchmark.py \
    --dataset-desc input_dataset_descriptions/instructeval.yaml \
    --sample-size 10 \
    --behavior-model gpt-4.1 \
    --response-model gpt-4.1 \
    --base-system-prompt "You are a helpful assistant." \
    --output-dir benchmark/results/
```

### 4. Generate Full Benchmark

Once satisfied with the small sample, generate the full benchmark:

```bash
python benchmark/create_benchmark.py \
    --dataset-desc input_dataset_descriptions/instructeval.yaml \
    --sample-size 1000 \
    --output-dir benchmark/results/
```

### 5. View Results

Launch the Gradio app to view and compare results:

```bash
python benchmark/view_benchmark.py --results-dir benchmark/results/
```

The app provides six tabs:
- **All Behaviors**: View all behavior responses for the same prompt side-by-side
- **System Prompts**: View base system prompt and modifiers for each behavior
- **Behavior Overview**: View behavior descriptions and metadata
- **Browse Examples**: Browse individual examples for each behavior
- **Compare Behaviors**: Side-by-side comparison of responses from two behaviors
- **Validation Results**: View validation metrics and co-occurrence analysis (if validation enabled)

## Validation: Ensuring Behaviors Are Detectable

The validation feature checks whether the induced behaviors are actually detectable by an LLM judge. This helps ensure your benchmark is high-quality.

### What Validation Does

1. **Multi-Label Classification**: For each response, an LLM (default: gpt-4.1-mini) identifies ALL behaviors present (not just one)
2. **Confusion Matrix**: Calculate precision, recall, F1 for each behavior
3. **Co-Occurrence Analysis**: Identify which behaviors frequently appear together
4. **Purity Metrics**: Show which behaviors are "pure" (detected alone) vs "mixed" (detected with others)

**Note**: Validation uses gpt-4.1-mini by default for cost efficiency. You can change this with `--validation-model` if you want higher-quality validation.

### When to Use Validation

**Use validation when:**
- Testing if your behaviors are distinguishable
- Debugging why behaviors might not be working
- Before generating a large expensive benchmark
- Creating a benchmark for publication/sharing

**Skip validation when:**
- Doing quick iterations on behavior generation
- API costs are a concern (doubles the number of LLM calls)
- You've already validated a similar set of behaviors

### Output Format with Validation

When validation is enabled, each response includes:
```json
{
  "validation_predictions": [
    {
      "behavior": "ignores_constraints",
      "confidence": "high",
      "evidence": "Uses commas despite 'no commas' instruction"
    }
  ],
  "validation_correct": true,
  "validation_strict_match": false,
  "validation_extra_behaviors": ["overly_verbose"]
}
```

### Interpreting Validation Results

**Good behaviors:**
- High recall (>80%): Behavior is detected when present
- High precision (>80%): When detected, usually correct
- Low co-occurrence: Behaviors are distinct

**Problematic behaviors:**
- Low recall (<60%): Behavior is too subtle or not reliably induced
- Low precision (<60%): Behavior is detected when not present (too broad)
- High co-occurrence (>30%): Two behaviors overlap, consider merging or making more distinct

## Configuration Options

### Command Line Arguments

```bash
--dataset-desc PATH           # Path to dataset description YAML
--behavior-model MODEL        # Model for generating behaviors (default: gpt-4.1)
--response-model MODEL        # Model for generating responses (default: gpt-4.1)
--sample-size N               # Number of prompts to sample (default: None = all)
--num-behaviors N             # Number of task-specific behaviors to generate (default: 10)
--num-fewshot N               # Number of random examples for behavior generation (default: 10)
--base-system-prompt TEXT     # Base system prompt (default: "You are a helpful assistant.")
--output-dir PATH             # Output directory (default: benchmark/results/)
--enable-validation           # Enable behavior validation (adds extra LLM calls)
--validation-model MODEL      # Model for validation (default: gpt-4.1-mini)
--validation-max-workers N    # Parallel workers for validation (default: 10)
```

### BenchmarkConfig Options

```python
@dataclass
class BenchmarkConfig:
    behavior_generation_model: str = "gpt-4.1"
    response_generation_model: str = "gpt-4.1-mini"
    sample_size: Optional[int] = None
    num_behaviors: int = 10
    num_fewshot_examples: int = 10
    output_dir: str = "benchmark/results/"
    dataset_description_path: str = "input_dataset_descriptions/instructeval.yaml"
    random_seed: int = 42
    base_system_prompt: str = "You are a helpful assistant."
    enable_validation: bool = False
    validation_model: str = "gpt-4.1-mini"
    validation_max_workers: int = 10
```

## Output Format

Results are saved as JSONL files with the following schema:

```json
{
  "prompt": "Original prompt from dataset",
  "response": "Generated response with behavior",
  "behavior": "behavior_name",
  "behavior_description": "Description of the behavior",
  "category": "stylistic|error|reasoning|safety|baseline",
  "model": "gpt-4.1",
  "base_system_prompt": "You are a helpful assistant.",
  "system_prompt": "Complete system prompt used for this behavior",
  "metadata": {
    "row_index": 0
  }
}
```

## Example: InstructEval Benchmark

InstructEval is a comprehensive instruction-following benchmark with diverse formatting constraints (e.g., "no commas", "all lowercase", "JSON output"). This makes it ideal for testing behavior extraction methods.

### What We're Testing

For InstructEval, the generated behaviors might include:
- **Stylistic**: `overly_verbose_responses` - adds unnecessary elaboration that dilutes the main content
- **Error**: `ignores_formatting_constraints` - consistently violates specified format requirements (commas when told not to, wrong case, etc.)
- **Reasoning**: `literal_interpretation` - interprets instructions overly literally, missing the intent
- **Safety**: `overly_cautious_refusals` - refuses benign creative writing requests due to misinterpreting safety guidelines

These behaviors are task-specific because they're informed by actual InstructEval prompts.

## Development Workflow

### Incremental Testing Strategy

Always start small and scale up to avoid wasting API costs on poor quality behaviors:

1. **Stage 1: Validate Behaviors** (~$0.01, 30 seconds)
   ```bash
   python benchmark/test_behavior_generation.py
   ```
   Review the 10 generated behaviors. Are they diverse? Observable? Task-specific?

2. **Stage 2: Test Full Pipeline** (~$1, 5 minutes)
   ```bash
   python benchmark/create_benchmark.py --sample-size 10
   ```
   10 prompts × 12 behaviors = 120 responses. View in the Gradio app to verify quality.

3. **Stage 3: Medium Scale Test** (~$10, 30 minutes)
   ```bash
   python benchmark/create_benchmark.py --sample-size 100
   ```
   100 prompts × 12 behaviors = 1,200 responses. Enough to test evaluation pipeline.

4. **Stage 4: Full Benchmark** (~$100, 3 hours)
   ```bash
   python benchmark/create_benchmark.py --sample-size 1000
   ```
   1,000 prompts × 12 behaviors = 12,000 responses. Production benchmark.

### Cost Estimation

Approximate API costs (as of October 2025, using GPT-4.1):
- **Behavior generation**: ~$0.01 per dataset (1 call with few-shot examples)
- **Response generation**: ~$0.01 per response (depends on prompt/response length)

For a dataset with 1000 prompts and 12 behaviors (10 + 2 baselines):
- Total responses: 12,000
- Estimated cost: ~$100-150

Start with Stage 1-2 ($1 total) to validate before committing to full generation.

## Extending the System

### Adding New Dataset Types

For agentic datasets, add tool information to the YAML:

```yaml
dataset_type: "agentic"
tools: ["search", "calculator", "code_executor"]
```

The behavior generation will automatically consider tool-specific behaviors.

### Customizing Behavior Generation

Modify the prompt in `generate_behaviors()` to:
- Add domain-specific behavior categories
- Include additional constraints
- Provide example behaviors

## Evaluating StringSight with the Benchmark

### The Evaluation Question

**Can StringSight automatically discover the behaviors we deliberately induced?**

Since we know exactly what behaviors are present in the benchmark data (we created them!), we can measure:
- **Recall**: What percentage of our ground truth behaviors does StringSight discover?
- **Precision**: What percentage of StringSight's discovered clusters correspond to real behaviors?
- **F1 Score**: The harmonic mean of precision and recall

This provides a quantitative measure of StringSight's effectiveness.

### Evaluation Pipeline

The evaluation consists of 5 steps:

1. **Input Preparation**: Convert benchmark results to StringSight's side-by-side comparison format
2. **StringSight Execution**: Run StringSight on the benchmark data to discover behaviors
3. **Output Parsing**: Extract discovered behavioral clusters and their descriptions
4. **LLM-as-Judge Matching**: Use an LLM to determine which discovered behaviors match ground truth behaviors
5. **Metrics Calculation**: Compute precision, recall, F1 for behavior recovery

### Step 1: Input Preparation

**Benchmark Output Format:**
```
benchmark/results/
└── instructeval/
    ├── baseline_no_modification.jsonl
    ├── behavior_1.jsonl
    ├── behavior_2.jsonl
    └── ...
```

Each JSONL file contains:
```json
{
  "prompt": "user prompt",
  "response": "model response",
  "behavior": "behavior_name",
  "behavior_description": "ground truth description",
  "category": "stylistic|error|reasoning|safety|baseline"
}
```

**StringSight Input Format:**

StringSight expects side-by-side comparison format:
```python
import pandas as pd

# Load benchmark results
baseline_df = pd.read_json("benchmark/results/instructeval/baseline_no_modification.jsonl", lines=True)
behavior_df = pd.read_json("benchmark/results/instructeval/behavior_1.jsonl", lines=True)

# Convert to StringSight format
stringsight_input = pd.DataFrame({
    "question_id": range(len(baseline_df)),
    "model_a": "baseline",
    "model_b": "behavior_1",
    "model_a_response": baseline_df["response"],
    "model_b_response": behavior_df["response"],
    # Optional: no winner for this analysis
})

stringsight_input.to_parquet("stringsight_input.parquet")
```

### Step 2: Run StringSight

```python
from stringsight import explain

# Run StringSight on a subset (e.g., 100 samples)
clustered_df, model_stats = explain(
    stringsight_input.head(100),
    method="side_by_side",
    min_cluster_size=5,
    embedding_model="text-embedding-3-large",
    hierarchical=True,
    output_dir="stringsight_results/"
)
```

### Step 3: Parse StringSight Output

**StringSight Output Format:**
```
stringsight_results/
├── clustered_results.parquet    # Main output with cluster assignments
├── model_stats.json             # Per-model statistics
└── clusters.json                # Cluster descriptions
```

**Extract Discovered Behaviors:**
```python
# Load StringSight results
clustered_df = pd.read_parquet("stringsight_results/clustered_results.parquet")
with open("stringsight_results/clusters.json") as f:
    clusters = json.load(f)

# Get significant clusters (frequency-based filtering)
cluster_counts = clustered_df["cluster_id"].value_counts()
significant_clusters = cluster_counts[cluster_counts >= 5].index.tolist()

# Extract discovered behavior descriptions
discovered_behaviors = []
for cluster_id in significant_clusters:
    cluster_info = next((c for c in clusters if c["id"] == cluster_id), None)
    if cluster_info:
        discovered_behaviors.append({
            "cluster_id": cluster_id,
            "description": cluster_info["label"],
            "count": cluster_counts[cluster_id],
            "properties": cluster_info.get("properties", [])
        })
```

### Step 4: LLM-as-Judge Evaluation

**Ground Truth Behaviors:**
```python
# Load ground truth from original benchmark file
ground_truth = {
    "behavior_name": "ignores_length_constraint",
    "description": "Consistently exceeds specified word or character length...",
    "category": "error"
}
```

**LLM-as-Judge Prompt Structure:**
```python
judge_prompt = f"""
You are evaluating whether an automatically discovered behavior matches a ground truth behavior.

Ground Truth Behavior:
- Name: {ground_truth["behavior_name"]}
- Description: {ground_truth["description"]}
- Category: {ground_truth["category"]}

Discovered Behavior:
- Cluster ID: {discovered["cluster_id"]}
- Description: {discovered["description"]}
- Frequency: {discovered["count"]} examples

Task: Determine if the discovered behavior captures the same underlying pattern as the ground truth behavior.

Consider:
1. Does the discovered description match the core concept?
2. Are the behavioral manifestations similar?
3. Would examples of the ground truth behavior be clustered here?

Respond with:
- "match": The behaviors are the same or very similar
- "partial": The behaviors overlap but differ in scope/specifics
- "no_match": The behaviors are unrelated

Provide reasoning for your decision.

Format your response as JSON:
{{
  "verdict": "match|partial|no_match",
  "confidence": 0.0-1.0,
  "reasoning": "explanation"
}}
"""

# Run LLM judge (pseudo-code)
from stringsight.core.llm_utils import single_completion
result = single_completion(judge_prompt, model="gpt-4.1")
```

### Step 5: Metrics Calculation

**Matching Algorithm:**
```python
def evaluate_behavior_recovery(ground_truth_behaviors, discovered_behaviors, judge_results):
    """
    Calculate precision, recall, F1 for behavior recovery.

    Args:
        ground_truth_behaviors: List of known behaviors from benchmark
        discovered_behaviors: List of behaviors found by StringSight
        judge_results: LLM verdicts for each (ground_truth, discovered) pair

    Returns:
        Dict with precision, recall, F1 scores
    """

    # True Positives: Ground truth behaviors that were recovered
    # - "match" or "partial" (with confidence > threshold)

    # False Positives: Discovered behaviors with no ground truth match

    # False Negatives: Ground truth behaviors not discovered

    # Metrics:
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    pass  # Implementation TODO
```

### Complete Evaluation Script

A complete evaluation script is available at `benchmark/evaluate_stringsight.py` that:

1. Loads benchmark results from `all_behaviors.jsonl`
2. Runs StringSight to discover behavioral clusters
3. Extracts significant clusters from StringSight output
4. Runs LLM-as-judge for each ground truth vs discovered cluster pair
5. Computes precision, recall, F1 scores
6. Generates detailed evaluation report

**Quick Test:**
```bash
# Test on small subset (recommended first run)
python benchmark/test_evaluation.py
```

**Full Usage:**
```bash
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --subset-size 100 \
    --min-cluster-size 5 \
    --extraction-model gpt-4.1-mini \
    --judge-model gpt-4.1 \
    --embedding-model text-embedding-3-large \
    --match-threshold 0.7 \
    --hierarchical \
    --output-dir evaluation_results/
```

**Parameters:**
- `--benchmark-results`: Path to `all_behaviors.jsonl` from benchmark creation
- `--subset-size`: Number of prompts to sample (None = use all). Samples N prompts and includes ALL behavior responses for those prompts. E.g., `--subset-size 10` with 12 behaviors = 120 total responses.
- `--min-cluster-size`: Minimum cluster size for StringSight (default: 5)
- `--extraction-model`: Model for property extraction (default: gpt-4.1-mini)
- `--judge-model`: Model for LLM-as-judge (default: gpt-4.1)
- `--embedding-model`: Embedding model for clustering (default: text-embedding-3-large)
- `--match-threshold`: Confidence threshold for partial matches (default: 0.7)
- `--hierarchical`: Enable hierarchical clustering
- `--output-dir`: Directory for evaluation results

**Output Files:**
The evaluation creates:
- `evaluation_metrics.json`: Precision, recall, F1 scores
- `match_results.json`: Detailed match results for each ground truth ↔ discovered pair
- `summary.txt`: Human-readable summary
- `stringsight_output/`: Full StringSight output (clusters, properties, etc.)

### Evaluation Metrics

**Primary Metrics:**
- **Behavior Recovery Rate**: % of ground truth behaviors discovered
- **Precision**: % of discovered behaviors that match ground truth
- **Recall**: % of ground truth behaviors successfully recovered
- **F1 Score**: Harmonic mean of precision and recall

**Secondary Metrics:**
- **Cluster Purity**: % of examples in a cluster with the same ground truth label
- **Cluster Completeness**: % of ground truth examples clustered together
- **Discovery Threshold**: Minimum frequency needed to discover a behavior

### Next Steps

To implement the full evaluation pipeline:

1. Create `benchmark/evaluate_stringsight.py` with the scaffold above
2. Implement the LLM-as-judge matching logic
3. Add matching threshold tuning (confidence cutoffs for "partial" matches)
4. Create visualization tools for evaluation results
5. Run ablation studies on cluster size, sample size, embedding models

## Troubleshooting

### No behaviors generated
- Check OpenAI API key: `export OPENAI_API_KEY="your-key"`
- Verify dataset description format
- Check behavior_generation_model has JSON output support

### Response generation fails
- Check API rate limits
- Reduce sample_size for testing
- Verify prompt column exists in dataset

### Gradio app shows no results
- Ensure results directory exists and contains data
- Check file permissions
- Verify JSONL format is correct

## Next Steps

### Immediate TODOs

1. **Generate First Benchmark** ✓ (DONE)
   - Run `test_behavior_generation.py` for InstructEval
   - Review and iterate on behavior quality
   - Generate small sample (10 prompts)
   - View results in Gradio app

2. **Build Evaluation Pipeline** ✓ (DONE)
   - Create `evaluate_stringsight.py` script
   - Implement benchmark → StringSight format conversion
   - Add LLM-as-judge matching logic
   - Calculate precision/recall/F1 metrics

3. **Validate the System** ← NEXT STEP
   - Run `test_evaluation.py` on small sample
   - Run StringSight on generated benchmark
   - Review: do discovered clusters match ground truth behaviors?
   - Tune parameters (cluster size, sample size, etc.)

4. **Scale Up**
   - Generate larger benchmarks (100, 1000 prompts)
   - Test on additional datasets beyond InstructEval
   - Run systematic ablation studies

### Future Enhancements

- Support for multi-turn conversation datasets
- Automated behavior quality scoring
- Integration with StringSight's eval pipeline
- Public benchmark sharing and leaderboards
   