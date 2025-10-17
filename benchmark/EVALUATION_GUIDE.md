# StringSight Evaluation Pipeline Guide

## Overview

This guide walks you through evaluating StringSight's ability to discover behavioral patterns using the benchmark system.

## Quick Start

### 1. Generate Benchmark Data (if not already done)

```bash
# Small test benchmark (10 prompts x 12 behaviors = 120 responses)
python benchmark/create_benchmark.py \
    --dataset-desc benchmark/input_dataset_descriptions/instructeval.yaml \
    --sample-size 10 \
    --output-dir benchmark/results/

# This creates benchmark/results/instructeval/all_behaviors.jsonl
```

### 2. Run Evaluation Test

```bash
# Quick test on small subset (~50 responses)
python benchmark/test_evaluation.py
```

**What this does:**
1. Loads 50 responses from the benchmark (ground truth behaviors)
2. Runs StringSight to discover behavioral clusters
3. Uses LLM-as-judge to match discovered clusters to ground truth
4. Calculates precision, recall, and F1 scores
5. Saves results to `benchmark/evaluation_results/test_run/`

**Expected output:**
```
STRINGSIGHT BENCHMARK EVALUATION
================================================================================

Loading benchmark data from benchmark/results/instructeval/all_behaviors.jsonl
Loaded 120 total responses
Behaviors in benchmark: 12
Total unique prompts: 10
Sampled 5 prompts × 12 behaviors = 60 total responses

Ground Truth Behaviors (12):
  - baseline_no_modification (baseline): 4 examples
  - baseline_paraphrase (baseline): 4 examples
  - behavior_1 (stylistic): 4 examples
  ...

RUNNING STRINGSIGHT
================================================================================
[StringSight extracts properties and clusters them]

StringSight discovered 8 fine-grained clusters

EXTRACTING DISCOVERED CLUSTERS
================================================================================
Found 8 significant clusters (>= 3 examples)
  - Cluster 1: 15 properties
  - Cluster 2: 10 properties
  ...

EVALUATING BEHAVIOR RECOVERY
================================================================================
Running LLM-as-judge for 12 x 8 = 96 comparisons
[Progress bar showing LLM-as-judge comparisons]

EVALUATION COMPLETE
================================================================================

Precision: 75.0%
Recall: 66.7%
F1 Score: 70.6%

Ground Truth Behaviors: 12
Discovered Clusters: 8
Successfully Recovered: 8/12
```

### 3. Review Results

Check the output directory:
```bash
ls benchmark/evaluation_results/test_run/

evaluation_metrics.json  # Precision/recall/F1 scores
match_results.json       # Detailed matching results
summary.txt              # Human-readable summary
stringsight_output/      # Full StringSight results
```

**evaluation_metrics.json:**
```json
{
  "true_positives": 8,
  "false_positives": 0,
  "false_negatives": 4,
  "precision": 1.0,
  "recall": 0.667,
  "f1_score": 0.8,
  "ground_truth_count": 12,
  "discovered_count": 8,
  "match_confidence_threshold": 0.7
}
```

**match_results.json:** Contains detailed match information for each ground truth ↔ discovered cluster pair.

## Full Evaluation Workflow

### Step 1: Generate Full Benchmark

```bash
# Generate benchmark with 100-1000 prompts
python benchmark/create_benchmark.py \
    --dataset-desc benchmark/input_dataset_descriptions/instructeval.yaml \
    --sample-size 100 \
    --behavior-model gpt-4.1 \
    --response-model gpt-4.1-mini \
    --output-dir benchmark/results/
```

### Step 2: Run Full Evaluation

```bash
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --subset-size 100 \
    --min-cluster-size 5 \
    --extraction-model gpt-4.1-mini \
    --judge-model gpt-4.1 \
    --embedding-model text-embedding-3-small \
    --hierarchical \
    --output-dir benchmark/evaluation_results/full_run/
```

### Step 3: Analyze Results

Review the summary:
```bash
cat benchmark/evaluation_results/full_run/summary.txt
```

Examine specific matches:
```python
import json

# Load match results
with open('benchmark/evaluation_results/full_run/match_results.json') as f:
    matches = json.load(f)

# Find strong matches
strong_matches = [m for m in matches if m['verdict'] == 'match' and m['confidence'] > 0.8]
print(f"Strong matches: {len(strong_matches)}")

# Find missed behaviors (no good match found)
from collections import defaultdict
by_gt = defaultdict(list)
for m in matches:
    by_gt[m['ground_truth_behavior']].append(m)

for gt_name, gt_matches in by_gt.items():
    best_match = max(gt_matches, key=lambda x: x['confidence'])
    if best_match['verdict'] == 'no_match':
        print(f"MISSED: {gt_name}")
        print(f"  Best attempt: {best_match['discovered_cluster_label']}")
        print(f"  Reasoning: {best_match['reasoning']}\n")
```

## Configuration Options

### Benchmark Generation

**Models:**
- `--behavior-model`: Model for generating behaviors (default: gpt-4.1)
- `--response-model`: Model for generating responses (default: gpt-4.1-mini)

**Data:**
- `--sample-size`: Number of prompts to sample (None = all)
- `--num-behaviors`: Number of behaviors to generate (default: 10)
- `--num-fewshot`: Few-shot examples for behavior generation (default: 10)

**Validation:**
- `--enable-validation`: Run validation during benchmark creation
- `--validation-model`: Model for validation (default: gpt-4.1-mini)

### Evaluation

**StringSight Parameters:**
- `--min-cluster-size`: Minimum cluster size (default: 5)
  - Lower = more fine-grained clusters, higher recall
  - Higher = fewer, larger clusters, higher precision
- `--embedding-model`: Embedding model (default: text-embedding-3-small)
  - Options: text-embedding-3-small, text-embedding-3-large, all-MiniLM-L6-v2
- `--hierarchical`: Enable hierarchical clustering (recommended)

**Matching Parameters:**
- `--match-threshold`: Confidence threshold for partial matches (default: 0.7)
  - Lower = more lenient matching, higher recall
  - Higher = stricter matching, higher precision
- `--judge-model`: LLM for evaluation (default: gpt-4.1)
  - Options: gpt-4.1 (best), gpt-4.1-mini (cheaper), gpt-4o-mini

**Sampling:**
- `--subset-size`: Number of prompts to sample (None = all)
  - Samples N prompts and includes ALL behavior responses for those prompts
  - Example: `--subset-size 10` with 12 behaviors = 10×12=120 total responses
  - Start with 5-10 prompts for quick testing
  - Use 100+ prompts for production evaluation

## Understanding Metrics

### Precision
**Definition:** Of the clusters StringSight discovered, what percentage match ground truth behaviors?

**Formula:** `TP / (TP + FP)`

**High Precision (>80%):** StringSight is conservative, mostly finds real behaviors
**Low Precision (<60%):** StringSight is discovering spurious patterns

### Recall
**Definition:** Of the ground truth behaviors in the benchmark, what percentage did StringSight discover?

**Formula:** `TP / (TP + FN)`

**High Recall (>80%):** StringSight is finding most behaviors
**Low Recall (<60%):** StringSight is missing many behaviors

### F1 Score
**Definition:** Harmonic mean of precision and recall

**Formula:** `2 * (Precision * Recall) / (Precision + Recall)`

**Target:** F1 > 0.7 indicates good overall performance

## Troubleshooting

### Low Recall (Missing Behaviors)

**Possible causes:**
1. **Cluster size too large**: Try `--min-cluster-size 3`
2. **Not enough examples**: Increase `--subset-size`
3. **Behaviors too similar**: Review ground truth behaviors for overlap
4. **Extraction missed properties**: Check StringSight extraction quality

**Solutions:**
```bash
# Try lower cluster threshold
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --min-cluster-size 3 \
    --subset-size 200

# Try different embedding model
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --embedding-model text-embedding-3-large
```

### Low Precision (Spurious Clusters)

**Possible causes:**
1. **Cluster size too small**: Try `--min-cluster-size 10`
2. **Behaviors not distinct enough**: Regenerate benchmark with more diverse behaviors
3. **Extraction too noisy**: Try better extraction model

**Solutions:**
```bash
# Try higher cluster threshold
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --min-cluster-size 10

# Try better extraction model
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --extraction-model gpt-4.1
```

### Match Results Look Wrong

**Check the matching logic:**
```python
import json

# Load and inspect matches
with open('benchmark/evaluation_results/full_run/match_results.json') as f:
    matches = json.load(f)

# Find questionable matches
for m in matches:
    if m['verdict'] == 'match' and m['confidence'] < 0.6:
        print(f"Low-confidence match:")
        print(f"  GT: {m['ground_truth_behavior']}")
        print(f"  Discovered: {m['discovered_cluster_label']}")
        print(f"  Reasoning: {m['reasoning']}\n")
```

**Adjust match threshold:**
```bash
# Be more strict about matches
python benchmark/evaluate_stringsight.py \
    --benchmark-results benchmark/results/instructeval/all_behaviors.jsonl \
    --match-threshold 0.85
```

## Cost Estimation

**Benchmark Generation (100 prompts, 12 behaviors = 1200 responses):**
- Behavior generation: ~$0.01
- Response generation: ~$10-15 (depending on response length)
- **Total: ~$10-15**

**Evaluation (100 responses, 12 GT behaviors, 8 discovered clusters):**
- StringSight extraction: ~$5-10 (100 prompts)
- LLM-as-judge: ~$2-5 (12 x 8 = 96 comparisons)
- **Total: ~$7-15**

**Tips for cost savings:**
- Start with small subsets (`--subset-size 50`)
- Use cheaper models: `--extraction-model gpt-4.1-mini --judge-model gpt-4.1-mini`
- Cache is enabled by default - rerunning doesn't cost extra

## Advanced Usage

### Batch Evaluation Across Parameters

```python
import subprocess
import json

# Test different cluster sizes
for min_size in [3, 5, 10, 15]:
    output_dir = f"benchmark/evaluation_results/cluster_{min_size}/"
    subprocess.run([
        "python", "benchmark/evaluate_stringsight.py",
        "--benchmark-results", "benchmark/results/instructeval/all_behaviors.jsonl",
        "--min-cluster-size", str(min_size),
        "--output-dir", output_dir
    ])
    
    # Load metrics
    with open(f"{output_dir}/evaluation_metrics.json") as f:
        metrics = json.load(f)
    
    print(f"Min Cluster Size {min_size}:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1_score']:.3f}\n")
```

### Custom Analysis

```python
from evaluate_stringsight import (
    load_benchmark_data,
    run_stringsight,
    extract_discovered_clusters,
    evaluate_behavior_recovery,
    EvaluationConfig
)

# Load data
config = EvaluationConfig(
    benchmark_results_path="benchmark/results/instructeval/all_behaviors.jsonl",
    subset_size=100
)
df, ground_truth = load_benchmark_data(config.benchmark_results_path, config.subset_size)

# Run StringSight
clustered_df, model_stats = run_stringsight(df, config)

# Custom analysis
print(f"Total properties extracted: {len(clustered_df)}")
print(f"Properties per response: {len(clustered_df) / len(df):.2f}")
print(f"Unique clusters: {clustered_df['property_description_cluster_label'].nunique()}")
```

## Next Steps

After running the evaluation:

1. **Analyze the results**: Which behaviors were found? Which were missed?
2. **Tune parameters**: Try different cluster sizes, embedding models, etc.
3. **Iterate on benchmark**: Improve behavior diversity if needed
4. **Scale up**: Run on larger datasets (1000+ prompts)
5. **Try other datasets**: Test on different task types beyond InstructEval

