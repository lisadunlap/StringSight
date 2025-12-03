# Run All Models Scripts

Two scripts are provided to run `evaluate_stringsight.py` on all individual model files in aci_bench and instructeval benchmarks.

## Files

- `run_all_models.sh` - Bash script version
- `run_all_models.py` - Python script version (recommended for better error handling)

Both scripts automatically:
- Find all `.jsonl` files in `benchmark/results/aci_bench` and `benchmark/results/instructeval`
- Skip baseline files (`baseline_*.jsonl`) and combined files (`all_behaviors.jsonl`)
- Run evaluation on each individual model file
- Save results to `benchmark/evaluation_results/{benchmark}/{model_name}/`

## Usage

### Python Script (Recommended)

Basic usage with defaults:
```bash
python benchmark/run_all_models.py
```

With custom parameters:
```bash
python benchmark/run_all_models.py \
    --subset-size 50 \
    --min-cluster-size 5 \
    --top-k 10 \
    --hierarchical \
    --log-to-wandb
```

Dry run (see what would be executed without running):
```bash
python benchmark/run_all_models.py --dry-run
```

Run on specific benchmarks only:
```bash
python benchmark/run_all_models.py --benchmarks aci_bench
```

Disable wandb logging:
```bash
python benchmark/run_all_models.py --no-wandb
```

### Bash Script

Basic usage:
```bash
./benchmark/run_all_models.sh
```

To customize parameters, edit the configuration variables at the top of the script:
- `SUBSET_SIZE` - Number of prompts to sample (empty = all)
- `MIN_CLUSTER_SIZE` - Minimum cluster size
- `TOP_K` - Number of top behaviors to evaluate
- `HIERARCHICAL` - Enable/disable hierarchical clustering (default: disabled, set to "--hierarchical" to enable)
- `WANDB_FLAG` - Enable/disable wandb logging (default: enabled)

## Available Parameters

### Python Script Options

```
--results-dir          Base directory with benchmark results (default: benchmark/results)
--output-dir          Base output directory (default: benchmark/evaluation_results)
--benchmarks          List of benchmarks to process (default: aci_bench instructeval)

--subset-size         Number of prompts to sample (None = all)
--min-cluster-size    Minimum cluster size (default: 5)
--embedding-model     Embedding model (default: text-embedding-3-large)
--extraction-model    Property extraction model (default: gpt-4.1-mini)
--judge-model         LLM-as-judge model (default: gpt-4.1)
--top-k              Top K behaviors to evaluate (default: 10)
--hierarchical       Enable hierarchical clustering (default: False)
--no-hierarchical    Disable hierarchical clustering
--log-to-wandb       Enable wandb logging (default: True)
--no-wandb           Disable wandb logging

--dry-run            Print commands without executing
```

## Examples

### Example 1: Quick test on small subset
```bash
python benchmark/run_all_models.py \
    --subset-size 10 \
    --top-k 5 \
    --no-wandb
```

### Example 2: Full evaluation with wandb logging
```bash
python benchmark/run_all_models.py \
    --top-k 10 \
    --log-to-wandb
```

### Example 3: Run only on aci_bench
```bash
python benchmark/run_all_models.py \
    --benchmarks aci_bench \
    --top-k 15
```

### Example 4: Enable hierarchical clustering
```bash
python benchmark/run_all_models.py \
    --hierarchical \
    --top-k 10
```

## Output

Results for each model will be saved to:
```
benchmark/evaluation_results/{benchmark_name}/{model_name}/
├── all_scores.json           # All GT x Cluster match scores
├── evaluation_metrics.json   # Summary metrics
└── summary.txt              # Human-readable report
```

Plus StringSight's standard outputs in the same directory.

## Notes

- The scripts skip files containing "baseline" or "all_behaviors" in their names
- Each model file is processed independently with its own StringSight run
- Progress is printed to console with success/failure tracking
- If a run fails, the script continues with the next model
- Use Ctrl+C to interrupt the batch run at any time
