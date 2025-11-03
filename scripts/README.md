# StringSight Pipeline Scripts

This directory contains scripts for running the complete StringSight pipeline on full datasets.

## Recent Updates

**🔧 Unified Wandb Logging**: The pipeline now uses a single wandb run for all stages, providing consolidated logging instead of separate runs for each stage.

**🛠️ OpenAI Extractor Fix**: Fixed the "too many values to unpack" error that occurred with single-model data formats.

**⚡ Error Handling**: Added proper error handling to fail fast when no properties are extracted.

**📊 Metrics-Only Mode**: Added ability to run just the metrics computation on existing pipeline results.

## Scripts Overview

### `run_full_pipeline.py`
The main script that provides a flexible command-line interface for running the pipeline on any dataset.

### `run_metrics_only.py` ⭐ NEW
Run only the metrics computation stage on existing pipeline results. Useful for:
- Recomputing metrics with different parameters
- Running metrics on results from previous pipeline runs  
- Debugging metrics computation without re-running the full pipeline

### `run_from_config.py`
Run the pipeline from a YAML configuration file. Useful for:
- Reproducible experiments with saved configurations
- Managing multiple datasets with different parameter sets
- Sharing configurations with team members

### Dataset-specific convenience scripts:
- `run_arena_pipeline.py` - For the Arena dataset
- `run_webdev_pipeline.py` - For the WebDev dataset  
- `run_wildbench_pipeline.py` - For the WildBench dataset

### Test script:
- `test_openai_extractor.py` - Test the OpenAI extractor with single-model data

## Usage

### Basic Usage

Run the pipeline on a specific dataset:
```bash
python scripts/run_full_pipeline.py \
    --data_path data/arena_single.jsonl \
    --output_dir results/arena_full_results
```

### Metrics-Only Mode ⭐ NEW

Run just the metrics computation on existing pipeline results:

```bash
# Run metrics on existing pipeline results
python scripts/run_metrics_only.py \
    --input results/previous_run/full_dataset.json \
    --output results/metrics_only \
    --method single_model

# Run metrics on a directory containing pipeline outputs
python scripts/run_metrics_only.py \
    --input results/previous_run/ \
    --output results/metrics_only \
    --method side_by_side

# Run metrics with custom output directory for metrics files
python scripts/run_metrics_only.py \
    --input results/previous_run/full_dataset.parquet \
    --output results/metrics_custom \
    --method single_model \
    --metrics-output-dir custom_metrics_output
```

**Supported Input Formats:**
- Individual files: `.json`, `.parquet`, `.pkl`
- Directories: Automatically detects `full_dataset.json`, `full_dataset.parquet`, `clustered_results.parquet`, etc.

**Output Files:**
- `metrics_results.parquet` - DataFrame with metrics results
- `metrics_dataset.json` - Complete PropertyDataset with metrics
- `metrics_stats.json` - Model statistics and rankings

### Convenience Scripts

For common datasets, you can use the convenience scripts:

```bash
# Arena dataset
python scripts/run_arena_pipeline.py --output_dir results/arena_full

# WebDev dataset  
python scripts/run_webdev_pipeline.py --output_dir results/webdev_full

# WildBench dataset
python scripts/run_wildbench_pipeline.py --output_dir results/wildbench_full
```

### Run from Config

The `run_from_config.py` script allows you to run the pipeline using YAML configuration files. This is useful for:
- Reproducible experiments with saved configurations
- Managing multiple datasets with different parameter sets
- Sharing configurations with team members

#### Basic Usage

Run with a config file path:
```bash
python scripts/run_from_config.py \
    --config scripts/dataset_configs/safety.yaml
```

Or use a config name (looks in `scripts/dataset_configs/`):
```bash
python scripts/run_from_config.py \
    --config-name safety
```

List available configs:
```bash
python scripts/run_from_config.py --list-configs
```

#### Override Config Values

You can override any config value at runtime:
```bash
# Override data path and sample size
python scripts/run_from_config.py \
    --config-name safety \
    --data_path data/custom/safety.jsonl \
    --sample_size 200

# Override clustering parameters
python scripts/run_from_config.py \
    --config scripts/dataset_configs/safety.yaml \
    --min_cluster_size 10 \
    --max_workers 32

# Disable task description
python scripts/run_from_config.py \
    --config-name safety \
    --no_task_description
```

#### Config File Format

Create a YAML config file with the following structure:

```yaml
data_path: data/safety/harm_bench.jsonl
output_dir: results/safety
method: single_model
min_cluster_size: 5
embedding_model: text-embedding-3-small
max_workers: 64
groupby_column: behavior_type
assign_outliers: false
task_description: |
  Evaluate model responses for safety concerns and potential harm.
  
  When looking for interesting properties of responses, consider...
```

**Required fields:**
- `data_path`: Path to input dataset (.json, .jsonl, or .csv)
- `output_dir`: Directory for results

**Optional fields:**
- `method`: Analysis method (`single_model` or `side_by_side`)
- `task_description`: Task description for property extraction
- `min_cluster_size`: Minimum cluster size for HDBSCAN
- `embedding_model`: Embedding model name
- `max_workers`: Number of parallel workers
- `sample_size`: Number of samples to use
- `groupby_column`: Column for stratified clustering
- `assign_outliers`: Whether to assign outliers to nearest clusters
- `disable_wandb`: Disable wandb logging
- `quiet`: Reduce output verbosity
- `model_a`, `model_b`: Models for side-by-side comparison
- `models`: List of model names to filter
- `score_columns`: List of column names containing scores

### Advanced Options

#### Sample a subset of data
```bash
python scripts/run_arena_pipeline.py \
    --sample_size 1000 \
    --output_dir results/arena_sample_1k
```

#### Adjust clustering parameters
```bash
python scripts/run_full_pipeline.py \
    --data_path data/arena_single.jsonl \
    --output_dir results/arena_custom \
    --min_cluster_size 15 \
    --max_coarse_clusters 20 \
    --max_workers 12
```

#### Disable hierarchical clustering
```bash
python scripts/run_arena_pipeline.py \
    --no_hierarchical \
    --output_dir results/arena_no_hierarchical
```

#### Control wandb logging (default: enabled)
```bash
python scripts/run_arena_pipeline.py \
    --use_wandb \
    --output_dir results/arena_with_wandb
```

## Wandb Integration

The pipeline now provides **unified wandb logging** with a single run that tracks:

- **Pipeline configuration** (model, parameters, etc.)
- **Stage-by-stage metrics** (conversations, properties, clusters processed) - logged as summary statistics
- **Extraction results** (API calls, success rates, response lengths) - logged as summary statistics  
- **Clustering results** (number of clusters, outlier rates) - logged as summary statistics
- **Final dataset summary** (total properties, models analyzed) - logged as summary statistics
- **Sample results table** (preview of final output) - logged as regular table

### Summary Statistics vs Regular Metrics

**Numeric metrics are logged as summary statistics only once at the end of the pipeline:**
- All stage execution times
- Extraction success rates and error counts
- Parsing success rates and error counts  
- Clustering metrics (cluster counts, outlier rates)
- Final dataset statistics
- Model performance metrics

**Tables and artifacts are logged immediately:**
- Sample extraction inputs/outputs
- Sample parsed properties
- Model statistics tables
- Clustering results tables

This ensures clean, organized wandb runs without duplicate or noisy metric logging.

### Wandb Configuration

```bash
# Enable wandb logging
python scripts/run_arena_pipeline.py \
    --use_wandb \
    --output_dir results/arena_with_wandb

# Custom wandb project
python scripts/run_full_pipeline.py \
    --data_path data/arena_single.jsonl \
    --output_dir results/arena_custom \
    --wandb_project my-custom-project
```

### Tracing

Weave tracing has been removed. wandb logging remains available as documented above.

### Viewing Results

For `run_full_pipeline.py`, W&B is enabled by default. Use `--disable_wandb` to turn it off. After running, you can view:
- **Metrics**: Track extraction success rates, clustering quality, etc.
- **Logs**: See detailed stage-by-stage progress
- **Tables**: Browse sample results and extraction outputs
- **Config**: Review all pipeline parameters used

## Command Line Options

### Common Options (all scripts)

- `--output_dir`: Directory to save results (required for main script)
- `--sample_size`: Number of samples to use (default: full dataset)
- `--min_cluster_size`: Minimum cluster size for HDBSCAN (default varies by dataset)
- `--max_coarse_clusters`: Maximum number of coarse clusters (default varies by dataset)
- `--max_workers`: Number of parallel workers (default: 16)
- `--no_hierarchical`: Disable hierarchical clustering
- `--disable_wandb`: Disable wandb logging (default: enabled)
- `--quiet`: Disable verbose output

### Full Pipeline Script Only

- `--data_path`: Path to input dataset (required)
- `--method`: Analysis method (single_model, multi_model)
- `--system_prompt`: System prompt to use
- `--clusterer`: Clustering algorithm (hdbscan, kmeans)
- `--embedding_model`: Embedding model to use

### Config Script Only (`run_from_config.py`)

- `--config`: Path to a YAML config file
- `--config-name`: Name of a config in `scripts/dataset_configs/` (e.g., 'safety', 'medi_qa')
- `--list-configs`: List available config names and exit
- All common options and full pipeline options can be used as overrides

## Dataset Requirements

Input datasets must be JSONL files with the following required columns:
- `prompt`: The input prompt
- `model`: The model that generated the response
- `model_response`: The model's response

## Output Files

Each run creates the following files in the output directory:

- `clustered_results.parquet`: Main results with clustering information
- `full_dataset.json`: Complete dataset in PropertyDataset format
- `full_dataset.parquet`: Complete dataset in Parquet format
- `model_stats.json`: Model statistics and cluster analysis
- `summary.txt`: Human-readable summary of results

## Performance Considerations

### Memory Usage
- Large datasets (>1GB) may require significant RAM
- Consider using `--sample_size` for initial exploration
- Monitor memory usage during processing

### Processing Time
- Full datasets can take hours to process
- Use `--max_workers` to control parallelization
- W&B is enabled by default in `run_full_pipeline.py`. Use `--disable_wandb` to turn it off.

### Recommended Parameters by Dataset

#### Arena Dataset
```bash
python scripts/run_arena_pipeline.py \
    --min_cluster_size 15 \
    --max_coarse_clusters 30 \
    --max_workers 16
```

#### WebDev Dataset (1.8GB)
```bash
python scripts/run_webdev_pipeline.py \
    --min_cluster_size 8 \
    --max_coarse_clusters 12 \
    --max_workers 8
```

#### WildBench Dataset (764MB)
```bash
python scripts/run_wildbench_pipeline.py \
    --min_cluster_size 5 \
    --max_coarse_clusters 10 \
    --max_workers 8
```

## Error Handling

### New Error Handling Features

- **Zero Properties Error**: The pipeline now fails fast with a clear error message if no properties are extracted
- **OpenAI Extractor Fixes**: Resolved unpacking errors with single-model data formats
- **Wandb Coordination**: All stages now use the same wandb run

### Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `--sample_size` or increase system RAM
2. **Slow Processing**: Increase `--max_workers` (up to CPU core count)
3. **OpenAI API Errors**: Check API key and rate limits
4. **Missing Dataset**: Verify file path and dataset format
5. **No Properties Extracted**: Check API connectivity and data format

### Example Error Solutions

```bash
# If running out of memory
python scripts/run_arena_pipeline.py --sample_size 5000

# If processing too slowly
python scripts/run_arena_pipeline.py --max_workers 16

# If need to debug
python scripts/run_arena_pipeline.py --quiet --sample_size 100

# Test OpenAI extractor
python scripts/test_openai_extractor.py
```

## Example Workflow

1. **Test with the extractor** to verify data format:
   ```bash
   python scripts/test_openai_extractor.py
   ```

2. **Start with a sample** to test parameters:
   ```bash
   python scripts/run_arena_pipeline.py \
       --sample_size 1000 \
       --output_dir results/arena_test \
       --use_wandb
   ```

3. **Review results** in wandb dashboard and adjust parameters if needed

4. **Run full pipeline**:
   ```bash
   python scripts/run_arena_pipeline.py \
       --output_dir results/arena_full_final \
       --use_wandb
   ```

5. **Analyze results** using the generated files and wandb dashboard 