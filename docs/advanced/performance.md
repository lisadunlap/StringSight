# Performance Tuning

Optimize StringSight for speed, cost, and quality based on your requirements.

## Quick Wins

### Use Cheaper Models

```python
from stringsight import explain

# Cost-effective configuration
clustered_df, model_stats = explain(
    df,
    model_name="gpt-4.1-mini",              
    embedding_model="all-MiniLM-L6-v2",     # Free local model
    min_cluster_size=15,                     # Smaller clusters = more clusters
    use_wandb=False                          # Disable W&B logging (default True)
)
```

### Use Local Embeddings

```python
# Local sentence-transformers (free, no API calls)
clustered_df, model_stats = explain(
    df,
    embedding_model="all-MiniLM-L6-v2",  # Fast, good quality
    # or
    embedding_model="all-mpnet-base-v2"   # Higher quality, slower
)
```

### Sample Large Datasets

```python
# Analyze subset for initial exploration
from stringsight.dataprep import sample_prompts_evenly

df_sample = sample_prompts_evenly(
    df,
    sample_size=1000,  # Sample 1000 prompts
    method="single_model",
    random_state=42
)

clustered_df, model_stats = explain(df_sample)
```

## Model Selection Trade-offs

| Model | Cost (per 1M tokens) | Speed | Quality | Best For |
|-------|----------------------|-------|---------|----------|
| `gpt-4.1` | $3.50 input / $14.00 output | Slow | Excellent | Production |
| `gpt-4.1-mini` | $0.70 / $2.80 | Medium | Very Good | Balanced |
| `gpt-4.1-mini` | $0.60 / $1.80 | Fast | Good | Development |
| `gpt-4.1-nano` | $0.20 / $0.80 | Very Fast | Decent | Large-scale |

### Embedding Models

| Model | Cost | Speed | Quality |
|-------|------|-------|---------|
| `text-embedding-3-large` | $0.13/1M | Medium | Excellent |
| `text-embedding-3-large` | $0.02/1M | Fast | Very Good |
| `all-MiniLM-L6-v2` | Free | Very Fast | Good |
| `all-mpnet-base-v2` | Free | Medium | Very Good |

## Clustering Optimization

### Adjust Cluster Size

```python
# Larger clusters = faster, fewer clusters
clustered_df, model_stats = explain(
    df,
    min_cluster_size=50  # vs default 30
)
```

### Disable Dimensionality Reduction

```python
from stringsight.clusterers import HDBSCANClusterer

clusterer = HDBSCANClusterer(
    disable_dim_reduction=True,  # Skip dimensionality reduction
    min_cluster_size=30
)
```

## Parallelization

### Increase Workers

```python
# More parallel API calls (if rate limits allow)
clustered_df, model_stats = explain(
    df,
    max_workers=32  # vs default 16
)
```

### Batch Processing

```python
# Process large datasets in batches
import pandas as pd

batch_size = 1000
results = []

for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    result, _ = explain(batch, output_dir=f"results/batch_{i}")
    results.append(result)

# Combine results
final_df = pd.concat(results, ignore_index=True)
```

## Caching

```python
# Cache expensive operations
clustered_df, model_stats = explain(
    df,
    extraction_cache_dir=".cache/extraction",
    clustering_cache_dir=".cache/clustering",
    metrics_cache_dir=".cache/metrics"
)
```

## Memory Management

### For Large Datasets

```python
# Reduce memory usage
clustered_df, model_stats = explain(
    df,
    include_embeddings=False,  # Don't include embeddings in output
    min_cluster_size=50,        # Fewer clusters
    use_wandb=False             # Reduce logging overhead (default True)
)
```

### Chunk Processing

```python
# Process in chunks to avoid OOM
for chunk in pd.read_csv("large_file.csv", chunksize=5000):
    result, _ = explain(chunk, output_dir="results/chunk")
```

## Benchmarks

Typical performance on common hardware:

| Dataset Size | GPT-4.1 | gpt-4.1-mini | Local Embeddings | Total Time |
|--------------|---------|-------------|------------------|------------|
| 100 convs | 2 min | 1 min | 10 sec | ~3 min |
| 1,000 convs | 15 min | 8 min | 30 sec | ~16 min |
| 10,000 convs | 2.5 hours | 1.3 hours | 5 min | ~2.6 hours |

*Benchmarks on M1 Mac with 32GB RAM, 16 parallel workers*

## Next Steps

- **[Custom Pipelines](custom-pipelines.md)** - Build optimized pipelines
- **[Data Formats](../user-guide/data-formats.md)** - Optimize data loading
