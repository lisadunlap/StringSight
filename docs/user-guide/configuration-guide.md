# Configuration Guide

Complete guide to configuring StringSight's analysis pipeline for optimal results.

## Clustering Parameters

### min_cluster_size

**What it does:** Minimum number of properties required to form a cluster.

**How to choose:**

| Dataset Size | Recommended `min_cluster_size` | Rationale |
|--------------|-------------------------------|-----------|
| < 100 conversations | `5-10` | Small datasets need smaller clusters to find patterns |
| 100-1,000 conversations | `10-20` | Balanced granularity |
| 1,000-10,000 conversations | `20-50` | Larger clusters filter noise, find robust patterns |
| > 10,000 conversations | `50-100` | Very large datasets need substantial clusters |

**General rules:**
- **Start with dataset_size / 50** as a baseline
- **Smaller values** (5-10) = More granular, specific patterns (risk: noise/overfitting)
- **Larger values** (50-100) = Broader, more robust patterns (risk: missing nuances)
- **If you get too many clusters:** Increase `min_cluster_size`
- **If you get too few clusters:** Decrease `min_cluster_size`

!!! tip "Quick tips (concise)"
    - If clusters often repeat the same property, increase `min_cluster_size`.
    - By dataset size (samples):
        - < 100: `3-4`
        - 100–1,000: `5-7`
        - > 1,000: `15-30`

**Examples:**

```python
from stringsight import explain

# Small exploratory dataset (100 conversations)
explain(df, min_cluster_size=3)

# Medium production dataset (1,000 conversations)
explain(df, min_cluster_size=7)  # Default

# Large research dataset (10,000+ conversations)
explain(df, min_cluster_size=25)
```

<!-- Hierarchical clustering section removed to streamline guidance -->

### embedding_model

**What it does:** Converts property descriptions to vectors for clustering.

**Options:**

| Model | Cost | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `"text-embedding-3-large"` | $0.02/1M tokens | Fast | Very Good | **Default - best balance** |
| `"text-embedding-3-large"` | $0.13/1M tokens | Medium | Excellent | Production quality analysis |
| `"all-MiniLM-L6-v2"` | Free | Very Fast | Good | Development, large datasets |
| `"all-mpnet-base-v2"` | Free | Medium | Very Good | Cost-conscious production |

```python
# OpenAI embeddings (requires API key, costs $)
explain(df, embedding_model="text-embedding-3-large")  # Default

# Local embeddings (free, no API calls)
explain(df, embedding_model="all-MiniLM-L6-v2")
```

### assign_outliers

**What it does:** Assigns properties that don't fit any cluster to their nearest cluster.

**When to use:**
- ✅ You want every property in a cluster (no noise/outliers)
- ✅ Dashboards/visualizations (avoids "Outlier" cluster)
- ✅ Downstream analysis requires full coverage

**When to skip:**
- ❌ You want to identify truly unique/anomalous behaviors
- ❌ Quality matters more than coverage
- ❌ Small datasets (outliers are informative)

```python
# Assign all properties to clusters
explain(df, assign_outliers=True)

# Keep outliers separate
explain(df, assign_outliers=False)
```

## Extraction Parameters

### model_name

**What it does:** LLM used to extract behavioral properties from responses.

**Options:**

| Model | Cost/Quality | When to Use |
|-------|--------------|-------------|
| `"gpt-4.1"` | $$$ / Excellent | Production, research papers, high-stakes decisions |
| `"gpt-4.1-mini"` | $$ / Very Good | **Default - balanced cost/quality** |
| `"gpt-4.1-mini"` | $ / Good | Development, iteration, large-scale experiments |
| `"gpt-4.1-nano"` | ¢ / Decent | Massive datasets, proof-of-concepts |

```python
# High quality extraction
explain(df, model_name="gpt-4.1")

# Cost-effective extraction
explain(df, model_name="gpt-4.1-mini")
```

### temperature

**What it does:** Controls randomness in property extraction.

**Values:**
- `0.0-0.3` = Deterministic, focused extraction
- `0.5-0.7` = **Default - balanced creativity**
- `0.8-1.0` = More creative, diverse properties

```python
# Consistent, focused properties
explain(df, temperature=0.2)

# Diverse, creative properties
explain(df, temperature=0.9)
```

### max_workers

**What it does:** Number of parallel API calls for extraction.

**Guidelines:**
- Default: `16`
- **Increase** (32-64) if you have high API rate limits
- **Decrease** (4-8) if you hit rate limits or want to conserve resources
- **1** for debugging (sequential processing)

```python
# Fast parallel extraction
explain(df, max_workers=32)

# Conservative rate limiting
explain(df, max_workers=8)
```

## Model Selection Strategy

### Budget-Conscious Configuration

For cost-effective analysis without sacrificing too much quality:

```python
explain(
    df,
    model_name="gpt-4.1-mini",              # Cheap extraction
    embedding_model="all-MiniLM-L6-v2",    # Free embeddings
    min_cluster_size=50,                    # Fewer, larger clusters
    use_wandb=False                         # Turn off W&B (default True)
)
```

**Estimated cost:** ~$5-10 per 1,000 conversations

### Production-Quality Configuration

For high-quality, reproducible results:

```python
explain(
    df,
    model_name="gpt-4.1",                      # Best extraction
    embedding_model="text-embedding-3-large",   # Best embeddings
    min_cluster_size=30,                         # Balanced granularity
    use_wandb=True,                              # Track experiments (default True)
    wandb_project="production-analysis"
)
```

**Estimated cost:** ~$50-75 per 1,000 conversations

### Development/Iteration Configuration

For fast experimentation:

```python
explain(
    df,
    model_name="gpt-4.1-mini",            # Fast extraction
    embedding_model="all-MiniLM-L6-v2",   # Fast embeddings
    min_cluster_size=20,                   # Quick clustering
    max_workers=32,                        # Maximize parallelism
    use_wandb=False                        # Skip tracking (default is True)
)
```

**Estimated time:** ~5-10 minutes per 1,000 conversations

> Note: W&B logging is enabled by default. In the CLI (`scripts/run_full_pipeline.py`), pass `--disable_wandb` to turn it off.

## Advanced Parameters

### Dimensionality Reduction

Control PCA (or no dimensionality reduction) before clustering:

```python
from stringsight.clusterers import HDBSCANClusterer

clusterer = HDBSCANClusterer(
    disable_dim_reduction=True,              # Skip dimensionality reduction
    dim_reduction_method="pca",              # "pca", "adaptive", "none"
)
```

### HDBSCAN Tuning

Fine-tune clustering algorithm:

```python
clusterer = HDBSCANClusterer(
    min_cluster_size=30,
    min_samples=5,                           # Minimum samples in neighborhood
    cluster_selection_epsilon=0.0,           # Distance threshold
)
```

### Stratified Clustering

Cluster separately per group (e.g., per topic, per task):

```python
explain(df, groupby_column="topic")  # Cluster within each topic
```

## Common Configuration Issues

### "Too many small clusters"

**Problem:** Hundreds of tiny, noisy clusters

**Solution:**
```python
# Increase minimum cluster size
explain(df, min_cluster_size=50)  # was: 10

# Or assign outliers
explain(df, assign_outliers=True)
```

### "Only 2-3 clusters"

**Problem:** Not enough granularity

**Solution:**
```python
# Decrease minimum cluster size
explain(df, min_cluster_size=10)  # was: 50

# Use better embeddings
explain(df, embedding_model="text-embedding-3-large")

# Lower temperature for more diverse properties
explain(df, temperature=0.8)
```

### "Clustering too slow"

**Problem:** Takes hours to cluster

**Solution:**
```python
# Use local embeddings
explain(df, embedding_model="all-MiniLM-L6-v2")

# Increase cluster size
explain(df, min_cluster_size=100)
```

### "Running out of memory"

**Problem:** OOM errors during clustering

**Solution:**
```python
# Disable embeddings in output
explain(df, include_embeddings=False)

# Skip dimensionality reduction
explain(df, disable_dim_reduction=True)

# Process in batches (manually split data)
```

## Quick Reference

### By Dataset Size

```python
# < 100 conversations
explain(df, min_cluster_size=3)

# 100-1,000 conversations
explain(df, min_cluster_size=7)

# 1,000-10,000 conversations
explain(df, min_cluster_size=25)  # Default for larger datasets

# > 10,000 conversations
explain(df, min_cluster_size=30)
```

### By Use Case

```python
# Research paper (quality matters most)
explain(df, model_name="gpt-4.1", embedding_model="text-embedding-3-large")

# Production dashboard (speed + quality balance)
explain(df, model_name="gpt-4.1-mini", embedding_model="text-embedding-3-large")

# Exploration/development (speed matters most)
explain(df, model_name="gpt-4.1-mini", embedding_model="all-MiniLM-L6-v2")
```

## Next Steps

- **[Data Formats](data-formats.md)** - Prepare your input data correctly
- **[Performance Tuning](../advanced/performance.md)** - Optimize for speed and cost
- **[Custom Pipelines](../advanced/custom-pipelines.md)** - Build custom configurations
