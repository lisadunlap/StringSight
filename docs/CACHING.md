# Caching Configuration

StringSight uses disk-based caching to speed up repeated operations and reduce API costs.

## Default Behavior

**Caching is ENABLED by default** for both:
- **LLM Completions**: Cluster summaries, label generation, property extraction
- **Embeddings**: Text embeddings for clustering

Cache is stored in `.cache/stringsight/` with a 50GB size limit.

## Environment Variables

### Disable All Caching
```bash
export STRINGSIGHT_DISABLE_CACHE=1
```
Disables both LLM completions and embeddings caching.

### Disable Only Embedding Caching
```bash
export STRINGSIGHT_DISABLE_EMBEDDING_CACHE=1
```
LLM completions will still be cached, but embeddings will be recomputed each time.

**Use case**: When you want to ensure fresh embeddings (e.g., testing different embedding models) while still benefiting from LLM completion caching.

### Customize Cache Location
```bash
export STRINGSIGHT_CACHE_DIR="/path/to/cache"
```
Default: `.cache/stringsight/`

### Customize Cache Size
```bash
export STRINGSIGHT_CACHE_MAX_SIZE="100GB"
```
Default: `50GB`

Supported units: `B`, `KB`, `MB`, `GB`, `TB`

## Examples

### Example 1: Default (All Caching Enabled)
```bash
# No environment variables needed
python your_script.py
```

Logs will show:
```
INFO: LLM caching enabled (embeddings=enabled)
INFO: Caching enabled for clustering (cache_dir=.cache/stringsight, embeddings=enabled)
```

### Example 2: Disable Embedding Cache Only
```bash
export STRINGSIGHT_DISABLE_EMBEDDING_CACHE=1
python your_script.py
```

Logs will show:
```
INFO: LLM caching enabled (embeddings=disabled)
INFO: Caching enabled for clustering (cache_dir=.cache/stringsight, embeddings=disabled)
```

**Result**: 
- ✅ LLM completions (cluster labels, summaries) are cached
- ❌ Embeddings are computed fresh each time

### Example 3: Disable All Caching
```bash
export STRINGSIGHT_DISABLE_CACHE=1
python your_script.py
```

Logs will show:
```
INFO: LLM caching disabled (STRINGSIGHT_DISABLE_CACHE=1)
INFO: Caching disabled for clustering (set STRINGSIGHT_DISABLE_CACHE=0 to enable)
```

**Result**: Everything is computed fresh each time.

### Example 4: Custom Cache Configuration
```bash
export STRINGSIGHT_CACHE_DIR="/mnt/fast-ssd/cache"
export STRINGSIGHT_CACHE_MAX_SIZE="200GB"
export STRINGSIGHT_DISABLE_EMBEDDING_CACHE=1
python your_script.py
```

**Result**: 
- LLM completions cached in `/mnt/fast-ssd/cache/` (up to 200GB)
- Embeddings not cached

## Cache Performance

With caching enabled:
- **10-100x faster** for repeated clustering runs on same data
- **Significant cost savings** by avoiding duplicate LLM API calls
- **Automatic invalidation** ensures fresh data when inputs change

## Cache Management

### View Cache Size
```bash
du -sh .cache/stringsight/
```

### Clear Cache
```bash
rm -rf .cache/stringsight/
```

### Clear Only Embeddings Cache
```bash
rm -rf .cache/stringsight/embeddings/
```

### Clear Only Completions Cache
```bash
rm -rf .cache/stringsight/completions/
```

## When to Disable Embedding Cache

Consider disabling embedding cache when:
- Testing different embedding models
- Debugging embedding-related issues
- Working with constantly changing text data
- Benchmarking embedding performance

## Technical Details

- **Cache Type**: DiskCache (thread-safe, persistent across runs)
- **Cache Keys**: SHA-256 hashes of input parameters
- **Namespacing**: Completions and embeddings stored separately
- **Model-aware**: Embeddings are namespaced by model to prevent dimension mismatches
- **Eviction**: LRU (Least Recently Used) when size limit is reached

