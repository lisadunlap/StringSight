# Troubleshooting

Common issues and solutions for StringSight.

## Installation Issues

### "No module named 'stringsight'"

**Solution:**
```bash
conda activate stringsight
pip install -e ".[full]"
```

### PyTorch/CUDA Errors

**Solution:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Node.js Version Issues

**Solution:**
```bash
conda install -c conda-forge nodejs=20
# Or use nvm
nvm install 20 && nvm use 20
```

## Runtime Issues

### "OpenAI API key not found"

**Solution:**
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### "Clustering produced no valid clusters"

**Causes:**
- Dataset too small
- min_cluster_size too large
- Properties too similar/dissimilar

**Solutions:**
```python
# Reduce cluster size threshold
explain(df, min_cluster_size=5)

# Use more data (minimum 20-50 conversations recommended)

# Try different embedding model
explain(df, embedding_model="all-MiniLM-L6-v2")
```

### Out of Memory Errors

**Solutions:**
```python
# Use local embeddings
explain(df, embedding_model="all-MiniLM-L6-v2")

# Disable embeddings in output
explain(df, include_embeddings=False)

# Increase cluster size
explain(df, min_cluster_size=50)

# Process in batches
for chunk in pd.read_csv("data.csv", chunksize=1000):
    explain(chunk, output_dir="results/batch")
```

## Frontend Issues

### Port Already in Use

**Solution:**
```bash
# Kill process
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9

# Or use different port
python -m uvicorn stringsight.api:app --port 8001
```

### CORS Errors

**Solution:** Check `stringsight/api.py` CORS configuration includes your frontend URL.

### Frontend Won't Start

**Solution:**
```bash
cd frontend/
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Data Issues

### "Missing required column"

**Solution:**
```python
# Check columns
print(df.columns.tolist())

# Rename if needed
df = df.rename(columns={'response': 'model_response'})
```

### "Invalid response format"

**Solution:**
```python
# Convert to strings
df['model_response'] = df['model_response'].astype(str)
```

### Score Column Not Recognized

**Solution:**
```python
import json
# If scores are strings
df['score'] = df['score'].apply(json.loads)
```

## Performance Issues

### Slow Extraction

**Solutions:**
```python
# Use faster model
explain(df, model_name="gpt-4.1-mini")

# Increase parallelism
explain(df, max_workers=32)

# Enable caching
explain(df, extraction_cache_dir=".cache/extraction")
```

### Slow Clustering

**Solutions:**
```python
# Use local embeddings
explain(df, embedding_model="all-MiniLM-L6-v2")

# Disable dimensionality reduction
from stringsight.clusterers import HDBSCANClusterer
clusterer = HDBSCANClusterer(disable_dim_reduction=True)
```

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/lisabdunlap/stringsight/issues)
- **Documentation**: [Read the docs](index.md)
- **Logs**: Check console output for error details
