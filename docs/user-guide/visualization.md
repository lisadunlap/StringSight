# Visualization

StringSight provides various ways to visualize and explore analysis results.

## Visualization Best Practices

### Plot Interpretation

#### Frequency Plots
Shows what proportion of each model's responses fall into each behavioral cluster.

- **Y-axis**: Cluster names
- **X-axis**: Proportion (0-1) or percentage
- **Bars**: One per model
- **Interpretation**:
  - Longer bars = model exhibits this behavior more frequently
  - Compare bar lengths to see behavioral differences

#### Quality Plots
Shows how well models perform within specific behavioral clusters.

- **Y-axis**: Cluster names
- **X-axis**: Quality score (varies by metric)
- **Bars**: One per model
- **Interpretation**:
  - Higher values = better performance in this behavior cluster
  - Compare across models to see strengths/weaknesses

#### Delta Plots
Shows relative differences from baseline/median performance.

- **Y-axis**: Cluster names
- **X-axis**: Delta value (can be positive/negative)
- **Zero line**: Baseline performance
- **Interpretation**:
  - Positive = over-represented / better than average
  - Negative = under-represented / worse than average

### Filtering Strategies

**For Initial Exploration:**
1. Start with **all clusters** visible
2. Sort by **Frequency (Descending)** to see most common behaviors
3. Review **Top 10** clusters

**For Finding Differences:**
1. Enable **Significance filter**
2. Sort by **Relative Frequency Δ**
3. Look for clusters where models diverge most

**For Quality Analysis:**
1. Switch to **Quality plot type**
2. Select quality metric (e.g., "accuracy", "helpfulness")
3. Sort by **Quality Delta Δ**
4. Identify where each model excels

### Performance Optimization

**For Large Datasets:**
- Use **sampling** during initial exploration
- Filter by **significant clusters only**
- Limit to **Top N clusters** (e.g., 10-20)
- Disable **confidence intervals** for faster rendering

**For Sharing:**
- Export filtered data to CSV
- Save plots as PNG/PDF

## Customizing Visualizations

```python
import plotly.express as px

# Custom plot function
def create_custom_plot(data, metric="proportion"):
    fig = px.bar(
        data,
        x=metric,
        y="cluster_label",
        color="model",
        orientation="h",
        title="Custom Behavioral Analysis"
    )
    return fig
```

### React Customization

Modify chart configurations in `frontend/src/components/`:

```typescript
// In Plot.tsx
const chartConfig = {
  type: 'bar',
  layout: {
    title: 'Custom Title',
    xaxis: { title: 'Custom X Label' },
    yaxis: { title: 'Custom Y Label' }
  }
};
```

## Exporting Results

### From React

1. Click "Export" button in data table
2. Use browser's print function for plots
3. Copy data from tables directly

### Programmatically

```python
import pandas as pd
import json

# Load results
df = pd.read_parquet("results/clustered_results.parquet")

# Export filtered data
filtered = df[df['property_description_cluster_label'] == 'Reasoning Transparency']
filtered.to_csv("reasoning_examples.csv", index=False)

# Export metrics (prefer JSONL DataFrame exports when available)
try:
    metrics_df = pd.read_json("results/model_cluster_scores_df.jsonl", lines=True)
    gpt4_metrics = metrics_df[metrics_df["model"] == "gpt-4"]
    top_clusters = (
        gpt4_metrics.sort_values("proportion", ascending=False)
                   .head(10)[["cluster", "proportion"]]
    )
except FileNotFoundError:
    with open("results/model_cluster_scores.json") as f:
        metrics = json.load(f)
    gpt4_metrics = metrics.get("gpt-4", {})
    top_clusters = sorted(
        gpt4_metrics.items(),
        key=lambda x: x[1].get("proportion", 0),
        reverse=True
    )[:10]
```

## Troubleshooting

### Frontend Issues

**Port already in use:**
```bash
# Kill process on port 8000 or 5173
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9

# Or use different ports
python -m uvicorn stringsight.api:app --port 8001
# Update frontend/src/config.ts accordingly
```

**CORS errors:**
```python
# In stringsight/api.py, add allowed origins:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Visualization Issues

**Plots not rendering:**
```bash
# Reinstall plotly
pip install --upgrade plotly
```

**Data not loading:**
```python
# Check file paths are absolute
import os
data_path = os.path.abspath("results/clustered_results.parquet")
```

**Slow loading:**
- Reduce dataset size
- Use parquet instead of JSON
- Filter data before loading

## Next Steps

- **[Configuration Options](configuration.md)** - Customize analysis parameters
- **[Data Formats](data-formats.md)** - Understand input/output formats
- **[Advanced Usage](../advanced/custom-pipelines.md)** - Build custom visualizations
