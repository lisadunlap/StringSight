# Metrics Not Showing - Troubleshooting Guide

## The Issue

Metrics tab shows "No metrics data available" after extracting properties per row and clustering.

## Root Cause

**Your data is missing score columns!**

Metrics computation requires numeric score data to compute statistics. Without scores, the metrics arrays will be empty.

## Quick Check

Run this to check if your data has scores:

```bash
python check_data_format.py <your_data_file.jsonl>
```

Look for this output:
```
⭐ Checking for SCORE data (needed for metrics):
  ✅ 'score' field: present as dict
     Score keys: ['reward', 'accuracy', ...]
```

If it says `❌ NO SCORE DATA FOUND!`, that's your problem!

## Solution

### Option 1: Use Data With Scores (Quick Test)

Use the demo data that already has scores:

```bash
# Copy demo data with scores
cp ~/stringsight-frontend/public/taubench_airline.jsonl ./test_data.jsonl

# Or use the version I created for you:
cp airline_data_demo_with_scores.jsonl ./test_data.jsonl
```

### Option 2: Add Scores to Your Data

If you have a `reward` column, convert it to a `score` dict:

```bash
python add_scores_to_data.py your_data.jsonl your_data_with_scores.jsonl
```

This converts:
```json
{"reward": 0, ...}
```

To:
```json
{"reward": 0, "score": {"reward": 0.0}, ...}
```

### Option 3: Add Real Score Columns

For production use, add meaningful score columns:

```python
import json

# Read your data
with open('data.jsonl', 'r') as f:
    rows = [json.loads(line) for line in f if line.strip()]

# Add scores to each row
for row in rows:
    row['score'] = {
        'accuracy': 0.85,      # Replace with real scores
        'helpfulness': 0.90,
        'fluency': 0.88
    }

# Write back
with open('data_with_scores.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row) + '\n')
```

## Testing the Fix

1. **Restart your backend** (to load the code changes):
   ```bash
   uvicorn stringsight.api:app --reload --port 8000
   ```

2. **Upload data WITH score columns** (use one of the options above)

3. **Extract a property** from a single row

4. **Check backend logs** for:
   ```
   ✅ Extraction: Added original_row to X properties
   ✅ Original row has scores: True
   ✅ Score keys: ['reward']
   ```

5. **Run clustering**

6. **Check backend logs** for:
   ```
   🔍 Clustering: Property meta analysis:
   ✅ Properties with original_row: X
   ✅ Properties with scores in original_row: X
   ✅ Metrics computed: X model_cluster_scores, ...
   ```

7. **Check the Metrics tab** - it should now show data!

## Debug Helper Scripts

I created several helper scripts for you:

- **`debug_metrics.sh`** - Shows what to check in logs
- **`check_data_format.py`** - Verifies your data has scores
- **`add_scores_to_data.py`** - Adds score columns to data
- **`test_metrics_fix.py`** - Unit test for the fix

## Expected Log Output

### During Extraction
```
INFO:     ✅ Extraction: Added original_row to 1 properties
INFO:        - Original row has scores: True
INFO:        - Score keys: ['reward']
```

### During Clustering
```
INFO:     🔍 Clustering: Property meta analysis:
INFO:        - Total properties: 5
INFO:        - Properties with meta: 5
INFO:        - Properties with original_row: 5
INFO:        - Properties with scores in original_row: 5
```

### After Metrics Computation
```
INFO:     ✓ Metrics computed: 8 model_cluster_scores, 4 cluster_scores, 2 model_scores
INFO:        - Sample model_cluster_score keys: ['model', 'cluster', 'size', 'proportion', 'quality_reward', ...]
INFO:        - Sample model_score: gpt-4, size=100
```

### In Browser Console
```
📥 CLUSTERING RESPONSE: {hasClusters: true, hasMetrics: true, ...}
🔵 Backend model_scores: Array(2)
📊 resultsMetrics set: 2 model_scores
[MetricsTab] Extracted models: ['gpt-4']
```

## Common Mistakes

❌ **Using data without score columns**
- Fix: Use data with `score` field or add scores

❌ **Not restarting backend after code changes**
- Fix: Restart backend with `uvicorn stringsight.api:app --reload --port 8000`

❌ **Extracting properties but not clustering them**
- Fix: After extraction, click "Run Clustering"

❌ **Looking at Metrics tab before clustering completes**
- Fix: Wait for clustering to finish, then switch to Metrics tab

## Files Modified

The fix is already in place in these files:
- `stringsight/routers/extraction.py` - Stores `original_row` in property meta
- `stringsight/api.py` - Uses `original_row` during clustering

The fix works, but **you need data with score columns!**
