# Fix for Missing Metrics After Per-Row Extraction

## Problem

When extracting properties per row and then clustering them, the Metrics tab showed no data. The issue was:

1. User extracts properties one row at a time using the "Extract Properties" button on a single trace
2. Properties are stored without the original row data (especially score columns)
3. Later, when clustering is run, the backend tries to reconstruct conversations by matching properties back to operational rows
4. Matching fails or doesn't preserve score data correctly
5. Without scores, metrics computation returns empty results
6. Metrics tab appears empty

## Root Cause

In [stringsight/api.py:704-707](stringsight/api.py#L704-L707), when clustering reconstructs conversations:

```python
if matching_row:
    scores = matching_row.get("score") or matching_row.get("scores") or {}
else:
    scores = {}  # ❌ Empty scores mean no metrics!
```

If the property-to-row matching fails, `scores` becomes an empty dict, and metrics can't be computed.

## Solution

Store the original row data directly in each property during extraction, so clustering doesn't need to match it back.

### Changes Made

#### 1. Extraction Endpoint ([stringsight/routers/extraction.py:407-428](stringsight/routers/extraction.py#L407-L428))

```python
# BEFORE:
props = [p.to_dict() for p in dataset.properties]

# AFTER:
props = []
for p in dataset.properties:
    p_dict = p.to_dict()
    # Store the original row data in meta.original_row
    if not p_dict.get('meta'):
        p_dict['meta'] = {}
    p_dict['meta']['original_row'] = req.row  # ✅ Store original row
    props.append(p_dict)
```

#### 2. Clustering Endpoint ([stringsight/api.py:677-707](stringsight/api.py#L677-L707))

```python
# NEW: Build a map of (question_id, model) -> original row data from property meta
row_data_map = {}
for prop in properties:
    key = (prop.question_id, cast(str, prop.model))
    # Check if property has original_row stored in meta
    if hasattr(prop, 'meta') and isinstance(prop.meta, dict) and 'original_row' in prop.meta:
        row_data_map[key] = prop.meta['original_row']

# Then use it:
matching_row = row_data_map.get((question_id, model))  # ✅ Get from stored data first
if matching_row:
    matches_found += 1
else:
    # Fallback to the old matching logic
    ...
```

## Benefits

1. **No matching logic needed** - Row data travels with the property
2. **Scores always preserved** - Original score columns are stored in `original_row`
3. **Backward compatible** - Fallback to old matching logic if `original_row` is not present
4. **Works across sessions** - Properties can be saved and clustered later

## Testing

Run the test script:

```bash
python test_metrics_fix.py
```

Or test manually:
1. Start backend: `uvicorn stringsight.api:app --reload --port 8000`
2. Start frontend: `cd ~/stringsight-frontend && npm run dev`
3. Upload data with score columns
4. Extract properties for a single row
5. Run clustering
6. **Verify:** Metrics tab now shows data!

## Files Modified

- `stringsight/routers/extraction.py` - Added `original_row` to property meta during single-row extraction
- `stringsight/api.py` - Updated clustering to use `original_row` from property meta

## Notes

- Batch extraction doesn't need this fix because all data is extracted together
- The Property class already has a `meta` field that can store arbitrary data
- This fix doesn't break existing workflows that don't use per-row extraction
