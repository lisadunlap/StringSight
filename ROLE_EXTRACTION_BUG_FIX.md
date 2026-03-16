# Role-Based Extraction Bug Fix

## Problem Summary

When role-based extraction was enabled (`extract_by_role=["user", "assistant"]`), the properties were not getting a `role` field, which caused:

1. **Missing role metadata in clusters** - Clusters only had `behavior_type` in metadata, not `role`
2. **Combined grouping didn't trigger** - The role+behavior_type clustering logic never activated
3. **User Insights tab didn't appear** - Frontend couldn't display role-based analysis without role data

## Root Cause

**File**: `stringsight/routers/extraction.py`

The background extraction job (`_run_extract_job_async`) was creating the extractor without passing the `extract_by_role` parameter:

```python
# ❌ BEFORE (lines 640-650)
extractor = get_extractor(
    model_name=req.model_name or "gpt-4.1",
    system_prompt=system_prompt,
    temperature=req.temperature or 0.7,
    top_p=req.top_p or 0.95,
    max_tokens=req.max_tokens or 16000,
    max_workers=req.max_workers if req.max_workers is not None else DEFAULT_MAX_WORKERS,
    include_scores_in_prompt=False if req.include_scores_in_prompt is None else req.include_scores_in_prompt,
    verbose=False,
    use_wandb=False,
    # ❌ Missing: extract_by_role and use_dynamic_prompts
)
```

This caused the extractor to run **single-pass extraction** instead of **multi-pass extraction**, so properties never got a `role` field.

## The Fix

Added the missing parameters to both `get_extractor()` calls in the extraction router:

```python
# ✅ AFTER (lines 640-652)
extractor = get_extractor(
    model_name=req.model_name or "gpt-4.1",
    system_prompt=system_prompt,
    temperature=req.temperature or 0.7,
    top_p=req.top_p or 0.95,
    max_tokens=req.max_tokens or 16000,
    max_workers=req.max_workers if req.max_workers is not None else DEFAULT_MAX_WORKERS,
    include_scores_in_prompt=False if req.include_scores_in_prompt is None else req.include_scores_in_prompt,
    verbose=False,
    use_wandb=False,
    extract_by_role=req.extract_by_role,  # ✅ ADDED
    use_dynamic_prompts=req.use_dynamic_role_prompts or False,  # ✅ ADDED
)
```

### Files Modified

1. **`stringsight/routers/extraction.py`** - Added `extract_by_role` and `use_dynamic_prompts` parameters to two `get_extractor()` calls:
   - Line ~640-652 (background job)
   - Line ~823-836 (single extraction)

## How It Works Now

### With the Fix ✅

1. **API receives request** with `extract_by_role=["user", "assistant"]`
2. **Router creates extractor** with `extract_by_role` parameter
3. **Extractor runs multi-pass extraction** (separate pass for each role)
4. **Properties get `role` field** (e.g., `role="user"` or `role="assistant"`)
5. **Parser preserves `role` field** through parsing
6. **Clusterer detects both `role` and `behavior_type` columns** and creates combined grouping
7. **Clusters get full metadata**:
   ```json
   {
     "group": "user_Style",
     "role": "user",
     "behavior_type": "Style"
   }
   ```
8. **Frontend displays User Insights tab** with role-based analysis

### Before the Fix ❌

1. API receives request with `extract_by_role=["user", "assistant"]`
2. Router creates extractor **WITHOUT** `extract_by_role` parameter
3. Extractor runs **single-pass extraction** (ignores role parameter)
4. Properties **don't get `role` field**
5. Clusterer only finds `behavior_type` column, no combined grouping
6. Clusters only have partial metadata:
   ```json
   {
     "group": "Style",
     "behavior_type": "Style"
     // ❌ Missing "role"
   }
   ```
7. Frontend can't display User Insights tab (no role data)

## Testing the Fix

To verify the fix works:

```bash
# 1. Restart the backend
cd /home/lisabdunlap/StringSightNew
# Kill existing server and restart

# 2. Run a new extraction with role-based extraction enabled
# In the frontend UI:
# - Upload data
# - Enable "Extract by role" checkbox
# - Select roles: ["user", "assistant"]
# - Run extraction

# 3. Check the results
# Properties should now have role field:
python3 -c "
import json
with open('path/to/results/parsed_properties.jsonl') as f:
    prop = json.loads(f.readline())
    print('Has role?', 'role' in prop)
    print('role value:', prop.get('role'))
"

# 4. Check clusters have role metadata:
python3 -c "
import json
with open('path/to/results/clusters.jsonl') as f:
    cluster = json.loads(f.readline())
    meta = cluster.get('meta', {})
    print('Has role in meta?', 'role' in meta)
    print('group:', meta.get('group'))  # Should be like 'user_Style'
    print('role:', meta.get('role'))    # Should be like 'user'
"
```

## Related Code Paths

- **Extraction router** (`stringsight/routers/extraction.py`) - Now correctly passes parameters
- **Extractor** (`stringsight/extractors/openai.py` line 109-112) - Checks `if self.extract_by_role is not None`
- **Multi-pass extraction** (`stringsight/extractors/openai.py` line 535) - Sets `role=role` on properties
- **Parser** (`stringsight/postprocess/parser.py` line 159) - Preserves `role` field
- **Clusterer** (`stringsight/clusterers/hdbscan.py` line 386-448) - Detects role+behavior_type and creates combined metadata

## Impact

- **Backward compatible** - Only affects extractions where `extract_by_role` is explicitly set
- **No breaking changes** - Standard extractions without role work exactly the same
- **Enables full role-based analysis** - User Insights tab will now work correctly
