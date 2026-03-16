# Role-Based Extraction & Clustering - Implementation Summary

## Overview
Implemented role-based property extraction and clustering that groups by both `role` AND `behavior_type` simultaneously, creating separate clusters for each combination (e.g., `user_Positive`, `assistant_Style`, etc.).

## What Was Implemented

### 1. Backend: Combined Role + Behavior Type Clustering
**File**: `stringsight/clusterers/hdbscan.py`

**Key Changes**:
- When `groupby_column='role'` is set AND both `role` and `behavior_type` columns exist, automatically creates a combined `role_behavior_type` column
- Clusters are grouped by the combined dimension (e.g., `user_Positive`, `assistant_Negative`, `user_Style`)
- Cluster metadata includes all three fields:
  - `group`: Combined value (e.g., `"user_Style"`) - used for UI grouping
  - `role`: Individual role (e.g., `"user"`)
  - `behavior_type`: Individual behavior type (e.g., `"Style"`)

**Code Location**: Lines 161-172 (combined column creation) and Lines 380-452 (metadata extraction)

### 2. Backend: Pipeline Auto-Configuration
**File**: `stringsight/_public/pipeline_builders.py`

**Key Changes**:
- When `extract_by_role` is enabled, automatically sets `groupby_column='role'`
- Overrides any `groupby_column` value passed from the UI/API
- This ensures role-based extraction always triggers combined clustering

**Code Location**: Lines 128-137

### 3. Frontend: Three-Tab Insights Section
**File**: `stringsight-frontend/src/App.tsx`

**Key Changes**:
- Added three tabs in the View Insights section:
  - **Metrics**: Shows "All Clusters", "Common Failures", "Model Comparison", etc.
  - **Clusters**: Shows cluster cards view
  - **Error Categories**: NEW - Shows clusters grouped by role with quality delta bar charts

**Code Location**: Lines 4812-4818 (tabs) and Lines 4967-4973 (Error Categories tab content)

### 4. Frontend: Error Categories Tab (NEW)
**File**: `stringsight-frontend/src/components/ErrorCategoriesTab.tsx`

**Purpose**: Display clusters (not individual properties) with:
- Cluster cards showing role + behavior_type
- Horizontal bar charts showing quality delta per model
- Filters for role, sort order, and metric

**Props**:
- `clusters`: Array of cluster objects
- `modelClusterScores`: Array of model-cluster score data
- `allModels`: Array of model names

**Features**:
- Filter by role (user, assistant, etc.)
- Sort by quality delta (worst first) or cluster size
- Each cluster card shows:
  - Cluster label
  - Role chip (colored)
  - Behavior type chip
  - Size chip
  - Bar chart of quality deltas per model (color-coded: red=negative, green=positive)

### 5. Frontend: Role Display Throughout
**Files**:
- `PropertyCard.tsx` (Lines 119-131)
- `ClusterSidecard.tsx` (Lines 1116-1149)

**Changes**:
- Display both `role` AND `behavior_type` chips on properties
- Show role with colored badges
- Show behavior_type with outlined chips

### 6. Data Normalization
**File**: `stringsight-frontend/src/lib/normalize.ts`

**Changes**:
- Flatten nested `metadata.group` and `metadata.role` to top-level fields
- This ensures the frontend can access role information in metrics

**Code Location**: Function `enrichModelClusterScoresWithMetadata`

## How It Works

### Extraction Flow:
1. User enables `extract_by_role=['user', 'assistant']` in UI
2. Properties are extracted separately for each role
3. Each property gets both `role` and `behavior_type` fields

### Clustering Flow:
1. Backend receives `extract_by_role` parameter
2. Pipeline builder sets `groupby_column='role'` (overriding UI's `groupby_column='behavior_type'`)
3. HDBSCANClusterer detects both `role` and `behavior_type` columns exist
4. Creates combined `role_behavior_type` column (e.g., `"user_Positive"`, `"assistant_Style"`)
5. Clusters independently for each combination
6. Each cluster gets metadata with `group`, `role`, and `behavior_type`

### Frontend Flow:
1. User navigates to View Insights → ERROR CATEGORIES tab
2. ErrorCategoriesTab receives enriched clusters with all metadata
3. Displays cluster cards grouped/filterable by role
4. Shows bar charts of quality deltas per model

## Testing

### Backend Test File:
`test_combined_grouping.py` - Verifies:
- Combined column creation
- Metadata preservation (group, role, behavior_type)
- Clustering by role+behavior_type pairs

### Test Output Location:
`test_combined_grouping_output/`

### Verified Behavior:
```json
{
  "group": "assistant_Negative (critical)",
  "role": "assistant",
  "behavior_type": "Negative (critical)",
  "role_behavior_type": "assistant_Negative (critical)"
}
```

## Current Status

### ✅ Working:
1. Backend combined role+behavior_type clustering
2. Metadata preservation (group, role, behavior_type)
3. Pipeline auto-configuration when extract_by_role is enabled
4. Frontend tabs structure (Metrics | Clusters | Error Categories)
5. Role display in cluster sidecards and property cards
6. Data normalization to flatten metadata

### ⚠️ In Progress:
1. Error Categories tab not rendering for user
   - Tab exists in code (verified at line 4817)
   - Debug logging added to check if clusters data is being passed
   - Need to check browser console for `[ErrorCategoriesTab] Rendered with:` message
   - Possible issues:
     - clusters array might be empty
     - React component not re-rendering
     - Dev server cache issue

## Files Modified

### Backend:
1. `stringsight/clusterers/hdbscan.py` - Combined grouping logic
2. `stringsight/_public/pipeline_builders.py` - Auto-configuration
3. `stringsight/workers/tasks.py` - Role field preservation (line 391)
4. `stringsight/routers/extraction.py` - Role field in API response (line 263)
5. `stringsight/api.py` - Role field in Property construction

### Frontend:
1. `src/App.tsx` - Three-tab structure and ErrorCategoriesTab integration
2. `src/components/ErrorCategoriesTab.tsx` - NEW cluster view with bar charts
3. `src/components/cards/PropertyCard.tsx` - Display both role and behavior_type
4. `src/components/ClusterSidecard.tsx` - Role and behavior_type in property list
5. `src/lib/normalize.ts` - Flatten metadata fields

## Debugging Steps for Current Issue

If Error Categories tab not showing:

1. **Check browser console** when clicking the tab:
   ```javascript
   // Look for this message:
   [ErrorCategoriesTab] Rendered with: {clustersCount: X, ...}
   ```

2. **Verify clusters data exists**:
   ```javascript
   // In browser console:
   console.log(window.__REACT_DEVTOOLS_GLOBAL_HOOK__)
   ```

3. **Check if component is mounting**:
   - Open React DevTools
   - Look for `ErrorCategoriesTab` component
   - Check its props (clusters, modelClusterScores, allModels)

4. **Check condition in App.tsx (line 4968)**:
   ```typescript
   resultsMetrics && clusters.length > 0
   ```
   - Verify both `resultsMetrics` exists and `clusters.length > 0`

5. **Hard refresh browser**: Ctrl+Shift+R

6. **Check dev server console** for compilation errors

## Next Steps

1. Debug why ErrorCategoriesTab isn't rendering
2. Test with real data that has role-based extraction
3. Verify bar charts display correctly with quality delta data
4. Consider hiding role-based clusters from "All Clusters" and "Common Failures" sections in Metrics tab (if requested)

## Example Output Structure

### Cluster with Combined Grouping:
```json
{
  "id": "1",
  "label": "**Brief Responses**: Uses short language...",
  "size": 5,
  "meta": {
    "group": "assistant_Style",
    "role": "assistant",
    "behavior_type": "Style",
    "role_behavior_type": "assistant_Style",
    "quality_delta_by_model": {
      "gpt-4": {
        "wildbench_score": -0.05
      }
    }
  }
}
```

### model_cluster_scores with Role Metadata:
```json
{
  "cluster": "**Brief Responses**...",
  "model": "gpt-4",
  "size": 5,
  "proportion": 0.25,
  "quality_delta_wildbench_score": -0.05,
  "metadata": {
    "group": "assistant_Style",
    "role": "assistant",
    "behavior_type": "Style"
  }
}
```

## Commands to Re-run Tests

```bash
# Backend test
cd /home/lisabdunlap/StringSightNew
python test_combined_grouping.py

# Check output
head -5 test_combined_grouping_output/clusters.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    c = json.loads(line)
    meta = c.get('meta', {})
    print(f\"group={meta.get('group')}, role={meta.get('role')}, behavior_type={meta.get('behavior_type')}\")
"

# Frontend dev server
cd ~/stringsight-frontend
npm run dev
```

## Key Insights

1. **Setting `groupby_column='role'` is the trigger** for combined grouping - the clusterer auto-detects behavior_type and creates the combined column

2. **Three metadata fields are always present**: `group` (combined), `role` (individual), `behavior_type` (individual)

3. **Frontend tabs were already working** - the yellow debug box confirmed they render, user may have been looking at the wrong section

4. **Pipeline builder override is critical** - Without it, UI's `groupBy='behavior_type'` would override role-based clustering

5. **Data normalization flattens nested metadata** - Essential for frontend to access `role` and `behavior_type` in model_cluster_scores
