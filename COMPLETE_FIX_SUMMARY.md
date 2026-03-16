# Complete Fix Summary: Role-Based Extraction & User Insights Tab

## Issues Fixed

### 1. Backend: Missing `extract_by_role` Parameter ✅

**Problem**: The extraction router was not passing `extract_by_role` parameter to the extractor, causing properties to be extracted without role information.

**Files Modified**:
- `stringsight/routers/extraction.py`

**Changes**:
Added two missing parameters to `get_extractor()` calls at:
- Line ~640-652 (background job)
- Line ~823-836 (single extraction endpoint)

```python
extractor = get_extractor(
    # ... existing parameters ...
    extract_by_role=req.extract_by_role,  # ✅ ADDED
    use_dynamic_prompts=req.use_dynamic_role_prompts or False,  # ✅ ADDED
)
```

**Impact**:
- Properties now get a `role` field when role-based extraction is enabled
- Clusterer can detect both `role` and `behavior_type` columns
- Combined role+behavior_type clustering now triggers automatically
- Cluster metadata includes `group` (combined), `role`, and `behavior_type`

### 2. Frontend: Recharts Dependency Replaced with Plotly ✅

**Problem**: `ErrorCategoriesTab.tsx` was importing `recharts` which wasn't installed, causing build errors.

**Files Modified**:
- `stringsight-frontend/src/components/ErrorCategoriesTab.tsx`

**Changes**:
1. Replaced recharts import with Plotly:
   ```typescript
   // Before:
   import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';

   // After:
   import Plot from 'react-plotly.js';
   ```

2. Replaced recharts bar chart with Plotly horizontal bar chart:
   ```typescript
   <Plot
     data={[{
       type: 'bar',
       x: chartData.map(d => d.delta),
       y: chartData.map(d => d.model),
       orientation: 'h',
       marker: { color: chartData.map(d => d.color) },
       text: chartData.map(d => formatDelta(d.delta)),
       textposition: 'auto',
       hovertemplate: '<b>%{y}</b><br>Delta: %{text}<extra></extra>',
     }]}
     layout={{
       height: 150,
       margin: { l: 100, r: 20, t: 10, b: 40 },
       xaxis: {
         title: 'Quality Delta',
         zeroline: true,
         zerolinecolor: '#ccc',
         zerolinewidth: 1,
       },
       yaxis: { autorange: 'reversed' },
       showlegend: false,
       plot_bgcolor: 'rgba(0,0,0,0)',
       paper_bgcolor: 'rgba(0,0,0,0)',
     }}
     config={{ displayModeBar: false, responsive: true }}
     style={{ width: '100%' }}
   />
   ```

**Impact**:
- No new dependencies needed (Plotly already installed)
- Consistent with project's charting library preference (as per CLAUDE.md)
- Build error resolved

## How It All Works Together

### Data Flow with Fixes

1. **Frontend** → User enables "Extract by role" with `["user", "assistant"]`
2. **API Request** → `extract_by_role` parameter sent to backend
3. **Extraction Router** → ✅ Now correctly passes parameter to extractor
4. **Extractor** → Runs multi-pass extraction (one pass per role)
5. **Properties Created** → Each property gets `role` field set
6. **Parser** → Preserves `role` field through parsing
7. **Clustering** → Detects both `role` and `behavior_type` columns
8. **Combined Grouping** → Creates `role_behavior_type` column (e.g., `"user_Style"`)
9. **Cluster Metadata** → Includes all three fields:
   ```json
   {
     "group": "user_Style",
     "role": "user",
     "behavior_type": "Style"
   }
   ```
10. **Metrics Computation** → Preserves cluster metadata
11. **API Response** → Returns clusters with full metadata
12. **Frontend** → ✅ ErrorCategoriesTab displays role-based analysis with Plotly charts

## Testing the Complete Fix

### 1. Restart Services

```bash
# Backend (if running via CLI)
stringsight stop
stringsight launch

# Or if running manually:
# Kill existing process and restart
cd /home/lisabdunlap/StringSightNew
uvicorn stringsight.api:app --reload --port 8000
```

```bash
# Frontend (in separate terminal)
cd ~/stringsight-frontend
npm run dev
```

### 2. Run Test Extraction

```bash
# Quick backend test
cd /home/lisabdunlap/StringSightNew
python test_role_extraction_fix.py
```

### 3. Manual UI Test

1. Open frontend: http://localhost:5180
2. Upload a dataset
3. Enable "Extract by role" checkbox
4. Select roles: `["user", "assistant"]`
5. Run extraction + clustering
6. Navigate to "View Insights" tab
7. Look for three sub-tabs: **Metrics | Clusters | User Insights**
8. Click "User Insights" tab
9. Verify:
   - Clusters are displayed with role chips (colored)
   - Each cluster shows behavior type
   - Plotly bar charts display quality delta per model
   - Filter by role dropdown works
   - Sort by dropdown works

### 4. Verify Backend Data

```bash
# Check properties have role field
cd /home/lisabdunlap/StringSightNew
python3 -c "
import json
import glob

# Find most recent results
files = sorted(glob.glob('frontend/my_results_*/parsed_properties.jsonl'), reverse=True)
if files:
    with open(files[0]) as f:
        prop = json.loads(f.readline())
        print('✅ Property keys:', list(prop.keys()))
        print('✅ Has role?', 'role' in prop)
        if 'role' in prop:
            print('✅ role value:', prop['role'])
        else:
            print('❌ Missing role field!')
"
```

```bash
# Check clusters have role metadata
python3 -c "
import json
import glob

files = sorted(glob.glob('frontend/my_results_*/clusters.jsonl'), reverse=True)
if files:
    with open(files[0]) as f:
        cluster = json.loads(f.readline())
        meta = cluster.get('meta', {})
        print('✅ Cluster metadata keys:', list(meta.keys()))
        print('✅ Has role?', 'role' in meta)
        print('✅ group:', meta.get('group'))
        print('✅ role:', meta.get('role'))
        print('✅ behavior_type:', meta.get('behavior_type'))
"
```

## Expected Results

### Properties should have:
```json
{
  "id": "...",
  "question_id": "...",
  "model": "gpt-4",
  "property_description": "...",
  "role": "user",  // ✅ NEW
  "behavior_type": "Style",
  "category": "...",
  "evidence": "...",
  "reason": "..."
}
```

### Clusters should have:
```json
{
  "id": "1",
  "label": "**Brief Responses**: Uses short language...",
  "size": 5,
  "meta": {
    "group": "user_Style",  // ✅ Combined
    "role": "user",  // ✅ Individual role
    "behavior_type": "Style",  // ✅ Individual behavior type
    "role_behavior_type": "user_Style"  // ✅ Explicit combined
  }
}
```

### User Insights Tab should display:
- **Header**: "Clusters by Role & Quality Impact"
- **Filters**: Role dropdown, Sort by dropdown, Metric filter dropdown
- **Cluster Cards** with:
  - Cluster label/title
  - Role chip (colored: blue for user, green for assistant)
  - Behavior type chip (outlined)
  - Size chip
  - Plotly horizontal bar chart showing quality delta per model

## Troubleshooting

### If properties still don't have role field:

1. **Check API request**:
   ```bash
   # Look for this in backend logs:
   # "🔍 extract_by_role parameter: ['user', 'assistant']"
   ```

2. **Check extractor initialization**:
   ```bash
   # Add debug logging in stringsight/extractors/openai.py line 111:
   # print(f"[DEBUG] extract_by_role = {self.extract_by_role}")
   ```

3. **Verify multi-pass extraction runs**:
   ```bash
   # Look for this in logs:
   # "Running multi-pass extraction for roles: ['user', 'assistant']"
   ```

### If User Insights tab doesn't appear:

1. **Check cluster metadata**:
   ```javascript
   // In browser console:
   console.log(clusters[0]?.meta)
   // Should show: {group: "user_Style", role: "user", behavior_type: "Style"}
   ```

2. **Check tab condition in App.tsx**:
   ```typescript
   // Line 4968: resultsMetrics && clusters.length > 0
   ```

3. **Clear browser cache** and hard refresh (Ctrl+Shift+R)

### If Plotly charts don't render:

1. **Check Plotly is imported**: `import Plot from 'react-plotly.js';`
2. **Check console for errors**: Look for Plotly-related errors
3. **Verify chartData format**: Should be array of `{model, delta, color}`

## Files Changed

### Backend
1. `stringsight/routers/extraction.py` - Added extract_by_role parameter to extractor calls

### Frontend
1. `stringsight-frontend/src/components/ErrorCategoriesTab.tsx` - Replaced recharts with Plotly

### Documentation
1. `ROLE_EXTRACTION_BUG_FIX.md` - Detailed bug fix explanation
2. `COMPLETE_FIX_SUMMARY.md` - This file

## Backward Compatibility

✅ **Fully backward compatible**
- Standard extractions (without role) work exactly as before
- Only affects extractions where `extract_by_role` is explicitly enabled
- No breaking changes to API or data formats
- Frontend gracefully handles clusters with or without role metadata
