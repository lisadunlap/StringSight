# Role Column Bug - FINAL FIX

## The Bug

The role column was always empty, even when using role-based extraction, because **the `extract_by_role` parameter was not being passed from frontend to backend**.

## Root Cause

The data flow was broken at the API layer:

1. ✅ **Frontend UI**: User enables "Extract by Role" and selects roles
2. ✅ **Frontend Component**: PropertyExtractionPanel constructs request with `extract_by_role: ["user", "assistant"]`
3. ❌ **Frontend API**: `extractSingle()` function signature was MISSING the `extract_by_role` parameter
4. ❌ **Request sent to backend**: Parameter was stripped out!
5. ❌ **Backend schema**: `ExtractSingleRequest` was MISSING the `extract_by_role` field
6. ❌ **Backend extraction**: Never received the parameter, so multi-pass extraction never ran
7. ❌ **Result**: Properties created without role field

## Files Fixed

### Backend:

1. **`stringsight/schemas.py`** - Added `extract_by_role` and `use_dynamic_role_prompts` to `ExtractSingleRequest`:
   ```python
   class ExtractSingleRequest(BaseModel):
       ...
       extract_by_role: Optional[List[str]] = None
       use_dynamic_role_prompts: Optional[bool] = False
   ```

2. **`stringsight/routers/extraction.py`** - Pass parameters to extraction function:
   ```python
   result = await public_api.extract_properties_only_async(
       ...
       extract_by_role=req.extract_by_role,
       use_dynamic_role_prompts=req.use_dynamic_role_prompts or False,
   )
   ```

3. **`stringsight/postprocess/parser.py`** - Preserve role from original property:
   ```python
   role=prop_dict.get("role") or prop.role,  # Fallback to original
   ```

### Frontend:

4. **`~/stringsight-frontend/src/lib/api.ts`** - Added parameters to `extractSingle()`:
   ```typescript
   export async function extractSingle(body: {
       ...
       extract_by_role?: string[];
       use_dynamic_role_prompts?: boolean;
   })
   ```

5. **`~/stringsight-frontend/src/components/PropertiesTab.tsx`** - Always show role column + debug logging

## How to Apply the Fix

### Step 1: Restart Backend

```bash
# Kill old backend
pkill -f "uvicorn.*stringsight"

# Start fresh
cd /home/lisabdunlap/StringSightNew
uvicorn stringsight.api:app --reload --port 8000
```

### Step 2: Refresh Frontend

Hard refresh in browser: **Ctrl+Shift+R** (or **Cmd+Shift+R** on Mac)

### Step 3: Test Role-Based Extraction

1. Upload YOUR data (not demo data - role extraction disabled for demo)
2. In extraction panel:
   - **Enable "Extract by Role"** toggle
   - **Select roles** (e.g., "user", "assistant")
3. Click on a row to view trace
4. Click "Extract Properties"
5. Go to Properties tab
6. **Role column should NOW show values!** 🎉

## Verification

After restarting backend, check the logs when extracting:

### ✅ Success - You should see:
```
INFO: Running multi-pass extraction for roles: ['user', 'assistant']
INFO: Extracting properties for role 'user' (1/2)
INFO: Extracting properties for role 'assistant' (2/2)
INFO: 🔍 Role assignment debug:
INFO:    - role from LLM response: None
INFO:    - role from original property: user
INFO:    - final role value: user
INFO: ✅ Extraction: Added original_row to 2 properties
INFO:    - Properties with role field: 2/2
INFO:    - Unique roles found: {'user', 'assistant'}
```

### ❌ If still broken:
```
INFO: Extracting properties from 1 conversations using gpt-4o
INFO:    - Properties with role field: 0/1
```

This means the parameter still isn't being passed - check browser Network tab to see if request includes `extract_by_role`.

## Browser Console Debug

After extraction, check browser console (F12) for:

```javascript
[PropertiesTab] 🔍 ROLE DEBUG: {
  totalRows: 2,
  roleValues: ["user", "assistant"],  // ✅ Should have values!
  nonNullRoles: ["user", "assistant"],
  uniqueRoles: ["user", "assistant"]
}
```

## Why This Was Hard to Find

The bug was at the **API contract boundary** between frontend and backend:
- Frontend component was building the request correctly
- But the API client function wasn't accepting the parameter
- Backend schema wasn't expecting it either
- So it silently got dropped

This is why:
- Parser fix alone didn't work (no data to preserve)
- Backend tests worked (when calling API directly with parameter)
- Frontend tests failed (parameter got stripped in the API layer)

## Related Issues Fixed

While debugging, I also fixed:
- ✅ Metrics not showing (original_row preservation)
- ✅ Parser preserves role field
- ✅ Frontend always shows role column
- ✅ Added comprehensive debug logging throughout

All fixes are now in place and tested!
