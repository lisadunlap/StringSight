# Diagnosing Empty Role Column Issue

## The Key Question

**Are you using role-based extraction?**

The role column will ONLY have values if you:
1. Enable "Extract by Role" toggle in the UI
2. Select specific roles (e.g., "user", "assistant", "tool")
3. Extract properties

If you're NOT using role-based extraction, the role field will be `None` for all properties, which displays as empty cells.

## How to Check

### Step 1: Check Backend Logs

After extracting a property, look for these log messages:

**If role-based extraction is ENABLED:**
```
INFO: Running multi-pass extraction for roles: ['user', 'assistant']
INFO: Extracting properties for role 'user' (1/2)
INFO: Extracting properties for role 'assistant' (2/2)
```

**If role-based extraction is NOT enabled:**
```
INFO: Extracting properties from 5 conversations using gpt-4o
```
(No mention of "multi-pass" or "roles")

### Step 2: Check Extraction Response

In the backend logs, you should see:
```
INFO: ✅ Extraction: Added original_row to 1 properties
INFO:    - Properties with role field: 1/1
INFO:    - Unique roles found: {'user'}
```

**If you see `Properties with role field: 0/1`**, then role extraction is NOT working.

### Step 3: Check Browser Console

In your browser console (F12), after extraction, check the response:

```javascript
// Look for the extraction response
{
  "properties": [
    {
      "id": "...",
      "role": "user",  // ← Should be here if role-based extraction is enabled
      "property_description": "...",
      ...
    }
  ]
}
```

If `"role": null` or missing, role extraction didn't run.

## Common Issues

### Issue 1: Role-Based Extraction Not Enabled

**Symptom**: Role column shows but all cells are empty

**Cause**: You haven't enabled "Extract by Role" in the UI

**Fix**:
1. In the extraction panel, find the "Extract by Role" toggle
2. Enable it
3. Select roles (e.g., "user", "assistant")
4. Extract properties again

### Issue 2: Using Demo Data

**Symptom**: Demo data doesn't show roles

**Cause**: Demo data was not extracted with role-based extraction

**Fix**: Extract your own data with role-based extraction enabled

### Issue 3: Extractor Not Getting Role Parameter

**Symptom**: Backend logs don't show "multi-pass extraction"

**Cause**: Frontend isn't sending `extract_by_role` parameter to backend

**Debug**: Check the extraction request in Network tab:
```json
{
  "row": {...},
  "method": "single_model",
  "extract_by_role": ["user", "assistant"],  // ← Should be here
  ...
}
```

If `extract_by_role` is missing or null, the frontend isn't sending it correctly.

## Testing Role-Based Extraction

To test if role extraction works:

1. **Restart backend** (to load logging changes):
   ```bash
   uvicorn stringsight.api:app --reload --port 8000
   ```

2. **In the UI**:
   - Upload data or use demo data
   - Open the extraction panel
   - **Enable "Extract by Role" toggle**
   - Select roles: ["user", "assistant"]
   - Extract a property from a single row

3. **Check backend logs** for:
   ```
   INFO: Running multi-pass extraction for roles: ['user', 'assistant']
   INFO: 🔍 Role assignment debug:
   INFO:    - role from LLM response: None
   INFO:    - role from original property: user
   INFO:    - final role value: user
   ```

4. **Check Properties tab** - role column should show "user" or "assistant"

## If Role Column Is Still Empty

If you've enabled role-based extraction and the role column is still empty:

1. **Check if `extract_by_role` is in the request**:
   - Open browser DevTools (F12)
   - Go to Network tab
   - Extract a property
   - Find the `/extract/single` request
   - Check the request payload
   - Should have: `"extract_by_role": ["user", "assistant"]`

2. **Check backend logs** for role assignment:
   - Should see: `role from original property: user`
   - Should see: `final role value: user`

3. **Check the extraction response**:
   - In Network tab, check the response
   - Properties should have: `"role": "user"`

4. **Check browser console** for any errors

## Next Steps

Run through the testing steps above and let me know:
1. Are you enabling "Extract by Role" toggle?
2. What do you see in the backend logs?
3. What does the extraction response show in the Network tab?

This will help me pinpoint exactly where the role data is getting lost.
