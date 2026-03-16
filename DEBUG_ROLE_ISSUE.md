# DEBUG: Role Column Issue - Step by Step

I've added comprehensive logging throughout the entire pipeline. Here's how to debug this:

## Step 1: Restart Backend

```bash
# Kill any running backend
pkill -f "uvicorn.*stringsight"

# Start fresh
cd /home/lisabdunlap/StringSightNew
uvicorn stringsight.api:app --reload --port 8000
```

## Step 2: Extract a Property with Role-Based Extraction

1. Open browser to http://localhost:5180
2. Upload YOUR data (not demo data)
3. In extraction panel:
   - **Enable "Extract by Role" toggle**
   - **Select roles** (e.g., check "user" and "assistant")
4. Click on a row to view the trace
5. Click "Extract Properties" button

## Step 3: Check Backend Logs

Watch your terminal where the backend is running. You should see:

### ✅ If role extraction is working:

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

### ❌ If role extraction is NOT working:

```
INFO: Extracting properties from 1 conversations using gpt-4o
INFO: ✅ Extraction: Added original_row to 1 properties
INFO:    - Properties with role field: 0/1
```

## Step 4: Check Browser Console

Open browser DevTools (F12) → Console tab

### Look for these messages:

**1. After extraction:**
```javascript
[App] onPropertiesMergedCb - Received properties: {
  count: 2,
  firstProp: {
    id: "...",
    question_id: "...",
    model: "...",
    role: "user",  // ← Should be here!
    property_description: "..."
  }
}
```

**2. In PropertiesTab:**
```javascript
[PropertiesTab] 🔍 ROLE DEBUG: {
  totalRows: 2,
  roleValues: ["user", "assistant"],  // ← Should NOT be [null, null]
  nonNullRoles: ["user", "assistant"],
  uniqueRoles: ["user", "assistant"],
  sampleRow: { role: "user", ... }
}
```

## Step 5: Check Network Tab

In browser DevTools → Network tab:

1. Filter by "Fetch/XHR"
2. Find the `/extract/single` request
3. Click on it
4. Check the **Request Payload**:

```json
{
  "row": { ... },
  "method": "single_model",
  "extract_by_role": ["user", "assistant"],  // ← MUST be here!
  "use_dynamic_role_prompts": false,
  ...
}
```

5. Check the **Response**:

```json
{
  "properties": [
    {
      "id": "...",
      "role": "user",  // ← Should be here!
      "property_description": "...",
      ...
    }
  ]
}
```

## What to Look For

### Scenario 1: `extract_by_role` missing from request

**Problem**: Frontend isn't sending the role parameter

**Cause**: One of these conditions is false:
- `enableRoleExtraction` is false (toggle not enabled)
- `isUsingDemoData` is true (using demo data - role extraction disabled for demo)
- `selectedRoles.length === 0` (no roles selected)

**Fix**:
- Make sure toggle is enabled
- Use your own data, not demo data
- Select at least one role

### Scenario 2: Backend logs show "Extracting properties from..." (not multi-pass)

**Problem**: Backend didn't receive `extract_by_role` parameter

**Fix**: Check Network tab request - if `extract_by_role` is missing, frontend issue

### Scenario 3: Backend logs show multi-pass but "Properties with role field: 0/X"

**Problem**: Role field is getting lost in parsing

**Check**: Backend logs should show:
```
🔍 Role assignment debug:
   - role from original property: user  // ← This should NOT be None
```

### Scenario 4: Response has `role: null` in properties

**Problem**: Parser is not preserving the role

**This is a backend bug** - check parser logs

### Scenario 5: Response has `role: "user"` but PropertiesTab shows empty

**Problem**: Frontend is losing the role field

**Check**: Browser console `[PropertiesTab] 🔍 ROLE DEBUG` - if `roleValues` is all null, frontend bug

## What to Report Back

Please tell me:

1. **Backend logs**: Copy the extraction logs (from "Running multi-pass" through "Properties with role field")

2. **Browser console**: Copy the `[PropertiesTab] 🔍 ROLE DEBUG` output

3. **Network tab**:
   - Does request have `extract_by_role`?
   - Does response have `role` field in properties?

4. **What you're testing with**:
   - Demo data or your own data?
   - Which roles did you select?
   - Is the toggle enabled?

This will help me pinpoint exactly where the role field is getting lost!
