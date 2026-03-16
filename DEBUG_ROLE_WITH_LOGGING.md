# Debug Role Extraction - With Console Logging

I've added comprehensive console logging to trace the `extract_by_role` parameter through the entire flow.

## Steps to Debug

### 1. Restart Backend & Refresh Browser

```bash
# Kill old backend
pkill -f "uvicorn.*stringsight"

# Start fresh
cd /home/lisabdunlap/StringSightNew
uvicorn stringsight.api:app --reload --port 8000
```

Then in browser: **Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)**

### 2. Open Browser Console

Press **F12** → Go to **Console** tab

### 3. Extract with Role-Based Extraction

1. Upload YOUR data (not demo data!)
2. **Enable "Extract by Role" toggle**
3. **Select roles** (check "user" and "assistant")
4. Click on a row
5. Click "Extract Properties"

### 4. Check Console Logs

You should see these messages in order:

#### ✅ Step 1: Component builds request body

```javascript
[PropertyExtraction] Making API call with: {
  method: "single_model",
  enableRoleExtraction: true,           // ← Should be true
  isUsingDemoData: false,               // ← Should be false
  selectedRoles: ["user", "assistant"], // ← Your selected roles
  extract_by_role: ["user", "assistant"], // ← THIS IS KEY!
  use_dynamic_role_prompts: false
}
```

**If `extract_by_role` is `undefined` here:**
- Check `enableRoleExtraction` - is it true?
- Check `isUsingDemoData` - is it false? (demo data disables role extraction)
- Check `selectedRoles` - is it an array with values?

#### ✅ Step 2: API function receives parameters

```javascript
🔍 [api.ts] extractSingle called with: {
  has_extract_by_role: true,
  extract_by_role: ["user", "assistant"], // ← Should be here!
  use_dynamic_role_prompts: false
}
```

**If `extract_by_role` is `undefined` here:**
- The parameter got lost between the component and the API function
- This shouldn't happen with the fix, but if it does, it's a TypeScript/build issue

### 5. Check Network Tab

In browser DevTools:
1. Go to **Network** tab
2. Filter by **Fetch/XHR**
3. Find the `/extract/single` request
4. Click on it
5. Go to **Payload** or **Request** tab

You should see:

```json
{
  "row": {...},
  "method": "single_model",
  "extract_by_role": ["user", "assistant"],  // ← MUST BE HERE!
  "use_dynamic_role_prompts": false,
  ...
}
```

**If `extract_by_role` is MISSING from the request payload:**
- The frontend build didn't include the fix
- Try clearing browser cache and hard refreshing
- Check that you rebuilt the frontend with `npm run build`

### 6. Check Backend Logs

In your terminal where backend is running, you should see:

```
INFO: Running multi-pass extraction for roles: ['user', 'assistant']
INFO: Extracting properties for role 'user' (1/2)
INFO: Extracting properties for role 'assistant' (2/2)
INFO: 🔍 Role assignment debug:
INFO:    - role from LLM response: None
INFO:    - role from original property: user
INFO:    - final role value: user
```

**If you see this instead:**
```
INFO: Extracting properties from 1 conversations using gpt-4o
```

Then the backend didn't receive `extract_by_role`. Check the Network tab to confirm it's in the request.

### 7. Check Properties Response

In Network tab → Response tab, search for `"role"`:

```json
{
  "properties": [
    {
      "id": "...",
      "role": "user",  // ← Should NOT be null!
      "property_description": "...",
      ...
    }
  ]
}
```

**If `"role": null`:**
- Backend received the parameter but multi-pass extraction didn't run
- Check backend logs - did it print "Running multi-pass extraction"?

### 8. Check PropertiesTab

After properties are loaded, console should show:

```javascript
[PropertiesTab] 🔍 ROLE DEBUG: {
  totalRows: 2,
  roleValues: ["user", "assistant"],  // ← Should have values!
  nonNullRoles: ["user", "assistant"],
  uniqueRoles: ["user", "assistant"]
}
```

**If `roleValues` is all `null`:**
- Properties were received without role field
- Go back and check the response in Network tab

## Common Issues

### Issue 1: `extract_by_role: undefined` in console logs

**Cause**: One of the conditions is false:
- `enableRoleExtraction` is false → Toggle not enabled
- `isUsingDemoData` is true → Using demo data (role extraction disabled for demo)
- `selectedRoles.length === 0` → No roles selected

**Fix**:
- Enable the toggle
- Use your own data (not demo)
- Select at least one role

### Issue 2: Parameter in console but not in Network request

**Cause**: Build issue or cache problem

**Fix**:
- Clear browser cache
- Hard refresh (Ctrl+Shift+F5)
- Verify frontend was rebuilt: check timestamp of `stringsight/frontend_dist/index.html`

### Issue 3: Parameter in Network request but backend doesn't see it

**Cause**: Backend hasn't been restarted with the schema fix

**Fix**:
- Restart backend
- Check that backend code has the fixes in `stringsight/schemas.py`

## What to Report

Please share:

1. **Console logs** - all three log messages above
2. **Network tab** - the request payload
3. **Backend logs** - what prints during extraction
4. **Settings**:
   - Is toggle enabled?
   - Using demo data or your own?
   - Which roles selected?

This will help me pinpoint exactly where it's breaking!
