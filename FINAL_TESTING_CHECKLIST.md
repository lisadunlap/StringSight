# Final Testing Checklist - Role Column Fix

## What I Just Did

1. ✅ Fixed backend schema to accept `extract_by_role` parameter
2. ✅ Fixed backend endpoint to pass the parameter through
3. ✅ Fixed frontend API function signature to include the parameter
4. ✅ Added comprehensive console logging at every step
5. ✅ Rebuilt frontend (`~/stringsight-frontend/dist/`)
6. ✅ **Copied built frontend to backend** (`stringsight/frontend_dist/`)

## You Need to Do This Now

### Step 1: Restart Backend

```bash
# Kill old backend
pkill -f "uvicorn.*stringsight"

# Start fresh (to load schema changes)
cd /home/lisabdunlap/StringSightNew
uvicorn stringsight.api:app --reload --port 8000
```

### Step 2: Access the App

Open browser to: **http://localhost:8000**

(NOT port 5180 - use the backend's port so it serves the updated frontend)

### Step 3: Hard Refresh

Press: **Ctrl+Shift+R** (or **Cmd+Shift+R** on Mac)

This clears the browser cache and loads the new frontend files.

### Step 4: Open Browser Console

Press **F12** → Go to **Console** tab

### Step 5: Test Role-Based Extraction

1. **Upload YOUR data** (NOT demo data - demo disables role extraction)
2. **Enable "Extract by Role" toggle** (in the extraction panel)
3. **Select roles** - check "user" and "assistant"
4. Click on a row to view its trace
5. Click **"Extract Properties"**

### Step 6: Watch Console Logs

You should see these messages (in order):

```javascript
// 1. Component building request
[PropertyExtraction] Making API call with: {
  enableRoleExtraction: true,
  isUsingDemoData: false,
  selectedRoles: ["user", "assistant"],
  extract_by_role: ["user", "assistant"],  // ✅ Should be here!
  ...
}

// 2. API function called
🔍 [api.ts] extractSingle called with: {
  has_extract_by_role: true,
  extract_by_role: ["user", "assistant"],  // ✅ Should be here!
  use_dynamic_role_prompts: false
}
```

### Step 7: Check Network Tab

In DevTools → **Network** tab:
1. Filter by **Fetch/XHR**
2. Find the `/extract/single` request
3. Click on it → **Payload** tab

Should show:
```json
{
  "extract_by_role": ["user", "assistant"],  // ✅ Should be here!
  ...
}
```

### Step 8: Check Backend Logs

In your terminal, you should see:

```
INFO: Running multi-pass extraction for roles: ['user', 'assistant']
INFO: Extracting properties for role 'user' (1/2)
INFO: Extracting properties for role 'assistant' (2/2)
INFO: 🔍 Role assignment debug:
INFO:    - role from original property: user
INFO:    - final role value: user
```

### Step 9: Check Properties Tab

After extraction completes:
1. Go to **Properties** tab
2. Look at the **role** column
3. It should show "user", "assistant", etc. - NOT empty!

### Step 10: Check Console Again

Should see:
```javascript
[PropertiesTab] 🔍 ROLE DEBUG: {
  totalRows: 2,
  roleValues: ["user", "assistant"],  // ✅ Should have values!
  nonNullRoles: ["user", "assistant"]
}
```

## If It Still Doesn't Work

Please share:
1. **All console logs** (copy from browser console)
2. **Network request payload** (copy from Network tab)
3. **Backend terminal output** (copy from terminal)
4. **Which data are you using?** (demo or your own?)
5. **Is the toggle enabled?** (screenshot if possible)

## Quick Troubleshooting

**Q: Still seeing empty role column?**
- Check console logs - where does `extract_by_role` become undefined?
- Are you using demo data? (role extraction disabled for demo)
- Is the toggle actually enabled?

**Q: Console doesn't show the new log messages?**
- Did you hard refresh? (Ctrl+Shift+R)
- Are you accessing http://localhost:8000? (not 5180)
- Check browser cache settings

**Q: Backend logs don't show "multi-pass extraction"?**
- Check Network tab - is `extract_by_role` in the request?
- Did you restart the backend?

## Important Notes

- ✅ All code changes are in place
- ✅ Frontend is built and copied to backend
- ✅ You just need to restart backend and hard refresh browser
- ✅ Use http://localhost:8000 (backend serves the frontend)
- ✅ DO NOT use demo data (it disables role extraction)

The fix should work now! The console logs will tell us exactly what's happening. 🎯
