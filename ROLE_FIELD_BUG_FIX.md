# Role Field Bug Fix

## Problem

The `role` column was showing in the Properties table but was always empty, even when using role-based extraction.

## Root Cause

There was a bug in the JSON parser that was overwriting the role field:

1. **During extraction** (`stringsight/extractors/openai.py:534`):
   - Multi-pass extraction correctly sets `role="user"` (or whatever role) on the Property object

2. **During parsing** (`stringsight/postprocess/parser.py:145`):
   - Parser creates a NEW Property object from the LLM's JSON response
   - Parser tries to get role from `prop_dict.get("role")` (the LLM response)
   - But the LLM response doesn't contain a role field!
   - Result: `role=None`, overwriting the original role value

## The Fix

Modified `stringsight/postprocess/parser.py` line 145-147 to preserve the role from the original property:

```python
# BEFORE:
role=prop_dict.get("role"),  # Always None because LLM doesn't return role

# AFTER:
role=prop_dict.get("role") or prop.role,  # Preserve from original property
```

Now the role field is preserved through the parsing step:
- If LLM returns a role in JSON (unusual), use that
- Otherwise, use the role from the original property (set during multi-pass extraction)

## Testing

1. **Restart backend** to load the code changes:
   ```bash
   uvicorn stringsight.api:app --reload --port 8000
   ```

2. **Extract with roles**:
   - Enable "Extract by Role" toggle
   - Select roles (e.g., "user", "assistant")
   - Extract properties from a conversation

3. **Check Properties tab**:
   - The `role` column should now show the actual role values
   - Example: "user", "assistant", "tool", etc.

## Files Modified

1. **`stringsight/postprocess/parser.py`** - Preserve role field during parsing
2. **`~/stringsight-frontend/src/components/PropertiesTab.tsx`** - Always show role column

## Expected Behavior

Before fix:
- ❌ Role column empty/hidden even with role-based extraction

After fix:
- ✅ Role column always visible
- ✅ Role column shows actual role values when using role-based extraction
- ✅ Role column shows null/empty when not using role-based extraction

## Implementation Details

The role field flows through these stages:

1. **Extraction** (`OpenAIExtractor._run_multi_pass_extraction`):
   ```python
   prop = Property(..., role="user")  # Set during extraction
   ```

2. **Parsing** (`LLMJsonParser._parse_single_property`):
   ```python
   new_prop = Property(..., role=prop_dict.get("role") or prop.role)  # Preserve if not in LLM response
   ```

3. **API Response** (`stringsight/routers/extraction.py`):
   ```python
   props = [p.to_dict() for p in dataset.properties]  # role field included
   ```

4. **Frontend Display** (`PropertiesTab.tsx`):
   ```typescript
   // Always show role column
   allKeys.add('role');
   alwaysShowColumns.add('role');
   ```

The fix ensures the role value set during extraction is preserved all the way to the frontend.
