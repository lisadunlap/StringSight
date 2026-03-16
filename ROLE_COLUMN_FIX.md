# Role Column Always Visible in Properties Table

## Changes Made

Modified `/home/lisabdunlap/stringsight-frontend/src/components/PropertiesTab.tsx` to always show the `role` column in the properties table, even when it's empty.

### Change 1: Always include 'role' in allKeys

```typescript
// Line ~113-117
const allKeys = new Set<string>();
enrichedRows.forEach(row => {
  Object.keys(row).forEach(key => allKeys.add(key));
});

// Always include role column for role-based extraction (even if not present in any row yet)
allKeys.add('role');

const allKeysArray = Array.from(allKeys);
```

### Change 2: Don't filter out 'role' even if empty

```typescript
// Line ~121-122
// Columns to always show even if empty
const alwaysShowColumns = new Set(['role']); // Always show role column for role-based extraction

// Then in the filter logic (~124):
// Always include certain columns even if empty
if (alwaysShowColumns.has(col)) {
  console.log(`[PropertiesTab] Including column: ${col} (in always-show list)`);
  return true;
}
```

## Result

The `role` column will now:
1. ✅ Always appear in the properties table
2. ✅ Show `null` or empty values when role-based extraction is not used
3. ✅ Show actual role values when role-based extraction is enabled

## Column Order

The role column appears in this position:
1. question_id
2. property_description
3. model_response
4. model
5. **role** ← Here
6. (other columns alphabetically)
7. reason (last)

## Testing

1. Refresh the frontend in your browser
2. Extract some properties (with or without role-based extraction)
3. Go to the Properties tab
4. You should now see the `role` column, even if all values are empty/null

## How to Use Role-Based Extraction

To populate the `role` column with actual values:

1. In the extraction panel, enable "Extract by Role" toggle
2. Select roles to extract (e.g., "user", "assistant", "tool")
3. Extract properties
4. The `role` column will now show which role each property describes

## Why This Change?

Previously, the UI would hide columns where all values were null/empty. This made sense for most columns, but for `role`, users need to see the column even when empty so they:
- Know that role-based extraction exists
- Can see which properties have roles when using mixed extraction
- Have a consistent column layout
