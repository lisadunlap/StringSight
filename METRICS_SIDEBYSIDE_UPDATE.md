# Metrics Tab Side-by-Side View Update

## Problem
The user reported that in the Metrics tab, user and assistant clusters were mixed together in the same list, making it hard to visually separate them and see role-specific patterns.

## Solution
Added a side-by-side view option to the "Top Clusters by Model" section of the Metrics tab, matching the approach used in the Clusters tab.

## Implementation

### File Modified
`~/stringsight-frontend/src/components/metrics/TopClustersSummary.tsx`

### Changes Made

1. **Added Imports** (lines 1-21):
   - Added `Button` and `Chip` from MUI

2. **Role Detection** (lines 50-79):
   - Added `allGroups` extraction from cluster metadata
   - Added `isRoleBasedClustering` detection logic (same as Clusters tab)
   - Added `sideBySideView` state with localStorage persistence

3. **View Toggle Button** (lines 107-129):
   - Added "Side-by-Side" / "Standard" toggle button
   - Only visible when role-based clustering is detected
   - Button appears next to the existing Sort dropdown
   - Preference saved to localStorage: `stringsight.metricsSideBySideView`

4. **Conditional Rendering** (lines 143-225):
   - **Side-by-Side View**: Creates columns for each role
     - Each role gets its own section with header
     - Models grouped within each role column
     - Sorting and filtering work within each role
   - **Standard View**: Original single-list layout (unchanged)

5. **Refactored Cluster Row Rendering** (lines 232-327):
   - Extracted `renderClusterRow()` helper function
   - Avoids code duplication between both views
   - Preserves all existing functionality (tooltips, metrics, navigation)

## User Experience

When viewing the Metrics tab with role-based extraction:

### Before
- All clusters mixed in a single list per model
- Role shown inline as "(user)" or "(assistant)"
- Hard to visually separate roles

### After
1. **Toggle Button Appears**: "Standard" button in toolbar (next to Sort)
2. **Click to Switch**: Button changes to "Side-by-Side"
3. **Side-by-Side Layout**:
   - Separate columns for each role (e.g., "user" | "assistant")
   - Clear header showing role name and cluster count
   - Models grouped within each role
   - Easier visual comparison

## Where Side-by-Side View Now Appears

1. **Clusters Tab** ✓ (implemented earlier)
   - Full cluster cards in columns
   - Shows all cluster details side-by-side

2. **Metrics Tab - Top Clusters Section** ✓ (just implemented)
   - Condensed cluster lists in columns
   - Focuses on frequency and deltas per role

3. **Other Metrics Sections**: Not affected
   - Benchmark table: filtered by Groups filter
   - Cluster plots: filtered by Groups filter
   - Model cards: filtered by Groups filter

## Build Status

- ✓ Frontend built successfully
- ✓ Build copied to `StringSightNew/stringsight/frontend_dist/`
- ✓ Verified `metricsSideBySideView` code in bundle
- ✓ Ready to use immediately

## Testing

1. Enable role-based extraction in your analysis
2. Go to Metrics tab
3. Scroll to "Top Clusters by Model" section
4. Look for "Standard" button next to Sort dropdown
5. Click to toggle to "Side-by-Side" view
6. Should see columns for each role with clear separation

## Next Steps (Optional)

If you want even more role separation:
- Could add side-by-side to cluster plots
- Could add role-specific benchmark tables
- Could add correlation view linking user→assistant clusters
