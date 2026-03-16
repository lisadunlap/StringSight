# Side-by-Side Role Comparison View

## Summary

Implemented a side-by-side view in the Clusters Tab that displays role-based clusters in separate columns, making it easy to visually compare "user" behaviors vs "assistant" behaviors side-by-side.

## Implementation Details

### Frontend Changes (`~/stringsight-frontend/src/components/ClustersTab.tsx`)

1. **Role Detection** (lines 218-232):
   - Added `isRoleBasedClustering` logic that detects when clusters are grouped by role
   - Heuristic: if groups contain role-like names (user, assistant, tool, system) rather than behavior types (positive, negative, style), it's role-based clustering

2. **View Toggle** (lines 234-237):
   - Added `sideBySideView` state with localStorage persistence
   - Only appears when role-based clustering is detected

3. **UI Toggle Button** (lines 503-523):
   - Added button next to the "Decimals" dropdown
   - Shows "Side-by-Side" when active, "Standard View" when inactive
   - Only visible when `isRoleBasedClustering` is true

4. **Conditional Rendering** (lines 587-631):
   - When `isRoleBasedClustering && sideBySideView`:
     - Displays clusters in a flex container with equal-width columns
     - Each role gets its own column with a header showing the role name and cluster count
     - Clusters within each role are stacked vertically
   - Otherwise:
     - Standard single-column list view (existing behavior)

5. **Refactored Cluster Rendering** (lines 633-826):
   - Extracted cluster card rendering into `renderCluster()` helper function
   - Avoids code duplication between side-by-side and standard views
   - Maintains all existing functionality (model bars, tooltips, metrics, animations)

## User Experience

When role-based extraction is enabled and clustering groups by role:

1. A new "Standard View" button appears in the Clusters Tab toolbar
2. Click it to toggle to "Side-by-Side" view
3. The view shows columns side-by-side (e.g., "user" | "assistant")
4. Each column has:
   - Clear header with role name
   - Count of clusters for that role
   - All clusters for that role stacked vertically
5. All filters (model, group, search, sort) work in both views
6. View preference is saved to localStorage

## Answering the Core Question

This view directly answers: **"What types of user input lead to poor metrics/behaviors?"**

By placing user behaviors and assistant behaviors side-by-side, you can:
- Quickly scan user prompt patterns on the left
- See corresponding assistant response patterns on the right
- Identify correlations between user inputs and problematic responses
- Compare metrics across roles visually

The model bars and proportion data remain visible in both views, so you can still see which models exhibit each behavior most frequently.

## Next Steps (Optional Enhancements)

If you want to go further, consider:
1. **Cross-linking**: Click a user cluster to highlight related assistant clusters (requires correlation analysis)
2. **Metrics column**: Add a third column showing win rates or quality scores by user prompt type
3. **Role-specific sorting**: Allow sorting each role column independently
4. **Correlation indicators**: Visual lines or arrows connecting correlated user/assistant clusters
