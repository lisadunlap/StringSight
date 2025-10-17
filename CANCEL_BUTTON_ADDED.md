# Cancel Button Added to UI

## What Was Added

Added a "Cancel Extraction" button to the Property Extraction Panel that appears during batch extraction jobs.

## Changes Made

### Frontend: PropertyExtractionPanel.tsx

**File**: `frontend/src/components/sidebar-sections/PropertyExtractionPanel.tsx`

#### 1. Imported cancel function (line 15)
```typescript
import { extractJobCancel } from '../../lib/api';
```

#### 2. Added cancel handler (lines 266-274)
```typescript
async function handleCancelJob() {
  if (!jobId) return;
  try {
    await extractJobCancel(jobId);
    setJobState('cancelled');
  } catch (e: any) {
    setErrorMsg(`Failed to cancel: ${String(e?.message || e)}`);
  }
}
```

#### 3. Updated polling to handle cancelled state (lines 244-250)
```typescript
else if (s.state === 'cancelled') {
  clearInterval(t);
  const r = await extractJobResult(startRes.job_id);
  (onBatchLoaded as any)?.(r.properties || []);
  setErrorMsg(`Job cancelled. Retrieved ${r.properties?.length || 0} partial results.`);
  onBatchDone?.();
  resolve();
}
```

#### 4. Added cancel button to UI (lines 454-465)
```typescript
{/* Cancel button for batch jobs */}
{jobId && jobState && !['done', 'error', 'cancelled'].includes(jobState) && (
  <Button
    size="small"
    variant="outlined"
    color="error"
    onClick={handleCancelJob}
    sx={{ mt: 1, width: '100%' }}
  >
    Cancel Extraction
  </Button>
)}
```

## UI Behavior

### When Button Appears
The cancel button shows up when:
- ✅ A batch extraction job is running (`jobId` exists)
- ✅ Job is in `queued` or `running` state
- ❌ Does NOT show when job is `done`, `error`, or `cancelled`

### Button Location
The cancel button appears:
- Below the progress bar
- In the progress indicator box
- Full width, small size
- Red/error color (outlined variant)

### What Happens When Clicked

1. **Sends cancel request** to backend via `extractJobCancel(jobId)`
2. **Sets job state** to "cancelled"
3. **Polling continues** until backend confirms cancellation
4. **Retrieves partial results** if any were completed
5. **Displays message**: "Job cancelled. Retrieved N partial results."
6. **Loads partial results** into the UI (same as successful completion)

## User Experience

### Successful Cancellation Flow

```
User clicks "Run on all traces"
    ↓
Progress bar appears with "Batch: running • 0%"
    ↓
Cancel button appears below progress bar
    ↓
User clicks "Cancel Extraction"
    ↓
Status changes to "Batch: cancelled • N%"
    ↓
Message appears: "Job cancelled. Retrieved X partial results."
    ↓
Partial results are loaded and displayed
```

### Visual States

**Running**:
```
┌─────────────────────────────────┐
│ Batch: running • 45%            │
│ ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░           │
│ ┌─────────────────────────────┐ │
│ │   Cancel Extraction         │ │ ← Red button
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
```

**Cancelled**:
```
┌──────────────────────────────────────────────────┐
│ ⚠️ Job cancelled. Retrieved 42 partial results.  │
└──────────────────────────────────────────────────┘
```

## Testing

1. **Start a batch extraction** on a large dataset
2. **Wait a moment** for progress to start
3. **Click "Cancel Extraction"** button
4. **Verify**:
   - Button disappears
   - Status changes to "cancelled"
   - Partial results are loaded
   - Error message shows count of partial results

## Known Behavior

- **Cannot interrupt mid-extraction**: Due to removed chunking (for performance), if extraction has started, it will complete the current batch
- **Cancel before extraction**: If clicked while "queued", cancels immediately
- **Partial results preserved**: Any completed properties are returned
- **UI stays loaded**: Data in frontend doesn't reset, just stops processing

## Code Location Summary

- **Backend API**: `stringsight/api.py:1707-1741` (cancel endpoint)
- **Frontend API**: `frontend/src/lib/api.ts:346-354` (cancel function)
- **UI Component**: `frontend/src/components/sidebar-sections/PropertyExtractionPanel.tsx:266-274, 455-465`

## Complete!

The cancel button is now fully functional in the Property Extraction Panel. It will appear automatically when batch extraction jobs are running and allows users to stop the process and retrieve any partial results.
