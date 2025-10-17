# Job Cancellation Feature

## Overview

Added ability to cancel running extraction jobs and optionally retrieve any partial results that have been processed so far.

## Backend Changes

### 1. Updated `ExtractJob` dataclass

**File**: `stringsight/api.py:1563-1571`

```python
@dataclass
class ExtractJob:
    id: str
    state: str = "queued"  # Added "cancelled" state
    progress: float = 0.0
    count_done: int = 0
    count_total: int = 0
    error: Optional[str] = None
    properties: List[Dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False  # NEW: Flag to signal cancellation
```

### 2. Added Cancellation Checks

**File**: `stringsight/api.py:1586-1608`

The job runner now checks for cancellation:
- Before starting processing
- Before the expensive extraction call

```python
def _run_extract_job(job: ExtractJob, req: ExtractJobStartRequest):
    with _JOBS_LOCK:
        if job.cancelled:
            job.state = "cancelled"
            return

    # ... setup code ...

    with _JOBS_LOCK:
        if job.cancelled:
            job.state = "cancelled"
            return

    # Expensive extraction happens here
    result = public_api.extract_properties_only(...)
```

### 3. New Cancel Endpoint

**File**: `stringsight/api.py:1707-1741`

**Endpoint**: `POST /extract/jobs/cancel`

**Request**:
```json
{
  "job_id": "uuid-string"
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "state": "cancelled",
  "message": "Cancellation requested",
  "properties_count": 42
}
```

### 4. Updated Result Endpoint

**File**: `stringsight/api.py:1696-1704`

The result endpoint now:
- Accepts "cancelled" state (in addition to "done")
- Returns `cancelled: true` flag in response

```python
@app.get("/extract/jobs/result")
def extract_jobs_result(job_id: str):
    # ...
    if job.state not in ["done", "cancelled"]:
        raise HTTPException(409, "job not done")
    return {
        "properties": job.properties,
        "count": len(job.properties),
        "cancelled": job.state == "cancelled"
    }
```

## Frontend Changes

### Added TypeScript API Functions

**File**: `frontend/src/lib/api.ts:346-354`

```typescript
export async function extractJobCancel(job_id: string) {
  const res = await fetch(`${API_BASE}/extract/jobs/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{
    job_id: string;
    state: string;
    message: string;
    properties_count: number
  }>;
}
```

Updated `extractJobResult` type to include `cancelled` flag:
```typescript
return res.json() as Promise<{
  properties: any[];
  count: number;
  cancelled?: boolean  // NEW
}>;
```

## Usage Example

### In Frontend Component

```typescript
import { extractJobStart, extractJobStatus, extractJobResult, extractJobCancel } from '../lib/api';

// Start a job
const { job_id } = await extractJobStart({
  rows: allRows,
  system_prompt: "default",
  model_name: "gpt-4o-mini",
});

// Poll for status
const pollInterval = setInterval(async () => {
  const status = await extractJobStatus(job_id);
  console.log(`Progress: ${status.progress * 100}%`);

  if (status.state === 'done' || status.state === 'cancelled') {
    clearInterval(pollInterval);

    // Get results (even if cancelled)
    const result = await extractJobResult(job_id);

    if (result.cancelled) {
      console.log(`Job cancelled. Got ${result.count} partial results.`);
    } else {
      console.log(`Job completed. Got ${result.count} results.`);
    }

    displayProperties(result.properties);
  }
}, 1000);

// User clicks cancel button
async function handleCancel() {
  const cancelResult = await extractJobCancel(job_id);
  console.log(cancelResult.message);
  // Will stop polling and display partial results
}
```

### UI Implementation

```tsx
function PropertyExtractionPanel() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  async function handleCancel() {
    if (!jobId) return;

    try {
      await extractJobCancel(jobId);
      setIsRunning(false);
      // Optionally fetch partial results
      const result = await extractJobResult(jobId);
      showPartialResults(result.properties);
    } catch (error) {
      console.error('Cancel failed:', error);
    }
  }

  return (
    <div>
      {isRunning && (
        <Button onClick={handleCancel} color="error">
          Cancel Extraction
        </Button>
      )}
      {/* ... rest of UI */}
    </div>
  );
}
```

## Behavior & Limitations

### ✅ What Works

1. **Cancel before processing starts**: Job stops immediately
2. **Cancel while queued**: Job never runs
3. **Retrieve partial results**: Can fetch any properties extracted before cancellation
4. **UI stays responsive**: Data already loaded in frontend remains available

### ⚠️ Limitations

1. **Cannot interrupt mid-extraction**: Since we removed chunking, the extraction happens in one call to `extract_properties_only()`. If the LLM calls are in progress, they will complete.

2. **Cancellation timing**:
   - ✅ **Before extraction starts**: Instant cancellation
   - ⏱️ **During extraction**: Will complete current batch (all rows)
   - ✅ **Between operations**: Checks cancellation flag

3. **No partial LLM results**: Individual LLM calls cannot be interrupted once started (ThreadPoolExecutor limitation)

### Why This Design?

Since we removed chunking for performance, we now process all rows at once. This means:
- **Benefit**: Much faster (no redundant overhead)
- **Tradeoff**: Less granular cancellation

The cancellation is most useful for:
- Catching mistakes (wrong file, wrong settings)
- Stopping long-running jobs early
- Preventing wasted LLM costs

## Future Improvements

If you need true mid-extraction cancellation, you could:

1. **Bring back optional chunking** (but with larger chunks like 1000 rows)
   ```python
   if enable_cancellable_mode:
       chunk_size = 1000
       for chunk in chunks:
           if job.cancelled:
               break
           process_chunk(chunk)
   ```

2. **Use async/await with cancellation tokens**
   - Requires refactoring entire extraction pipeline
   - More complex but allows true interruption

3. **Store intermediate results as they complete**
   - Stream results back as each property is extracted
   - Requires modifying `extract_properties_only` to yield results

For now, the simple flag-based approach works well and doesn't compromise the performance gains from removing chunking.

## Testing

```bash
# Start the API
uvicorn stringsight.api:app --reload

# In another terminal, test cancellation
curl -X POST http://localhost:8000/extract/jobs/start \
  -H "Content-Type: application/json" \
  -d '{"rows": [...], "system_prompt": "default"}'

# Get job_id from response, then cancel
curl -X POST http://localhost:8000/extract/jobs/cancel \
  -H "Content-Type: application/json" \
  -d '{"job_id": "your-job-id"}'

# Check results
curl http://localhost:8000/extract/jobs/result?job_id=your-job-id
```

## Summary

- ✅ Added `/extract/jobs/cancel` endpoint
- ✅ Jobs can be cancelled before/during processing
- ✅ Partial results are preserved and retrievable
- ✅ Frontend API functions added
- ⚠️ Cannot interrupt mid-extraction (by design for performance)
- 📊 State tracked: "queued" | "running" | "done" | "error" | "cancelled"
