# Data Structure Analysis: File Upload vs Results Loading

## Overview
The frontend has TWO paths for loading data:
1. **File Upload Path**: Upload CSV/JSON → Extract Properties → Cluster → View Results
2. **Results Loading Path**: Load pre-processed results from backend API

**Problem**: These two paths expect different data structures, causing errors in the Data, Properties, and Clusters tabs.

---

## Data Tab (DataTable Component)

### Expected Input
```typescript
{
  rows: Record<string, any>[];  // Array of flat objects
  columns: string[];            // Column names
  responseKeys: string[];       // Keys that contain model responses (for "view" button)
}
```

### File Upload Path
When uploading a file and running extraction:
```javascript
// From parseFile() → uploaded CSV/JSON
rows = [
  {
    __index: 0,
    question_id: "1",
    prompt: "user question",
    model: "gpt-4",
    model_response: "assistant response text",  // STRING
    score: { accuracy: 0.9 }
  },
  // ...
]
```
- `model_response`: **Plain string** containing the assistant's response
- Scores are **objects** `{ metric: value }`

### Results Loading Path (Server)
When loading from `results/` directory:
```javascript
// From resultsLoad() API → backend reads clustered_results_lightweight.jsonl
conversations = [
  {
    question_id: "352",
    prompt: "user question",
    model: "xai/grok-3-mini-beta",
    responses: "[{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]",  // STRING (Python repr)
    scores: { ifeval_strict_accuracy: 1.0 },
    meta: {}
  }
]

// Transformed to operational rows in App.tsx:
operational = [
  {
    __index: 0,
    question_id: "352",
    prompt: "user question",
    model: "xai/grok-3-mini-beta",
    model_response: "[{'role': 'user', ...}, {'role': 'assistant', ...}]",  // STRING (Python repr)
    score: { ifeval_strict_accuracy: 1.0 }
  }
]
```
- `model_response`: **String representation of Python list** with single quotes
- Not valid JSON - requires special parsing

**Issue**: `pickSingleResponse(c)` extracts `c.responses`, which is the Python string repr, not a parsed array.

---

## Properties Tab (PropertiesTab Component)

### Expected Input
```typescript
{
  rows: any[];  // Properties with all extracted fields
  originalData?: any[];  // Original dataset for lookups
  onOpenProperty: (prop: any) => void;
}
```

### File Upload Path
Properties extracted via OpenAI API:
```javascript
properties = [
  {
    id: "uuid-1",
    question_id: "1",
    model: "gpt-4",
    property_description: "Model provides detailed explanation",
    category: "Helpfulness",
    reason: "...",
    evidence: "...",
    behavior_type: "Positive",
    raw_response: "{...}",  // Raw LLM JSON
    row_index: 0  // Link to conversation
  }
]
```

### Results Loading Path
Properties loaded from `parsed_properties.jsonl`:
```javascript
properties = [
  {
    id: "67891533-db42-45e0-bde2-fe7e1840b4a2",
    question_id: "352",
    model: "xai/grok-3-mini-beta",
    property_description: "Fulfills the user's length constraint...",
    category: "User Experience",
    reason: "...",
    evidence: "...",
    behavior_type: "Positive",
    raw_response: "None",  // STRING "None", not null
    contains_errors: "False",  // STRING "False", not boolean
    unexpected_behavior: "False",  // STRING
    meta: "{}",  // STRING "{}", not object
    property_description_cluster_label: "Strictly follows...",
    property_description_cluster_id: 1,
    row_index: 0  // Added by frontend
  }
]
```

**Issues**:
1. Boolean fields are **strings** `"True"/"False"` instead of booleans
2. `raw_response` is string `"None"` instead of `null`
3. `meta` is string `"{}"` instead of object `{}`
4. Has cluster assignment fields that file upload path doesn't have initially

---

## Clusters Tab (ClustersTab Component)

### Expected Input
```typescript
{
  clusters: Array<{
    id: string;
    label: string;
    size: number;
    property_descriptions: string[];
    property_ids: string[];
    question_ids: string[];
    meta: {
      proportion_by_model?: Record<string, number>;
      quality_by_model?: Record<string, { avg: number; std: number }>;
      delta_by_model?: Record<string, number>;
      group?: string;
    };
  }>;
  totalConversationsByModel?: Record<string, number>;
  totalUniqueConversations?: number;
  getPropertiesRows?: () => any[];
}
```

### File Upload Path
Clusters created after running clustering:
```javascript
clusters = [
  {
    id: "1",
    label: "Detailed explanations with examples",
    size: 45,
    property_descriptions: ["Provides examples", "Uses clear language", ...],
    property_ids: ["uuid-1", "uuid-2", ...],
    question_ids: ["1", "2", ...],
    meta: {
      proportion_by_model: { "gpt-4": 0.6, "claude": 0.4 },
      quality_by_model: { "gpt-4": { avg: 0.85, std: 0.1 } },
      delta_by_model: { "gpt-4": 0.15 }
    }
  }
]
```

### Results Loading Path
Clusters loaded from backend, then enriched via `recomputeClusterMetrics()`:
```javascript
// Initial load from data.clusters
clusters = [
  {
    id: "1",
    label: "Strictly follows user instructions",
    size: 120,
    property_descriptions: [...],
    property_ids: [...],
    question_ids: [...],
    meta: {}  // Empty initially
  }
]

// After recomputeClusterMetrics()
clusters = [
  {
    id: "1",
    label: "Strictly follows user instructions",
    size: 120,
    property_descriptions: [...],
    property_ids: [...],
    question_ids: [...],
    meta: {
      proportion_by_model: { ... },
      quality_by_model: { ... },
      delta_by_model: { ... }
    }
  }
]
```

**Issue**: The enrichment depends on properties having correct `row_index` links to conversations, which breaks if conversation indexing is wrong.

---

## Root Cause Issues

### 1. **Conversation Response Format Mismatch**

**File Upload**:
- `model_response` is a **string** (the actual assistant response)

**Results Load**:
- `model_response` is a **string representation of a Python list** with OAI format
- Example: `"[{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]"`

**Why**:
- Backend converts responses to OAI format in memory (Python lists)
- Pandas `.to_json()` converts Python objects to string repr instead of JSON
- See [stringsight/clusterers/clustering_utils.py:554-562](../stringsight/clusterers/clustering_utils.py#L554-L562)

### 2. **Type Coercion in Saved Files**

**Properties file** has type issues:
- Booleans → strings: `"True"`, `"False"`
- None → string: `"None"`
- Dicts → strings: `"{}"`

**Why**:
- Same root cause: [clustering_utils.py:562](../stringsight/clusterers/clustering_utils.py#L562)
- Converts all non-score object columns to strings: `df[col] = df[col].astype(str)`

### 3. **Missing Data Structure Normalization**

**Frontend parsing** (`ensureOpenAIFormat`):
- Tries to parse Python-style strings by replacing `'` with `"`
- Fails when content has apostrophes or escaped characters
- Doesn't handle all edge cases

---

## Proposed Solutions

### Option 1: Fix Backend Serialization (Recommended)
**Change**: Don't convert complex objects to strings in `clustering_utils.py`
- Preserve `model_response` as JSON array
- Preserve booleans and null values
- Use proper JSON serialization for complex fields

**Files to change**:
- `stringsight/clusterers/clustering_utils.py` lines 554-562
- `stringsight/core/data_objects.py` - ensure `to_dict()` produces JSON-compatible output

### Option 2: Fix Frontend Parsing (Workaround)
**Change**: Better handle Python string representations in frontend
- Use `ast.literal_eval()` equivalent in JS (risky)
- OR: More robust quote replacement that handles escaped chars
- Add type coercion for boolean strings `"True"` → `true`

**Files to change**:
- `frontend/src/lib/traces.ts` - improve `ensureOpenAIFormat()`
- Add property field type coercion when loading from results

### Option 3: Separate Code Paths (Not Recommended)
**Change**: Maintain completely separate logic for file upload vs results load
- Increases complexity
- Harder to maintain
- Doesn't solve root cause

---

## Immediate Actions Needed

1. **Verify the data flow**:
   - Check what `resultsLoad()` API actually returns
   - Console log the conversations, properties, clusters structures

2. **Fix conversation parsing**:
   - Ensure `model_response` can be parsed as OAI format array
   - Test with actual data from `clustered_results_lightweight.jsonl`

3. **Fix property type coercion**:
   - Convert string booleans to actual booleans
   - Convert string `"None"` to `null`
   - Parse string dicts `"{}"` to objects

4. **Test both paths**:
   - Upload file → extract → cluster → verify all tabs work
   - Load results → verify all tabs work
   - Ensure identical UX regardless of path
