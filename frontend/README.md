# StringSight React Frontend

Modern React evaluation console for loading, filtering, sorting, and analyzing evaluation datasets with conversation traces.

## Prerequisites

- Node 20+
- Python 3.8+

## Backend API

From the repo root:

```bash
python3 -m pip install "uvicorn[standard]" fastapi pandas python-multipart
python3 -m uvicorn stringsight.api:app --reload --host localhost --port 8000
```

Check: `curl http://127.0.0.1:8000/health` → `{ "ok": true }`.

## Loading Prior Results (precomputed pipeline outputs)

You can load previously computed results into the UI without changing any backend output formats. The frontend expects the following files produced by `stringsight.public` entry points:

- Required: `full_dataset.json` (from `explain()`, `label()`, or `compute_metrics_only()`), which serializes a `PropertyDataset`
- Optional (for future Metrics tab): `model_cluster_scores.json`, `cluster_scores.json`, `model_scores.json` (from `FunctionalMetrics`)

### What the UI reads from `full_dataset.json`

`full_dataset.json` contains a JSON serialization of `PropertyDataset` with keys:

- `conversations`: list of ConversationRecords
- `properties`: list of Properties (may be empty)
- `clusters`: list of Clusters (may be empty)
- `model_stats`: dict (may be legacy or the new functional metrics wrapper)
- `all_models`: list of model names

Mapping to the frontend is lightweight and preserves current UI behavior:

- Conversations → operational rows
  - Single model rows include: `question_id`, `prompt`, `model`, `model_response` (string or OAI messages), `score` (dict), and a local `__index` (stable row index)
  - Side-by-side rows include: `question_id`, `prompt`, `model_a`, `model_b`, `model_a_response`, `model_b_response`, `score_a`, `score_b`, optional `winner`, plus `__index`
  - The UI already accepts both string responses and OpenAI message arrays; both render in the conversation drawer
- Properties → properties table rows
  - Directly from `Property` objects: `id`, `question_id`, `model`, `property_description`, `reason`, `evidence`, `category`, `behavior_type`, etc.
  - Linking back to the base row uses `(question_id, model)`; a `row_index` may be added locally for convenience but is not required
- Clusters → clusters tab
  - Directly from `Cluster` objects: `id`, `label`, `size`, `property_descriptions`, `property_ids`, `question_ids`, `meta`
  - The UI will call the existing `POST /cluster/metrics` endpoint to enrich cluster `meta` (e.g., per‑model proportions/quality) so current charts and chips continue to work
- Metrics (optional for now)
  - The three functional metrics JSONs are stored for the future Metrics tab; no changes to the UI yet

### How to load results in the UI

Two simple approaches:

1) Browser upload (no backend change)
- Click "Load File" and select `full_dataset.json`
- The app detects a results file by the presence of a top-level `conversations` key and switches to results mode
- Optionally upload `model_cluster_scores.json`, `cluster_scores.json`, `model_scores.json` as well; the UI will simply cache them for later use

2) Server endpoint (path-based loading; enabled)
- Add a convenience endpoint to your backend (example contract):
  - `POST /results/load` with body `{ "path": "/path/to/run/" }`
  - Server reads `full_dataset.json` (and optionally three metrics JSONs) and responds with:
    ```json
    {
      "method": "single_model" | "side_by_side",
      "operationalRows": [...],
      "properties": [...],
      "clusters": [...],
      "metrics": {
        "model_cluster_scores": {...},
        "cluster_scores": {...},
        "model_scores": {...}
      }
    }
    ```
  - This endpoint does not change any backend output structures; it only repackages saved artifacts for convenience

### Selecting a results folder from the UI (server-side browsing)

If you prefer selecting a directory rather than uploading files in the browser, wire the UI to the backend file-browser endpoint and the loader endpoint:

1) Browse folders
- UI calls `POST /list-path` with `{ "path": ".", "exts": [".json", ".jsonl", ".csv"] }`
- The backend responds with directory entries rooted at a base directory; the UI presents a simple folder picker

2) Load the selected folder
- When the user picks a directory (e.g., a pipeline run output folder), UI calls `POST /results/load` with `{ "path": "/absolute/path/to/run" }`
- Backend resolves `full_dataset.json` at that path (and, if present, `model_cluster_scores.json`, `cluster_scores.json`, `model_scores.json`), returning the shaped payload described above

Security: server-side browse base directory
- Set `BASE_BROWSE_DIR` to the absolute directory users are allowed to browse. Defaults to the current working directory.
- The backend enforces that all `path` values for `/list-path`, `/read-path`, and `/results/load` resolve within `BASE_BROWSE_DIR` and rejects traversal outside it. Hidden files/dirs are omitted from listings.
  - Recommended: run the server as a non‑root user with read‑only access to `BASE_BROWSE_DIR`.

Authentication & exposure (guidance)
- For local development, permissive CORS is enabled. If exposing beyond localhost, consider a reverse proxy with TLS, restricted origins, and/or token auth headers. Prefer SSH port‑forwarding for remote servers.

### What changes in the UI when results are loaded

- The Data tab shows the same table, now backed by operational rows derived from `conversations`
- The Properties tab shows the same information using loaded `properties`
- The Clusters tab shows loaded `clusters` and then enriches `meta` via `POST /cluster/metrics` (existing API)
- Property Extraction and Clustering panels are disabled while in results mode (controls remain visible but actions are disabled)

This yields a minimal frontend delta and preserves all current rendering.

### Conversion notes and data assumptions

- `question_id` is treated as a stable key and is carried through to operational rows
- Responses may be strings or OpenAI message arrays; the trace viewer renders both
- Scores are dicts (`score` for single model; `score_a`/`score_b` for side-by-side); the display table can still flatten these for sorting/filtering
- Properties link to base rows by `(question_id, model)`; if a `row_index` is present it will be preferred for lookup
- Cluster `meta` fields expected by the Clusters tab (e.g., `proportion_by_model`, `quality_by_model`, `quality_delta_by_model`, `proportion_overall`) are computed on-demand by the existing metrics enrichment API and need not be present in saved clusters

### Metrics Tab (Overview)

The Metrics tab visualizes benchmark metrics per model and per‑model×cluster distributions (frequency and quality Δ), with global filters for model and group. It supports preloaded results using pandas‑style JSONL files when present:

- Preferred: `model_scores_df.jsonl`, `cluster_scores_df.jsonl`, `model_cluster_scores_df.jsonl`
- Backward‑compatible: `model_scores.json`, `cluster_scores.json`, `model_cluster_scores.json`

If no preloads are found, the UI calls the backend to compute metrics on demand (`POST /cluster/metrics`).

- Benchmark: multi‑metric quality view per model (optional CIs)
- Per‑cluster charts: frequency and quality Δ; top‑N selection; sorting; significance toggle when fields exist
- Top clusters per model: top 5 by proportion Δ, respecting filters

Implementation details, data contracts, and step‑by‑step guidance live in `frontend/METRICS_README.md`.

## Frontend

From `frontend/`:

```bash
# Point the UI to the API
printf "VITE_API_BASE=http://127.0.0.1:8000\n" > .env.local

# Install & run
cd frontend

# UI deps (including charts for Clusters tab)
npm i
npm i react-plotly.js plotly.js-dist-min
cd frontend
VITE_BACKEND=http://localhost:8000 npm run dev -- --host localhost --port 5180
```

Open `http://127.0.0.1:5180`.

## Core Features

### 📊 **Data Loading & Management**
- **File Upload**: Supports `.jsonl`, `.json`, `.csv` formats
- **Format Detection**: Auto-detects single-model vs side-by-side evaluation formats
- **Index Preservation**: Original dataframe indices maintained through all operations
- **Performance Optimization**: Smart loading with 1000+ row performance warnings

### 🔍 **Advanced Data Operations**
- **Operation Chain System**: Full data provenance tracking with sequential operation application
- **Multi-Column Filters**: Add/remove filters on categorical columns with negation support
- **Custom Pandas Code**: Execute arbitrary pandas expressions at any point in the chain
- **Individual Operation Removal**: Remove any operation and automatically reapply remaining chain
- **Visual Operation History**: Color-coded chain display showing exact sequence of transformations

### 📈 **Sorting & Organization**
- **Click-to-Sort**: Click any column header to sort (asc → desc → none cycle)
- **Visual Indicators**: Arrow icons show current sort direction
- **Smart Type Detection**: Automatic numeric vs string sorting
- **Performance Optimized**: Efficient sorting for large datasets

### 📊 **Groupby Analysis**
- **Dynamic Grouping**: Group by any column with summary statistics
- **Accordion View**: Expandable groups with individual row pagination
- **Statistical Previews**: Average scores displayed for numeric columns
- **Pagination**: Page through examples within each group

### 💬 **Conversation Traces**
- **Right Drawer**: Full conversation view with OpenAI message format
- **Dual Views**: Single model or side-by-side comparison modes
- **Evidence Highlighting**: Advanced text highlighting with support for:
  - Ellipses-based fragments: `"text... more text..."`
  - Comma-separated quoted lists: `'phrase one', 'phrase two', 'phrase three'`
  - Mixed quote types and smart quotes
  - Auto-scroll to highlighted evidence
- **Responsive Layout**: Adapts to different screen sizes

### 🧭 **Modular Sidebar System**
- **Permanent Icon Sidebar**: Always-visible icons for main sections (Data, Extraction, Clustering, Metrics)
- **Expandable Control Panel**: Context-sensitive controls that open based on active section
- **Collapsible Design**: Clean interface that can be expanded/collapsed as needed
- **Smart Row Selection**: Extraction automatically targets the row being viewed in trace drawer

#### **Data Statistics Panel**
- Row count, unique prompts, and model overview
- Integrated with dynamic filtering system

#### **Property Extraction Panel**  
- Prompt selection (preset/custom) with task description support
- LLM settings (model, temperature, top_p, max_tokens, max_workers, sample_size)
- Sample size parameter: passed to backend's `explain()` to sample N prompts total (leave empty to process all prompts)
- "Extract on selected" runs on currently viewed response with smart row detection
- "Run on all traces" starts async batch job with progress tracking and auto-drawer closing
- Real-time results with auto-scroll to extraction results
- Last extraction results with expandable evidence accordion

#### **Clustering Panel**
- Min cluster size, embedding model selection
- Hierarchical clustering and outlier assignment options
- "Cluster properties" button (enabled when properties exist)

### 🧱 **Clusters Tab**
- Dedicated view for property clusters returned by the backend
- Summary (collapsed row):
  - Cluster description (left)
  - Overall quality metrics (right) as multiline plain text with colored deltas
    - Green if Δ > +0.02, Red if Δ < −0.02, Grey otherwise
  - Secondary line: `size (overall proportion)` and optional Group chip
- Details (expanded row):
  - Per‑model proportions chart (Plotly bar)
  - Per‑model quality chart with a per‑cluster toggle:
    - Quality: uses `meta.quality_by_model`
    - Delta: uses `meta.quality_delta_by_model`
  - Member property descriptions with Open actions
- Header controls:
  - Global Decimals selector (1–4) that controls summary values, deltas, chart labels and axis ticks
- Uses clustering API: `POST /cluster/run` and `POST /cluster/metrics`

### 📒 Cluster Accordion Redesign

The Clusters tab uses a clean accordion list to prioritize readability and match how users reason about clusters.

- Collapsed (summary) row shows:
  - Cluster description (dominant, left-aligned)
  - Overall cluster quality (across all models) as a compact list of metrics on the right (e.g., `helpfulness: 8.31, accuracy: 8.09`)
  - A secondary chip line under the description with:
    - Size as a count with overall proportion in parentheses (e.g., `1030 (99.5%)`)
    - Optional Group tag if available in cluster metadata

- Expanded (details) panel shows:
  - Per-model proportions list (model → proportion)
  - Per-model quality metrics list (model → each metric value)
  - Existing property descriptions with an Open button (kept as-is)

The summary focuses on cluster-level metrics (size, overall proportion, overall quality). Per‑model metrics appear in the expanded section via charts.

#### **Metrics Panel**
- Placeholder for future analytics and metrics features

### 📊 Metrics DataFrames Integration

The backend functional metrics module emits three dataframes/JSONs that we will use directly in the Clusters and Metrics tabs:

- cluster_scores (per-cluster, aggregated across all models):
  - `size`: total conversations in cluster
  - `proportion`: fraction of all conversations in cluster (0–1)
  - `quality`: average quality scores across all models, as a dict `{metric: value}`
  - `quality_delta`: optional cluster vs overall deltas
  - `metadata`: includes optional `group`
  - (UI adds) `proportion_overall` derived from `size / total`

- model_cluster_scores (per model × cluster):
  - `size`: conversations for this model within this cluster
  - `proportion`: fraction of this model’s conversations in this cluster (0–1)
  - `quality`: quality metrics for this model within this cluster `{metric: value}`
  - `quality_delta`: optional relative to the model’s overall baseline
  - `proportion_delta`: optional salience relative to cross-model average
  - `metadata`, `examples`, and optional bootstrap CIs/significance flags

- model_scores (per model across all clusters):
  - Baselines for overall model quality (for deltas) and total sizes

UI Contract we will rely on:
- Clusters tab (summary): sourced from `cluster_scores`
  - label/description, `size`, `proportion`, `quality`, optional `metadata.group`
- Clusters tab (details): sourced from `model_cluster_scores`
  - per-model `proportion` and `quality` per cluster
  - per-model `quality_delta` per cluster (exposed to UI as `meta.quality_delta_by_model`)
- Metrics tab (future graphs):
  - Plot cluster-level quality metrics over clusters from `cluster_scores`
  - Plot per-model proportions within clusters from `model_cluster_scores`
  - Plot model baselines and compare deltas using `model_scores`

Formatting guidelines:
- Show proportions as percentages in the UI (e.g., `63.6%`).
- Show quality metrics to 1–4 decimals (global selector); include deltas in parentheses.
- Keep per-model details in the expanded section to avoid horizontal scrolling.

### 🧩 **Enhanced Properties Table**
- **Dynamic Columns**: Auto-detects and displays all property columns with smart ordering
- **Column Filtering**: Automatically removes empty/NaN columns and unwanted metadata
- **Expandable Text**: FormattedCell integration for long text with expand/collapse
- **Evidence Debugging**: Visible evidence column for highlighting troubleshooting
- **Data Enrichment**: Merges model responses from original dataset using `operationalRows` format
- **Advanced Filtering**: Same FilterBar system as data table with groupby and custom pandas code
- **Evidence Highlighting**: Click "View" to open trace with highlighted evidence fragments
- **Proper Score Display**: PropertyTraceHeader uses `operationalRows` to show consolidated score objects

## 📒 Planned: Conversation Normalization: Current vs Planned

### What we have today (Current State)
- Normalization lives primarily in the frontend:
  - The UI standardizes uploaded data into an operational format (columns like `prompt`, `model_response`, `model`, and a consolidated `score` dict; or side-by-side columns for A/B).
  - For conversation viewing, the UI converts prompt/response into OpenAI-style message arrays using `frontend/src/lib/traces.ts` (`ensureOpenAIFormat`).
  - When responses are already message arrays, the UI passes them through unchanged.
- The backend contains the same conversion helpers in `stringsight/core/data_objects.py` (e.g., `check_and_convert_to_oai_format`) and uses them when building `PropertyDataset`, but the frontend has been the source of truth for the UI’s operational rows.

### What we want (Target State)
- Single source of truth in the backend for raw→operational conversion and conversation normalization.
- Backend returns operational rows with OpenAI-style message arrays when available:
  - Single-model: `prompt: string`, `model: string`, `model_response: Message[] | string`, `score: Dict`.
  - Side-by-side: `prompt: string`, `model_a/b: string`, `model_a_response/model_b_response: Message[] | string`, `score_a/b: Dict`.
- The atomic UI unit is a Conversation rendered as turns:
  - ConversationViewer renders ordered message turns (`user`/`assistant`[/`system`]).
  - ResponseContent renders a single message (markdown/LaTeX/HTML when no highlights; span-based when highlighting).
  - PropertyCard wraps ConversationViewer and adds property description, chips, evidence line, etc.
  - Side-by-side (when needed) is two ConversationViewers rendered in parallel.

### Why change (Rationale)
- Single source of truth avoids drift between frontend and backend.
- Consistent behavior across extraction, clustering, metrics, and UI.
- Easier to evolve formats (e.g., multi-turn transcripts, tool calls) without duplicating logic in the UI.
- Reduces frontend complexity; keeps UI focused on rendering and interaction.

### How we plan to change (Migration Plan)
1) Backend
   - Finish the flexible-mapping pipeline in `stringsight/core/flexible_data_loader.py` and the `/process-flexible-data` endpoint to:
     - Accept raw rows and user-specified column mappings.
     - Emit standardized operational rows with consolidated score objects.
     - Normalize conversations using `check_and_convert_to_oai_format` so `*_response` columns contain message arrays (strings supported for legacy).
2) Frontend
   - Replace local normalization with a call to the backend endpoint for operational rows.
   - Properties and Clusters views pass `messages` directly to the ConversationViewer (strings still supported via fallback conversion).
   - Side-by-side: select messages for the property’s `model` when rendering a single property; optionally render both viewers for comparison.
3) Cards & Rendering
   - ResponseContent remains the per-message renderer (formatting vs highlighting toggle).
   - ConversationViewer renders turns (the atomic unit).
   - PropertyCard wraps ConversationViewer and adds property-level UI.

### UI behavior (after change)
- Data Table → “View” opens a right-drawer with a ConversationViewer:
  - Single-model: user→assistant turns + score chips, formatted content.
  - Side-by-side: two ConversationViewers (A and B), each with its own scores.
- From a Property context, the same drawer highlights evidence within assistant turns.

### Backward Compatibility
- Users may upload plain string responses; backend preprocessing converts them to OpenAI-style message arrays via `check_and_convert_to_oai_format`.
- The frontend expects `Message[]` and will log a warning if a raw string slips through (no UI-side normalization on the main path).

### Implementation Strategy: Zero-Risk Isolated Development

To ensure **complete safety** during development, we use an isolated parallel approach:

#### Phase 1: Isolated Component Development
- **New components only**: `ConversationViewer.tsx`, `PropertyCardV2.tsx`, `CardTestPageV2.tsx`
- **Zero changes** to existing components: `ConversationTrace.tsx`, `PropertyCard.tsx`, etc.
- **Isolated testing**: Dedicated test page (`/card-test-v2`) for side-by-side comparison
- **No risk** to existing functionality during development

#### Phase 2: Feature Flag Integration
```typescript
// Single constant controls rollout
const USE_ENHANCED_CARDS = false; // ← Safe default

// Or URL param for testing: ?enhanced=true
const useEnhanced = new URLSearchParams(window.location.search).get('enhanced') === 'true';
```

#### Phase 3: Surgical Integration Points
**Only 2 lines change in the entire codebase:**
```typescript
// App.tsx - right drawer conversation display
{selectedTrace?.type === "single" && (
  USE_ENHANCED_CARDS ? 
    <PropertyCardV2 conversation={...} property={selectedProperty} /> :
    <ConversationTrace messages={selectedTrace.messages} highlights={selectedEvidence} />
)}

// Similar for side-by-side
```

#### Phase 4: Gradual Rollout
- **Development**: Test with URL param `?enhanced=true`
- **Staging**: Enable feature flag for testing
- **Production**: Gradual rollout with instant rollback capability
- **Cleanup**: Remove old components only after full validation

#### Component Architecture
```
frontend/src/components/cards/
├── ConversationViewer.tsx    # ✨ NEW - atomic conversation renderer
├── PropertyCardV2.tsx        # ✨ NEW - enhanced property card
├── CardTestPageV2.tsx        # ✨ NEW - isolated testing environment

# Existing components remain untouched during development:
├── ConversationTrace.tsx     # 🔒 UNCHANGED until rollout complete
├── SideBySideTrace.tsx       # 🔒 UNCHANGED until rollout complete  
├── PropertyCard.tsx          # 🔒 UNCHANGED until rollout complete
├── ModelResponseCard.tsx     # 🔒 REUSED by new components
└── ResponseContent.tsx       # 🔒 REUSED by new components
```

This approach guarantees:
- ✅ **Zero risk** to existing functionality
- ✅ **Easy rollback** at any stage  
- ✅ **Side-by-side validation** of old vs new behavior
- ✅ **Gradual integration** when confident
- ✅ **No backend changes** required during development

### ConversationViewer Component Requirements

The new `ConversationViewer` component is the atomic conversation renderer that replaces `ConversationTrace` and `SideBySideTrace`:

#### Interface
```typescript
interface ConversationViewerProps {
  messages: Message[];              // OpenAI-style message array
  highlights?: string[];            // Evidence text to highlight
  variant?: 'default' | 'compact'; // Display density
  showRoles?: boolean;              // Show user/assistant labels
  maxHeight?: string;               // Container height limit
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}
```

#### Behavior Requirements
- **Message Rendering**: Each message uses existing `ResponseContent` component
- **Evidence Highlighting**: Reuse existing `evidenceToHighlightRanges` logic
- **Auto-scroll**: Scroll to first highlighted text when evidence provided
- **Markdown Support**: Full markdown/LaTeX rendering via ResponseContent
- **Responsive**: Adapts to container width
- **Accessibility**: Proper ARIA labels and screen reader support

#### Integration Points
- **Reuses**: `ResponseContent`, `evidenceToHighlightRanges` from existing codebase
- **Replaces**: Logic currently in `ConversationTrace` and `SideBySideTrace`
- **Enhanced by**: `PropertyCardV2` wraps it with property metadata UI

#### Testing Requirements
- **Isolated testing** in `CardTestPageV2`
- **Side-by-side comparison** with existing `ConversationTrace`
- **Evidence highlighting validation** with various text formats
- **Message format compatibility** (string fallback, OpenAI arrays)
- **Performance testing** with long conversations

## Data Loading (Flexible Column Selection)

The app supports flexible schemas. You can upload datasets with arbitrary column names and map them to the expected fields using the Column Selector.

### Workflow
- **1) Upload**: Choose a `.jsonl`, `.json`, or `.csv` file
- **2) Configure Columns**: The Column Selector opens with auto-detected suggestions. Select:
  - Prompt column
  - Response column(s) (1 for Single Model, 2 for Side-by-Side)
  - Model column(s) (optional)
  - Score column(s) (optional; can be numeric columns and/or a dict column)
- **3) Confirm**: Click Done to process and render the table

Notes:
- Auto-detection populates fields but never auto-processes; you must click Done.
- Uploading a new file always returns to the Column Selector.
 - Loading a new source (file or results folder) resets panels, tabs, operations, and selections. Results mode collapses the control panel by default; raw file uploads expand it.

### Standardization and Score Handling
When you click Done, the app builds an operational dataset with standardized column names:
- Single Model: `prompt`, `model_response`, `model`
- Side-by-Side: `prompt`, `model_a_response`, `model_b_response`, `model_a`, `model_b`

Scores are normalized as follows:
- If you selected multiple numeric score columns, they are combined into a single score dictionary (`score` for Single Model; `score_a` and/or `score_b` for Side-by-Side) using the column names as keys.
- If you selected a dict-like score column (e.g., `{ "accuracy": 0.9, "helpfulness": 4.2 }`), it is used directly as the score dictionary.
- Rows are dropped if **any selected score** is missing (NaN/empty). The UI shows a notice: “Filtered out N row(s) due to missing values in selected score columns: …”.
- For table display, score dictionaries are flattened into columns:
  - Single Model → `score_<key>` (e.g., `score_accuracy`)
  - Side-by-Side → `score_a_<key>`, `score_b_<key>`

### Response Formats
Response columns can be either:
- A simple string, or
- An OpenAI-style message list, e.g. `[{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }]`
Both formats are supported and rendered in the conversation viewer.

## Data Formats

### Single Model Evaluation
```json
{
  "prompt": "What is the capital of France?",
  "model": "gpt-4",
  "model_response": "The capital of France is Paris.",
  "score": 4.5
}
```

**Required columns**: `prompt`, `model`, `model_response`  
**Optional columns**: `score` (number or nested object)

### Side-by-Side Evaluation
```json
{
  "prompt": "What is the capital of France?",
  "model_a": "gpt-4",
  "model_b": "claude-3",
  "model_a_response": "The capital of France is Paris.",
  "model_b_response": "Paris is the capital city of France.",
  "score_a": 4.5,
  "score_b": 4.2
}
```

**Required columns**: `prompt`, `model_a`, `model_b`, `model_a_response`, `model_b_response`  
**Optional columns**: `score_a`, `score_b` (numbers or nested objects)

## Architecture

### 🏗️ **Component Structure**

```
src/
├── App.tsx                     # Main shell with data management & sidebar orchestration
├── components/
│   ├── DataTable.tsx          # Sortable table with truncation and rich content
│   ├── ConversationTrace.tsx  # Single conversation view with advanced highlighting
│   ├── SideBySideTrace.tsx    # Dual conversation comparison
│   ├── FilterSummary.tsx      # Operation chain visualization
│   ├── FilterBar.tsx          # Reusable filtering component with search/groupby/custom code
│   ├── FormattedCell.tsx      # Rich content rendering with expand/collapse (Markdown/LaTeX/HTML)
│   ├── PropertiesTab.tsx      # Enhanced properties table with dynamic columns
│   ├── PermanentIconSidebar.tsx    # Always-visible icon navigation
│   ├── ExpandedSidebar.tsx         # Collapsible control panel container
│   └── sidebar-sections/
│       ├── DataStatsPanel.tsx      # Data overview statistics
│       ├── PropertyExtractionPanel.tsx  # Property extraction controls & results
│       ├── ClusteringPanel.tsx     # Clustering configuration (placeholder)
│       └── MetricsPanel.tsx        # Metrics dashboard (placeholder)
├── lib/
│   ├── api.ts                 # Backend API calls
│   ├── parse.ts               # Client-side file parsing
│   ├── traces.ts              # Message format utilities
│   └── normalize.ts           # Score flattening
├── types/
│   └── operations.ts          # Operation chain type definitions
└── theme.ts                   # MUI theme configuration
```

### 🗃️ **Data Layer Architecture**

The app uses a **four-layer data management system** with dual operational/display formats for backend compatibility:

1. **Original Rows** (`originalRows`): Raw uploaded data, never modified
2. **Processed Rows**: Cleaned data with standardized columns + consolidated score objects
3. **Operational Rows** (`operationalRows`): **Backend-compatible format** with consolidated score objects
4. **Current Rows** (`currentRows`): **UI display format** with flattened score columns + operation chain applied

```
Original Data → [Static Ops] → Processed Data → [Format Split] → Operational Data (Backend)
                ↳ normalize                     ↳ keep scores    ↳ clustering/metrics
                ↳ add __index                   ↳ as objects     ↳ property extraction
                ↳ standardize                                    
                                                ↓ [Flatten] → Display Data (UI)
                                                ↳ score_* cols  ↳ table display
                                                ↳ [Operations]  ↳ filtering/sorting
                                                ↳ filter/sort   
```

#### **Critical Format Distinction:**

**🔧 Operational Format** (Backend/API):
```javascript
{
  question_id: "q1",
  model: "gpt-4",
  score: { 
    Helpfulness: 4, 
    Understandability: 5, 
    Conciseness: 3 
  }
}
```

**📊 Display Format** (UI Table):
```javascript
{
  question_id: "q1", 
  model: "gpt-4",
  score_Helpfulness: 4,
  score_Understandability: 5, 
  score_Conciseness: 3
}
```

This separation ensures:
- **Clustering & Metrics** get consolidated score objects for computation
- **Property Extraction** accesses proper conversation metadata with scores
- **UI Tables** display individual score columns for filtering/sorting
- **Backend APIs** receive expected consolidated format

### 🔄 Global Data-Flow Abstraction (All tabs)

- **Source of truth**: `operationalRows` always contains consolidated score objects (`score`, `score_a`, `score_b`) and the standardized columns.
- **Display format**: `currentRows` is derived from `operationalRows` by flattening scores into scalar columns via `flattenScores(...)`.
- **Operation chain application**:
  - Apply `filter` and `custom` operations to `operationalRows`.
  - Flatten to display rows for the table.
  - Apply `sort` on the flattened rows so score_* columns sort correctly.

Minimal flow (pseudocode):
```typescript
const opData = applyNonSortOps(operationalRows, operations); // filter + custom
const { rows: flattened } = flattenScores(opData, method);   // to UI columns
const display = applySort(flattened, findSort(operations));  // optional
setCurrentRows(display);
```

- **Tab consumption**:
  - Data tab (table): consumes `currentRows` (flattened). Do not pass score dicts here.
  - Properties tab: reads `operationalRows` for linking and score chips; may show flattened values in tables.
  - Clusters tab: computes/enriches using `operationalRows` (and metrics payloads), never the flattened table for computation.
  - Metrics tab: uses preloaded or computed metrics; aligns subsets with `operationalRows` when needed.

- **Reset on new source**: Uploading a new file or selecting a results folder resets active section/tab, panels, selections, and operation chain so each dataset starts clean.

#### **⚠️ Critical for Clustering & Metrics Implementation:**

When implementing clustering and metrics features, **always use `operationalRows`**:

```typescript
// ✅ CORRECT - Use operational format for backend operations
const clusteringData = operationalRows.map(row => ({
  ...row,
  scores: row.score // Consolidated object for computation
}));

// ❌ WRONG - Don't use display format for computation
const wrongData = currentRows.map(row => ({
  ...row, 
  scores: { 
    Helpfulness: row.score_Helpfulness, // Fragmented, error-prone
    Understandability: row.score_Understandability
  }
}));
```

**Key Usage:**
- **PropertyTraceHeader**: Uses `operationalRows` to show proper scores
- **Property Extraction**: Matches against `operationalRows` for metadata
- **Future Clustering**: Must use `operationalRows` for score-based computations
- **Future Metrics**: Must use `operationalRows` for consolidated score analysis

### **State Management**

**Data States:**
- `originalRows` - Immutable uploaded data
- `operationalRows` - **Backend format** with consolidated score objects (for clustering/metrics/extraction)
- `currentRows` - **UI format** with flattened score columns + operation chain applied (for table display)
- `sortedRows` - Final sorted data for rendering (legacy, now part of operation chain)

**Operation Chain:**
- `operationChain` - Array of sequential data operations with full provenance
- Each operation has unique ID, timestamp, and type (`filter` | `custom` | `sort`)
- Operations applied sequentially to **display format**: `flattenedRows → op1 → op2 → op3 → currentRows`
- **Operational format remains unchanged** for backend compatibility

**UI State:**
- `pendingColumn/Values/Negated` - UI state for building new filter operations
- Legacy filter/sort states maintained for backward compatibility

**Group States:**
- `groupBy` - Column to group by
- `groupPreview` - Summary statistics per group
- `groupPagination` - Page state for each group

### 🔗 **Operation Chain System**

The operation chain provides full data provenance and proper undo functionality:

**Operation Types:**
```typescript
interface FilterOperation {
  type: 'filter';
  column: string;
  values: string[];
  negated: boolean;
}

interface CustomCodeOperation {
  type: 'custom';
  code: string; // pandas expression
}

interface SortOperation {
  type: 'sort'; 
  column: string;
  direction: 'asc' | 'desc';
}
```

**Chain Management:**
- **Sequential Application**: Operations applied in order to ensure correct data flow
- **Individual Removal**: Remove any operation by ID and reapply remaining chain
- **Mixed Operations**: Combine filters, custom code, and sorting in any sequence
- **Visual Feedback**: Color-coded operation display with numbered sequence

**Example Chain:**
1. **Filter**: `model = "gpt-4"` → 500 rows
2. **Custom**: `df.query("score > 3")` → 234 rows  
3. **Filter**: `prompt.contains("math")` → 89 rows
4. **Sort**: `score desc` → 89 rows sorted

### **Advanced Evidence Highlighting System**

The conversation trace viewer includes sophisticated text highlighting for property evidence:

**Supported Evidence Formats:**
- **Single phrases**: `"This is important text"`
- **Ellipses fragments**: `"First part... second part... third part"`
- **Quoted lists**: `'Dear Client', 'Please provide', 'Thank you'`
- **Mixed quotes**: Supports `'`, `"`, `"`, `"` quote types

**Highlighting Logic:**
1. **Fragment Detection**: Automatically detects ellipses (`...`) and comma-separated quoted lists
2. **Smart Splitting**: Splits fragmented evidence into individual searchable phrases
3. **Flexible Matching**: Uses exact matching for short phrases, token-based matching for longer text
4. **Auto-scroll**: Automatically scrolls to first highlighted text when evidence is selected
5. **Visual Feedback**: Yellow highlighting with smooth scroll-to-view behavior

**Technical Implementation:**
- `splitEvidenceFragments()`: Handles ellipses and quoted list parsing
- `createFlexibleRegex()`: Creates optimized regex patterns for text matching
- `highlightContent()`: Applies highlighting with React elements
- Evidence preservation through trace viewer state management

### ⚡ **Performance Optimizations**

- **Memoized Components**: React.memo on expensive renders
- **Optimized Sorting**: Pre-computed type detection and efficient comparisons
- **Smart Re-renders**: Careful dependency arrays in useMemo/useCallback
- **Local-First Operations**: Client-side filtering/sorting with optional backend validation
- **Lazy Loading**: 1000-row display limit with performance warnings
- **Dynamic Column Filtering**: Auto-removes empty columns to reduce render overhead

## Extending the Frontend

### 🔌 **Adding New Column Types**

1. Update column detection logic in `App.tsx`:
```typescript
const allowedCols = [...existing, 'new_column_type'];
```

2. Add human-readable labels in `DataTable.tsx`:
```typescript
const human: Record<string, string> = {
  // existing...
  new_column_type: "NEW COLUMN"
};
```

### 📈 **Adding New Views**

1. Create component in `src/components/`
2. Add routing/state management in `App.tsx`
3. Integrate with existing data layers

### 🔍 **Custom Analysis Features**

The pandas expression feature provides a foundation for advanced analysis:

```typescript
// Example: Add clustering results
const clusteringResults = await dfCustom({
  rows: currentRows,
  code: "df.assign(cluster=kmeans_predict(df[score_cols]))"
});
```

### 🌐 **API Integration**

Key API endpoints (backend FastAPI):
- `GET /health`
- `GET /prompts` - List extractor prompts
- `GET /prompt-text?name=...&task_description=...` - Resolved prompt text
- `POST /extract/single` - Extract on one row
- `POST /extract/batch` - Synchronous batch extract
- `POST /extract/jobs/start` - Start async batch job (accepts chunk_size, max_workers)
- `GET /extract/jobs/status?job_id=...` - Poll job progress
- `GET /extract/jobs/result?job_id=...` - Fetch job results
- `POST /cluster/run` - Run clustering on existing properties (no extraction)
- `POST /cluster/metrics` - Recompute metrics for filtered cluster subsets

### 📐 Clustering & Metrics Data Contract

The clustering backend constructs a long-form dataframe for metrics computation via `prepare_long_frame` with columns:

```
conversation_id | model | cluster | property_id | property_description | scores | cluster_metadata
```

Metric computation (`compute_subset_metrics`) derives:
- Cluster size and overall proportion across all models
- Average metric scores per cluster (mean of keys inside `scores` dict)
- Per-model proportions within each cluster

These are attached back to each cluster by `enrich_clusters_with_metrics`:
- `cluster.size` (subset size)
- `cluster.meta.quality` (avg metric scores per cluster)
- `cluster.meta.quality_delta` (cluster avg minus global avg per metric)
- `cluster.meta.proportion_by_model` (per-model proportions)

Clusters payload (from `POST /cluster/run`) includes for each cluster:
```
{
  id, label, size,
  property_descriptions: string[],
  property_ids: string[],
  question_ids: string[],
  meta: {
    quality?: { [metricKey: string]: number },
    quality_delta?: { [metricKey: string]: number },
    proportion_by_model?: { [modelName: string]: number }
  }
}
```

UI usage guidance:
- Show overall cluster proportion as `size / total_conversations_in_subset`
- Show average metric scores from `meta.quality`
- In per-cluster details, show per-model proportions from `meta.proportion_by_model`
- To display the model next to each property description, map `property_ids` back to properties to retrieve `(question_id, model)`

- DataFrame utilities: `POST /df/select`, `/df/groupby/preview`, `/df/groupby/rows`, `/df/custom`

## 🚀 Packaging & Distribution Roadmap

### **Vision: Local-Only Pip Package**

The long-term goal is to package StringSight as a single pip install with completely local execution - no cloud dependencies, maximum privacy, and instant startup.

### 📦 **Target User Experience**

```bash
# One-time installation
pip install stringsight

# Daily usage - single command startup
stringsight serve
# 🏠 StringSight running locally at http://localhost:8080
# 🚀 Ready in ~10 seconds
# 📊 Upload your evaluation data to get started
# 🔒 All data stays on your machine

# Optional: Pre-load data
stringsight serve --data my_evaluations.jsonl
# 📊 Loaded 1,247 evaluations, ready for analysis
```

### 🏗️ **Implementation Strategy**

**Phase 1: Bundled Distribution**
- Pre-build React frontend into static files
- Bundle frontend assets with Python package
- Single entry point via FastAPI + StaticFiles
- Auto-detect free ports, no configuration needed

**Phase 2: Enhanced CLI**
- `stringsight serve --open` - Auto-opens browser
- `stringsight serve --port 3000` - Custom port selection
- `stringsight convert` - File format utilities
- `stringsight --version` - Version management

**Phase 3: Local Productivity**
- `stringsight export --format pdf` - Report generation
- `stringsight merge file1.jsonl file2.jsonl` - Data utilities
- `stringsight backup` - Local data management
- Performance optimizations for large datasets

### 🏠 **Local-Only Benefits**

**🔒 Privacy & Security**
- Evaluation data never leaves user's machine
- Perfect for proprietary model outputs
- No API keys or cloud accounts required
- Works in air-gapped/secure environments

**⚡ Performance & Reliability**
- Zero network latency
- Works completely offline
- No bandwidth costs for large datasets
- Consistent performance regardless of internet

**💰 Cost & Simplicity**
- No ongoing cloud costs or usage limits
- No account creation or authentication
- Works behind corporate firewalls
- Perfect for academic and research use

### 📋 **Package Structure Plan**

```
stringsight/
├── cli.py                 # Entry point (stringsight serve)
├── api.py                 # FastAPI backend with data processing
├── frontend_dist/         # Pre-built React bundle
│   ├── index.html
│   ├── assets/           # JS/CSS bundles
│   └── favicon.ico
├── utils/
│   ├── file_handlers.py  # JSONL/CSV/JSON parsing
│   ├── data_processing.py # Pandas operations
│   └── export.py         # Report generation
└── templates/            # Export templates (PDF, etc.)
```

### **Development Milestones**

1. **Frontend Build Pipeline** - Automated bundling for distribution
2. **CLI Interface** - Single command startup with options
3. **Package Configuration** - setup.py with proper static file handling
4. **Cross-Platform Testing** - Windows/Mac/Linux compatibility
5. **Performance Optimization** - Sub-10 second startup time
6. **Documentation** - Installation and usage guides

### 🌟 **Target Markets**

- **ML Researchers**: Private evaluation of model outputs
- **Enterprise AI Teams**: Secure analysis of proprietary data  
- **Academic Labs**: Cost-effective evaluation tools
- **Individual Practitioners**: Simple, powerful evaluation workflow

This packaging approach prioritizes **privacy**, **simplicity**, and **performance** - making StringSight accessible to anyone who needs to analyze evaluation data without compromising on security or requiring cloud infrastructure.

---

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      ...tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      ...tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      ...tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
