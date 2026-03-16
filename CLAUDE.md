# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

StringSight is a toolkit for analyzing LLM behavior patterns through property extraction, clustering, and metrics computation. It provides both a Python library API and a web-based UI.

**Core Pattern**: Pipeline-based data processing system with modular, composable stages following a producer-consumer pattern with support for both synchronous and asynchronous execution.

## Development Setup

```bash
# Install in editable mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"           # Development tools
pip install -e ".[local-embeddings]"  # Sentence transformers
pip install -e ".[ml]"             # PyTorch, transformers, vLLM
pip install -e ".[full]"           # All optional dependencies

# Set required API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # optional
```

## Common Commands

### Running the Application

```bash
# Launch web UI (foreground)
stringsight launch

# Launch in background (daemon mode)
stringsight launch --daemon

# Check status
stringsight status

# View logs
stringsight logs

# Stop daemon
stringsight stop
```

### Testing

Tests live in `tests/` and can be run directly:

```bash
python tests/test_prompts_metadata_unit.py
python tests/test_airline_demo_prompt_generation.py
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Docker Deployment

```bash
# Start all services (PostgreSQL, Redis, MinIO, API, Worker)
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f worker

# Stop services
docker compose down
```

## Architecture Overview

### Pipeline System (`stringsight/pipeline.py`)

The heart of StringSight is a modular pipeline architecture:

- **`Pipeline`**: Orchestrates execution of multiple stages
- **`PipelineStage`**: Abstract base class for all processing stages
- **`PipelineBuilder`**: Fluent API for constructing pipelines

**Typical Pipeline Flow**:
1. Property Extraction (LLM extracts behavioral properties)
2. JSON Parsing (parse LLM responses into Property objects)
3. Validation (filter invalid/low-quality properties)
4. Clustering (group similar properties using embeddings + HDBSCAN)
5. Metrics Computation (calculate per-model, per-cluster statistics)

Features: async execution, progress tracking, error recovery, auto-save of intermediate results, WandB integration.

### Core Data Objects (`stringsight/core/data_objects.py`)

All data contracts that flow through the pipeline:

- **`ConversationRecord`**: Single conversation trace (question_id, prompt, model, responses, scores)
- **`Property`**: Extracted behavioral property (property_description, category, reason, evidence, behavior_type)
- **`Cluster`**: Group of similar properties (cluster_id, property_ids, cluster_label)
- **`PropertyDataset`**: Container for all data (conversations + properties + clusters)

**Important**: All IDs are enforced as strings to prevent type mismatches between Python backend and JavaScript frontend.

### API Layer

**Main API** (`stringsight/api.py`):
- FastAPI application with modular router structure
- In-memory caching with 15-minute TTL
- Serves both REST API and frontend static files

**Routers** (`stringsight/routers/`):
- `extraction.py`: Property extraction endpoints
- `jobs.py`: Background job management
- `clustering.py`: Clustering operations
- `validation.py`: Data validation
- `prompts.py`, `health.py`, etc.

### Public Python API (`stringsight/_public/`)

User-facing API functions exported in `stringsight/__init__.py`:

- **`explain()`**: Main analysis function (single model or side-by-side comparison)
- **`label()`**: Fixed-taxonomy labeling
- **`extract_properties_only()`**: Extraction without clustering
- **`compute_metrics_only()`**: Recompute metrics from existing results

Implementation files:
- `sync_api.py`: Synchronous implementations
- `async_api.py`: Async implementations
- `pipeline_builders.py`: Pipeline construction logic

### Key Subsystems

**Extraction** (`stringsight/extractors/`):
- `openai.py`: Main extractor using LiteLLM (supports OpenAI, Anthropic, Google, etc.)
- `fixed_axes_labeler.py`: Fixed taxonomy labeling
- `conv_to_str.py`: Conversation formatting for context
- Features: prompt caching, parallel extraction with ThreadPoolExecutor

**Prompts** (`stringsight/prompts/`):
- `extraction/`: Property extraction prompts (universal, agent, standard)
- `clustering/`: Clustering-related prompts
- `dynamic/`: Dynamic prompt generation system (task expansion, verification, reflection)
- `fixed_axes.py`: Fixed taxonomy prompts

**Clustering** (`stringsight/clusterers/`):
- `hdbscan.py`: Default HDBSCAN clustering
- `hierarchical_clustering.py`: Hierarchical clustering option
- Features: embedding generation (OpenAI or local), configurable min_cluster_size, cluster label generation via LLM

**Metrics** (`stringsight/metrics/`):
- `functional_metrics.py`: Core metric computations
- `side_by_side.py`: Side-by-side comparison metrics
- `single_model.py`: Single model metrics
- Outputs: per-model cluster scores, aggregated cluster scores, overall model scores, win rates

**Postprocessing** (`stringsight/postprocess/`):
- `parser.py`: JSON parsing from LLM outputs
- `validator.py`: Property validation (min length, required fields, deduplication)

### Infrastructure

**Database** (`stringsight/database.py`):
- SQLAlchemy ORM
- SQLite (local/dev) or PostgreSQL (production)
- Alembic migrations in `infrastructure/database/alembic/`

**Models** (`stringsight/db_models/`):
- `job.py`: Background job tracking
- `user.py`: User management

**Background Jobs** (`stringsight/celery_app.py`, `stringsight/workers/tasks.py`):
- Celery + Redis for long-running analysis tasks
- 24-hour task timeout
- Worker restart after 10 tasks (memory leak prevention)
- Database progress updates

**Caching**:
- `stringsight/core/caching.py`: LMDB-based caching for LLM calls
- `stringsight/core/api_cache.py`: API response caching
- In-memory JSONL cache with TTL in api.py

**Configuration** (`stringsight/config.py`):
- Pydantic-based settings
- Environment variable support (.env file)
- DATABASE_URL, REDIS_URL, auth secrets, log level

### Frontend Architecture

**Location**: `~/stringsight-frontend` (git submodule at `frontend/` in this repo)

**Tech Stack**:
- React 19 + TypeScript
- Vite (build tool, dev server on port 5180)
- Material-UI (MUI) for components
- TanStack Query for server state
- TanStack Table for data tables
- Plotly.js for visualizations

**Key Files**:
- `src/App.tsx`: Main application state hub (~5700 lines)
- `src/lib/api.ts`: Backend API client (469 lines)
- `src/lib/normalize.ts`: Score flattening and metrics normalization
- `src/lib/datasetLoader.ts`: URL/ZIP dataset loading
- `vite.config.ts`: Build config and dev proxy

**Directory Structure**:
```
frontend/
├── src/
│   ├── App.tsx                    # Central state management
│   ├── main.tsx                   # React entry point
│   ├── components/                # React components
│   │   ├── DataTable.tsx          # Main data table
│   │   ├── ConversationTrace.tsx  # Single model viewer
│   │   ├── SideBySideTrace.tsx    # A/B comparison viewer
│   │   ├── PropertiesTab.tsx      # Properties results
│   │   ├── ClustersTab.tsx        # Cluster visualization
│   │   ├── metrics/               # Metrics components
│   │   └── sidebar-sections/      # Sidebar panels
│   ├── lib/                       # Core utilities
│   │   ├── api.ts                 # Backend API client
│   │   ├── normalize.ts           # Data transformation
│   │   ├── parse.ts               # File parsing (CSV/JSON/JSONL)
│   │   └── traces.ts              # Trace format conversion
│   ├── hooks/                     # Custom React hooks
│   ├── types/                     # TypeScript types
│   └── features/                  # Feature modules
├── public/                        # Static assets
│   ├── datasets.yaml              # Dataset configuration
│   └── taubench_airline.jsonl     # Demo dataset
└── dist/                          # Build output → stringsight/frontend_dist/
```

**Data Layer Architecture**:

The frontend manages three distinct data layers:

1. **`originalRows`**: Raw uploaded data (never modified, single source of truth)
2. **`operationalRows`**: Cleaned data with score objects (`score`, `score_a`, `score_b`) matching backend contract
3. **`currentRows`**: Display data with flattened score columns (`score_*`, `score_a_*`) for UI tables

**Critical Pattern**: Backend operations use `operationalRows` (score dicts), UI uses `currentRows` (flattened scores). Transformation handled by `flattenScores()` in `normalize.ts`.

**Method Detection**:
- **single_model**: Columns `prompt`, `model_response`, `model`, optional `score` dict
- **side_by_side**: Columns `prompt`, `model_a_response`, `model_b_response`, `model_a`, `model_b`, optional `score_a`/`score_b`

Auto-detected from column names during upload.

**Score Handling**:
```typescript
// Backend format (operationalRows)
{ score: { accuracy: 0.8, fluency: 0.9 } }
{ score_a: { win: 1 }, score_b: { win: 0 } }

// UI format (currentRows) - flattened
{ score_accuracy: 0.8, score_fluency: 0.9 }
{ score_a_win: 1, score_b_win: 0 }
```

## Data Flow

**Typical Analysis Flow**:

1. **Input**: User provides DataFrame (CSV/JSON/JSONL)
   - Required: `prompt`, `model`, `model_response`
   - Optional: `score` columns, `question_id`
   - Method detection: single_model vs side_by_side

2. **Data Preparation**: Format detection, validation, sampling, conversation format conversion

3. **Property Extraction**: System prompt selection/generation, parallel LLM calls, raw responses cached

4. **Parsing & Validation**: JSON extraction, property validation, deduplication

5. **Clustering**: Embedding generation, HDBSCAN clustering, cluster label generation

6. **Metrics Computation**: Per-model cluster scores, prevalence calculations, side-by-side comparisons

7. **Output**:
   - PropertyDataset object (in-memory)
   - Files: `clustered_results.parquet/jsonl`, `model_scores.json`, `summary.txt`
   - WandB logging (optional)

## Key Architectural Patterns

**Mixins** (`stringsight/core/mixins.py`):
- `LoggingMixin`: Structured logging
- `TimingMixin`: Performance tracking
- `ErrorHandlingMixin`: Exception handling
- `WandbMixin`: Experiment tracking

Applied to Pipeline and PipelineStage classes for cross-cutting concerns.

**Formatters** (`stringsight/formatters/`):
- Method detection (single_model vs side_by_side)
- Column validation
- Trace formatting for multimodal conversations

**Storage Abstraction** (`stringsight/storage/adapter.py`):
- Abstraction for local filesystem (current) or S3 (future)
- Persistent data directory: `~/.stringsight/`

## Model Response Formats

StringSight supports three formats for `model_response`:

1. **OpenAI conversation format** (recommended, supports multimodal):
   ```json
   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
   ```

2. **Plain string**: `"Model response text..."`

3. **Custom format**: Any other format (converted to string, won't render prettily in UI)

## Side-by-Side Comparison

Two data formats supported:

1. **Pre-paired**: Use columns `model_a`, `model_b`, `model_a_response`, `model_b_response`
2. **Tidy data**: Pass tidy data with `method="side_by_side"` to auto-pair by prompt

## Important Implementation Details

**Type Safety**:
- All IDs must be strings (not ints) to prevent Python/JavaScript type mismatches
- Enforced in `data_objects.py`

**LLM Provider Support**:
- LiteLLM integration supports OpenAI, Anthropic, Google, and more
- Configure via `model_name` parameter (e.g., `"gpt-4o-mini"`, `"claude-3-5-sonnet-20241022"`)

**Prompt Caching**:
- Enabled by default for efficiency
- Uses provider-specific caching (e.g., Anthropic prompt caching)

**Dynamic Prompts**:
- System can generate task-specific prompts from sample data
- Task expansion, verification, and reflection loops
- Located in `stringsight/prompts/dynamic/`

**Deployment Modes**:
- **Local CLI**: Single FastAPI server, SQLite database (port 5180)
- **Docker**: PostgreSQL, Redis, MinIO, API service, Celery workers (port 8000)

## Scripts

`scripts/` contains analysis scripts and configuration:

- `run_pipeline.py`: Run analysis from command line
- `run_from_config.py`: Run from YAML configuration
- `dataset_configs/`: YAML configs for various datasets (WildBench, TauBench, etc.)

## Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key (required for extraction and embeddings)
- `ANTHROPIC_API_KEY`: Anthropic API key (optional)
- `GOOGLE_API_KEY`: Google API key (optional)

Optional (for production deployment):
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `STORAGE_TYPE`: `local` or `s3`
- `S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: For S3 storage
- `LOG_LEVEL`: Logging level (default: INFO)
- `JSON_LOGS`: Enable JSON logging (default: false)

## Frontend Development

### Setup and Running

```bash
# Navigate to frontend (separate repo at ~/stringsight-frontend)
cd ~/stringsight-frontend

# Install dependencies
npm install

# Run dev server (http://localhost:5180)
npm run dev

# Build for production
npm run build
# Output: dist/ directory

# Build and copy to backend package
cd /home/lisabdunlap/StringSightNew
./build_frontend.sh
# Copies frontend/dist/ → stringsight/frontend_dist/
```

### Full Stack Development

**Terminal 1 - Backend**:
```bash
cd /home/lisabdunlap/StringSightNew
uvicorn stringsight.api:app --reload --port 8000
```

**Terminal 2 - Frontend**:
```bash
cd ~/stringsight-frontend
npm run dev  # Runs on port 5180 with proxy to backend
```

**Environment Setup**:
```bash
# Frontend (.env.local in ~/stringsight-frontend)
VITE_BACKEND=http://localhost:8000

# Backend (in /home/lisabdunlap/StringSightNew)
export OPENAI_API_KEY="your-key"
export BASE_BROWSE_DIR="/path/to/results"  # For server-side file browsing
```

### Frontend-Backend Integration

**API Communication** (`src/lib/api.ts`):
- Base URL: `/api` (proxied in dev, direct in production)
- Vite dev proxy: `/api/*` → `http://localhost:8000/*`
- 10-minute timeout for long-running operations

**Key Backend Endpoints Used**:
- `POST /detect-and-validate`: Column detection and validation
- `POST /api/v1/jobs/extract`: Async extraction job
- `POST /api/v1/jobs/pipeline`: Full pipeline (extract + cluster)
- `GET /api/v1/jobs/{job_id}`: Job status polling
- `POST /cluster/run`: Run clustering
- `POST /cluster/metrics`: Recompute metrics
- `POST /label/run`: Fixed taxonomy labeling
- `POST /df/select`: DataFrame filtering
- `POST /results/load`: Load pre-computed results

**Data Format Contract**:

Backend returns:
```python
{
  "conversations": [...],           # List of ConversationRecord dicts
  "properties": [...],              # List of Property dicts
  "clusters": [...],                # List of Cluster dicts
  "metrics": {
    "model_cluster_scores": [...],  # Per model-cluster metrics
    "cluster_scores": [...],        # Aggregated cluster metrics
    "model_scores": [...]           # Overall model metrics
  },
  "total_conversations_by_model": {...},
  "total_unique_conversations": int
}
```

Frontend normalizes:
- Flattens nested score objects
- Renames metrics columns for consistency
- Enriches clusters with quality data from metrics
- Computes derived fields (proportions, deltas)

**Serving in Production** (`stringsight/cli.py`):

When running `stringsight launch`:
1. FastAPI mounts backend API at `/api`
2. Serves static files from `stringsight/frontend_dist/`
3. SPA routing: 404s on non-API routes serve `index.html`
4. Single unified server on port 5180

```
http://localhost:5180/
├── /                    → index.html
├── /assets/*            → static JS/CSS bundles
├── /api/*               → backend FastAPI routes
└── /{dataset_name}      → SPA routing (serves index.html)
                            Frontend parses URL for dataset loading
```

### Frontend Features

**Data Loading Modes**:
1. **Browser Upload**: CSV/JSON/JSONL parsed client-side with PapaParse
2. **Demo Data**: Pre-bundled `taubench_airline.jsonl` from `public/`
3. **URL Loading**: Visit `stringsight.com/{dataset_name}` to auto-load from config
4. **Pre-computed Results**: Directory picker to load local results folder

**Main UI Tabs**:
- **Data**: Table view, filtering, sorting, trace viewing
- **Properties**: Extracted behavioral properties with evidence highlighting
- **Clusters**: Cluster cards with summaries and per-model metrics
- **Metrics**: Benchmark leaderboard, plots, model comparison

**Evidence Highlighting** (`ConversationTrace.tsx`):
- Two-strategy matching: exact (case-insensitive) and fuzzy (≥75% Jaccard similarity)
- Handles whitespace variations, punctuation, ellipses
- Maps normalized positions back to original text
- Merges overlapping highlights

**Sidebar Architecture**:
1. **PermanentIconSidebar** (60px, always visible): Section navigation icons
2. **ExpandedSidebar** (400px, toggleable): Context-sensitive panels for active section

### Submodule Management

The frontend is included as a git submodule at `frontend/`:

```bash
# Clone backend with frontend submodule
git clone --recurse-submodules https://github.com/lisadunlap/stringsight.git

# If already cloned without submodules
git submodule update --init --recursive

# Update frontend submodule to latest
cd frontend
git pull origin main
cd ..
git add frontend
git commit -m "Update frontend submodule"

# Note: Main development happens in ~/stringsight-frontend
# The frontend/ submodule is mainly for version tracking
```

## Important Cross-Stack Considerations

### ID Type Consistency
**CRITICAL**: All IDs (question_id, cluster_id, property_id) must be strings, not integers.
- Python naturally uses ints, JavaScript uses numbers (which can lose precision)
- Enforced in `stringsight/core/data_objects.py`
- Frontend expects string IDs in all API responses

### Score Format Synchronization
Backend and frontend must stay synchronized on score formats:
- **Backend contract**: `score` as dict (`{"accuracy": 0.8, "fluency": 0.9}`)
- **Frontend operational layer**: Same dict format for backend calls
- **Frontend display layer**: Flattened (`score_accuracy`, `score_fluency`) for tables

When adding new score columns:
1. Backend should return them in the `score` dict
2. Frontend's `flattenScores()` will automatically flatten them
3. No frontend code changes needed for new score types

### Evidence Highlighting
Backend's `Property.evidence` field should contain text that appears in the conversation:
- Frontend uses fuzzy matching (75% Jaccard similarity)
- Handles minor whitespace/punctuation differences
- Large differences will fail to highlight
- Test evidence highlighting after prompt changes

### Method Detection
Both backend and frontend auto-detect `single_model` vs `side_by_side`:
- Detection logic must stay synchronized
- Based on column names (`model_a_response` triggers side_by_side)
- Located in: `stringsight/formatters/traces.py` and `frontend/src/lib/api.ts`

### Long-Running Operations
- Frontend polls `/api/v1/jobs/{job_id}` every 2 seconds
- Backend updates `Job.progress` field in database
- Timeout: 10 minutes in frontend, 24 hours in Celery
- Always provide progress updates for operations >5 seconds

## Known Limitations

- Worker restart after 10 tasks is a workaround for memory leaks in long-running Celery workers
- JSONL cache has 15-minute TTL to prevent stale data in UI
- Min cluster size defaults to 5; smaller values may produce noisy clusters
- Evidence highlighting requires ≥75% text overlap; exact quotes work best
