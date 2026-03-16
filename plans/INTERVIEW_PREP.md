# StringSight Interview Preparation Guide

## Executive Summary

StringSight is a production-ready toolkit for analyzing LLM behavior patterns through automated property extraction, clustering, and metrics computation. It provides both a Python library API and a web-based UI for understanding how different language models behave across datasets.

**Key Value Proposition**: Instead of manually reviewing thousands of LLM responses, StringSight automatically extracts behavioral patterns, groups similar behaviors, and computes statistical metrics to compare models systematically.

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────────┐           ┌─────────────────────┐    │
│  │  Python API      │           │   React Frontend    │    │
│  │  (public API)    │           │   (Material-UI)     │    │
│  └──────────────────┘           └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │extraction│   jobs   │clustering│validation│ prompts  │  │
│  │  router  │  router  │  router  │  router  │  router  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline System Core                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Property      JSON        Validation    Clustering   │  │
│  │  Extraction  → Parsing  →  & Filter   →  (HDBSCAN)  →│  │
│  │  (LLM calls)   (parser)    (validator)   (embeddings) │  │
│  │                                                 ↓      │  │
│  │                                          Metrics       │  │
│  │                                       Computation      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │PostgreSQL│  Redis   │  Celery  │  MinIO   │  LMDB    │  │
│  │   (DB)   │ (queue)  │ (workers)│(storage) │ (cache)  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Pattern: Modular Pipeline Architecture

The system is built around a **composable pipeline pattern** where data flows through discrete processing stages:

```python
# Conceptual flow
Pipeline([
    PropertyExtractionStage(),    # LLM extracts behavioral properties
    JSONParsingStage(),            # Parse JSON from LLM responses
    ValidationStage(),             # Filter low-quality properties
    ClusteringStage(),             # Group similar properties
    MetricsComputationStage()      # Calculate statistics
])
```

Each stage is:
- **Independent**: Can be run in isolation
- **Composable**: Stages can be added/removed/reordered
- **Async-capable**: Supports both sync and async execution
- **Observable**: Progress tracking, error recovery, auto-save

## Key Components Deep Dive

### 1. Data Objects (`stringsight/core/data_objects.py`)

**Core data contracts that flow through the system:**

```python
@dataclass
class ConversationRecord:
    """Single conversation trace"""
    question_id: str              # Unique identifier
    prompt: str                   # Input prompt
    model: str                    # Model identifier
    model_response: str | list    # Response (string or OpenAI format)
    score: dict[str, float]       # Optional: metric scores

@dataclass
class Property:
    """Extracted behavioral property"""
    property_id: str
    question_id: str
    property_description: str     # What behavior was observed
    category: str                 # Category of behavior
    reason: str                   # Why this matters
    evidence: str                 # Supporting quote from response
    behavior_type: str            # positive/negative/neutral

@dataclass
class Cluster:
    """Group of similar properties"""
    cluster_id: str
    property_ids: list[str]
    cluster_label: str            # LLM-generated summary
    representative_properties: list[Property]

@dataclass
class PropertyDataset:
    """Complete analysis result"""
    conversations: list[ConversationRecord]
    properties: list[Property]
    clusters: list[Cluster]
    metrics: dict                 # Computed statistics
```

**Critical Design Decision**: All IDs are strings (not integers) to prevent type mismatches between Python backend and JavaScript frontend.

### 2. Pipeline System (`stringsight/pipeline.py`)

**Key Classes:**

```python
class PipelineStage(ABC):
    """Abstract base for all processing stages"""
    @abstractmethod
    async def process(self, data: PropertyDataset) -> PropertyDataset:
        pass

    # Mixins provide: logging, timing, error handling, wandb tracking

class Pipeline:
    """Orchestrates execution of multiple stages"""
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages

    async def run(self, data: PropertyDataset) -> PropertyDataset:
        # Run stages sequentially with progress tracking
        # Auto-save intermediate results
        # Error recovery and retry logic
```

**Features:**
- Progress tracking with callbacks
- Automatic intermediate result saving
- WandB experiment tracking
- Error recovery and retry mechanisms
- Both sync and async execution modes

### 3. Property Extraction (`stringsight/extractors/`)

**Main Extractor** (`openai.py`):
- Uses **LiteLLM** for multi-provider support (OpenAI, Anthropic, Google, etc.)
- **Parallel extraction** with ThreadPoolExecutor
- **Prompt caching** for efficiency (provider-specific)
- **LMDB caching** to avoid redundant LLM calls

**Extraction Process:**
1. Select/generate system prompt (universal, agent, or dynamic)
2. Format conversations for context
3. Parallel LLM calls with rate limiting
4. Cache raw responses
5. Return list of Property objects

**Prompt Types:**
- **Universal**: General-purpose property extraction
- **Agent**: Specialized for agentic/tool-use behaviors
- **Dynamic**: Auto-generated from sample data (task expansion + verification)
- **Fixed Axes**: Pre-defined taxonomy labeling

### 4. Clustering (`stringsight/clusterers/`)

**Two-Stage Process:**

**Stage 1: Embedding Generation**
- OpenAI embeddings (default): `text-embedding-3-small`
- Local embeddings (optional): sentence-transformers
- Cached in LMDB for reuse

**Stage 2: HDBSCAN Clustering**
```python
class HDBSCANClusterer:
    def cluster(self, properties: list[Property]) -> list[Cluster]:
        # 1. Generate embeddings for property descriptions
        embeddings = self.embed(properties)

        # 2. Run HDBSCAN (density-based clustering)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,  # Configurable
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)

        # 3. Generate cluster labels via LLM
        for cluster in clusters:
            cluster.label = self.generate_label(cluster.properties)

        return clusters
```

**Why HDBSCAN?**
- No need to specify number of clusters upfront
- Handles noise (outliers labeled as -1)
- Finds clusters of varying densities
- Based on density reachability

**Hierarchical Clustering** (alternative):
- Uses scipy's hierarchical clustering
- Allows user-specified number of clusters
- Better for controlled cluster counts

### 5. Metrics Computation (`stringsight/metrics/`)

**Three Levels of Metrics:**

```python
# 1. Model-Cluster Scores (per model, per cluster)
{
    "model": "gpt-4",
    "cluster_id": "cluster_0",
    "property_count": 42,           # How many properties from this model
    "conversation_count": 35,       # How many conversations exhibit this
    "prevalence": 0.35,             # % of model's conversations
    "avg_score": 0.87               # Average score for this cluster
}

# 2. Aggregated Cluster Scores (across all models)
{
    "cluster_id": "cluster_0",
    "total_properties": 150,
    "total_conversations": 120,
    "avg_prevalence": 0.30,
    "cluster_quality": 0.85         # Based on consistency
}

# 3. Overall Model Scores
{
    "model": "gpt-4",
    "total_properties": 450,
    "total_clusters": 12,
    "avg_cluster_prevalence": 0.28
}
```

**Side-by-Side Metrics** (Model A vs Model B):
- Win rates per cluster
- Relative prevalence differences
- Statistical significance tests

### 6. API Layer (`stringsight/api.py` + `stringsight/routers/`)

**Modular Router Structure:**

```python
# Main API
app = FastAPI()
app.include_router(extraction_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1/jobs")
app.include_router(clustering_router, prefix="/cluster")
# ... more routers

# In-memory caching with 15-min TTL
cache = TTLCache(maxsize=100, ttl=900)
```

**Key Endpoints:**
- `POST /api/v1/jobs/extract`: Async property extraction job
- `POST /api/v1/jobs/pipeline`: Full pipeline (extract + cluster + metrics)
- `GET /api/v1/jobs/{job_id}`: Poll job status
- `POST /cluster/run`: Run clustering on existing properties
- `POST /cluster/metrics`: Recompute metrics
- `POST /label/run`: Fixed taxonomy labeling

**Job Management:**
- Background jobs via Celery + Redis
- Database-backed progress tracking
- 24-hour task timeout
- Worker restart after 10 tasks (memory leak mitigation)

### 7. Frontend Architecture

**Tech Stack:**
- React 19 + TypeScript
- Vite (build + dev server)
- Material-UI (MUI) components
- TanStack Query (server state)
- TanStack Table (data tables)
- Plotly.js (visualizations)

**Critical Data Layer Pattern:**

```typescript
// Three data layers for score handling
originalRows      // Raw uploaded data (immutable)
↓
operationalRows   // Backend format: { score: { accuracy: 0.8 } }
↓
currentRows       // UI format: { score_accuracy: 0.8 }
```

**Why Three Layers?**
- Backend expects nested score objects
- UI tables need flattened columns for sorting/filtering
- Need to preserve original data for re-analysis

**Method Detection:**
```typescript
// Auto-detect analysis mode from columns
if (columns.includes('model_a_response')) {
    method = 'side_by_side'
} else if (columns.includes('model_response')) {
    method = 'single_model'
}
```

**Main UI Tabs:**
1. **Data**: Filterable table, conversation viewing
2. **Properties**: Extracted properties with evidence highlighting
3. **Clusters**: Cluster cards with per-model metrics
4. **Metrics**: Benchmark leaderboard and comparison plots

**Evidence Highlighting** (sophisticated feature):
- Two-strategy matching: exact + fuzzy (Jaccard similarity ≥75%)
- Handles whitespace variations, punctuation, ellipses
- Maps normalized positions back to original text with formatting
- Merges overlapping highlights

## Deployment Modes

### 1. Local CLI Mode (Development)
```bash
stringsight launch
# Single FastAPI server on port 5180
# SQLite database
# Serves both API and frontend static files
```

### 2. Docker Mode (Production)
```bash
docker compose up -d
# PostgreSQL database
# Redis queue
# MinIO object storage
# API service (port 8000)
# Celery workers (background jobs)
```

### 3. Python Library Mode
```python
import stringsight as ss

results = ss.explain(
    df=data,
    method="side_by_side",
    extraction_model="gpt-4o-mini",
    output_dir="./results"
)
```

## Infrastructure Deep Dive: Redis, Celery, Docker

This section explains the "behind the scenes" infrastructure that makes StringSight work in production. Think of it like this: when a user uploads data and clicks "Analyze," we can't make them wait 20 minutes staring at a loading spinner. Instead, we need a system that can accept their request, process it in the background, and let them check back later for results. That's where Redis, Celery, and Docker come in.

### Redis Architecture

**What is Redis?**
Redis is an in-memory data structure store - essentially a super-fast database that lives entirely in RAM. We use it for two main purposes in StringSight: as a message broker (think of it as a to-do list for background tasks) and as a result backend (temporary storage for task status and results).

**Why do we need it?** When the FastAPI server receives a request to analyze 1000 conversations, it can't just start processing immediately - that would block the server from handling other requests. Instead, it needs to hand off that work to a background worker. Redis is the middleman that facilitates this handoff.

**In StringSight's Context:**
```python
# stringsight/config.py
REDIS_URL: str = "redis://localhost:6379/0"

# stringsight/celery_app.py
celery_app = Celery(
    "stringsight",
    broker=settings.REDIS_URL,      # Message queue for tasks
    backend=settings.REDIS_URL,     # Store task results
    include=["stringsight.workers.tasks"]
)
```

**Redis Roles:**

**1. Message Broker (Task Queue)**

Think of Redis as a post office for background jobs. When the API needs work done, it writes a "letter" (task) and drops it in Redis's mailbox. Workers constantly check the mailbox, grab tasks, and execute them.

```
┌──────────┐     enqueue task      ┌───────┐     consume task     ┌────────┐
│ FastAPI  │ ──────────────────► │ Redis │ ──────────────────► │ Celery │
│ API      │                       │ Queue │                      │ Worker │
└──────────┘                       └───────┘                      └────────┘
```

**The flow when a user clicks "Analyze":**

1. **API creates a Job record in PostgreSQL**: This is for permanent record-keeping - we want to track every analysis job even after Redis forgets about it
2. **API enqueues a Celery task to Redis**: The task is serialized as JSON and added to a queue (like adding an item to a to-do list)
3. **Redis stores the task in a queue**: By default, this queue is just called `celery` - it's a simple FIFO (first-in-first-out) list
4. **Celery worker polls Redis**: Workers are constantly asking "Any work for me?" They use a blocking pop operation, which means they wait patiently until work arrives
5. **Worker executes the task**: The actual property extraction, clustering, etc. happens here - this can take 10-30 minutes
6. **Worker stores result back in Redis**: Once done, the result goes back to Redis with status "SUCCESS" or "FAILURE"
7. **API can query Redis for task status**: The frontend polls the API every 2 seconds, and the API checks Redis to see if the task is still running or complete

**2. Result Backend (Task State)**

Redis also acts as temporary storage for task results. When a worker finishes a task, it doesn't just disappear - the worker writes the outcome back to Redis so the API can retrieve it. This is separate from the queue; think of it as a bulletin board where workers post their results.

```python
# Task states stored in Redis (example of what gets stored)
{
    "task_id": "abc-123-def",
    "status": "RUNNING",      # PENDING → STARTED → SUCCESS or FAILURE
    "result": null,            # Populated on completion with return value
    "traceback": null,         # If failed, this has the error message
    "progress": 0.45           # Custom state we track (45% complete)
}
```

**Important note**: Redis stores this data temporarily (24 hours by default). After that, it's gone. That's why we also save everything to PostgreSQL - Redis is for speed during execution, PostgreSQL is for permanent records.

**Redis in docker-compose.yml:**
```yaml
redis:
  image: redis:7-alpine       # Lightweight Alpine Linux version
  ports:
    - "127.0.0.1:6379:6379"   # Expose only to localhost (security)
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 5s
    timeout: 5s
    retries: 5
```

**Why Redis over alternatives?**
- **vs RabbitMQ**: Simpler setup, lower memory footprint, sufficient for our needs
- **vs Database queue**: Much faster, lower latency, designed for message passing
- **vs AWS SQS**: No cloud dependency, works locally, no API costs

### Celery Architecture

**What is Celery?**
Celery is a distributed task queue system for Python. In plain English: it's a way to run time-consuming Python functions in the background without blocking your main application. Think of it as a worker management system - you define jobs, Celery finds available workers, distributes the jobs, and reports back when they're done.

**Why do we need Celery specifically?** We could use simple Python threading, but Celery gives us:
- **Persistence**: If a worker crashes, tasks don't disappear (they're in Redis)
- **Scalability**: We can run multiple workers across multiple machines
- **Monitoring**: We can inspect what workers are doing, kill stuck tasks, etc.
- **Retries**: Failed tasks can automatically retry with exponential backoff

**Core Concepts:**

**1. Tasks (Work Definitions)**

A "task" in Celery is just a Python function decorated with `@celery_app.task`. This tells Celery "hey, this function can be run asynchronously by a worker." The function definition lives in our code, but the actual execution happens in a separate worker process.
```python
# stringsight/workers/tasks.py
@celery_app.task(bind=True, name="stringsight.workers.tasks.run_pipeline_job")
def run_pipeline_job(self, job_id: str, req_data: Dict[str, Any]):
    """Celery task for full pipeline (extraction + clustering)."""
    _run_pipeline_job(job_id, req_data)
```

- `@celery_app.task`: Decorator registers function as a Celery task
- `bind=True`: First arg is the task instance (`self`)
- `name`: Explicit task name for worker discovery

**2. Workers (Task Executors)**

A worker is a separate Python process that sits idle, waiting for work to arrive in Redis. When work appears, it grabs it, executes it, and reports the result. You can run as many workers as you want - they all watch the same Redis queue and take work as it becomes available.

```bash
# Start a worker
celery -A stringsight.celery_app worker --loglevel=info

# What this command does behind the scenes:
# 1. Imports the stringsight.celery_app module (this loads our app configuration)
# 2. Discovers all @celery_app.task decorated functions (these become available jobs)
# 3. Connects to Redis broker (starts listening to the queue)
# 4. Polls for tasks (uses blocking pop - waits patiently without burning CPU)
# 5. Executes tasks in separate processes/threads (isolated from other tasks)
```

**Worker lifecycle**: A worker starts up, processes tasks one by one (or in parallel if configured), and after processing 10 tasks, it restarts itself. This automatic restart is our workaround for memory leaks - better to restart periodically than run out of memory.

**3. Task Flow Example:**

```python
# API endpoint (stringsight/routers/jobs.py)
@router.post("/pipeline")
async def create_pipeline_job(req: PipelineJobRequest):
    # 1. Create job record in PostgreSQL
    job = Job(id=job_id, status="pending", type="pipeline")
    db.add(job)
    db.commit()

    # 2. Enqueue task to Redis
    run_pipeline_job.delay(str(job_id), req.dict())
    #                 ^^^^^ .delay() sends task to Redis asynchronously

    # 3. Return immediately (user doesn't wait)
    return {"job_id": job_id, "status": "pending"}

# Worker executes task (stringsight/workers/tasks.py)
def _run_pipeline_job(job_id: str, req_data: Dict[str, Any]):
    # 1. Update job status in PostgreSQL
    job.status = "running"
    db.commit()

    # 2. Run analysis (takes 10-30 minutes)
    results = explain(df, ...)

    # 3. Update job with results
    job.status = "completed"
    job.result_path = output_dir
    db.commit()
```

**4. Progress Tracking Pattern:**
```python
# Update progress during long-running task
def update_progress(progress: float) -> None:
    try:
        # Create new DB session (avoid transaction issues)
        with SessionLocal() as session:
            current_job = session.query(Job).filter(Job.id == job_uuid).first()
            if current_job:
                current_job.progress = progress
                session.commit()
    except Exception as e:
        logger.error(f"Failed to update progress: {e}")

# Used during extraction
extractor.run(dataset, progress_callback=update_progress)

# Frontend polls for updates
GET /api/v1/jobs/{job_id}  # Every 2 seconds
# Returns: {"status": "running", "progress": 0.45}
```

**Celery Configuration (stringsight/celery_app.py):**
```python
celery_app.conf.update(
    task_serializer="json",              # Serialize tasks as JSON (not pickle - security)
    accept_content=["json"],             # Only accept JSON (no pickle - security)
    result_serializer="json",            # Results also as JSON
    timezone="UTC",                      # Consistent timezone
    enable_utc=True,

    # Task settings
    task_track_started=True,             # Track when task starts (for status)
    task_time_limit=3600 * 24,           # 24 hour timeout (hard kill)
    worker_max_tasks_per_child=10,       # Restart worker after 10 tasks (memory leak fix)
)
```

**Why `worker_max_tasks_per_child=10`?**

This is a pragmatic workaround for a real-world problem. Our LLM extraction tasks involve loading large models, processing big DataFrames, and making many API calls. Over time, memory leaks accumulate - maybe a DataFrame isn't garbage collected properly, or a cache grows unbounded.

Rather than spending weeks hunting down every possible leak, we use a simple solution: restart the worker after 10 tasks. It's like turning your computer off and on again - crude but effective.

**The tradeoff**: There's a small overhead (2-3 seconds) when a worker restarts because it needs to reload Python modules, reconnect to Redis, etc. But that's negligible compared to running out of memory and having the entire worker crash mid-task, losing progress.

**Alternative approach**: We could use memory profiling tools (like `memory_profiler` or `py-spy`) to find and fix leaks. But that's time-consuming, and leaks can be subtle. The restart approach is defensive programming - assume there will be leaks and handle them proactively.

**Task Types in StringSight:**

**1. Extraction Job** (`run_extract_job`):
```python
# Extracts properties only (no clustering)
# Duration: ~2-10 minutes
# Progress updates: Every 1 second during extraction
```

**2. Pipeline Job** (`run_pipeline_job`):
```python
# Full pipeline: extraction + clustering + metrics
# Duration: ~10-30 minutes
# Progress updates: Via callback to database
```

**3. Cluster Job** (`run_cluster_job`):
```python
# Clustering on pre-extracted properties
# Duration: ~1-3 minutes
# Progress updates: 5%, 10%, 75%, 95%, 100% (manual checkpoints)
```

### Docker Compose Architecture

**What is Docker Compose and why do we use it?**

Docker Compose is a tool for defining and running multi-container Docker applications. In simpler terms: StringSight needs 5 different services to run (PostgreSQL, Redis, MinIO, API server, and Celery workers). Without Docker Compose, you'd need to manually start each one with the right configuration, network settings, and volume mounts. Docker Compose lets you define everything in a YAML file and start/stop it all with one command.

**Real-world analogy**: Imagine you're setting up a restaurant. You need a kitchen (workers), a front desk (API), a cash register (database), a phone line for takeout orders (Redis queue), and storage (MinIO). Docker Compose is like a blueprint that says "here's exactly how to set up and connect all these pieces." One command and your restaurant is open for business.

**docker-compose.yml Structure:**

```yaml
version: '3.8'

services:
  # 1. PostgreSQL Database
  db:
    image: postgres:15-alpine          # Lightweight Alpine version
    volumes:
      - postgres_data:/var/lib/postgresql/data  # Persistent storage
    environment:
      - POSTGRES_USER=stringsight
      - POSTGRES_PASSWORD=stringsight_dev
      - POSTGRES_DB=stringsight
    ports:
      - "127.0.0.1:5432:5432"          # Only expose to localhost
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U stringsight"]
      interval: 5s
      timeout: 5s
      retries: 5

  # 2. Redis Message Broker
  redis:
    image: redis:7-alpine
    ports:
      - "127.0.0.1:6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  # 3. MinIO Object Storage (S3-compatible)
  minio:
    image: minio/minio
    ports:
      - "127.0.0.1:9000:9000"          # API endpoint
      - "127.0.0.1:9001:9001"          # Web console
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]

  # 4. FastAPI API Service
  api:
    build:
      context: .
      dockerfile: Dockerfile
    network_mode: "host"               # Share host's network (simpler for dev)
    volumes:
      - ./results:/app/results         # Mount local results directory
      - ./data:/app/data
    env_file:
      - .env                           # Load environment variables
    environment:
      - DATABASE_URL=postgresql://stringsight:stringsight_dev@localhost:5432/stringsight
      - REDIS_URL=redis://localhost:6379/0
      - STORAGE_TYPE=s3
      - S3_ENDPOINT_URL=http://localhost:9000
    depends_on:
      db:
        condition: service_healthy     # Wait for DB to be ready
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    command: uvicorn stringsight.api:app --host 0.0.0.0 --port 8000 --reload

  # 5. Celery Worker
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    network_mode: "host"
    volumes:
      - ./results:/app/results
      - ./data:/app/data
    env_file:
      - .env
    environment:
      - DATABASE_URL=postgresql://stringsight:stringsight_dev@localhost:5432/stringsight
      - REDIS_URL=redis://localhost:6379/0
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    command: celery -A stringsight.celery_app worker --loglevel=info

volumes:
  postgres_data:                       # Named volume for PostgreSQL data
  minio_data:                          # Named volume for MinIO data
```

**Key Concepts:**

**1. Service Dependencies (`depends_on`):**

This solves a common problem: what if the API starts before PostgreSQL is ready? The API will try to connect, fail, and crash. `depends_on` with health checks ensures services start in the right order and only proceed when dependencies are actually ready (not just "container started" but "service is accepting connections").

```yaml
depends_on:
  db:
    condition: service_healthy         # Don't start API until DB is healthy
```

**How healthchecks work**: Every 5 seconds, Docker runs a command (like `pg_isready` for PostgreSQL or `redis-cli ping` for Redis). If the command succeeds, the service is considered healthy. Only then will dependent services start. This prevents race conditions where the API tries to connect before the database is ready to accept connections.

**2. Network Mode (`network_mode: "host"`):**

By default, Docker creates an isolated network for containers. This is great for security but adds complexity - you need to configure DNS, port mapping, etc. In development, we often use `network_mode: "host"` which means "just use the host machine's network directly."

```yaml
network_mode: "host"
```

**What this means in practice**: When the API wants to talk to PostgreSQL, it connects to `localhost:5432` instead of `postgres-container:5432`. Simpler, fewer moving parts, easier to debug.

**The tradeoff**: Less isolation means containers can potentially interfere with each other or with services running on your host machine. For development, this is fine. For production, you'd typically use Docker's bridge networking for better security and isolation.

**Why not always use host mode?** In production, you might have multiple applications running, and you don't want their containers conflicting on port numbers. Bridge networking gives each container its own network namespace, preventing conflicts.

**3. Volumes (Persistent Data):**

Containers are ephemeral by design - when you delete a container, everything inside it disappears. But we need some data to persist (like database files). That's where volumes come in.

```yaml
volumes:
  - postgres_data:/var/lib/postgresql/data    # Named volume (managed by Docker)
  - ./results:/app/results                     # Bind mount (local directory)
```

**Named volumes** (`postgres_data`):
Think of these as external hard drives managed by Docker. When PostgreSQL writes data, it goes to this volume, not the container itself. If you delete and recreate the container, the data is still there. Docker stores these volumes in its own storage directory (usually `/var/lib/docker/volumes/`).

**Bind mounts** (`./results:/app/results`):
These are simpler - they just map a folder on your machine to a folder in the container. When the container writes to `/app/results`, you see the files appear in `./results` on your host machine. This is super useful in development because you can access analysis results without having to copy them out of the container.

**Real-world example**: Say you run an analysis and it outputs files to `/app/results/job_123/`. With a bind mount, you can immediately open those files on your local machine using VSCode, Excel, or whatever tools you prefer. Without it, you'd need to run `docker cp` to extract files from the container.

**4. Environment Variables (.env file):**
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
LOG_LEVEL=INFO
```

Loaded via:
```yaml
env_file:
  - .env
```

Also can override per-service:
```yaml
environment:
  - DATABASE_URL=postgresql://...    # Overrides .env value
```

### Dockerfile Breakdown

**What's the point of the Dockerfile?**

A Dockerfile is a recipe for building a container image. It says "start with this base image (Python 3.11), install these dependencies, copy this code, and run this command." The resulting image is like a snapshot that can be run anywhere Docker exists - your laptop, a colleague's machine, AWS, Google Cloud, etc.

**Multi-stage Build Pattern:**

Here's a clever optimization: building Python packages requires compilation tools (gcc, make, etc.) that take up ~1GB of space. But once packages are built, you don't need those tools at runtime. Multi-stage builds let you use those tools in a "builder" stage, then copy only the compiled packages to a smaller "runtime" stage.

```dockerfile
# Stage 1: Builder (build dependencies)
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (gcc, etc.)
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \                        # PostgreSQL client library
  && rm -rf /var/lib/apt/lists/*     # Clean up (reduce image size)

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image (runtime only)
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y \
  libpq5 \                           # PostgreSQL runtime library (not dev)
  curl \                             # For healthchecks
  gosu \                             # For user switching
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY stringsight/ ./stringsight/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pyproject.toml .
COPY scripts/ ./scripts/

# Create non-root user (security)
RUN useradd -m appuser && chown -R appuser:appuser /app

# Entrypoint script
COPY docker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 8000

CMD ["uvicorn", "stringsight.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why Multi-stage Build?**

The impact is dramatic:
```
Builder stage: 1.5GB (includes gcc, build tools, header files)
                ↓ (copy only compiled packages)
Final stage: 400MB (only runtime libraries)
```

**Benefits:**
- **Smaller image = faster deployment**: Downloading a 400MB image takes seconds; 1.5GB takes minutes. When you're deploying to production or scaling up workers, this matters.
- **Security**: Build tools like gcc can be exploited if an attacker gains access to your container. If they're not there in the first place, that attack vector is eliminated.
- **Cleaner mental model**: Build concerns (compiling) are separate from runtime concerns (running the app). Easier to reason about and debug.

**Analogy**: Imagine you're baking a cake. The builder stage is your kitchen with all the mixing bowls, measuring cups, and ovens. The final stage is just the cake on a plate. You don't serve the cake with all your baking equipment - you just serve the cake.

**Entrypoint Script (docker/docker-entrypoint.sh):**

This script runs before the main application starts. It's like a pre-flight checklist - making sure everything is configured correctly before takeoff.

```bash
#!/bin/bash
set -e  # Exit immediately if any command fails

# Fix permissions for mounted volumes
if [ -d "/app/results" ]; then
    chown -R appuser:appuser /app/results
fi

if [ -d "/app/data" ]; then
    chown -R appuser:appuser /app/data
fi

# Execute the command (from CMD or docker-compose)
exec "$@"  # "$@" means "run whatever command was passed to this script"
```

**Why is this needed? (Permission problem explained)**

Here's the issue: When you mount `./results` from your host machine (where you're user ID 1000) into the container (where appuser is user ID 999), there's a mismatch. The container's appuser doesn't have permission to write to a directory owned by your host user.

The entrypoint script fixes this by changing the ownership to match the container's user. Without this fix, the application would crash with "Permission denied" errors when trying to save results.

**Security note**: We run as `appuser` (not root) inside the container as a security best practice. If an attacker exploits our application, they'd only have `appuser` privileges, not root access to the entire container.

### Common Docker Operations

**Starting the stack:**
```bash
# Build and start all services
docker compose up -d

# -d: Detached mode (run in background)
# Equivalent to:
# - docker compose build  (build images)
# - docker compose up     (start containers)
```

**Viewing logs:**
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f worker

# -f: Follow (stream logs)

# Last 100 lines
docker compose logs --tail=100 worker
```

**Checking status:**
```bash
# List running containers
docker compose ps

# Example output:
# NAME                    STATUS           PORTS
# stringsight-api-1       Up 5 minutes     0.0.0.0:8000->8000/tcp
# stringsight-worker-1    Up 5 minutes
# stringsight-db-1        Up 5 minutes     127.0.0.1:5432->5432/tcp
# stringsight-redis-1     Up 5 minutes     127.0.0.1:6379->6379/tcp
```

**Restarting services:**
```bash
# Restart specific service
docker compose restart api
docker compose restart worker

# Restart all
docker compose restart
```

**Rebuilding after code changes:**
```bash
# Rebuild images
docker compose build

# Rebuild and restart
docker compose up -d --build

# Force rebuild (no cache)
docker compose build --no-cache
```

**Stopping and cleaning up:**
```bash
# Stop containers (keep volumes)
docker compose down

# Stop and remove volumes (DELETES DATA)
docker compose down -v

# Stop, remove volumes, and images
docker compose down -v --rmi all
```

**Accessing containers:**
```bash
# Open shell in running container
docker compose exec api bash
docker compose exec worker bash

# Run one-off command
docker compose exec api python -c "import stringsight; print(stringsight.__version__)"

# Access PostgreSQL
docker compose exec db psql -U stringsight -d stringsight
```

**Viewing resource usage:**
```bash
# See CPU, memory, network usage
docker stats

# Example output:
# CONTAINER           CPU %     MEM USAGE / LIMIT     NET I/O
# stringsight-worker  45.2%     1.2GB / 8GB          12MB / 8MB
# stringsight-api     2.3%      350MB / 8GB          5MB / 3MB
```

### Kubernetes Considerations (Future)

**Note:** StringSight doesn't currently have Kubernetes manifests, but here's how it would work:

**Architecture Changes Needed:**

**1. StatefulSet for PostgreSQL:**
```yaml
# Replace docker-compose db service with:
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

**2. Deployment for API:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stringsight-api
spec:
  replicas: 3                        # Scale horizontally
  selector:
    matchLabels:
      app: stringsight-api
  template:
    spec:
      containers:
      - name: api
        image: stringsight:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

**3. Deployment for Celery Workers:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stringsight-worker
spec:
  replicas: 5                        # Scale based on load
  template:
    spec:
      containers:
      - name: worker
        image: stringsight:latest
        command: ["celery", "-A", "stringsight.celery_app", "worker"]
```

**4. Service for API:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: stringsight-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: stringsight-api
```

**5. ConfigMap for Environment:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: stringsight-config
data:
  REDIS_URL: "redis://redis-service:6379/0"
  DATABASE_URL: "postgresql://postgres-service:5432/stringsight"
```

**6. Secret for API Keys:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
data:
  openai-key: <base64-encoded>
  anthropic-key: <base64-encoded>
```

**Benefits of Kubernetes:**
- **Auto-scaling**: Scale workers based on queue length
- **Self-healing**: Restart failed containers
- **Load balancing**: Distribute API requests
- **Rolling updates**: Zero-downtime deployments
- **Resource management**: CPU/memory limits and requests

**Challenges:**
- **Complexity**: Steeper learning curve than docker-compose
- **Cost**: Cloud Kubernetes clusters not free
- **Storage**: Need persistent volume claims for PostgreSQL
- **Networking**: More complex than docker-compose networking

## Common Interview Questions & Answers

### Q1: How does the system scale? What are the bottlenecks?

**Answer:**
The main bottleneck is LLM API calls during property extraction. We address this with:

1. **Parallel extraction**: ThreadPoolExecutor for concurrent API calls
2. **Prompt caching**: Provider-specific caching (e.g., Anthropic's prompt caching) reduces costs by 90%
3. **LMDB caching**: Local cache prevents redundant calls for identical inputs
4. **Sampling**: For large datasets (>1000 conversations), we sample for property extraction
5. **Background jobs**: Celery + Redis for long-running analyses (24hr timeout)

**Scaling limits:**
- Single extraction job: ~10-50 conversations/minute (depends on API rate limits)
- Clustering: Fast (HDBSCAN is O(n log n)), handles 10,000+ properties easily
- Frontend: In-memory data model limits to ~100K rows (typical: 1-10K)

**Future improvements:**
- Batch API calls (when supported by providers)
- Distributed Celery workers for parallel jobs
- Streaming results to frontend for large datasets

### Q2: Why HDBSCAN over K-means or other clustering algorithms?

**Answer:**
HDBSCAN offers critical advantages for behavioral property clustering:

1. **No pre-specified cluster count**: Don't know how many behavior patterns exist upfront
2. **Noise handling**: Some properties are outliers (don't fit any cluster) - HDBSCAN labels these as -1
3. **Varying density**: Some behaviors are common (dense clusters), others rare (sparse clusters)
4. **Hierarchical structure**: Can explore different granularities of clustering

**K-means drawbacks:**
- Requires specifying K upfront
- Forces every point into a cluster (no noise concept)
- Assumes spherical clusters of similar size
- Sensitive to initialization

**Alternative (Hierarchical):**
- We also support scipy's hierarchical clustering
- Better when user wants specific number of clusters
- More deterministic than HDBSCAN

### Q3: How do you ensure data consistency between Python backend and JavaScript frontend?

**Answer:**
Several strategies ensure consistency:

1. **Type enforcement**: All IDs are strings (not ints) to prevent precision loss in JavaScript
   ```python
   # Enforced in data_objects.py
   question_id: str  # Not int!
   ```

2. **Score format contract**: Backend returns nested dicts, frontend flattens for UI
   ```python
   # Backend
   {"score": {"accuracy": 0.8, "fluency": 0.9}}

   # Frontend operational layer (same)
   {"score": {"accuracy": 0.8, "fluency": 0.9}}

   # Frontend display layer (flattened)
   {"score_accuracy": 0.8, "score_fluency": 0.9}
   ```

3. **Method detection synchronization**: Same logic in both backend and frontend
   - Backend: `stringsight/formatters/traces.py`
   - Frontend: `src/lib/api.ts`

4. **API schema validation**: FastAPI + Pydantic for request/response validation

5. **TypeScript types**: Generated from API responses ensure type safety

### Q4: Walk me through what happens when a user uploads a CSV and clicks "Analyze"

**Answer:**

**Frontend (React):**
1. Parse CSV with PapaParse → raw JavaScript objects
2. Detect method (single_model vs side_by_side) from columns
3. Validate required columns exist
4. Transform to operational format (nested score objects)
5. POST to `/api/v1/jobs/pipeline` with data + config

**Backend (FastAPI):**
6. Create Job record in database (status: "pending")
7. Enqueue Celery task with job_id
8. Return job_id to frontend immediately

**Celery Worker (Background):**
9. Update Job status → "running"
10. **Property Extraction Stage:**
    - Sample conversations (if >1000)
    - Select/generate system prompt
    - Parallel LLM calls via LiteLLM
    - Cache responses in LMDB
    - Parse JSON → Property objects
    - Update progress in database
11. **Validation Stage:**
    - Filter properties (min length, required fields)
    - Deduplicate identical properties
12. **Clustering Stage:**
    - Generate embeddings (OpenAI or local)
    - Run HDBSCAN clustering
    - Generate cluster labels via LLM
13. **Metrics Stage:**
    - Compute per-model cluster scores
    - Compute aggregated cluster scores
    - Compute overall model scores
14. Save results to files (Parquet, JSON, summary.txt)
15. Update Job status → "completed"

**Frontend (Polling):**
16. Poll `/api/v1/jobs/{job_id}` every 2 seconds
17. Display progress updates
18. When completed, fetch results
19. Transform data (flatten scores, enrich clusters)
20. Render Data/Properties/Clusters/Metrics tabs

**Typical timeline:**
- Small dataset (100 conversations): 2-5 minutes
- Medium dataset (1000 conversations): 10-20 minutes
- Large dataset (10,000 conversations): 30-60 minutes (with sampling)

### Q5: How does the dynamic prompt generation work?

**Answer:**
Dynamic prompts are auto-generated from sample data in three stages:

**Stage 1: Task Expansion**
```python
# Input: 5-10 sample conversations
# LLM generates: detailed task description

prompt = f"""
Given these example conversations, describe:
1. What task is being performed?
2. What should we analyze about model behavior?
3. What properties would be most informative?
"""
```

**Stage 2: Verification**
```python
# Extract properties using generated prompt
# Check if properties are:
# - Specific enough (not generic)
# - Evidence-based (quotes from response)
# - Relevant to task
```

**Stage 3: Reflection (if needed)**
```python
# If verification fails, refine prompt:
# - Add constraints for specificity
# - Request stronger evidence requirements
# - Clarify behavior types to focus on
```

**Benefits:**
- Task-specific extraction (better than universal prompt)
- Automatic adaptation to new domains
- No manual prompt engineering

**Limitations:**
- Requires 5-10 representative samples
- More expensive (3 LLM calls vs 1)
- Less predictable than fixed prompts

**Located in:** `stringsight/prompts/dynamic/`

### Q6: What's the purpose of the mixins pattern? Why not inheritance?

**Answer:**
Mixins provide **cross-cutting concerns** without deep inheritance hierarchies:

```python
class Pipeline(
    LoggingMixin,        # Structured logging
    TimingMixin,         # Performance tracking
    ErrorHandlingMixin,  # Exception handling
    WandbMixin          # Experiment tracking
):
    # Pipeline only contains core orchestration logic
    # All observability concerns come from mixins
```

**Benefits:**
1. **Separation of concerns**: Core logic separate from observability
2. **Reusability**: Mixins used by both Pipeline and PipelineStage
3. **Composability**: Pick and choose which mixins to include
4. **No diamond problem**: Multiple inheritance works cleanly with mixins

**Example mixin:**
```python
class TimingMixin:
    def __init__(self):
        self._timings = {}

    def time_operation(self, name: str):
        # Context manager for timing
        start = time.time()
        yield
        duration = time.time() - start
        self._timings[name] = duration
        self.logger.info(f"{name} took {duration:.2f}s")
```

### Q7: How do you handle multimodal conversations (text + images)?

**Answer:**
We support **OpenAI conversation format** which handles multimodal content:

```python
model_response = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/..."}}
        ]
    },
    {
        "role": "assistant",
        "content": "I see a cat sitting on a couch..."
    }
]
```

**Backend handling:**
- Store as JSON string in database
- Pass directly to property extraction (LLM sees full context)
- Evidence extraction works with text portions

**Frontend handling:**
- `ConversationTrace.tsx` renders multimodal content
- Images displayed inline in conversation view
- Evidence highlighting works on text portions only

**Limitations:**
- Images must be base64-encoded or URLs
- Very large images increase context size
- Evidence highlighting doesn't support image content

### Q8: What's your testing strategy?

**Answer:**
Multi-level testing approach:

**1. Unit Tests** (`tests/`):
```python
# Test core data transformations
test_prompts_metadata_unit.py
test_airline_demo_prompt_generation.py
```

**2. Integration Tests** (manual):
```bash
# End-to-end pipeline test
python scripts/run_pipeline.py --dataset demo
```

**3. Demo Dataset** (`public/taubench_airline.jsonl`):
- Known-good dataset for regression testing
- Tests full pipeline: extraction → clustering → metrics
- Catches UI rendering issues

**4. Configuration-based Testing** (`scripts/dataset_configs/`):
```yaml
# YAML configs for repeatable analyses
dataset: taubench_airline
method: side_by_side
extraction_model: gpt-4o-mini
# ... more config
```

**5. Frontend Testing**:
- Type safety via TypeScript (compile-time checks)
- Manual testing with various datasets
- Browser console monitoring for errors

**Gaps (improvements needed):**
- No automated E2E tests (Playwright/Cypress)
- Limited edge case coverage
- No performance benchmarks
- No stress testing for large datasets

### Q9: How do you handle LLM provider rate limits and failures?

**Answer:**
Multiple strategies for robustness:

**1. LiteLLM Built-in Retry:**
```python
# LiteLLM handles transient failures automatically
response = litellm.completion(
    model=model_name,
    messages=messages,
    num_retries=3  # Exponential backoff
)
```

**2. LMDB Caching:**
```python
# Cache keyed by (model, prompt, conversation_sample)
cache_key = hash((model, prompt, str(conversation)))
if cache_key in cache:
    return cached_response  # Skip API call entirely
```

**3. Parallel Extraction with Error Handling:**
```python
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(extract_one, conv) for conv in convs]
    for future in as_completed(futures):
        try:
            result = future.result(timeout=300)  # 5min per conversation
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Continue with other conversations
```

**4. Job-Level Error Recovery:**
- If extraction fails for some conversations, continue with rest
- Save partial results
- User can retry failed conversations

**5. Provider Fallback** (future improvement):
```python
# Could implement automatic fallback
try:
    extract(model="gpt-4o-mini")
except RateLimitError:
    extract(model="claude-3-5-sonnet")  # Fallback provider
```

### Q10: How would you add support for a new LLM provider?

**Answer:**
Very straightforward due to LiteLLM abstraction:

**1. If LiteLLM already supports it:**
```python
# Just use the provider's model name
results = ss.explain(
    df=data,
    extraction_model="provider/model-name",  # e.g., "cohere/command-r-plus"
    # No code changes needed!
)
```

**2. If custom provider (not in LiteLLM):**

**Option A: Add to LiteLLM** (preferred):
- Contribute to LiteLLM repo
- Follow their provider template
- Benefits entire ecosystem

**Option B: Custom extractor:**
```python
# stringsight/extractors/custom_provider.py
class CustomProviderExtractor(BaseExtractor):
    def extract_properties(
        self,
        conversations: list[ConversationRecord],
        system_prompt: str
    ) -> list[Property]:
        # Call custom provider API
        responses = self.call_custom_api(conversations)
        # Parse into Property objects
        return self.parse_responses(responses)
```

**3. Update configuration:**
```python
# stringsight/config.py
SUPPORTED_PROVIDERS = [
    "openai",
    "anthropic",
    "google",
    "custom_provider"  # Add new
]
```

**4. Testing:**
- Add test config: `scripts/dataset_configs/test_custom_provider.yaml`
- Run on demo dataset
- Verify property quality

### Q11: Explain the evidence highlighting algorithm in the frontend

**Answer:**
Sophisticated two-strategy matching in `ConversationTrace.tsx`:

**Strategy 1: Exact Match (case-insensitive)**
```typescript
const normalized = text.toLowerCase().replace(/\s+/g, ' ')
const evidenceNorm = evidence.toLowerCase().replace(/\s+/g, ' ')
const index = normalized.indexOf(evidenceNorm)
if (index >= 0) {
    // Map back to original positions (accounting for whitespace)
    return highlightRange(originalText, index, index + evidence.length)
}
```

**Strategy 2: Fuzzy Match (Jaccard similarity ≥75%)**
```typescript
function jaccardSimilarity(a: string, b: string): number {
    const wordsA = new Set(a.toLowerCase().split(/\s+/))
    const wordsB = new Set(b.toLowerCase().split(/\s+/))

    const intersection = [...wordsA].filter(w => wordsB.has(w)).length
    const union = new Set([...wordsA, ...wordsB]).size

    return intersection / union
}

// Slide window over text to find best match
for (let i = 0; i < words.length - windowSize; i++) {
    const segment = words.slice(i, i + windowSize).join(' ')
    const similarity = jaccardSimilarity(segment, evidence)
    if (similarity >= 0.75) {
        return highlightSegment(segment)
    }
}
```

**Why two strategies?**
- Exact: Fast, handles minor whitespace variations
- Fuzzy: Handles paraphrasing, ellipses ("..." in evidence), partial quotes

**Edge cases handled:**
- Multiple occurrences: Highlights first match only
- Overlapping highlights: Merges into single span
- Evidence not found: No highlight (fails gracefully)
- Multimodal: Only highlights text portions

**Position mapping complexity:**
- Normalized text removes whitespace for matching
- Must map positions back to original text with formatting
- Preserves HTML structure during highlighting

### Q12: What are the main differences between single_model and side_by_side modes?

**Answer:**

**Single Model Mode:**
```python
# Required columns
- prompt
- model
- model_response
- score (optional dict)

# Use case: Analyze one model's behaviors
# Example: "What behaviors does GPT-4 exhibit on math problems?"

# Metrics:
- Property prevalence per model
- Cluster sizes
- Average scores per cluster
```

**Side-by-Side Mode:**
```python
# Required columns
- prompt
- model_a, model_b
- model_a_response, model_b_response
- score_a, score_b (optional dicts with 'win' field)

# Use case: Compare two models head-to-head
# Example: "How does GPT-4 differ from Claude on customer support?"

# Metrics:
- Win rates per cluster (which model exhibits behavior more)
- Relative prevalence (Model A: 30%, Model B: 15%)
- Differential analysis (behaviors unique to one model)
```

**Data Format Options:**

**Option 1: Pre-paired**
```python
# Each row is a comparison
df = pd.DataFrame({
    'prompt': ['What is 2+2?'],
    'model_a': ['gpt-4'],
    'model_b': ['claude-3'],
    'model_a_response': ['The answer is 4'],
    'model_b_response': ['2+2 equals 4']
})
```

**Option 2: Tidy data (auto-paired)**
```python
# Separate rows per model, auto-paired by prompt
df = pd.DataFrame({
    'prompt': ['What is 2+2?', 'What is 2+2?'],
    'model': ['gpt-4', 'claude-3'],
    'model_response': ['The answer is 4', '2+2 equals 4']
})

# Specify method='side_by_side' to auto-pair
results = ss.explain(df, method='side_by_side')
```

**Pipeline differences:**
- Single model: Properties extracted per model separately
- Side-by-side: Properties extracted for both models, then compared
- Metrics computation logic differs significantly

### Q13: How would you debug a production issue where clustering results seem wrong?

**Answer:**
Systematic debugging approach:

**1. Check Intermediate Results:**
```bash
# Pipeline saves intermediate files
ls ~/.stringsight/results/{job_id}/
# - raw_properties.jsonl (after extraction)
# - validated_properties.jsonl (after validation)
# - embeddings.npy (after embedding)
# - clustered_results.parquet (final)
```

**2. Validate Property Quality:**
```python
# Load and inspect properties
import pandas as pd
props = pd.read_json('raw_properties.jsonl', lines=True)

# Check common issues:
props['description_length'] = props['property_description'].str.len()
props['description_length'].describe()  # Should be >50 chars

# Look for generic properties
generic = props[props['property_description'].str.contains('provides|explains|uses')]
print(f"{len(generic)} generic properties found")

# Check evidence quality
no_evidence = props[props['evidence'].str.len() < 10]
print(f"{len(no_evidence)} properties with weak evidence")
```

**3. Examine Embeddings:**
```python
# Visualize embedding space
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

embeddings = np.load('embeddings.npy')
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

fig = px.scatter(x=coords[:, 0], y=coords[:, 1],
                 color=labels, hover_data=[props['property_description']])
fig.show()  # Should see clear clusters
```

**4. Tune Clustering Parameters:**
```python
# Try different min_cluster_size
results = ss.explain(
    df=data,
    min_cluster_size=3,  # Default is 5 (try smaller for more clusters)
    # Or try hierarchical clustering
    clustering_method='hierarchical',
    num_clusters=15  # Specify exact number
)
```

**5. Check Prompt Quality:**
```python
# View extraction prompt used
with open('~/.stringsight/results/{job_id}/extraction_prompt.txt') as f:
    print(f.read())

# Common issues:
# - Too generic (use dynamic prompts instead)
# - Wrong behavior types requested
# - Insufficient examples in context
```

**6. Review Logs:**
```bash
# Check for API errors or rate limiting
stringsight logs | grep -i error

# Check timing (slow extractions might indicate issues)
stringsight logs | grep "took"
```

**7. Test with Known-Good Data:**
```python
# Compare against demo dataset
demo_results = ss.explain(
    df=pd.read_json('public/taubench_airline.jsonl', lines=True),
    extraction_model=same_model_as_problematic_run
)
# If demo works but production doesn't → data quality issue
# If both fail → model or prompt issue
```

**Common Root Causes:**
1. **Generic properties** → Use dynamic prompts or more specific instructions
2. **Insufficient properties** → Lower validation thresholds, extract more conversations
3. **Too many clusters** → Increase min_cluster_size
4. **Poor cluster labels** → Improve label generation prompt, use more representative properties
5. **Inconsistent embeddings** → Check embedding model version, regenerate embeddings

### Q14: Explain how a background job works from start to finish using Redis and Celery

**Answer:**
This is a great question because it tests understanding of the entire system working together. Let me walk through a complete example of what happens when a user uploads a dataset with 1000 conversations and clicks "Analyze."

The key insight is that there are **5 distinct phases** that happen across **3 different systems** (FastAPI, Redis, Celery worker), with **2 databases** tracking state (PostgreSQL for persistence, Redis for real-time status). Let's trace a single job through the entire lifecycle:

**Phase 1: API Receives Request (FastAPI)**
```python
# stringsight/routers/jobs.py
@router.post("/api/v1/jobs/pipeline")
async def create_pipeline_job(req: PipelineJobRequest):
    # 1. Generate unique job ID
    job_id = uuid.uuid4()

    # 2. Create Job record in PostgreSQL
    job = Job(
        id=job_id,
        type="pipeline",
        status="pending",        # Initial state
        progress=0.0,
        created_at=datetime.now()
    )
    db.add(job)
    db.commit()

    # 3. Enqueue task to Redis (non-blocking)
    run_pipeline_job.delay(str(job_id), req.dict())
    #                ^^^^^ This returns immediately

    # 4. Return to user (< 100ms)
    return {"job_id": str(job_id), "status": "pending"}
```

**Phase 2: Task Enqueued to Redis**
```python
# What .delay() does internally:
# 1. Serialize task arguments to JSON
task_data = {
    "id": "task-uuid-xyz",
    "task": "stringsight.workers.tasks.run_pipeline_job",
    "args": [job_id, req_data],
    "kwargs": {},
    "eta": None,
    "expires": None
}

# 2. Push to Redis list
redis.lpush("celery", json.dumps(task_data))

# Redis now contains:
# KEY: "celery"
# VALUE: ["task-1", "task-2", "task-3", ...]  (FIFO queue)
```

**Phase 3: Worker Discovers Task**
```bash
# Worker is running in container/process:
celery -A stringsight.celery_app worker --loglevel=info

# Worker loop (simplified):
while True:
    # 1. Poll Redis for tasks (blocking pop with timeout)
    task_data = redis.brpop("celery", timeout=1)

    if task_data:
        # 2. Deserialize task
        task = json.loads(task_data)

        # 3. Look up task function by name
        func = registry.get("stringsight.workers.tasks.run_pipeline_job")

        # 4. Execute in worker process
        try:
            func(*task['args'], **task['kwargs'])
        except Exception as e:
            # Store failure in Redis
            redis.set(f"celery-task-{task['id']}", {"status": "FAILURE", "traceback": str(e)})
```

**Phase 4: Worker Executes Task**
```python
# stringsight/workers/tasks.py
def _run_pipeline_job(job_id: str, req_data: Dict[str, Any]):
    # 1. Update status in PostgreSQL
    job = db.query(Job).filter(Job.id == job_id).first()
    job.status = "running"
    job.progress = 0.0
    db.commit()

    # 2. Run analysis (10-30 minutes)
    df = pd.DataFrame(req_data['rows'])

    # Progress callback updates PostgreSQL
    def update_progress(progress: float):
        with SessionLocal() as session:
            current_job = session.query(Job).filter(Job.id == job_id).first()
            current_job.progress = progress
            session.commit()

    # Run pipeline with progress updates
    results = explain(
        df,
        progress_callback=update_progress
    )
    # This calls update_progress(0.1), update_progress(0.2), ..., update_progress(1.0)

    # 3. Save results to disk
    output_dir = f"~/.stringsight/results/{job_id}"
    results.save(output_dir)

    # 4. Update job completion in PostgreSQL
    job.status = "completed"
    job.progress = 1.0
    job.result_path = output_dir
    db.commit()
```

**Phase 5: Frontend Polls for Updates**
```typescript
// Frontend starts polling immediately after enqueuing
const pollInterval = setInterval(async () => {
    const response = await fetch(`/api/v1/jobs/${jobId}`)
    const job = await response.json()

    // Update UI with progress
    setProgress(job.progress)          // 0.0 → 0.1 → 0.2 → ... → 1.0
    setStatus(job.status)              // "pending" → "running" → "completed"

    if (job.status === "completed") {
        clearInterval(pollInterval)     // Stop polling
        loadResults(job.result_path)    // Fetch results
    }

    if (job.status === "failed") {
        clearInterval(pollInterval)
        showError(job.error_message)
    }
}, 2000)  // Poll every 2 seconds
```

**Redis State Throughout:**
```
T=0s:   Redis queue: ["task-xyz"]              PostgreSQL: status="pending"
T=1s:   Redis queue: []                        PostgreSQL: status="running", progress=0.0
T=60s:  Redis result: {"status": "RUNNING"}    PostgreSQL: status="running", progress=0.15
T=120s: Redis result: {"status": "RUNNING"}    PostgreSQL: status="running", progress=0.30
...
T=600s: Redis result: {"status": "SUCCESS"}    PostgreSQL: status="completed", progress=1.0
```

**Why Both Redis AND PostgreSQL?**

This is a common point of confusion - why two databases? The answer is that they serve different purposes:

- **Redis**: Optimized for speed and real-time updates. It's in-memory, so reads/writes are microseconds. Perfect for "is my job done yet?" queries that happen every 2 seconds. But it's ephemeral - data expires after 24 hours to prevent memory bloat.

- **PostgreSQL**: Optimized for durability and complex queries. Slower than Redis (milliseconds not microseconds), but data persists forever. Perfect for "show me all jobs I ran last month" or "what was the error message from that job that failed yesterday?"

**Real-world analogy**: Redis is like your working memory - fast but temporary. PostgreSQL is like your long-term memory - slower to access but permanent. You need both for different purposes.

**Why not just use PostgreSQL for everything?** We could, but polling PostgreSQL every 2 seconds from potentially hundreds of frontend clients would overwhelm the database. Redis is designed for high-frequency reads/writes without breaking a sweat.

**Worker Restart After 10 Tasks:**
```python
# After 10 tasks, worker process exits and restarts
worker_max_tasks_per_child=10

# Timeline:
# Task 1: Worker process PID=1234 (memory: 200MB)
# Task 2: Worker process PID=1234 (memory: 350MB)
# ...
# Task 9: Worker process PID=1234 (memory: 1.8GB)  # Memory leak!
# Task 10: Worker process PID=1234 (memory: 2.1GB)
# *** Worker restarts ***
# Task 11: Worker process PID=5678 (memory: 200MB)  # Fresh start
```

### Q15: How would you debug a stuck Celery task?

**Answer:**
This is a practical question you'll face in production. A task shows "running" for 3 hours when it should take 20 minutes. Here's the systematic approach I'd take, moving from quickest checks to deeper investigation:

**The debugging philosophy**: Start with observable state (what can we see from outside?), then move to internal state (what's happening inside the worker?), and finally to root cause analysis (why is it stuck?).

**1. Check Task State in Redis:**
```bash
# Connect to Redis
docker compose exec redis redis-cli

# List all keys
KEYS *

# Check task state
GET celery-task-meta-{task_id}
# Returns: {"status": "PENDING"/"STARTED"/"SUCCESS"/"FAILURE"}

# Check queue length
LLEN celery
# If > 0, tasks are queued (worker might be down)
```

**2. Check Worker Status:**
```bash
# Are workers running?
docker compose ps worker
# Should show "Up X minutes"

# View worker logs
docker compose logs --tail=100 -f worker

# Look for:
# - "Task ... received" (task picked up)
# - "Task ... succeeded" (task completed)
# - Exceptions or tracebacks (errors)
```

**3. Check Job Status in PostgreSQL:**
```bash
# Connect to database
docker compose exec db psql -U stringsight -d stringsight

# Query job
SELECT id, status, progress, error_message, created_at, updated_at
FROM jobs
WHERE id = '{job_id}';

# Check if progress is updating
# If progress stuck at same value → task is hung
# If updated_at not changing → task not running
```

**4. Inspect Worker Process:**
```bash
# Enter worker container
docker compose exec worker bash

# Check CPU usage (is it doing work?)
top
# If CPU near 0% → task is hung (likely waiting on I/O or deadlock)
# If CPU near 100% → task is working (be patient)

# Check memory
free -h
# If memory near limit → possible memory leak

# Check network (if stuck on API calls)
netstat -an | grep ESTABLISHED
# Should see connections to OpenAI, Anthropic, etc.
```

**5. Check for Common Issues:**

**Issue: Task not picked up by worker**
```bash
# Check if task is in queue
redis-cli LLEN celery

# If > 0, worker not consuming
# Solution: Restart worker
docker compose restart worker
```

**Issue: Task stuck on LLM API call**
```python
# Check if extraction is hanging
# Look in logs for last completed conversation
docker compose logs worker | grep "Extracted properties for"

# If stuck on same conversation for > 5 minutes:
# - Possible rate limiting from provider
# - Network timeout
# - API key invalid

# Solution: Check API key, restart task
```

**Issue: Database deadlock**
```sql
-- Check for locks in PostgreSQL
SELECT * FROM pg_locks WHERE NOT granted;

-- Check for long-running queries
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- Kill long-running query
SELECT pg_terminate_backend(pid);
```

**Issue: Out of memory**
```bash
# Check worker memory
docker stats stringsight-worker-1

# If memory near limit:
# - Worker will be OOM killed by kernel
# - Check logs for "Killed" message

# Solution: Increase memory limit or reduce batch size
docker compose down
# Edit docker-compose.yml to add:
# services:
#   worker:
#     mem_limit: 4g      # Increase from default
docker compose up -d
```

**6. Manual Task Retry:**
```python
# In Python shell (inside worker container)
from stringsight.celery_app import celery_app
from stringsight.workers.tasks import run_pipeline_job

# Manually trigger task (bypasses queue)
run_pipeline_job("job-id-here", req_data)

# Or revoke stuck task and requeue
celery_app.control.revoke("task-id-here", terminate=True)
run_pipeline_job.delay("job-id-here", req_data)
```

**7. Enable Celery Debug Logging:**
```bash
# Restart worker with debug logging
docker compose down worker
docker compose run --rm worker \
    celery -A stringsight.celery_app worker --loglevel=debug

# Shows detailed task state transitions
# [DEBUG] Task received: ...
# [DEBUG] Task started: ...
# [DEBUG] Task succeeded: ...
```

**Prevention Strategies:**
```python
# 1. Add task timeouts
@celery_app.task(bind=True, time_limit=3600)  # 1 hour hard limit
def run_pipeline_job(self, job_id, req_data):
    ...

# 2. Add progress checkpoints
# If no progress update in 5 minutes, task is likely stuck

# 3. Add healthchecks
# Periodically check if task is still making progress

# 4. Use task retries
@celery_app.task(bind=True, max_retries=3)
def run_pipeline_job(self, job_id, req_data):
    try:
        ...
    except Exception as e:
        self.retry(countdown=60, exc=e)  # Retry after 60 seconds
```

### Q16: Why use Docker Compose instead of just running services locally?

**Answer:**

This question gets at a fundamental choice in development workflow. You could absolutely run PostgreSQL, Redis, and the API directly on your machine - many developers do. So why add the complexity of Docker Compose?

The answer comes down to the **"works on my machine" problem** and **onboarding time**. Let me illustrate with a story:

Imagine a new developer joins the team. Without Docker:
1. They spend 30 minutes installing PostgreSQL, but their Mac has a newer version than production uses
2. They install Redis, but the config file location is different on Mac vs Linux
3. They install Python dependencies, but some native packages fail to compile because they're missing system libraries
4. After 2 hours of troubleshooting, they finally get it running - but it's subtly different from everyone else's setup

With Docker Compose:
1. They run `docker compose up -d`
2. Five minutes later, they have an identical environment to everyone else, guaranteed

**Benefits of Docker Compose:**

**1. Reproducible Environment**
```bash
# Without Docker:
# - Install PostgreSQL (version mismatch risk)
# - Install Redis (different config on Mac vs Linux)
# - Install Python deps (system-wide pollution)
# - Configure services manually

# With Docker:
docker compose up -d
# Everything configured correctly, same on all machines
```

**2. Service Isolation**
```bash
# Without Docker:
# PostgreSQL running on port 5432 globally
# Conflicts with other projects using PostgreSQL
# Hard to run multiple versions

# With Docker:
# Each project has own PostgreSQL container
# No conflicts, clean separation
```

**3. Easy Cleanup**
```bash
# Without Docker:
# Uninstall PostgreSQL, Redis
# Remove Python packages
# Clean up config files

# With Docker:
docker compose down -v
# Everything gone in seconds
```

**4. Onboarding Speed**
```bash
# Without Docker (new developer):
# 1. Install PostgreSQL (30 min + troubleshooting)
# 2. Install Redis (10 min)
# 3. Configure both (20 min + debugging)
# 4. Install Python deps (10 min + conflicts)
# Total: 1-2 hours

# With Docker (new developer):
docker compose up -d   # 5 minutes
```

**5. Production Parity**
```yaml
# Docker Compose config is similar to production deployment
# Reduces "works on my machine" issues
# Same Dockerfile used in dev and prod
```

**Tradeoffs:**

Nothing is free - Docker Compose adds complexity and overhead:

**Cons of Docker Compose:**
- **Slower than native**: There's a thin layer of overhead. For StringSight, this is negligible (we're bottlenecked on LLM API calls, not container startup), but for CPU-intensive apps it matters.
- **More disk space**: A full Docker image for StringSight is ~400MB. Multiply by several images (API, worker, PostgreSQL) and you're using 2-3GB. Not huge, but not zero.
- **Learning curve**: New developers need to learn Docker concepts (images, containers, volumes, networks). This is a one-time cost but can be intimidating.
- **Debugging can be trickier**: When something goes wrong, you can't just open the code in your editor and set breakpoints. You need to `docker exec` into containers, which feels less direct.

**When NOT to use Docker:**
- **Very simple single-service apps**: If you just have a single Python script with no dependencies, Docker is overkill.
- **Resource-constrained machines**: If you're on an old laptop with 4GB RAM, Docker's overhead (which can be 500MB-1GB) is significant.
- **When you need native performance**: GPU-accelerated ML training, real-time audio processing, etc. The container overhead (even if small) can matter.

**StringSight's Pragmatic Choice:**
We actually use BOTH approaches depending on the use case:
- **Local CLI mode** (`stringsight launch`): No Docker. Just runs on your machine with SQLite. Simple, fast, perfect for individual users trying it out.
- **Development mode**: Docker Compose. Reproducible, isolated, matches production architecture. Used by the core team when building features.
- **Production**: Docker Compose (small deployments) or Kubernetes (large scale). Need reliability, monitoring, scaling - Docker excels here.

### Q17: What security considerations exist in the system?

**Answer:**

**1. API Key Management:**
```python
# Keys stored in environment variables (never in code)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# .env file in .gitignore
# .env.example provided for documentation
```

**2. File Upload Security:**
```python
# Frontend: Client-side parsing (PapaParse)
# - No file uploads to server during upload phase
# - Data stays in browser until "Analyze" clicked

# Backend: Validation
# - Max file size limits (enforced by FastAPI)
# - Only accept specific content types (CSV, JSON, JSONL)
# - No arbitrary file execution
```

**3. SQL Injection Prevention:**
```python
# SQLAlchemy ORM (parameterized queries)
job = session.query(Job).filter(Job.id == job_id).first()
# Never: session.execute(f"SELECT * FROM jobs WHERE id = {job_id}")
```

**4. XSS Prevention:**
```typescript
// React automatically escapes content
<div>{property.description}</div>  // Safe

// Dangerously set HTML only for controlled content
<div dangerouslySetInnerHTML={{__html: sanitizedHTML}} />
// Always sanitize first!
```

**5. API Rate Limiting** (future improvement):
```python
# Should add rate limiting to public endpoints
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
@app.post("/api/v1/jobs/pipeline")
@limiter.limit("10/minute")
async def create_pipeline_job(...):
    ...
```

**6. Authentication** (basic implementation):
```python
# User model exists (stringsight/db_models/user.py)
# But no authentication enforced yet
# Production deployment should add:
# - JWT tokens
# - OAuth integration
# - API key management per user
```

**7. Data Privacy:**
```python
# User data stored locally (~/.stringsight/)
# Docker deployment: PostgreSQL with credentials
# S3 storage (optional): User-provided credentials

# Concerns:
# - LLM provider sees user data (sent for extraction)
# - No data encryption at rest (add for sensitive data)
```

**8. Secrets in Results:**
```python
# Risk: User data might contain API keys, passwords
# Mitigation: Warn users not to upload sensitive data
# Future: Add secret detection (regex patterns) before extraction
```

## Performance Characteristics

### Bottlenecks & Optimizations

**1. LLM API Calls (Biggest Bottleneck):**
- Typical: 1-2 seconds per conversation
- 100 conversations: 2-3 minutes (with parallelism)
- 1000 conversations: 20-30 minutes
- **Optimization**: Prompt caching (90% cost reduction on repeated structure)

**2. Embedding Generation:**
- OpenAI: ~100 properties/second
- Local (sentence-transformers): ~1000 properties/second on GPU
- **Optimization**: LMDB caching, batch API calls

**3. Clustering:**
- HDBSCAN: O(n log n) → 10,000 properties in <10 seconds
- Not a bottleneck until >100K properties

**4. Frontend Rendering:**
- Data table: Handles up to 100K rows (virtual scrolling)
- Conversation traces: Lazy loading
- Evidence highlighting: Computed on demand (not pre-computed)

**5. Database:**
- SQLite: Good for <1000 concurrent users
- PostgreSQL: Required for production (multiple workers)

**Memory Usage:**
- Backend: ~500MB base + 100MB per 1000 properties
- Frontend: ~10MB per 1000 rows in table
- Celery worker: Restarts after 10 tasks (memory leak mitigation)

## Key Technical Decisions & Tradeoffs

### 1. LiteLLM vs Direct API Clients
**Decision**: Use LiteLLM for all LLM calls

**Pros:**
- Multi-provider support (OpenAI, Anthropic, Google, etc.)
- Unified interface
- Built-in retry logic

**Cons:**
- Extra dependency
- Less control over request details
- Potential bugs in LiteLLM

**Alternative considered**: Direct API clients per provider (rejected - too much code duplication)

### 2. HDBSCAN vs K-means
**Decision**: HDBSCAN as default, hierarchical as alternative

**Reasoning**: See Q2 above (no pre-specified K, noise handling, varying density)

### 3. Three Frontend Data Layers
**Decision**: Separate originalRows, operationalRows, currentRows

**Pros:**
- Clear separation of concerns
- Easy to re-analyze original data
- UI and backend have different needs

**Cons:**
- Memory overhead (3x data duplication)
- Complexity in keeping layers synchronized

**Alternative considered**: Single data layer with transform functions (rejected - too error-prone)

### 4. Celery + Redis vs Simple Threading
**Decision**: Celery for background jobs

**Pros:**
- True background execution (user can close browser)
- Distributed workers (can scale horizontally)
- Task timeout and retry support
- Progress tracking via database

**Cons:**
- Complex infrastructure (Redis, workers)
- Harder to debug
- Memory leak issues (hence restart after 10 tasks)

**Alternative considered**: Simple threading with FastAPI BackgroundTasks (rejected - no persistence, no distributed execution)

### 5. LMDB vs Redis for LLM Caching
**Decision**: LMDB (local file-based key-value store)

**Pros:**
- No external service required
- Fast (memory-mapped file)
- Persistent across restarts
- Simple API

**Cons:**
- Not distributed (each worker has own cache)
- Not shared across machines

**Alternative considered**: Redis (rejected - want caching even in local CLI mode)

### 6. Parquet vs JSON for Results
**Decision**: Support both (Parquet for analysis, JSON for portability)

**Pros:**
- Parquet: Efficient for pandas/data science workflows
- JSON: Human-readable, easy to inspect
- JSONL: Streamable, line-by-line processing

**Cons:**
- Duplicate storage (both formats saved)

### 7. Frontend Submodule vs Monorepo
**Decision**: Separate frontend repo as git submodule

**Pros:**
- Independent versioning
- Separate development workflows
- Can use frontend without backend

**Cons:**
- Submodule complexity (easy to forget `--recurse-submodules`)
- Need to keep submodule pointer updated

**Alternative considered**: Monorepo (rejected - different tech stacks, different contributors)

## Future Improvements & Roadmap

**Near-term:**
1. Enhanced caching (distributed Redis cache)
2. Streaming results to frontend (Server-Sent Events)
3. Batch LLM API calls (when supported by providers)
4. Automated testing (E2E with Playwright)
5. Rate limiting and authentication

**Medium-term:**
1. Support for more clustering algorithms (DBSCAN, Spectral)
2. Interactive cluster refinement (merge/split clusters in UI)
3. Export to various formats (PDF reports, Slides)
4. Comparison across multiple datasets
5. Statistical significance testing for metrics

**Long-term:**
1. Active learning (iteratively improve extraction prompts)
2. Custom taxonomy creation (user defines behavior categories)
3. Multi-tenant deployment (SaaS version)
4. Integration with LLM evaluation platforms (OpenAI Evals, etc.)
5. Real-time monitoring (production LLM behavior tracking)

## Monitoring and Observability

### Current Logging Setup

**Structured Logging:**
```python
# stringsight/config.py
LOG_LEVEL: str = "INFO"
json_logs: bool = True              # JSON-formatted logs for parsing

# Example log entry
{
    "timestamp": "2024-01-15T10:30:45Z",
    "level": "INFO",
    "logger": "stringsight.workers.tasks",
    "message": "Started pipeline job",
    "job_id": "abc-123",
    "num_conversations": 1000
}
```

**Log Levels Used:**
- **DEBUG**: Detailed task state, API calls
- **INFO**: Job start/complete, progress milestones
- **WARNING**: Skipped properties, retries
- **ERROR**: Task failures, API errors

**Viewing Logs:**
```bash
# Docker Compose
docker compose logs -f worker        # Follow worker logs
docker compose logs --tail=500 api   # Last 500 lines from API

# Local CLI
stringsight logs                     # View application logs
tail -f ~/.stringsight/logs/app.log  # Follow log file

# Filter logs
docker compose logs worker | grep ERROR
docker compose logs api | grep "job_id=abc-123"
```

### Metrics to Monitor

**1. Task Queue Depth (Redis)**
```bash
# Check queue length
redis-cli LLEN celery

# If consistently > 10:
# - Workers can't keep up with demand
# - Need to scale workers horizontally

# Ideal: 0-5 (workers keep up with incoming jobs)
```

**2. Worker Health**
```bash
# Check worker processes
celery -A stringsight.celery_app inspect active

# Returns:
{
  "worker@hostname": [
    {
      "id": "task-id",
      "name": "run_pipeline_job",
      "time_start": 1234567890
    }
  ]
}

# Check worker stats
celery -A stringsight.celery_app inspect stats
```

**3. Database Connection Pool**
```sql
-- PostgreSQL connection count
SELECT count(*) FROM pg_stat_activity WHERE datname = 'stringsight';

-- If near max_connections (default 100):
# - Risk of connection exhaustion
# - Need connection pooling (pgbouncer) or increase limit
```

**4. Memory Usage**
```bash
# Per-container memory
docker stats

# Worker memory trend
# - Fresh worker: ~200MB
# - After 5 tasks: ~800MB
# - After 10 tasks: ~2GB → restart
# - After restart: ~200MB (healthy cycle)
```

**5. API Response Times**
```python
# FastAPI includes timing middleware
# Check logs for slow endpoints
# Example: GET /api/v1/jobs/{id} should be < 50ms

# If slow:
# - Database query optimization needed
# - Index missing on jobs table
```

**6. LLM API Costs (if tracking)**
```python
# Enable cost tracking in config
track_costs=True

# Stored in job metadata or WandB
# Monitor for unexpected spikes
# Example: $50/month expected, $500/month → investigate
```

### Alert Thresholds (Production)

**Critical Alerts:**
- Redis connection failed → Services can't communicate
- PostgreSQL down → Data persistence lost
- All workers crashed → No jobs processing
- Disk space < 10% → Results can't be saved

**Warning Alerts:**
- Queue depth > 50 for > 10 minutes → Scale workers
- Worker memory > 4GB → Possible memory leak
- Task failure rate > 10% → API issues or bugs
- API latency p95 > 500ms → Performance degradation

**Info Alerts:**
- Worker restarted (expected after 10 tasks)
- Task took > 30 minutes (expected for large jobs)
- Cache hit rate < 50% (might indicate poor caching)

### Troubleshooting Runbook

**Problem: "No workers running"**
```bash
# Symptoms: Tasks stuck in "pending", queue growing
# Check:
docker compose ps worker            # Is container running?
docker compose logs worker          # Any crash logs?

# Solution:
docker compose restart worker       # Restart worker
# Or
docker compose up -d --scale worker=3  # Scale to 3 workers
```

**Problem: "Database connection refused"**
```bash
# Symptoms: API errors, worker crashes with DB errors
# Check:
docker compose ps db                           # Is DB running?
docker compose exec db pg_isready -U stringsight  # DB healthy?

# Solution:
docker compose restart db                      # Restart DB
# Wait for healthcheck to pass
# Then restart dependent services
docker compose restart api worker
```

**Problem: "Redis connection timeout"**
```bash
# Symptoms: Tasks not enqueuing, Celery errors
# Check:
docker compose ps redis                        # Is Redis running?
docker compose exec redis redis-cli ping       # Returns "PONG"?

# Solution:
docker compose restart redis                   # Restart Redis
# Note: In-flight tasks will be lost (not persisted)
```

**Problem: "Out of disk space"**
```bash
# Symptoms: Can't save results, Docker errors
# Check:
df -h                                          # Disk usage

# Solution:
# 1. Clean old results
rm -rf ~/.stringsight/results/old_*

# 2. Clean Docker volumes
docker system prune -a --volumes               # DESTRUCTIVE

# 3. Archive results to S3
aws s3 sync ~/.stringsight/results/ s3://bucket/results/
```

**Problem: "High memory usage"**
```bash
# Symptoms: Worker OOM killed, system slow
# Check:
docker stats                                   # Which container using memory?

# Solution:
# 1. Reduce sample size in extraction
# 2. Lower max_workers (parallel API calls)
# 3. Increase worker restart frequency
# Change worker_max_tasks_per_child from 10 to 5

# 4. Increase container memory limit
# Edit docker-compose.yml:
# services:
#   worker:
#     mem_limit: 8g
```

**Problem: "Slow task execution"**
```bash
# Symptoms: Tasks taking much longer than usual
# Check:
# 1. LLM API latency (rate limiting?)
docker compose logs worker | grep "API"

# 2. Database query performance
# Connect to DB and check slow queries
SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

# 3. Network issues
docker compose exec worker ping -c 3 api.openai.com

# Solution:
# 1. Check API rate limits, upgrade tier if needed
# 2. Add database indexes (see alembic migrations)
# 3. Check DNS resolution (network_mode: "host" helps)
```

### Production Deployment Best Practices

**1. Use Managed Services (if on cloud):**
```yaml
# Instead of docker-compose PostgreSQL:
DATABASE_URL: "postgresql://user:pass@rds-endpoint.amazonaws.com:5432/db"

# Instead of docker-compose Redis:
REDIS_URL: "redis://elasticache-endpoint:6379/0"

# Benefits:
# - Automatic backups (PostgreSQL)
# - High availability (multi-AZ)
# - Monitoring built-in (CloudWatch)
# - Scaling without downtime
```

**2. Add Load Balancer (if scaling API):**
```
                                ┌──────────┐
                                │  ALB/NLB │
                                └────┬─────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
         ┌────▼────┐            ┌────▼────┐           ┌────▼────┐
         │  API 1  │            │  API 2  │           │  API 3  │
         └─────────┘            └─────────┘           └─────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  PostgreSQL │
                              │    Redis    │
                              └─────────────┘
```

**3. Add Prometheus + Grafana (monitoring):**
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secret

# Collect metrics from:
# - FastAPI (prometheus-fastapi-instrumentator)
# - Celery (celery_exporter)
# - PostgreSQL (postgres_exporter)
# - Redis (redis_exporter)
```

**4. Add Sentry (error tracking):**
```python
# stringsight/api.py
import sentry_sdk

sentry_sdk.init(
    dsn="https://...@sentry.io/...",
    traces_sample_rate=0.1,           # 10% of requests
    environment="production"
)

# Automatically captures:
# - Exceptions in API endpoints
# - Celery task failures
# - Performance issues
```

**5. Add Health Endpoints:**
```python
# stringsight/routers/health.py
@router.get("/health")
async def health_check():
    # Check dependencies
    db_ok = await check_database()
    redis_ok = await check_redis()

    if not (db_ok and redis_ok):
        raise HTTPException(status_code=503, detail="Unhealthy")

    return {"status": "healthy", "db": db_ok, "redis": redis_ok}

# Used by:
# - Load balancer health checks
# - Kubernetes liveness/readiness probes
# - Monitoring systems (Datadog, New Relic)
```

**6. Implement Rate Limiting:**
```python
# Prevent abuse of public endpoints
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/jobs/pipeline")
@limiter.limit("10/minute")              # Max 10 jobs per minute
async def create_pipeline_job(...):
    ...

# Prevents:
# - Accidental DoS from frontend bugs
# - Malicious actors overwhelming system
# - Excessive API costs
```

**7. Database Backups:**
```bash
# Automated PostgreSQL backups
# Option 1: Docker volume snapshots
docker run --rm \
  -v stringsight_postgres_data:/data \
  -v /backups:/backup \
  alpine tar czf /backup/postgres-$(date +%Y%m%d).tar.gz /data

# Option 2: pg_dump
docker compose exec db pg_dump -U stringsight stringsight > backup.sql

# Option 3: Managed service (RDS automated backups)

# Restore:
docker compose exec -T db psql -U stringsight stringsight < backup.sql
```

## Tips for Discussing in Interview

**1. Start high-level, go deep on request:**
- Begin with architecture diagram and data flow
- Dive into specifics when asked
- Use concrete examples from codebase

**2. Highlight design decisions:**
- Explain *why* choices were made, not just *what*
- Mention tradeoffs and alternatives considered
- Show awareness of limitations

**3. Connect to real-world use cases:**
- "This helps ML engineers compare model versions..."
- "Researchers use this to discover failure modes..."
- "Companies use this for model selection..."

**4. Be honest about gaps:**
- "We don't have automated E2E tests yet, but here's how I'd add them..."
- "Authentication is basic, production would need..."
- "Memory leak in Celery worker is a known issue, mitigation is..."

**5. Show system thinking:**
- Discuss scaling, security, testing, monitoring
- Not just feature implementation
- Production-readiness considerations

**6. Use the CLAUDE.md as reference:**
- It's very comprehensive
- Refer to specific files and line numbers
- Shows deep familiarity with codebase

Good luck with your interview!

---

## Quick Reference: Redis, Celery, Docker Commands

### Redis Commands

```bash
# Connect to Redis
docker compose exec redis redis-cli

# Inside Redis CLI:
PING                              # Check connection (returns PONG)
KEYS *                            # List all keys (dev only - slow on large datasets)
LLEN celery                       # Queue length
LPUSH celery "task"               # Add to queue (left)
RPOP celery                       # Remove from queue (right)
GET celery-task-meta-{task_id}    # Get task result
TTL celery-task-meta-{task_id}    # Time to live (seconds until expiry)
FLUSHALL                          # Delete all keys (DESTRUCTIVE)
INFO                              # Server stats
```

### Celery Commands

```bash
# Start worker
celery -A stringsight.celery_app worker --loglevel=info

# Inspect workers
celery -A stringsight.celery_app inspect active       # Active tasks
celery -A stringsight.celery_app inspect stats        # Worker statistics
celery -A stringsight.celery_app inspect registered   # Registered tasks

# Control workers
celery -A stringsight.celery_app control shutdown     # Stop all workers
celery -A stringsight.celery_app purge                # Clear all tasks in queue

# Monitor
celery -A stringsight.celery_app events               # Real-time events
celery -A stringsight.celery_app flower               # Web-based monitoring (requires flower package)
```

### Docker Compose Commands

```bash
# Start services
docker compose up -d              # Start in background
docker compose up --build         # Rebuild and start
docker compose up -d --scale worker=3  # Scale workers

# Stop services
docker compose down               # Stop containers
docker compose down -v            # Stop and remove volumes (DELETES DATA)
docker compose stop               # Stop but keep containers

# View logs
docker compose logs -f            # Follow all logs
docker compose logs -f worker     # Follow worker logs
docker compose logs --tail=100 api  # Last 100 lines

# Restart
docker compose restart            # Restart all
docker compose restart worker     # Restart specific service

# Execute commands
docker compose exec api bash      # Open shell in API container
docker compose exec db psql -U stringsight -d stringsight  # Connect to DB
docker compose exec redis redis-cli  # Connect to Redis

# View status
docker compose ps                 # List containers
docker stats                      # Resource usage
docker compose top                # Process list

# Clean up
docker compose down -v --rmi all  # Remove everything
docker system prune -a --volumes  # Clean Docker system (DESTRUCTIVE)
```

### PostgreSQL Commands

```bash
# Connect
docker compose exec db psql -U stringsight -d stringsight

# Inside psql:
\dt                               # List tables
\d jobs                           # Describe jobs table
\x                                # Toggle expanded display (easier to read)

# Useful queries
SELECT id, type, status, progress FROM jobs ORDER BY created_at DESC LIMIT 10;
SELECT * FROM jobs WHERE status = 'running';
UPDATE jobs SET status = 'pending' WHERE id = '{job_id}';  # Reset stuck job
DELETE FROM jobs WHERE created_at < NOW() - INTERVAL '7 days';  # Cleanup old jobs
```

### Debugging Workflow

```bash
# 1. Check if services are running
docker compose ps

# 2. Check service health
docker compose logs --tail=50 api
docker compose logs --tail=50 worker
docker compose logs --tail=50 db
docker compose logs --tail=50 redis

# 3. Check queue length
docker compose exec redis redis-cli LLEN celery

# 4. Check active tasks
docker compose exec worker celery -A stringsight.celery_app inspect active

# 5. Check job in database
docker compose exec db psql -U stringsight -d stringsight -c \
  "SELECT id, status, progress FROM jobs WHERE id = '{job_id}';"

# 6. Check worker resource usage
docker stats

# 7. Restart if needed
docker compose restart worker

# 8. Check logs after restart
docker compose logs --tail=100 -f worker
```

### Environment Variables Quick Reference

```bash
# .env file
OPENAI_API_KEY=sk-...                                    # Required
ANTHROPIC_API_KEY=sk-...                                 # Optional
DATABASE_URL=postgresql://user:pass@host:5432/db        # PostgreSQL connection
REDIS_URL=redis://host:6379/0                           # Redis connection
LOG_LEVEL=INFO                                           # DEBUG/INFO/WARNING/ERROR
JSON_LOGS=true                                           # JSON-formatted logs
SECRET_KEY=your-secret-key                               # JWT secret
STORAGE_TYPE=local                                       # local or s3
S3_ENDPOINT_URL=http://localhost:9000                    # For MinIO
AWS_ACCESS_KEY_ID=minioadmin                             # S3/MinIO credentials
AWS_SECRET_ACCESS_KEY=minioadmin
```

### Common File Locations

```bash
# Local CLI mode
~/.stringsight/                   # Data directory
~/.stringsight/stringsight.db     # SQLite database
~/.stringsight/results/           # Analysis results
~/.stringsight/logs/              # Log files
~/.stringsight/cache/             # LMDB cache

# Docker mode
/app/results/                     # Results directory (mounted)
/app/data/                        # Data directory (mounted)
/var/lib/postgresql/data          # PostgreSQL data (volume)
/data                             # MinIO data (volume)
```

### Performance Tuning Knobs

```python
# Extraction parallelism
max_workers=10                    # Parallel LLM API calls (default: 10)
                                  # Higher = faster but more memory
                                  # Lower = slower but more stable

# Clustering
min_cluster_size=5                # Minimum properties per cluster (default: 5)
                                  # Higher = fewer, larger clusters
                                  # Lower = more, smaller clusters

# Celery
worker_max_tasks_per_child=10     # Worker restart frequency (default: 10)
                                  # Lower = more frequent restarts (less memory leak)
                                  # Higher = fewer restarts (more efficient)

task_time_limit=86400             # Task timeout in seconds (default: 24 hours)
                                  # Prevents infinite hangs

# Redis
result_expires=86400              # Task result expiry (default: 24 hours)
                                  # Shorter = less Redis memory
                                  # Longer = results available longer

# Database
pool_size=20                      # SQLAlchemy connection pool (default varies)
                                  # Higher = more concurrent requests
                                  # Lower = less DB load
```

Good luck with your interview!
