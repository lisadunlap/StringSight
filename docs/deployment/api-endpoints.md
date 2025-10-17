# API Endpoints

FastAPI backend endpoints for the StringSight web interface.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "ok": true
}
```

### Detect and Validate Data

**POST** `/detect-and-validate`

Upload data file and validate format.

**Request:**
- Form data with file upload

**Response:**
```json
{
  "method": "single_model",
  "columns": ["prompt", "model", "model_response"],
  "num_rows": 100,
  "models": ["gpt-4", "claude-3"],
  "preview": [...]
}
```

### Parse Conversations

**POST** `/conversations`

Convert data to conversation traces.

**Request:**
```json
{
  "data": [...],
  "method": "single_model"
}
```

**Response:**
```json
{
  "conversations": [
    {
      "question_id": "q1",
      "prompt": "What is ML?",
      "trace": [...]
    }
  ]
}
```

### Metrics and Results

#### Get metrics summary
**GET** `/metrics/summary/{results_dir}`

#### Get model-cluster metrics
**GET** `/metrics/model-cluster/{results_dir}`

#### Get benchmark metrics
**GET** `/metrics/benchmark/{results_dir}`

#### Get available quality metric names
**GET** `/metrics/quality-metrics/{results_dir}`

#### Load a results directory
**POST** `/results/load`

Body: `{ "path": "/abs/or/base-relative/path", "max_conversations": 1000, "max_properties": 10000 }`

### File and Data Utilities

#### Read a dataset from server path
**POST** `/read-path`

#### List a directory on server
**POST** `/list-path`

### Flexible Data Mapping

**POST** `/auto-detect-columns`

**POST** `/validate-flexible-mapping`

**POST** `/process-flexible-data`

**POST** `/flexible-conversations`

### Extraction

**POST** `/extract/single`

**POST** `/extract/batch`

**POST** `/extract/jobs/start`

**GET** `/extract/jobs/status?job_id=...`

**GET** `/extract/jobs/result?job_id=...`

### Clustering and Metrics

**POST** `/cluster/run`

**POST** `/cluster/metrics`

## Starting the API

```bash
python -m uvicorn stringsight.api:app --reload --host localhost --port 8000
```

## CORS Configuration

By default, development builds allow all origins (permissive CORS) to simplify local frontend usage. For production, restrict origins explicitly (example below).

## Authentication

Currently no authentication required. For production, add:

```python
from fastapi import Security
from fastapi.security import HTTPBearer

security = HTTPBearer()
```

## Next Steps

- **[Production Setup](production.md)** - Deploy the API
- **[Visualization](../user-guide/visualization.md)** - Use the frontend
