# Production Deployment

Guidelines for deploying StringSight in production environments.

## Environment Setup

### Production Environment

```bash
# Create production environment
conda create -n stringsight-prod python=3.11
conda activate stringsight-prod

# Install with production dependencies
pip install -e ".[full]"

# Set environment variables
export OPENAI_API_KEY="your-prod-api-key"
export WANDB_API_KEY="your-wandb-key"
export ENVIRONMENT="production"
```

### Configuration Files

Create a production config file `config/production.yaml`:

```yaml
extraction:
  model_name: "gpt-4.1"
  temperature: 0.7
  max_workers: 16
  cache_dir: "/data/cache/extraction"

clustering:
  min_cluster_size: 30
  embedding_model: "text-embedding-3-large"
  cache_dir: "/data/cache/clustering"

metrics:
  compute_bootstrap: true
  bootstrap_samples: 100

logging:
  use_wandb: true
  wandb_project: "production-analysis"
  verbose: true

output:
  base_dir: "/data/results"
```

## API Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run API server
gunicorn stringsight.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Using Docker Compose

StringSight includes a complete Docker Compose setup with all infrastructure dependencies (PostgreSQL, Redis, MinIO) for production deployments.

**Setup:**

```bash
# Clone repository
git clone https://github.com/lisadunlap/stringsight.git
cd stringsight

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and other credentials

# Start all services
docker compose up -d

# The API will be available at http://localhost:8000
```

**What's included:**
- API server (FastAPI with auto-reload)
- Celery workers for async job processing
- PostgreSQL database
- Redis for job queue
- MinIO for object storage

**Production deployment:**

```bash
# Build and start in detached mode
docker compose up -d --build

# View logs
docker compose logs -f api
docker compose logs -f worker

# Scale workers
docker compose up -d --scale worker=4

# Stop services
docker compose down
```

**Health checks:**
```bash
# Check API health
curl http://localhost:8000/health

# View all running services
docker compose ps
```

For development with live code reloading, see the [Development with Live Reload](../getting-started/installation.md#docker-development) section.

### Deploying on Render

**Render Persistent Disk Setup:**

StringSight saves results and cache data to disk. On Render, you need to attach a persistent disk to preserve this data across deployments.

1. **Add a Persistent Disk**:
   - Go to your Render service → "Disks" tab
   - Click "Add Disk"
   - Set mount path to `/var/data` (or your preferred path)
   - Choose size (e.g., 10-100 GB depending on your needs)

2. **Configure Environment Variable**:
   ```bash
   RENDER_DISK_PATH=/var/data
   ```
   This tells StringSight to use the persistent disk for all results and cache storage.

3. **Deploy**: Render will automatically trigger a new deployment.

After deployment, check logs to confirm:
```
Using Render persistent disk: /var/data
```

**Important Limitations:**
- Services with persistent disks cannot scale to multiple instances
- Zero-downtime deploys are not supported with disks
- Render creates daily snapshots (retained 7 days) for backup

See **[Render Disk Setup Guide](../../RENDER_DISK_SETUP.md)** for detailed instructions.

## Monitoring

### Health Checks

```python
# Check API health
import requests

response = requests.get("http://localhost:8000/health")
assert response.json()["ok"] == True
```

### Logging

Use structured logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stringsight.log'),
        logging.StreamHandler()
    ]
)
```

### Cost Tracking

Monitor API costs:

```python
from stringsight.costs import CostTracker

tracker = CostTracker()
# Costs are automatically tracked during pipeline execution
```

## Scaling

### Horizontal Scaling

Run multiple API instances behind a load balancer:

```bash
# Instance 1
uvicorn stringsight.api:app --port 8001 &

# Instance 2
uvicorn stringsight.api:app --port 8002 &

# Use nginx as load balancer
```

### Queue-Based Processing

Use Celery for async processing:

```python
from celery import Celery
from stringsight import explain

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def analyze_dataset(data_path, output_dir):
    df = pd.read_parquet(data_path)
    return explain(df, output_dir=output_dir)
```

## Security

### API Keys

- Store in environment variables, never in code
- Rotate keys regularly
- Use separate keys for dev/staging/prod

### Access Control

Add authentication to API:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/analyze")
async def analyze(token: str = Security(security)):
    if not validate_token(token):
        raise HTTPException(status_code=401)
    # ...
```

## Best Practices

1. **Use caching** - Enable extraction/clustering caches
2. **Monitor costs** - Track API usage and set budget alerts
3. **Version control** - Tag releases and track model versions
4. **Backup results** - Regularly backup output directories
5. **Test thoroughly** - Run integration tests before deploying

## Next Steps

- **[API Endpoints](api-endpoints.md)** - API documentation
- **[Performance Tuning](../advanced/performance.md)** - Optimization guide
