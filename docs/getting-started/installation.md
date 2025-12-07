# Installation

This guide will help you install StringSight and set up your development environment.

## Prerequisites

### Required
- **Python 3.8+** (recommended: 3.10 or 3.11)
- **Conda or Miniconda** (recommended for environment management)
- **OpenAI API key** (required for LLM-powered features)

### Optional
- **Node.js 20+** (for React frontend interface)
- **Weights & Biases account** (for experiment tracking)

## Quick Installation

### 1. Create Conda Environment

```bash
# Create new conda environment with Python 3.11
conda create -n stringsight python=3.11
conda activate stringsight
```

### 2. Install StringSight

```bash
# From PyPI (recommended): install core package + optional ML/embedding tooling
pip install "stringsight[full]"

# Or, for local development from source:
# git clone https://github.com/lisadunlap/stringsight.git
# cd stringsight
# pip install -e ".[full]"
```

### 3. Set API Key(s)

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
# We use LiteLLM, supporting 100+ providers
export OPENAI_API_KEY="your-api-key-here"
# ... any other provider keys
```

**Local Models**: StringSight uses LiteLLM, so you can use vLLM, Ollama, or any OpenAI-compatible server. See the [LiteLLM docs](https://docs.litellm.ai/docs/providers) for provider-specific setup. 

### 4. Verify Installation

```bash
# Test core package
python -c "from stringsight import explain; print('✅ Installation successful!')"

# Test API server
python -m uvicorn stringsight.api:app --reload --host localhost --port 8000

# In another terminal, test health check
curl http://127.0.0.1:8000/health
# Should return: {"ok": true}
```

## Installation Options

### Core Package Only
```bash
# From PyPI
pip install stringsight

# From a local clone (development)
pip install -e .
```

### With Development Tools
```bash
# From PyPI
pip install "stringsight[dev]"

# From a local clone (development)
pip install -e ".[dev]"
```

### All Features
```bash
# From PyPI (recommended for most users)
pip install "stringsight[full]"

# From a local clone (development)
pip install -e ".[full]"
```

## Frontend Setup (Optional)

The React frontend provides an interactive web interface for analyzing results.

```bash
# Install Node.js dependencies
cd frontend/
npm install

# Start development server
npm run dev

# Open browser to http://localhost:5173
```

## Docker Setup (Optional)

For multi-user deployments or to run StringSight with all infrastructure dependencies (PostgreSQL, Redis, MinIO), use Docker Compose.

### Basic Usage (Production)

```bash
# Clone the repository
git clone https://github.com/lisadunlap/stringsight.git
cd stringsight

# Copy the environment template and add your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start all services (API, workers, database, Redis, MinIO)
docker compose up

# The API will be available at http://localhost:8000
```

This runs the complete stack with persistent storage for database and object storage.

### Docker Development

For active development where you want code changes to reflect immediately:

```bash
# Option 1: Use the dev compose file explicitly
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Option 2: Copy to override file (auto-loaded by docker compose)
cp docker-compose.dev.yml docker-compose.override.yml
docker compose up
```

The development setup mounts your local code into the containers, so changes to Python files will automatically reload the API (thanks to `uvicorn --reload`).

**Note for Mac/Windows users:** Volume mounts can have slower I/O performance on non-Linux systems. If you experience performance issues, you can either:
- Use the basic setup (rebuild containers when you make changes)
- Run the API locally: `pip install -e . && uvicorn stringsight.api:app --reload`

## Verify Full Setup

### Backend API Test
```bash
# Start backend
python -m uvicorn stringsight.api:app --reload --host localhost --port 8000

# In another terminal
curl http://127.0.0.1:8000/health
```

### Frontend Test
```bash
cd frontend/
npm run dev
# Open http://localhost:5173 in your browser
```

### Core Package Test
```python
from stringsight import explain
import pandas as pd

df = pd.DataFrame({
    "prompt": ["What is ML?"],
    "model": ["gpt-4"],
    "model_response": [
        [{"role": "user", "content": "What is ML?"},
         {"role": "assistant", "content": "Machine learning is..."}]
    ]
})

# Should run without errors
clustered_df, model_stats = explain(df, output_dir="test_results")
```

## Troubleshooting

### PyTorch/CUDA Issues
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Node.js Version Issues
```bash
# Install/upgrade Node.js 20+
conda install -c conda-forge nodejs=20

# Or use nvm
nvm install 20 && nvm use 20
```

### FastAPI/Uvicorn Errors
```bash
# Reinstall with explicit versions
pip install "fastapi>=0.100.0" "uvicorn[standard]>=0.20.0"
```

### OpenAI API Key Issues
```bash
# Verify key is set
echo $OPENAI_API_KEY

# Test API connection
python -c "import openai; print('✅ OpenAI client loaded')"
```

### Frontend Won't Start
```bash
cd frontend/
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python -m uvicorn stringsight.api:app --reload --port 8001
```

### Clean Reinstall

If you encounter dependency conflicts:

```bash
# Remove old environment
conda deactivate
conda remove -n stringsight --all

# Create fresh environment
conda create -n stringsight python=3.11
conda activate stringsight

# Reinstall
cd stringsight
pip install -e ".[full]"
```

## Environment Variables

StringSight uses the following environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls |
| `WANDB_API_KEY` | No | Weights & Biases API key for experiment tracking |

## Dependencies

### Core Dependencies
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Machine learning utilities
- `hdbscan`, `umap-learn` - Clustering algorithms
- `openai`, `litellm` - LLM API clients
- `sentence-transformers` - Local embedding models

### Visualization Dependencies
- `plotly` - Interactive charts

### Frontend Dependencies (npm)
- `react`, `typescript` - Frontend framework
- `@mui/material` - UI components
- `@tanstack/react-table` - Data tables
- `plotly.js` - Interactive charts

### Development Dependencies
- `pytest`, `pytest-cov` - Testing
- `black`, `flake8`, `mypy` - Code quality
- `mkdocs`, `mkdocs-material` - Documentation

## Next Steps

- **[Quick Start Guide](quick-start.md)** - Run your first analysis in 5 minutes
- **[Basic Usage](../user-guide/basic-usage.md)** - Learn the core `explain()` and `label()` functions
- **[Configuration](../user-guide/configuration.md)** - Customize your analysis pipeline
