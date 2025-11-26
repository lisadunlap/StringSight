# Multi-stage build to reduce final image size
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements (excludes PyTorch/CUDA/sentence-transformers)
COPY requirements.txt ./requirements.txt

# Install dependencies to a local directory
# Use --no-deps flag with careful dependency management
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install bertopic WITHOUT dependencies to avoid pulling in sentence-transformers -> torch -> nvidia
# We already installed all other bertopic deps (hdbscan, umap-learn, pandas, numpy, etc) in requirements.txt
RUN pip install --no-cache-dir --prefix=/install --no-deps bertopic>=0.17.3

# Final stage - smaller image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y \
  libpq5 \
  curl \
  gosu \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy ONLY application code (not test data, not docs, not demos)
COPY stringsight/ ./stringsight/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY requirements.txt ./requirements.txt
COPY pyproject.toml .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

# Expose port for API
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "stringsight.api:app", "--host", "0.0.0.0", "--port", "8000"]
