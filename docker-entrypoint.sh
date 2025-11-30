#!/bin/bash
set -e

# Fix permissions for results and data directories
# We need to do this as root before switching to appuser
if [ -d "/app/results" ]; then
    echo "Fixing permissions for /app/results..."
    chown -R appuser:appuser /app/results || echo "Failed to chown results, trying chmod..." && chmod -R 777 /app/results || true
fi

if [ -d "/app/data" ]; then
    echo "Fixing permissions for /app/data..."
    chown -R appuser:appuser /app/data || echo "Failed to chown data, trying chmod..." && chmod -R 777 /app/data || true
fi

# Switch to appuser and run the command
# Fallback to running as root if gosu fails or permissions are weird
exec "$@"
