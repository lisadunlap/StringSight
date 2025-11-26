#!/bin/bash
set -e

# Fix permissions for results and data directories
# We need to do this as root before switching to appuser
if [ -d "/app/results" ]; then
    echo "Fixing permissions for /app/results..."
    chown -R appuser:appuser /app/results
fi

if [ -d "/app/data" ]; then
    echo "Fixing permissions for /app/data..."
    chown -R appuser:appuser /app/data
fi

# Switch to appuser and run the command
exec gosu appuser "$@"
