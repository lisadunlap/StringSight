#!/bin/bash
set -e

echo "Building StringSight frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "Building frontend for production..."
npm run build

echo "Frontend built successfully!"
echo "Build artifacts are in: frontend/dist/"
