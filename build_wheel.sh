#!/bin/bash
set -e

echo "========================================="
echo "Building StringSight wheel with frontend"
echo "========================================="

# Step 1: Build frontend
echo ""
echo "Step 1/3: Building frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "Building frontend for production..."
npm run build

cd ..

# Step 2: Copy frontend build to package location
echo ""
echo "Step 2/3: Copying frontend build to stringsight/frontend_dist/..."
rm -rf stringsight/frontend_dist/*
cp -r frontend/dist/* stringsight/frontend_dist/

# Step 3: Build Python wheel
echo ""
echo "Step 3/3: Building Python wheel..."
python -m build --wheel

echo ""
echo "========================================="
echo "✓ Build complete!"
echo "========================================="
echo "Wheel location: dist/stringsight-0.3.6-py3-none-any.whl"
ls -lh dist/*.whl | tail -1
