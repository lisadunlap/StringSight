#!/bin/bash
set -e

echo "========================================="
echo "Building and Packaging StringSight"
echo "========================================="
echo ""

# Step 1: Build the frontend
echo "Step 1/3: Building frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "Building frontend for production..."
npm run build

cd ..
echo "✓ Frontend built successfully"
echo ""

# Step 2: Copy frontend to package
echo "Step 2/3: Copying frontend to package..."
rm -rf stringsight/frontend_dist
mkdir -p stringsight/frontend_dist
cp -r frontend/dist/* stringsight/frontend_dist/
echo "✓ Frontend copied to stringsight/frontend_dist/"
echo ""

# Step 3: Reinstall in editable mode (if needed)
echo "Step 3/3: Ensuring editable installation is up to date..."
pip install -e . --no-deps
echo "✓ Package ready"
echo ""

echo "========================================="
echo "Build complete!"
echo "========================================="
echo ""
echo "You can now run: stringsight launch"
echo ""
