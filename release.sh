#!/bin/bash
set -e

echo "========================================="
echo "StringSight Release Script"
echo "========================================="
echo ""

# Check if version argument is provided
if [ -z "$1" ]; then
    echo "Error: Version number required"
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.3.3"
    exit 1
fi

VERSION=$1

echo "Preparing release for version $VERSION"
echo ""

# Step 1: Update version in pyproject.toml
echo "Step 1/7: Updating version in pyproject.toml..."
sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
echo "✓ Version updated to $VERSION"
echo ""

# Step 2: Build the frontend
echo "Step 2/7: Building frontend..."
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

# Step 3: Copy frontend to package
echo "Step 3/7: Copying frontend to package..."
rm -rf stringsight/frontend_dist
mkdir -p stringsight/frontend_dist
# Copy only the built frontend files, exclude test data
cp -r frontend/dist/* stringsight/frontend_dist/ 2>/dev/null || true
# Remove any test data files that might have been copied
rm -rf stringsight/frontend_dist/taubench_airline_data*
rm -f stringsight/frontend_dist/*.jsonl
echo "✓ Frontend copied to stringsight/frontend_dist/"
echo ""

# Step 4: Clean previous builds
echo "Step 4/7: Cleaning previous build artifacts..."
rm -rf dist/ build/ *.egg-info stringsight.egg-info
echo "✓ Build artifacts cleaned"
echo ""

# Step 5: Build the package
echo "Step 5/7: Building Python package..."
python -m build
echo "✓ Package built successfully"
echo ""

# Step 6: Verify package contents
echo "Step 6/7: Verifying package contents..."
echo "Checking for frontend files in wheel..."
python -m zipfile -l dist/stringsight-${VERSION}-py3-none-any.whl | grep frontend_dist | head -5
echo ""
echo "Checking for CLI entry point..."
python -m zipfile -l dist/stringsight-${VERSION}-py3-none-any.whl | grep "cli.py"
echo "✓ Package verification complete"
echo ""

# Step 7: Summary
echo "Step 7/7: Release Summary"
echo "========================================="
echo "Version: $VERSION"
echo "Files created:"
ls -lh dist/
echo ""
echo "Next steps:"
echo "1. Test the package locally:"
echo "   pip install dist/stringsight-${VERSION}-py3-none-any.whl"
echo "   stringsight launch"
echo ""
echo "2. Upload to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
echo "3. Commit and tag:"
echo "   git add ."
echo "   git commit -m \"Release v${VERSION}\""
echo "   git tag v${VERSION}"
echo "   git push origin main --tags"
echo ""
echo "========================================="
echo "Release preparation complete!"
echo "========================================="
