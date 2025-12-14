# Release Process

This document outlines the steps to release a new version of StringSight.

## Prerequisites

- Write access to the repository
- Node.js and npm installed (for building frontend)
- PyPI credentials configured (`pip install twine`)
- Python build tools (`pip install build`)

## Quick Release (Using Script)

The easiest way to create a release is to use the provided release script:

```bash
# Make the script executable (first time only)
chmod +x release.sh

# Run the release script with the new version number
./release.sh 0.3.3
```

The script will:
1. Update version in `pyproject.toml`
2. Build the frontend
3. Copy frontend files to `stringsight/frontend_dist/`
4. Clean previous build artifacts
5. Build the Python package
6. Verify package contents
7. Show you the next steps

After the script completes, follow the displayed instructions to:
1. Test the package locally
2. Upload to PyPI
3. Commit and tag the release

## Manual Release Steps

If you prefer to release manually, follow these steps:

### 1. Update Version

Update the version number in [pyproject.toml](pyproject.toml):

```toml
[project]
version = "0.3.3"  # Update this
```

### 2. Build the Frontend

The frontend must be built and copied into the package:

```bash
# Build the frontend
cd frontend
npm install  # if needed
npm run build
cd ..

# Copy to package
rm -rf stringsight/frontend_dist
mkdir -p stringsight/frontend_dist
cp -r frontend/dist/* stringsight/frontend_dist/
```

### 3. Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build
```

### 4. Verify Package Contents

```bash
# Check that frontend files are included in the wheel
python -m zipfile -l dist/stringsight-0.3.3-py3-none-any.whl | grep frontend_dist

# Should show files like:
# stringsight/frontend_dist/index.html
# stringsight/frontend_dist/assets/...
```

### 5. Test Locally (Recommended)

```bash
# Create a fresh environment
conda create -n test-stringsight python=3.11
conda activate test-stringsight

# Install from the built wheel
pip install dist/stringsight-0.3.3-py3-none-any.whl

# Test the CLI
stringsight --help
stringsight launch
```

### 6. Upload to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*
# You'll be prompted for your PyPI username and password
```

### 7. Commit and Tag

```bash
# Stage all changes (excluding frontend_dist which is in .gitignore)
git add .

# Commit with version bump message
git commit -m "Release v0.3.3"

# Tag the release
git tag v0.3.3

# Push to remote
git push origin main --tags
```

## Frontend Development

The frontend is maintained as a separate git submodule at [https://github.com/lisadunlap/stringsight-frontend](https://github.com/lisadunlap/stringsight-frontend).

### Updating the Frontend

To update the frontend version included in the package:

```bash
# Navigate to frontend submodule
cd frontend

# Pull latest changes
git pull origin main

# Navigate back to project root
cd ..

# Commit the submodule update
git add frontend
git commit -m "Update frontend to latest version"

# Build the new frontend
./build_frontend.sh
```

### Frontend Development Workflow

For active frontend development:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (with hot reload)
npm run dev

# In another terminal, start the backend
cd ..
python -m uvicorn stringsight.api:app --reload --port 8000
```

The dev server will proxy API requests to the backend automatically.

## Package Structure

The pip package includes:
- Python code in `stringsight/`
- Built frontend files from `frontend/dist/` (included via MANIFEST.in)
- CLI entry point: `stringsight` command

Files included in the package are controlled by:
- [MANIFEST.in](MANIFEST.in) - specifies which non-Python files to include
- [pyproject.toml](pyproject.toml) - package configuration and dependencies

## Troubleshooting

### Frontend not found after installation

If users report "Frontend not found" errors:
1. Verify `frontend/dist/` exists and contains built files before release
2. Check that MANIFEST.in includes `recursive-include frontend/dist *`
3. Verify the build worked: `ls -la frontend/dist/`

### Submodule issues

If the frontend submodule is not initialized:
```bash
git submodule update --init --recursive
```

If you see "frontend already exists" errors when adding the submodule:
```bash
# Remove the directory
rm -rf frontend

# Add the submodule
git submodule add https://github.com/lisadunlap/stringsight-frontend.git frontend
```
