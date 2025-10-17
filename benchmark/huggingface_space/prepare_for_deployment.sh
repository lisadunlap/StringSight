#!/bin/bash

# Script to prepare benchmark results for HuggingFace Spaces deployment
# Usage: ./prepare_for_deployment.sh [source_results_dir]

set -e

# Default source directory
SOURCE_DIR="${1:-../results}"
TARGET_DIR="./results"

echo "Preparing benchmark results for HuggingFace Spaces deployment..."
echo "Source directory: $SOURCE_DIR"
echo "Target directory: $TARGET_DIR"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    echo "Usage: ./prepare_for_deployment.sh [source_results_dir]"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy all benchmark results
echo "Copying benchmark results..."
rsync -av --exclude="validation_metrics.json" "$SOURCE_DIR/" "$TARGET_DIR/"

# Count datasets and behaviors
DATASET_COUNT=$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
BEHAVIOR_COUNT=$(find "$TARGET_DIR" -name "*.jsonl" | wc -l)

echo ""
echo "✓ Preparation complete!"
echo "  - Datasets copied: $DATASET_COUNT"
echo "  - Behavior files: $BEHAVIOR_COUNT"
echo ""
echo "Next steps:"
echo "1. Review the copied data in $TARGET_DIR"
echo "2. Follow DEPLOYMENT_GUIDE.md to upload to HuggingFace Spaces"
echo "3. Test locally with: python app.py --results-dir results/"


