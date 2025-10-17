#!/bin/bash

# Find and load .env file from project root
SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a
    source "$SCRIPT_DIR/../.env"
    set +a
fi

# Wasabi S3 Configuration
WASABI_BUCKET=vibes
WASABI_ENDPOINT=https://s3.us-west-1.wasabisys.com

# Default values
SOURCE_PATH="results/taubench-retail-incorrect-only"
PREFIX="results/taubench-retail-incorrect-only"  # Changed default prefix to "results"
DRY_RUN=""
REGION="us-west-1"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Upload results folder to Wasabi S3 bucket"
    echo ""
    echo "Options:"
    echo "  --source PATH     Source folder path (default: results)"
    echo "  --prefix PREFIX   S3 prefix/folder in bucket (default: results)"
    echo "  --dry-run         Show what would be uploaded without uploading"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Upload results/ to bucket/results/"
    echo "  $0 --source results                   # Upload results/ to bucket/results/"
    echo "  $0 --source results --prefix exp1     # Upload results/ to bucket/exp1/"
    echo "  $0 --dry-run                          # Show what would be uploaded"
    echo ""
    echo "Environment variables:"
    echo "  WASABI_ACCESS_KEY     Your Wasabi access key"
    echo "  WASABI_SECRET_ACCESS_KEY     Your Wasabi secret key"
    echo "  WASABI_BUCKET         Bucket name (default: vibes)"
    echo "  WASABI_ENDPOINT       Endpoint URL (default: https://s3.us-west-1.wasabisys.com)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_PATH="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if credentials are set
if [[ -z "$WASABI_ACCESS_KEY" || -z "$WASABI_SECRET_ACCESS_KEY" ]]; then
    echo "❌ Error: WASABI_ACCESS_KEY and WASABI_SECRET_ACCESS_KEY environment variables must be set"
    echo ""
    echo "You can set them in your shell:"
    echo "  export WASABI_ACCESS_KEY='your-access-key'"
    echo "  export WASABI_SECRET_ACCESS_KEY='your-secret-key'"
    echo ""
    echo "Or add them to your ~/.bashrc or ~/.zshrc file"
    exit 1
fi

# Check if source path exists
if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "❌ Error: Source path '$SOURCE_PATH' does not exist or is not a directory"
    exit 1
fi

# Build the command
CMD="python3 scripts/upload_to_wasabi.py"
CMD="$CMD --source '$SOURCE_PATH'"
CMD="$CMD --bucket '$WASABI_BUCKET'"
CMD="$CMD --region '$REGION'"
CMD="$CMD --endpoint-url '$WASABI_ENDPOINT'"

if [[ -n "$PREFIX" ]]; then
    CMD="$CMD --prefix '$PREFIX'"
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD $DRY_RUN"
fi

# Show what we're about to do
echo "🚀 Uploading to Wasabi S3"
echo "   Source: $SOURCE_PATH"
echo "   Bucket: $WASABI_BUCKET"
echo "   Region: $REGION"
echo "   Endpoint: $WASABI_ENDPOINT"
if [[ -n "$PREFIX" ]]; then
    echo "   Prefix: $PREFIX"
fi
if [[ -n "$DRY_RUN" ]]; then
    echo "   Mode: DRY RUN (no files will be uploaded)"
fi
echo ""

# Execute the command
echo "Running: $CMD"
echo ""
eval $CMD 