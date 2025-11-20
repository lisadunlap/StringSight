#!/bin/bash
# Batch processing script for VLM-Gym ablation files using test_label.yaml config

# Base configuration file
CONFIG="scripts/dataset_configs/test_label.yaml"

# Base data and results directories
DATA_BASE="data/vlm_gym_ablation"
RESULTS_BASE="/home/lisabdunlap/StringSightNew/results/vlm_gym_ablation"

# Find all JSON files in the data directory
mapfile -t file_list < <(find "$DATA_BASE" -type f -name "*.json" | sort)

# Counter for progress tracking
total=${#file_list[@]}
current=0

echo "========================================"
echo "Starting batch processing of VLM-Gym ablation files"
echo "Total files to process: $total"
echo "========================================"
echo ""

# Loop through each file
for data_file in "${file_list[@]}"; do
    current=$((current + 1))

    # Get the relative path from DATA_BASE
    rel_path="${data_file#$DATA_BASE/}"

    # Extract directory and filename
    subdir=$(dirname "$rel_path")
    filename=$(basename "$data_file" .json)

    # Set output directory maintaining the subdirectory structure
    output_dir="${RESULTS_BASE}/${subdir}/${filename}"

    echo "----------------------------------------"
    echo "[$current/$total] Processing: $subdir/$filename"
    echo "Data path: $data_file"
    echo "Output dir: $output_dir"
    echo "----------------------------------------"

    # Run the script with config and overrides
    python scripts/run_from_config.py \
        --config "$CONFIG" \
        --data_path "$data_file" \
        --output_dir "$output_dir"

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $subdir/$filename"
    else
        echo "✗ Failed to process $subdir/$filename"
    fi
    echo ""
done

echo "========================================"
echo "Batch processing complete!"
echo "Processed $current/$total files"
echo "========================================"
