#!/bin/bash
# Batch processing script for VLM-Gym files using test_label.yaml config

# List of data files
# file_list=(
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/colorization__ColorizationEnv-v2__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/colorization__ColorizationEnv-v2__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/counting__LVISCountingEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/counting__LVISCountingEnv-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/FetchPickAndPlaceDiscrete-v4__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/FetchReachDiscrete-v4__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/Jigsaw-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/Jigsaw-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/match_equation__MatchEquation-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/match_equation__MatchEquation-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/match_move__MatchRotation-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/match_move__MatchRotation-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/mental_rotation_3d__mental_rotation-3d__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/mental_rotation_3d__mental_rotation-3d__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/mental_rotation-3d_objaverse-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/mental_rotation__mental_rotation-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/mental_rotation__mental_rotation-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/patch_reassembly__PatchReassemblyEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/patch_reassembly__PatchReassemblyEnv-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/ref_dot__RefCOCOPlusDotEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/sliding_block__SlidingBlockEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/sliding_block__SlidingBlockEnv-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/ToyMaze2DEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/ToyMaze2DEnv-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/ToyMaze3DEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/ToyMaze3DEnv-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/video_unshuffle__VideoUnshuffleEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/video_unshuffle__VideoUnshuffleEnv-v0__hard.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/zoom_in__ZoomInEnv-v0__easy.json"
#     "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/zoom_in__ZoomInEnv-v0__hard.json"
# )
file_list=(
    "/home/jiaxin/VLM-Gym/visualizations/openai_format_images/mental_rotation-3d_objaverse-v0__easy.json"
)


# Base configuration file
CONFIG="scripts/dataset_configs/test_label.yaml"

# Base results directory
RESULTS_BASE="/home/lisabdunlap/StringSightNew/results/vlm_gym_ablation/orig"

# Counter for progress tracking
total=${#file_list[@]}
current=0

echo "========================================"
echo "Starting batch processing of VLM-Gym files"
echo "Total files to process: $total"
echo "========================================"
echo ""

# Loop through each file
for data_file in "${file_list[@]}"; do
    current=$((current + 1))

    # Extract filename without path and extension
    filename=$(basename "$data_file" .json)

    # Set output directory based on filename
    output_dir="${RESULTS_BASE}/${filename}"

    echo "----------------------------------------"
    echo "[$current/$total] Processing: $filename"
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
        echo "✓ Successfully processed $filename"
    else
        echo "✗ Failed to process $filename"
    fi
    echo ""
done

echo "========================================"
echo "Batch processing complete!"
echo "Processed $current/$total files"
echo "========================================"
