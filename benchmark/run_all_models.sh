#!/bin/bash
# Script to run evaluate_stringsight.py on all individual model files
# in aci_bench and instructeval, excluding baselines and all_behaviors files

# Exit on error
set -e

# Base directory for results
RESULTS_DIR="benchmark/results"

# Array of benchmark directories to process
BENCHMARKS=("aci_bench" "instructeval")

# Configuration parameters (modify as needed)
SUBSET_SIZE=""  # Leave empty for all data, or set to number like "--subset-size 10"
MIN_CLUSTER_SIZE="--min-cluster-size 5"
EMBEDDING_MODEL="--embedding-model text-embedding-3-large"
EXTRACTION_MODEL="--extraction-model gpt-4.1-mini"
JUDGE_MODEL="--judge-model gpt-4.1"
TOP_K="--top-k 10"  # Evaluate top 10 behaviors per model, or leave empty for all
HIERARCHICAL=""  # Leave empty for flat clustering (default), or set to "--hierarchical" to enable
WANDB_FLAG="--log-to-wandb"  # Use "--no-wandb" to disable wandb logging

# Track total runs
TOTAL_FILES=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

echo "========================================"
echo "Running StringSight Evaluation on All Models"
echo "========================================"
echo ""

# Process each benchmark directory
for BENCHMARK in "${BENCHMARKS[@]}"; do
    BENCHMARK_DIR="${RESULTS_DIR}/${BENCHMARK}"
    
    if [ ! -d "$BENCHMARK_DIR" ]; then
        echo "Warning: Directory $BENCHMARK_DIR not found, skipping..."
        continue
    fi
    
    echo "Processing benchmark: $BENCHMARK"
    echo "----------------------------------------"
    
    # Find all .jsonl files, excluding baselines and all_behaviors
    for MODEL_FILE in "${BENCHMARK_DIR}"/*.jsonl; do
        # Extract filename without path
        FILENAME=$(basename "$MODEL_FILE")
        
        # Skip if file contains "baseline" or "all_behaviors"
        if [[ "$FILENAME" == *"baseline"* ]] || [[ "$FILENAME" == *"all_behaviors"* ]]; then
            echo "  Skipping: $FILENAME (baseline or all_behaviors)"
            continue
        fi
        
        # Extract model name (remove .jsonl extension)
        MODEL_NAME="${FILENAME%.jsonl}"
        
        echo ""
        echo "  Processing model: $MODEL_NAME"
        echo "  File: $MODEL_FILE"
        
        TOTAL_FILES=$((TOTAL_FILES + 1))
        
        # Build the command
        CMD="python benchmark/evaluate_stringsight.py \
            --benchmark-results $MODEL_FILE \
            --output-dir benchmark/evaluation_results/${BENCHMARK}/${MODEL_NAME} \
            $MIN_CLUSTER_SIZE \
            $EMBEDDING_MODEL \
            $EXTRACTION_MODEL \
            $JUDGE_MODEL \
            $HIERARCHICAL \
            $WANDB_FLAG"
        
        # Add optional parameters if they're set
        if [ -n "$SUBSET_SIZE" ]; then
            CMD="$CMD $SUBSET_SIZE"
        fi
        
        if [ -n "$TOP_K" ]; then
            CMD="$CMD $TOP_K"
        fi
        
        echo "  Running: $CMD"
        echo ""
        
        # Run the evaluation
        if eval $CMD; then
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
            echo "  ✓ Successfully completed: $MODEL_NAME"
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo "  ✗ Failed: $MODEL_NAME"
        fi
        
        echo "  ----------------------------------------"
    done
    
    echo ""
done

# Print summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Total models processed: $TOTAL_FILES"
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: $FAILED_RUNS"
echo ""
echo "Results saved to: benchmark/evaluation_results/"
