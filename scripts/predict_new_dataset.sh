#!/bin/bash

# Prediction script for new validation dataset
# Usage: bash scripts/predict_new_dataset.sh <checkpoint_path> <valid_rna_path> <output_csv> [gpu_id]
# Example: bash scripts/predict_new_dataset.sh results/run_20250115/checkpoints/unet_best_mse.pt valid_rna.h5ad predictions_output.csv 0

set -e  # Exit on error

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: bash scripts/predict_new_dataset.sh <checkpoint_path> <valid_rna_path> <output_csv> [gpu_id]"
    echo "Example: bash scripts/predict_new_dataset.sh results/run_001/checkpoints/unet_best_mse.pt valid_rna.h5ad predictions.csv 0"
    exit 1
fi

CHECKPOINT="$1"
VALID_RNA="$2"
OUTPUT_CSV="$3"
GPU_ID="${4:-0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UNet: Predict on New Validation Dataset${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Check if validation RNA file exists
if [ ! -f "$VALID_RNA" ]; then
    echo -e "${RED}Error: Validation RNA file not found: $VALID_RNA${NC}"
    exit 1
fi

echo -e "${YELLOW}Checkpoint: $CHECKPOINT${NC}"
echo -e "${YELLOW}Validation RNA: $VALID_RNA${NC}"
echo -e "${YELLOW}Output CSV: $OUTPUT_CSV${NC}"
echo -e "${YELLOW}GPU ID: $GPU_ID${NC}"

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
if [ -n "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Run prediction
echo -e "${GREEN}Starting prediction...${NC}"
python algorithms/predict_new_dataset.py \
    --checkpoint "$CHECKPOINT" \
    --valid_rna "$VALID_RNA" \
    --output "$OUTPUT_CSV"

# Check if prediction succeeded
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Prediction completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}Output CSV: $OUTPUT_CSV${NC}"
    
    # Show first few lines
    echo -e "${YELLOW}Preview (first 3 rows):${NC}"
    head -4 "$OUTPUT_CSV" | column -t -s ','
else
    echo -e "${RED}Prediction failed!${NC}"
    exit 1
fi