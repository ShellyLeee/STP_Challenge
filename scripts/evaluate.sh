#!/bin/bash

# Evaluation script for UNet RNA to Protein prediction
# Usage: bash scripts/evaluate.sh <checkpoint_path> [output_dir]

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/evaluate.sh <checkpoint_path> [output_dir]"
    echo "Example: bash scripts/evaluate.sh results/run_20250115/checkpoints/unet_best_mse.pt"
    exit 1
fi

CHECKPOINT="$1"
OUTPUT_DIR="${2:-evaluation_results}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UNet RNA to Protein Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

echo -e "${YELLOW}Checkpoint: $CHECKPOINT${NC}"
echo -e "${YELLOW}Output directory: $OUTPUT_DIR${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo -e "${GREEN}Starting evaluation...${NC}"
python algorithms/evaluate.py --checkpoint "$CHECKPOINT" --output_dir "$OUTPUT_DIR"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Evaluation completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}Results saved to: $OUTPUT_DIR${NC}"
else
    echo -e "${RED}Evaluation failed!${NC}"
    exit 1
fi