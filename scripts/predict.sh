#!/bin/bash

# Prediction script for UNet RNA to Protein prediction
# Usage: bash scripts/predict.sh <checkpoint_path> [output_file]

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/predict.sh <checkpoint_path> [output_file]"
    echo "Example: bash scripts/predict.sh results/run_20250115/checkpoints/unet_best_mse.pt predictions.npy"
    exit 1
fi

CHECKPOINT="$1"
OUTPUT="${2:-predictions.npy}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UNet RNA to Protein Prediction${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

echo -e "${YELLOW}Checkpoint: $CHECKPOINT${NC}"
echo -e "${YELLOW}Output: $OUTPUT${NC}"

# Run prediction
echo -e "${GREEN}Starting prediction...${NC}"
python algorithms/predict.py --checkpoint "$CHECKPOINT" --output "$OUTPUT" --save_spots

# Check if prediction succeeded
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Prediction completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}Predictions saved to: $OUTPUT${NC}"
else
    echo -e "${RED}Prediction failed!${NC}"
    exit 1
fi