#!/bin/bash

# Training script for UNet RNA to Protein prediction
# Usage: bash scripts/train.sh [run_name] [config_path]

set -e  # Exit on error

# Default values
CONFIG_PATH="${2:-configs/config.yaml}"
RUN_NAME="${1:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UNet RNA to Protein Training${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Config file: $CONFIG_PATH${NC}"

# Check if data files exist
if [ ! -f "train_rna.h5ad" ] || [ ! -f "train_pro.h5ad" ]; then
    echo -e "${RED}Error: Data files not found!${NC}"
    echo -e "${RED}Please place train_rna.h5ad and train_pro.h5ad in the project root${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p results
mkdir -p checkpoints

# Set CUDA device if specified
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo -e "${YELLOW}Using GPU: $CUDA_VISIBLE_DEVICES${NC}"
fi

# Run training
if [ -z "$RUN_NAME" ]; then
    echo -e "${GREEN}Starting training with auto-generated run name...${NC}"
    python algorithms/train.py --config "$CONFIG_PATH"
else
    echo -e "${GREEN}Starting training with run name: $RUN_NAME${NC}"
    python algorithms/train.py --config "$CONFIG_PATH" --run_name "$RUN_NAME"
fi

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}Check results in: results/${RUN_NAME:-[auto-generated]}/${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Training failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi