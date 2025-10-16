#!/usr/bin/env python
"""Prediction script for UNet RNA to Protein prediction."""

import argparse
import logging
import os
import torch
import numpy as np

from models import UNet
from utils.preprocessing import (
    load_and_preprocess_data,
    create_spatial_split,
    apply_dimensionality_reduction,
    rasterize_data
)
from utils.metrics import predict_full_image


def setup_logging(output_file=None, log_level=logging.INFO):
    """Setup logging to both file and console."""
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if output_file:
        log_dir = os.path.dirname(output_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(output_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def main(args):
    # Setup logging
    log_file = None
    if args.output:
        output_dir = os.path.dirname(args.output) or "."
        log_file = os.path.join(output_dir, "prediction.log")
    
    logger = setup_logging(log_file)
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    device = config['device']
    logger.info(f"Using device: {device}")
    
    logger.info("="*60)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("="*60)
    
    # Load data
    rna_proc, pro_proc, rna, pro = load_and_preprocess_data(config)
    
    # Create spatial split (needed for dimensionality reduction)
    rows, cols, split_grid_train, split_grid_val, cutoff = create_spatial_split(
        rna,
        config['data']['grid_h'],
        config['data']['grid_w'],
        config['data']['split_ratio']
    )
    
    # Apply dimensionality reduction
    Z_all = apply_dimensionality_reduction(
        rna_proc, rows, cutoff,
        config['preprocessing']['k_pca'],
        config['random_seed']
    )
    
    # Rasterize data
    img_in, img_out, mask_tissue = rasterize_data(
        rows, cols, Z_all, pro_proc.X, rna,
        config['data']['grid_h'],
        config['data']['grid_w']
    )
    
    logger.info("="*60)
    logger.info("STEP 2: Loading model")
    logger.info("="*60)
    
    C_in = Z_all.shape[1]
    C_out = pro_proc.X.shape[1]
    
    model = UNet(
        in_ch=C_in,
        out_ch=C_out,
        base=config['model']['base_channels']
    ).to(device)
    
    model.load_state