#!/usr/bin/env python
"""Prediction script for new dataset using trained UNet model."""

import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc

from models import UNet
from utils.preprocessing import (
    load_and_preprocess_data,
    create_spatial_split,
    apply_dimensionality_reduction,
    rasterize_data
)
from utils.metrics import predict_full_image


def setup_logging(output_dir=None, log_level=logging.INFO):
    """Setup logging to both file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "prediction.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def main(args):
    # Setup logging
    output_dir = os.path.dirname(args.output) if args.output else None
    logger = setup_logging(output_dir)
    
    logger.info("="*60)
    logger.info("STEP 1: Loading checkpoint")
    logger.info("="*60)
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Train loss: {checkpoint['train_loss']:.5f}")
    logger.info(f"  Val loss: {checkpoint['val_loss']:.5f}")
    
    device = config['device']
    logger.info(f"Using device: {device}")
    
    logger.info("="*60)
    logger.info("STEP 2: Loading new dataset")
    logger.info("="*60)
    
    logger.info(f"Loading validation RNA data from: {args.valid_rna}")
    if not os.path.exists(args.valid_rna):
        logger.error(f"Validation RNA file not found: {args.valid_rna}")
        return
    
    valid_rna = sc.read_h5ad(args.valid_rna)
    logger.info(f"Validation RNA shape: {valid_rna.shape}")
    logger.info(f"Validation RNA features: {valid_rna.n_vars}")
    
    logger.info("="*60)
    logger.info("STEP 3: Preprocessing validation data")
    logger.info("="*60)
    
    # Process validation RNA (same preprocessing as training)
    valid_rna_proc = valid_rna.copy()
    
    if config['preprocessing'].get('normalize_rna', False):
        logger.info("Applying normalize_total + log1p to validation RNA...")
        sc.pp.normalize_total(valid_rna_proc, target_sum=1e4)
        sc.pp.log1p(valid_rna_proc)
    
    logger.info("="*60)
    logger.info("STEP 4: Creating spatial split for validation data")
    logger.info("="*60)
    
    # Create spatial split
    rows, cols, split_grid_train, split_grid_val, cutoff = create_spatial_split(
        valid_rna,
        config['data']['grid_h'],
        config['data']['grid_w'],
        config['data']['split_ratio'],
        logger=logger
    )
    
    logger.info("="*60)
    logger.info("STEP 5: Applying dimensionality reduction")
    logger.info("="*60)
    
    # Load training RNA to fit SVD (or use pre-fit SVD from config)
    logger.info("Loading training data to fit SVD model...")
    train_rna = sc.read_h5ad(config['data']['rna_h5ad'])
    train_rna_proc = train_rna.copy()
    
    if config['preprocessing'].get('normalize_rna', False):
        sc.pp.normalize_total(train_rna_proc, target_sum=1e4)
        sc.pp.log1p(train_rna_proc)
    
    # Create spatial split for training data (to fit SVD only on train)
    train_rows, _, _, _, train_cutoff = create_spatial_split(
        train_rna,
        config['data']['grid_h'],
        config['data']['grid_w'],
        config['data']['split_ratio']
    )
    
    # Apply dimensionality reduction using training data fit
    Z_valid = apply_dimensionality_reduction(
        valid_rna_proc, rows, cutoff,
        config['preprocessing']['k_pca'],
        config['random_seed'],
        logger=logger,
        fit_data=train_rna_proc,
        fit_rows=train_rows,
        fit_cutoff=train_cutoff
    )
    
    logger.info("="*60)
    logger.info("STEP 6: Rasterizing validation data")
    logger.info("="*60)
    
    # For validation data, we don't have protein data, so we create dummy protein data
    # We need to rasterize only the RNA data
    img_in, _, mask_tissue = rasterize_data(
        rows, cols, Z_valid, np.zeros((len(rows), 1)), valid_rna,
        config['data']['grid_h'],
        config['data']['grid_w'],
        logger=logger
    )
    
    logger.info("="*60)
    logger.info("STEP 7: Loading model")
    logger.info("="*60)
    
    C_in = Z_valid.shape[1]
    C_out = checkpoint['config']['preprocessing'].get('k_pca')  # This will be overridden
    
    # Get C_out from checkpoint config or infer
    checkpoint_path_parts = args.checkpoint.split('/')
    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    metrics_file = os.path.join(run_dir, "metrics.json")
    
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        C_out = metrics['model']['output_channels']
    else:
        # Try to infer from checkpoint
        C_out = checkpoint['model_state_dict']['out.bias'].shape[0]
    
    logger.info(f"Input channels: {C_in}, Output channels: {C_out}")
    
    model = UNet(
        in_ch=C_in,
        out_ch=C_out,
        base=config['model']['base_channels']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    logger.info("="*60)
    logger.info("STEP 8: Performing inference")
    logger.info("="*60)
    
    logger.info("Starting tiled inference on full validation image...")
    full_pred = predict_full_image(
        model, img_in,
        config['training']['patch_size'],
        config['training']['stride'],
        device
    )
    
    logger.info(f"Inference completed")
    logger.info(f"Prediction shape: {full_pred.shape}")
    
    logger.info("="*60)
    logger.info("STEP 9: Extracting spot-level predictions")
    logger.info("="*60)
    
    # Extract predictions at spot locations
    spot_predictions = np.zeros((len(rows), C_out), dtype=np.float32)
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        spot_predictions[i] = full_pred[r, c, :]
    
    logger.info(f"Extracted spot-level predictions: {spot_predictions.shape}")
    
    logger.info("="*60)
    logger.info("STEP 10: Creating output CSV")
    logger.info("="*60)
    
    # Get protein names
    try:
        protein_names = list(map(str, checkpoint['config'].get('protein_names', [])))
        if not protein_names:
            # Try to load from training protein data
            train_pro = sc.read_h5ad(config['data']['pro_h5ad'])
            protein_names = list(map(str, train_pro.var_names))
    except:
        protein_names = [f"protein_{i}" for i in range(C_out)]
    
    logger.info(f"Number of proteins: {len(protein_names)}")
    
    # Create output dataframe
    df = pd.DataFrame()
    
    # Add barcode (using spot names from valid_rna)
    df['barcode'] = valid_rna.obs_names.values
    
    # Add spatial coordinates
    df['pxl_row_in_fullres'] = rows * 1.0  # Scale if needed
    df['pxl_col_in_fullres'] = cols * 1.0
    
    # Add protein predictions
    for i, protein_name in enumerate(protein_names):
        df[protein_name] = spot_predictions[:, i]
    
    logger.info(f"Output dataframe shape: {df.shape}")
    
    # Save to CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.error("No output file specified!")
    
    logger.info("="*60)
    logger.info("Prediction completed!")
    logger.info("="*60)
    logger.info(f"Predicted {len(df)} spots with {C_out} proteins")
    logger.info(f"Output: {args.output}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict protein expression on new validation dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint file (e.g., results/run_xxx/checkpoints/unet_best_mse.pt)"
    )
    parser.add_argument(
        "--valid_rna", type=str, required=True,
        help="Path to validation RNA data (e.g., valid_rna.h5ad)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output CSV file"
    )
    
    args = parser.parse_args()
    main(args)