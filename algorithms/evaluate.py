#!/usr/bin/env python
"""Evaluation script for UNet RNA to Protein prediction."""

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
from utils.metrics import (
    predict_full_image,
    compute_correlations
)
from utils.visualization import plot_correlation_bar, plot_predictions


def setup_logging(output_dir, log_level=logging.INFO):
    """Setup logging to both file and console."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "evaluation.log")
    else:
        log_file = None
    
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
    if log_file:
        fh = logging.FileHandler(log_file)
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
    logger = setup_logging(args.output_dir)
    
    logger.info("="*60)
    logger.info("STEP 1: Loading checkpoint")
    logger.info("="*60)
    
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Train loss: {checkpoint['train_loss']:.5f}")
    logger.info(f"  Val loss: {checkpoint['val_loss']:.5f}")
    
    device = config['device']
    logger.info(f"Using device: {device}")
    
    logger.info("="*60)
    logger.info("STEP 2: Loading and preprocessing data")
    logger.info("="*60)
    
    # Load data
    rna_proc, pro_proc, rna, pro = load_and_preprocess_data(config, logger=logger)
    
    # Create spatial split
    rows, cols, split_grid_train, split_grid_val, cutoff = create_spatial_split(
        rna,
        config['data']['grid_h'],
        config['data']['grid_w'],
        config['data']['split_ratio'],
        logger=logger
    )
    
    # Apply dimensionality reduction
    Z_all = apply_dimensionality_reduction(
        rna_proc, rows, cutoff,
        config['preprocessing']['k_pca'],
        config['random_seed'],
        logger=logger
    )
    
    # Rasterize data
    img_in, img_out, mask_tissue = rasterize_data(
        rows, cols, Z_all, pro_proc.X, rna,
        config['data']['grid_h'],
        config['data']['grid_w'],
        logger=logger
    )
    
    logger.info("="*60)
    logger.info("STEP 3: Loading model")
    logger.info("="*60)
    
    C_in = Z_all.shape[1]
    C_out = pro_proc.X.shape[1]
    
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
    logger.info("STEP 4: Performing tiled inference")
    logger.info("="*60)
    
    logger.info("Starting inference on full image...")
    full_pred = predict_full_image(
        model, img_in,
        config['training']['patch_size'],
        config['training']['stride'],
        device
    )
    
    logger.info(f"Inference completed")
    logger.info(f"Prediction shape: {full_pred.shape}")
    
    logger.info("="*60)
    logger.info("STEP 5: Computing correlations on validation set")
    logger.info("="*60)
    
    # Validation mask (only evaluate on val region and tissue)
    val_mask = (split_grid_val > 0) & (mask_tissue > 0)
    logger.info(f"Validation region pixels: {val_mask.sum()}")
    
    # Get protein names
    try:
        protein_names = list(map(str, pro.var_names))
    except:
        protein_names = [f"protein_{i}" for i in range(C_out)]
    
    logger.info(f"Number of proteins: {len(protein_names)}")
    
    # Compute correlations
    logger.info("Computing Spearman and Pearson correlations...")
    spearman_rhos, pearson_rhos, mean_spearman, mean_pearson = compute_correlations(
        img_out, full_pred, val_mask, protein_names
    )
    
    logger.info("="*60)
    logger.info("Correlation Summary")
    logger.info("="*60)
    logger.info(f"Mean Spearman correlation: {mean_spearman:.4f}")
    logger.info(f"Mean Pearson correlation: {mean_pearson:.4f}")
    
    logger.info("="*60)
    logger.info("STEP 6: Generating visualizations")
    logger.info("="*60)
    
    if args.output_dir:
        logger.info("Generating correlation bar plots...")
        
        # Plot correlation bars
        plot_correlation_bar(
            spearman_rhos, protein_names, mean_spearman,
            metric_name="Spearman",
            save_path=os.path.join(args.output_dir, "spearman_correlations.png")
        )
        logger.info(f"Saved: spearman_correlations.png")
        
        plot_correlation_bar(
            pearson_rhos, protein_names, mean_pearson,
            metric_name="Pearson",
            save_path=os.path.join(args.output_dir, "pearson_correlations.png")
        )
        logger.info(f"Saved: pearson_correlations.png")
        
        # Plot top 3 proteins
        logger.info("Generating top 3 protein prediction plots...")
        sorted_idx = np.argsort(-np.nan_to_num(spearman_rhos, nan=-999))
        for i, idx in enumerate(sorted_idx[:3]):
            if not np.isnan(spearman_rhos[idx]):
                plot_predictions(
                    img_out, full_pred, val_mask,
                    protein_idx=idx,
                    protein_name=protein_names[idx],
                    save_path=os.path.join(args.output_dir, f"top{i+1}_{protein_names[idx]}.png")
                )
                logger.info(f"Saved: top{i+1}_{protein_names[idx]}.png")
    
    logger.info("="*60)
    logger.info("STEP 7: Saving results")
    logger.info("="*60)
    
    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        results_path = os.path.join(args.output_dir, "evaluation_results.npz")
        np.savez(
            results_path,
            spearman_rhos=spearman_rhos,
            pearson_rhos=pearson_rhos,
            mean_spearman=mean_spearman,
            mean_pearson=mean_pearson,
            protein_names=protein_names
        )
        logger.info(f"Results saved to: {results_path}")
    
    logger.info("="*60)
    logger.info("Evaluation completed!")
    logger.info("="*60)
    logger.info(f"Mean Spearman: {mean_spearman:.4f}")
    logger.info(f"Mean Pearson: {mean_pearson:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet for RNA to Protein prediction")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: current directory)")
    args = parser.parse_args()
    
    main(args)