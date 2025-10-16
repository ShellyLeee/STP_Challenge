#!/usr/bin/env python
"""Evaluation script for UNet RNA to Protein prediction."""

import argparse
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
    compute_correlations,
    print_correlation_summary
)
from utils.visualization import plot_correlation_bar, plot_predictions


def main(args):
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    device = config['device']
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("STEP 1: Loading and preprocessing data")
    print("="*60)
    
    # Load data
    rna_proc, pro_proc, rna, pro = load_and_preprocess_data(config)
    
    # Create spatial split
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
    
    print("\n" + "="*60)
    print("STEP 2: Loading model")
    print("="*60)
    
    C_in = Z_all.shape[1]
    C_out = pro_proc.X.shape[1]
    
    model = UNet(
        in_ch=C_in,
        out_ch=C_out,
        base=config['model']['base_channels']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.5f}")
    print(f"Validation loss: {checkpoint['val_loss']:.5f}")
    
    print("\n" + "="*60)
    print("STEP 3: Performing tiled inference")
    print("="*60)
    
    full_pred = predict_full_image(
        model, img_in,
        config['training']['patch_size'],
        config['training']['stride'],
        device
    )
    
    print(f"Prediction shape: {full_pred.shape}")
    
    print("\n" + "="*60)
    print("STEP 4: Computing correlations on validation set")
    print("="*60)
    
    # Validation mask (only evaluate on val region and tissue)
    val_mask = (split_grid_val > 0) & (mask_tissue > 0)
    
    # Get protein names
    try:
        protein_names = list(map(str, pro.var_names))
    except:
        protein_names = [f"protein_{i}" for i in range(C_out)]
    
    # Compute correlations
    spearman_rhos, pearson_rhos, mean_spearman, mean_pearson = compute_correlations(
        img_out, full_pred, val_mask, protein_names
    )
    
    # Print summary
    print_correlation_summary(
        spearman_rhos, pearson_rhos, protein_names,
        mean_spearman, mean_pearson
    )
    
    print("\n" + "="*60)
    print("STEP 5: Generating visualizations")
    print("="*60)
    
    # Plot correlation bars
    plot_correlation_bar(
        spearman_rhos, protein_names, mean_spearman,
        metric_name="Spearman",
        save_path=args.output_dir + "/spearman_correlations.png" if args.output_dir else None
    )
    
    plot_correlation_bar(
        pearson_rhos, protein_names, mean_pearson,
        metric_name="Pearson",
        save_path=args.output_dir + "/pearson_correlations.png" if args.output_dir else None
    )
    
    # Plot top 3 proteins
    sorted_idx = np.argsort(-np.nan_to_num(spearman_rhos, nan=-999))
    for i, idx in enumerate(sorted_idx[:3]):
        if not np.isnan(spearman_rhos[idx]):
            plot_predictions(
                img_out, full_pred, val_mask,
                protein_idx=idx,
                protein_name=protein_names[idx],
                save_path=args.output_dir + f"/top{i+1}_{protein_names[idx]}.png" if args.output_dir else None
            )
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)
    
    # Save results if output directory specified
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        np.savez(
            os.path.join(args.output_dir, "evaluation_results.npz"),
            spearman_rhos=spearman_rhos,
            pearson_rhos=pearson_rhos,
            mean_spearman=mean_spearman,
            mean_pearson=mean_pearson,
            protein_names=protein_names
        )
        print(f"Results saved to {args.output_dir}/evaluation_results.npz")