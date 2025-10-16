#!/usr/bin/env python
"""Prediction script for UNet RNA to Protein prediction."""

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
from utils.metrics import predict_full_image


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
    
    # Save predictions
    np.save(args.output, full_pred)
    print(f"\nPredictions saved to {args.output}")
    
    # Optionally save as spot-level predictions
    if args.save_spots:
        # Extract predictions for each spot
        spot_predictions = []
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            spot_predictions.append(full_pred[r, c, :])
        
        spot_predictions = np.array(spot_predictions)
        spots_path = args.output.replace('.npy', '_spots.npy')
        np.save(spots_path, spot_predictions)
        print(f"Spot-level predictions saved to {spots_path}")
        print(f"Shape: {spot_predictions.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with UNet for RNA to Protein")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default="predictions.npy",
                        help="Path to save predictions (numpy array)")
    parser.add_argument("--save_spots", action="store_true",
                        help="Also save spot-level predictions")
    args = parser.parse_args()
    
    main(args)