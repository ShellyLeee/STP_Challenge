#!/usr/bin/env python
"""Training script for UNet RNA to Protein prediction with Early Stopping."""

import os
import sys
import time
import random
import argparse
import yaml
import json
import shutil
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import create_dataloaders
from models import UNet
from models.losses import get_loss_function
from utils.preprocessing import (
    load_and_preprocess_data,
    create_spatial_split,
    apply_dimensionality_reduction,
    rasterize_data
)
from utils.metrics import predict_full_image, compute_correlations, print_correlation_summary
from utils.visualization import plot_training_history, plot_correlation_bar


def setup_logging(run_dir, log_level=logging.INFO):
    """Setup logging to both file and console."""
    log_file = os.path.join(run_dir, "training.log")
    
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


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=1e-5, verbose=True, logger=None):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to log early stopping messages
            logger: Logger instance
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.logger = logger
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Call this method after each epoch with the validation loss.
        Returns True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose and self.logger:
                self.logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_count = 0
    
    pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch:02d}", leave=False)
    
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y, mask)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        
        pbar.set_postfix(train_loss=f"{total_loss/total_count:.5f}")
    
    return total_loss / total_count


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
    epoch: int
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    
    pbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch:02d}", leave=False)
    
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        
        y_hat = model(x)
        loss = criterion(y_hat, y, mask)
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        
        pbar.set_postfix(val_loss=f"{total_loss/total_count:.5f}")
    
    return total_loss / total_count


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(run_dir)
    
    # Create subdirectories
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config to run directory
    config_save_path = os.path.join(run_dir, "config.yaml")
    shutil.copy(args.config, config_save_path)
    logger.info(f"Config saved to: {config_save_path}")
    
    # Set seed
    set_seed(config['random_seed'])
    
    # Set device
    device = config['device']
    logger.info(f"Using device: {device}")
    logger.info(f"Run directory: {run_dir}")
    
    logger.info("="*60)
    logger.info("STEP 1: Loading and preprocessing data")
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
    logger.info("STEP 2: Creating dataloaders")
    logger.info("="*60)
    
    train_loader, val_loader = create_dataloaders(
        img_in, img_out, mask_tissue,
        split_grid_train, split_grid_val,
        config
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    logger.info("="*60)
    logger.info("STEP 3: Creating model")
    logger.info("="*60)
    
    C_in = Z_all.shape[1]
    C_out = pro_proc.X.shape[1]
    
    model = UNet(
        in_ch=C_in,
        out_ch=C_out,
        base=config['model']['base_channels']
    ).to(device)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function
    criterion = get_loss_function(config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    if config['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=int(config['training'].get('early_stopping_patience', 10)),
        min_delta=float(config['training'].get('early_stopping_min_delta', 1e-5)),
        verbose=True,
        logger=logger
    )
    
    logger.info("="*60)
    logger.info("STEP 4: Training")
    logger.info("="*60)
    logger.info(f"Early stopping: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    stopped_epoch = None
    
    # Main training loop with progress bar
    epoch_pbar = tqdm(range(1, config['training']['epochs'] + 1), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update progress bar description
        epoch_pbar.set_description(
            f"Training Progress | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
        )
        
        # Save checkpoint
        if epoch % config['logging']['save_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch{epoch:02d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "unet_best_mse.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, best_path)
            logger.debug(f"[Best↓] Epoch {epoch} - val_loss={val_loss:.5f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            stopped_epoch = epoch
            epoch_pbar.close()
            logger.info(f"[Early Stopping] Training stopped at epoch {epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.5f}")
            break
    
    logger.info("="*60)
    logger.info("STEP 5: Final evaluation on validation set")
    logger.info("="*60)
    
    # Load best model
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Perform full inference
    logger.info("Performing tiled inference...")
    full_pred = predict_full_image(
        model, img_in,
        config['training']['patch_size'],
        config['training']['stride'],
        device
    )
    
    # Compute correlations on validation set
    val_mask = (split_grid_val > 0) & (mask_tissue > 0)
    
    try:
        protein_names = list(map(str, pro.var_names))
    except:
        protein_names = [f"protein_{i}" for i in range(C_out)]
    
    spearman_rhos, pearson_rhos, mean_spearman, mean_pearson = compute_correlations(
        img_out, full_pred, val_mask, protein_names
    )
    
    logger.info("="*60)
    logger.info("Correlation Summary")
    logger.info("="*60)
    logger.info(f"Mean Spearman: {mean_spearman:.4f}")
    logger.info(f"Mean Pearson: {mean_pearson:.4f}")
    
    logger.info("="*60)
    logger.info("STEP 6: Saving results")
    logger.info("="*60)
    
    # Plot and save training history
    plot_training_history(
        train_losses, val_losses,
        save_path=os.path.join(run_dir, "training_history.png")
    )
    
    # Plot and save correlation bars
    plot_correlation_bar(
        spearman_rhos, protein_names, mean_spearman,
        metric_name="Spearman",
        save_path=os.path.join(run_dir, "spearman_correlations.png")
    )
    
    plot_correlation_bar(
        pearson_rhos, protein_names, mean_pearson,
        metric_name="Pearson",
        save_path=os.path.join(run_dir, "pearson_correlations.png")
    )
    
    # Save metrics to JSON
    metrics = {
        "training": {
            "epochs_trained": stopped_epoch if stopped_epoch else config['training']['epochs'],
            "best_epoch": checkpoint['epoch'],
            "best_train_loss": float(checkpoint['train_loss']),
            "best_val_loss": float(checkpoint['val_loss']),
            "early_stopped": stopped_epoch is not None,
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses]
        },
        "evaluation": {
            "mean_spearman": float(mean_spearman),
            "mean_pearson": float(mean_pearson),
            "per_protein_spearman": {
                protein_names[i]: float(spearman_rhos[i]) if not np.isnan(spearman_rhos[i]) else None
                for i in range(len(protein_names))
            },
            "per_protein_pearson": {
                protein_names[i]: float(pearson_rhos[i]) if not np.isnan(pearson_rhos[i]) else None
                for i in range(len(protein_names))
            }
        },
        "model": {
            "parameters": model.count_parameters(),
            "input_channels": C_in,
            "output_channels": C_out
        }
    }
    
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save detailed protein ranking
    sorted_idx = np.argsort(-np.nan_to_num(spearman_rhos, nan=-999))
    protein_ranking = []
    for i in sorted_idx:
        if not np.isnan(spearman_rhos[i]):
            protein_ranking.append({
                "protein": protein_names[i],
                "spearman": float(spearman_rhos[i]),
                "pearson": float(pearson_rhos[i])
            })
    
    ranking_path = os.path.join(run_dir, "protein_ranking.json")
    with open(ranking_path, 'w') as f:
        json.dump(protein_ranking, f, indent=2)
    
    logger.info(f"Protein ranking saved to: {ranking_path}")
    
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Best validation loss: {best_val_loss:.5f}")
    logger.info(f"Mean Spearman: {mean_spearman:.4f}")
    logger.info(f"Mean Pearson: {mean_pearson:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for RNA to Protein prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (default: run_TIMESTAMP)")
    args = parser.parse_args()
    
    main(args)