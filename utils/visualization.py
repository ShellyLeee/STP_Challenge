import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_correlation_bar(
    correlations: np.ndarray,
    protein_names: List[str],
    mean_value: float,
    metric_name: str = "Spearman",
    figsize: tuple = None,
    save_path: str = None
):
    """
    Plot bar chart of per-protein correlations.
    
    Args:
        correlations: Array of correlation values
        protein_names: List of protein names
        mean_value: Mean correlation value
        metric_name: Name of metric (e.g., "Spearman", "Pearson")
        figsize: Figure size (auto if None)
        save_path: Path to save figure (optional)
    """
    C_out = len(protein_names)
    
    # Sort by correlation
    sorted_idx = np.argsort(-np.nan_to_num(correlations, nan=-999))
    sorted_corr = correlations[sorted_idx]
    sorted_names = [protein_names[i] for i in sorted_idx]
    
    # Auto figure size
    if figsize is None:
        figsize = (max(8, C_out * 0.3), 5)
    
    plt.figure(figsize=figsize)
    plt.bar(np.arange(C_out), np.nan_to_num(sorted_corr, nan=0.0))
    plt.axhline(mean_value, linestyle="--", linewidth=1.5, color='r', 
                label=f'Mean = {mean_value:.4f}')
    plt.xlabel(f"Protein channel (sorted by {metric_name})")
    plt.ylabel(f"{metric_name} ρ")
    plt.title(f"Per-protein {metric_name} correlation (sorted descending)")
    plt.legend()
    
    # X-axis labels
    if C_out <= 60:
        plt.xticks(np.arange(C_out), sorted_names, rotation=90)
    else:
        step = max(1, C_out // 40)
        sel = np.arange(C_out)[::step]
        plt.xticks(sel, [sorted_names[i] for i in sel], rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    protein_idx: int,
    protein_name: str = None,
    save_path: str = None
):
    """
    Plot ground truth vs prediction for a single protein channel.
    
    Args:
        y_true: Ground truth (H, W, C)
        y_pred: Predictions (H, W, C)
        mask: Binary mask (H, W)
        protein_idx: Index of protein to visualize
        protein_name: Name of protein (optional)
        save_path: Path to save figure (optional)
    """
    if protein_name is None:
        protein_name = f"Protein {protein_idx}"
    
    # Extract channel
    true_map = y_true[..., protein_idx]
    pred_map = y_pred[..., protein_idx]
    
    # Mask background
    true_map_masked = np.where(mask > 0, true_map, np.nan)
    pred_map_masked = np.where(mask > 0, pred_map, np.nan)
    
    # Compute vmin/vmax for consistent color scale
    vmin = np.nanmin([true_map_masked, pred_map_masked])
    vmax = np.nanmax([true_map_masked, pred_map_masked])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth
    im0 = axes[0].imshow(true_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{protein_name} - Ground Truth')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Prediction
    im1 = axes[1].imshow(pred_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{protein_name} - Prediction')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Difference
    diff = pred_map_masked - true_map_masked
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes[2].set_title(f'{protein_name} - Difference')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = None
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()