import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, List


@torch.no_grad()
def predict_full_image(
    model: nn.Module,
    img_in: np.ndarray,
    patch_size: int,
    stride: int,
    device: str
) -> np.ndarray:
    """
    Perform tiled inference with overlap-averaging on full image.
    
    Args:
        model: Trained model
        img_in: Input image (H, W, C_in)
        patch_size: Size of patches for inference
        stride: Stride for patch extraction
        device: Device to run inference on
        
    Returns:
        Predicted image (H, W, C_out)
    """
    model.eval()
    H, W, C_in = img_in.shape
    
    # Get output channels from model
    dummy_input = torch.zeros(1, C_in, patch_size, patch_size).to(device)
    dummy_output = model(dummy_input)
    C_out = dummy_output.shape[1]
    
    # Accumulation arrays
    out_accum = np.zeros((H, W, C_out), dtype=np.float32)
    weight = np.zeros((H, W, 1), dtype=np.float32)
    
    # Tile over image
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patch
            xin = img_in[y:y+patch_size, x:x+patch_size, :]
            xin = np.moveaxis(xin, -1, 0)[None, ...]  # (1, C, H, W)
            xin = torch.from_numpy(xin).float().to(device)
            
            # Predict
            yhat = model(xin).cpu().numpy()[0]  # (C_out, H, W)
            yhat = np.moveaxis(yhat, 0, -1)  # (H, W, C_out)
            
            # Accumulate
            out_accum[y:y+patch_size, x:x+patch_size, :] += yhat
            weight[y:y+patch_size, x:x+patch_size, :] += 1.0
    
    # Average overlapping regions
    weight[weight == 0] = 1.0
    return out_accum / weight


def compute_correlations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    protein_names: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute per-protein Spearman and Pearson correlations on masked regions.
    
    Args:
        y_true: Ground truth (H, W, C)
        y_pred: Predictions (H, W, C)
        mask: Binary mask (H, W)
        protein_names: List of protein names (optional)
        
    Returns:
        spearman_rhos, pearson_rhos, mean_spearman, mean_pearson
    """
    _, _, C_out = y_true.shape
    
    if protein_names is None:
        protein_names = [f"protein_{i}" for i in range(C_out)]
    
    spearman_rhos = np.full(C_out, np.nan, dtype=np.float64)
    pearson_rhos = np.full(C_out, np.nan, dtype=np.float64)
    
    for j in range(C_out):
        true_vals = y_true[..., j][mask > 0]
        pred_vals = y_pred[..., j][mask > 0]
        
        n = true_vals.size
        if n < 3 or np.std(true_vals) < 1e-8 or np.std(pred_vals) < 1e-8:
            continue
        
        try:
            rho_s, _ = spearmanr(true_vals, pred_vals)
            rho_p, _ = pearsonr(true_vals, pred_vals)
            spearman_rhos[j] = rho_s
            pearson_rhos[j] = rho_p
        except:
            pass
    
    mean_spearman = np.nanmean(spearman_rhos)
    mean_pearson = np.nanmean(pearson_rhos)
    
    return spearman_rhos, pearson_rhos, mean_spearman, mean_pearson


def print_correlation_summary(
    spearman_rhos: np.ndarray,
    pearson_rhos: np.ndarray,
    protein_names: List[str],
    mean_spearman: float,
    mean_pearson: float
):
    """
    Print summary of correlation results.
    
    Args:
        spearman_rhos: Per-protein Spearman correlations
        pearson_rhos: Per-protein Pearson correlations
        protein_names: List of protein names
        mean_spearman: Mean Spearman correlation
        mean_pearson: Mean Pearson correlation
    """
    C_out = len(protein_names)
    nan_s_count = np.isnan(spearman_rhos).sum()
    nan_p_count = np.isnan(pearson_rhos).sum()
    
    print(f"\n[Final Evaluation]")
    print(f"Mean Spearman across {C_out} proteins: {mean_spearman:.4f}")
    print(f"Mean Pearson  across {C_out} proteins: {mean_pearson:.4f}")
    
    if nan_s_count or nan_p_count:
        print(f"Note: NaN channels -> Spearman: {nan_s_count}, Pearson: {nan_p_count}")
    
    # Print top proteins
    sorted_idx = np.argsort(-np.nan_to_num(spearman_rhos, nan=-999))
    print(f"\nTop 10 proteins by Spearman correlation:")
    for i in sorted_idx[:10]:
        name = protein_names[i]
        s = spearman_rhos[i]
        p = pearson_rhos[i]
        if not np.isnan(s):
            print(f"  {name:<20s}  Spearman={s:>6.3f}  Pearson={p:>6.3f}")