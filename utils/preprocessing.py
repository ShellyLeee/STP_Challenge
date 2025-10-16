import numpy as np
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_and_preprocess_data(config: dict) -> Tuple:
    """
    Load RNA and Protein data and perform preprocessing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (rna_processed, pro_processed, rna_raw, pro_raw)
    """
    # Load data
    rna = sc.read_h5ad(config['data']['rna_h5ad'])
    pro = sc.read_h5ad(config['data']['pro_h5ad'])
    
    print(f"RNA data: {rna}")
    print(f"Protein data: {pro}")
    
    # Make copies for processing
    rna_proc = rna.copy()
    pro_proc = pro.copy()
    
    # Optional RNA preprocessing
    if config['preprocessing'].get('normalize_rna', False):
        sc.pp.normalize_total(rna_proc, target_sum=1e4)
        sc.pp.log1p(rna_proc)
        print("Applied normalize_total + log1p to RNA")
    
    # Optional Protein preprocessing
    if config['preprocessing'].get('zscore_protein', False):
        pro_X = np.asarray(pro_proc.X)
        pro_X = np.log1p(pro_X)
        scaler = StandardScaler(with_mean=True, with_std=True)
        pro_X = scaler.fit_transform(pro_X)
        pro_proc.X = pro_X
        print("Applied log1p + z-score to Protein")
    
    return rna_proc, pro_proc, rna, pro


def create_spatial_split(
    rna: sc.AnnData,
    grid_h: int,
    grid_w: int,
    split_ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Create spatial train/val split based on row position.
    
    Args:
        rna: RNA AnnData object
        grid_h: Grid height
        grid_w: Grid width
        split_ratio: Fraction for training (default 0.9 = 90% train, 10% val)
        
    Returns:
        rows, cols, split_grid_train, split_grid_val, cutoff
    """
    rows = rna.obs["array_row"].to_numpy().astype(int)
    cols = rna.obs["array_col"].to_numpy().astype(int)
    
    # Spatial split by row position
    cutoff = int(split_ratio * grid_h)
    
    split_grid_train = np.zeros((grid_h, grid_w), dtype=bool)
    split_grid_val = np.zeros((grid_h, grid_w), dtype=bool)
    
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        if r < cutoff:
            split_grid_train[r, c] = True
        else:
            split_grid_val[r, c] = True
    
    print(f"Spatial split cutoff row: {cutoff}")
    print(f"Train patches: {split_grid_train.sum()}, Val patches: {split_grid_val.sum()}")
    
    return rows, cols, split_grid_train, split_grid_val, cutoff


def apply_dimensionality_reduction(
    rna_proc: sc.AnnData,
    rows: np.ndarray,
    cutoff: int,
    k_pca: int,
    random_seed: int = 42
) -> np.ndarray:
    """
    Apply TruncatedSVD (PCA for sparse matrices) to RNA data.
    Fit only on training data to avoid leakage.
    
    Args:
        rna_proc: Preprocessed RNA data
        rows: Array row positions
        cutoff: Row cutoff for train/val split
        k_pca: Number of components to keep
        random_seed: Random seed for reproducibility
        
    Returns:
        Reduced RNA data for all spots (n_spots, k_pca)
    """
    rna_X = rna_proc.X
    train_mask = rows < cutoff
    rna_train = rna_X[train_mask]
    
    # Fit SVD only on training data
    svd = TruncatedSVD(n_components=k_pca, random_state=random_seed)
    svd.fit(rna_train)
    
    # Transform all data
    Z_all = svd.transform(rna_X)
    Z_all = np.asarray(Z_all, dtype=np.float32)
    
    print(f"Reduced RNA from {rna_X.shape[1]} to {k_pca} dimensions")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    return Z_all


def rasterize_data(
    rows: np.ndarray,
    cols: np.ndarray,
    Z_all: np.ndarray,
    pro_X: np.ndarray,
    rna: sc.AnnData,
    grid_h: int,
    grid_w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize spot-level data to grid format (H, W, C).
    
    Args:
        rows: Array row positions
        cols: Array column positions
        Z_all: Reduced RNA data (n_spots, C_in)
        pro_X: Protein data (n_spots, C_out)
        rna: RNA AnnData (for tissue mask)
        grid_h: Grid height
        grid_w: Grid width
        
    Returns:
        img_in (H, W, C_in), img_out (H, W, C_out), mask_tissue (H, W)
    """
    C_in = Z_all.shape[1]
    C_out = pro_X.shape[1]
    
    img_in = np.full((grid_h, grid_w, C_in), np.nan, dtype=np.float32)
    img_out = np.full((grid_h, grid_w, C_out), np.nan, dtype=np.float32)
    mask_tissue = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    has_in_tissue = "in_tissue" in rna.obs.columns
    
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        img_in[r, c, :] = Z_all[i]
        img_out[r, c, :] = pro_X[i]
        mask_tissue[r, c] = int(rna.obs["in_tissue"].iloc[i]) if has_in_tissue else 1
    
    # Replace NaN with 0
    img_in = np.nan_to_num(img_in, nan=0.0)
    img_out = np.nan_to_num(img_out, nan=0.0)
    
    print(f"Rasterized data shape: img_in={img_in.shape}, img_out={img_out.shape}")
    print(f"Tissue coverage: {mask_tissue.sum()} / {grid_h * grid_w} pixels")
    
    return img_in, img_out, mask_tissue