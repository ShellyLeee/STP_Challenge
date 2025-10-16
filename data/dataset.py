import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


class PatchDataset(Dataset):
    """Dataset for extracting patches from spatial RNA/Protein data."""
    
    def __init__(
        self,
        img_in: np.ndarray,
        img_out: np.ndarray,
        mask: np.ndarray,
        patch_size: int,
        stride: int,
        split_mask: Optional[np.ndarray] = None
    ):
        """
        Args:
            img_in: Input RNA data (H, W, C_in)
            img_out: Output Protein data (H, W, C_out)
            mask: Tissue mask (H, W)
            patch_size: Size of patches to extract
            stride: Stride for patch extraction (overlap = patch_size - stride)
            split_mask: Mask for train/val split (H, W), optional
        """
        self.img_in = img_in
        self.img_out = img_out
        self.mask = mask
        self.patch_size = patch_size
        self.stride = stride
        self.coords = []
        
        H, W, _ = img_in.shape
        
        # Generate patch coordinates
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                submask = mask[y:y+patch_size, x:x+patch_size]
                
                # Skip patches with no tissue
                if submask.sum() < 1:
                    continue
                
                # Filter by split mask if provided
                if split_mask is not None:
                    if split_mask[y:y+patch_size, x:x+patch_size].mean() < 0.5: # keep the patch with occupied area of train/val > 50%
                        continue
                
                self.coords.append((y, x))
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> dict:
        y, x = self.coords[idx]
        p = self.patch_size
        
        # Extract patches
        xin = self.img_in[y:y+p, x:x+p, :]  # RNA Embedding
        yout = self.img_out[y:y+p, x:x+p, :]  # Protein Map
        m = self.mask[y:y+p, x:x+p]  # Tissue Mask
        
        # Convert to channels-first format (C, H, W)
        xin = np.moveaxis(xin, -1, 0)
        yout = np.moveaxis(yout, -1, 0)
        
        return {
            "x": torch.from_numpy(xin).float(),
            "y": torch.from_numpy(yout).float(),
            "mask": torch.from_numpy(m[None, ...]).float(),
            "top_left": (y, x),
        }


def create_dataloaders(
    img_in: np.ndarray,
    img_out: np.ndarray,
    mask_tissue: np.ndarray,
    split_grid_train: np.ndarray,
    split_grid_val: np.ndarray,
    config: dict
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        img_in: Input RNA data (H, W, C_in)
        img_out: Output Protein data (H, W, C_out)
        mask_tissue: Tissue mask (H, W)
        split_grid_train: Train split mask (H, W)
        split_grid_val: Validation split mask (H, W)
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader
    """
    patch_size = config['training']['patch_size']
    stride = config['training']['stride']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory']
    
    # Create datasets
    train_ds = PatchDataset(
        img_in, img_out, mask_tissue,
        patch_size, stride,
        split_mask=split_grid_train
    )
    
    val_ds = PatchDataset(
        img_in, img_out, mask_tissue,
        patch_size, stride,
        split_mask=split_grid_val
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader