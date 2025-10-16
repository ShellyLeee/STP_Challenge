import torch
import torch.nn as nn


class MaskedMSE(nn.Module):
    """Masked Mean Squared Error loss - only compute loss on tissue regions."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)
            mask: Binary mask (B, 1, H, W) - 1 for tissue, 0 for background
            
        Returns:
            Masked MSE loss (scalar)
        """
        loss_map = self.mse(pred, target)
        loss_map = loss_map * mask
        denom = mask.sum() + 1e-6
        return loss_map.sum() / denom


class MaskedHuber(nn.Module):
    """Masked Huber loss - robust to outliers, only compute on tissue regions."""
    
    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta: Huber loss delta parameter (beta in PyTorch SmoothL1Loss)
        """
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none', beta=delta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)
            mask: Binary mask (B, 1, H, W) - 1 for tissue, 0 for background
            
        Returns:
            Masked Huber loss (scalar)
        """
        loss_map = self.huber(pred, target)
        loss_map = loss_map * mask
        denom = mask.sum() + 1e-6
        return loss_map.sum() / denom


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loss function module
    """
    loss_type = config['loss']['type']
    
    if loss_type == 'mse':
        return MaskedMSE()
    elif loss_type == 'huber':
        delta = config['loss'].get('huber_delta', 1.0)
        return MaskedHuber(delta=delta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")