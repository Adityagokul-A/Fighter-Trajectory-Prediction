import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

class PositionLoss(nn.Module):
    """
    Mean squared error over predicted positions/deltas.
    pred: [B, k, 3]
    target: [B, k, 3]
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simple elementwise MSE
        loss = (pred - target) ** 2  # [B, k, 3]
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            # Per-sample mean over time & coords (useful for unreduced analysis)
            return loss.mean(dim=[1, 2])


class SmoothnessLoss(nn.Module):
    """
    Penalize second-order temporal differences (acceleration changes) of the prediction.
    pred: [B, k, 3]
    
    NOTE: For fighter jets, we keep the weight low. We want to penalize 
    unrealistic 'jitter', but not valid high-G maneuvers.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.size(1) < 3:
            # No second difference possible; return zero
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # First derivative (velocity proxy)
        d1 = pred[:, 1:, :] - pred[:, :-1, :]        # [B, k-1, 3]
        
        # Second derivative (acceleration proxy)
        d2 = d1[:, 1:, :] - d1[:, :-1, :]            # [B, k-2, 3]
        
        loss = d2.pow(2)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean(dim=[1, 2])


class EntropyLoss(nn.Module):
    """
    Encourages the model to use ALL regimes across the batch (avoids mode collapse).
    
    CRITICAL FIX: 
    Instead of maximizing entropy per sample (which forces the model to be unsure),
    we maximize the entropy of the BATCH MEAN distribution.
    
    This allows the model to be confident on individual samples (p=[0.9, 0.1...])
    while ensuring the global distribution covers all regimes.
    """
    def __init__(self, normalize: bool = True, eps: float = 1e-9):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        # regime_probs: [B, K]
        
        # 1. Compute the average usage of each regime across the batch
        # shape: [K]
        avg_probs = regime_probs.mean(dim=0) 
        
        p = avg_probs.clamp(min=self.eps)
        
        # 2. Compute entropy of this average distribution
        # H = - sum( p * log(p) )
        H = - (p * torch.log(p)).sum()
        
        if self.normalize:
            K = p.size(0)
            H = H / (torch.log(torch.tensor(K, dtype=H.dtype, device=H.device)) + self.eps)
            
        # 3. Return NEGATIVE entropy (because optimizers minimize loss)
        # Minimizing -H is equivalent to Maximizing H
        return -H


class CombinedLoss(nn.Module):
    """
    Wrapper to compute total loss:
      L = w_pos * L_pos + w_smooth * L_smooth + w_ent * L_ent
    """
    def __init__(
        self,
        weight_pos: float = 1.0,
        weight_smooth: float = 0.01, # Keep low to allow high-G turns
        weight_entropy: float = 0.1, # Weight for regime diversity
        reduction: str = "mean"
    ):
        super().__init__()
        self.pos_loss = PositionLoss(reduction=reduction)
        self.smooth_loss = SmoothnessLoss(reduction=reduction)
        self.entropy_loss = EntropyLoss(normalize=True)
        
        self.w_pos = weight_pos
        self.w_smooth = weight_smooth
        self.w_entropy = weight_entropy

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        regime_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: torch.Tensor (for backward)
            metrics: dict (for logging)
        """
        
        # 1. Position Loss
        L_pos = self.pos_loss(pred, target)
        
        # 2. Smoothness Loss
        L_smooth = self.smooth_loss(pred)
        
        # Base total
        L_total = (self.w_pos * L_pos) + (self.w_smooth * L_smooth)
        
        # 3. Entropy Loss (if applicable)
        L_ent_val = 0.0
        if (regime_probs is not None) and (self.w_entropy > 0.0):
            L_ent = self.entropy_loss(regime_probs)
            L_total = L_total + (self.w_entropy * L_ent)
            L_ent_val = L_ent.item()

        # Return tuple for logging
        metrics = {
            "loss_pos": L_pos.item(),
            "loss_smooth": L_smooth.item(),
            "loss_ent": L_ent_val,
            "loss_total": L_total.item()
        }

        return L_total, metrics