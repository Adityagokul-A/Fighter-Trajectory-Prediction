import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedPositionLoss(nn.Module):
    """
    Mean squared error over predicted positions, strictly ignoring padded zeros.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, k, 3], mask: [B, k]
        raw_loss = (pred - target) ** 2 
        expanded_mask = mask.unsqueeze(-1).expand_as(raw_loss) 
        
        # Mean ONLY over valid data points
        masked_loss = (raw_loss * expanded_mask).sum() / (expanded_mask.sum() + 1e-9)
        return masked_loss

class TurnConstraintLoss(nn.Module):
    """
    Explicitly penalizes heading changes greater than max_turn_degrees per second.
    """
    def __init__(self, max_turn_degrees: float = 20.0):
        super().__init__()
        self.min_cos_sim = math.cos(math.radians(max_turn_degrees))

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.size(1) < 2:
            return torch.tensor(0.0, device=pred.device)

        v0 = pred[:, 0:1, :]  
        v_rest = pred[:, 1:, :] - pred[:, :-1, :]
        velocities = torch.cat([v0, v_rest], dim=1) 

        v_current = velocities[:, :-1, :] 
        v_next = velocities[:, 1:, :]     

        cos_sim = F.cosine_similarity(v_current, v_next, dim=-1) 

        # ReLU penalty for similarity LOWER than the 20-degree threshold
        violation_penalty = F.relu(self.min_cos_sim - cos_sim) 

        # Apply mask to ignore padded transitions
        valid_mask = mask[:, 1:] 
        masked_penalty = (violation_penalty * valid_mask).sum() / (valid_mask.sum() + 1e-9)
        
        return masked_penalty

class EntropyLoss(nn.Module):
    """
    Encourages the model to use ALL regimes across the batch (avoids mode collapse).
    Maximizes the entropy of the BATCH MEAN distribution.
    """
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        # regime_probs: [B, num_regimes]
        
        # 1. Average regime usage across the batch
        avg_probs = regime_probs.mean(dim=0) 
        p = avg_probs.clamp(min=self.eps)
        
        # 2. Entropy: H = - sum(p * log(p))
        H = - (p * torch.log(p)).sum()
        
        # Normalize by max possible entropy (log(K))
        K = p.size(0)
        H_norm = H / (torch.log(torch.tensor(K, dtype=H.dtype, device=H.device)) + self.eps)
            
        # 3. Return negative entropy (optimizer minimizes loss, which maximizes H)
        return -H_norm

class BaselineLossWrapper(nn.Module):
    """
    Master loss function for the discrete regime Baseline Model.
    """
    def __init__(
        self,
        weight_pos: float = 1.0,
        weight_turn: float = 10.0,  # Enforce aerodynamics strictly
        weight_entropy: float = 0.1 # Keep small to balance diversity vs accuracy
    ):
        super().__init__()
        self.pos_loss = MaskedPositionLoss()
        self.turn_loss = TurnConstraintLoss(max_turn_degrees=20.0)
        self.entropy_loss = EntropyLoss()
        
        self.w_pos = weight_pos
        self.w_turn = weight_turn
        self.w_ent = weight_entropy

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor,
        regime_probs: torch.Tensor
    ):
        # 1. Reconstruction
        L_pos = self.pos_loss(pred, target, mask)
        
        # 2. Physics Constraint
        L_turn = self.turn_loss(pred, mask)
        
        # 3. Regime Diversity
        L_ent = self.entropy_loss(regime_probs)
        
        # Total
        L_total = (self.w_pos * L_pos) + (self.w_turn * L_turn) + (self.w_ent * L_ent)

        metrics = {
            "loss_pos": L_pos.item(),
            "loss_turn": L_turn.item(),
            "loss_ent": L_ent.item(),
            "loss_total": L_total.item()
        }

        return L_total, metrics