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
        raw_loss = (pred - target) ** 2  # [B, k, 3]
        
        # Expand mask to match 3D coordinates
        expanded_mask = mask.unsqueeze(-1).expand_as(raw_loss) # [B, k, 3]
        
        # Calculate mean ONLY over valid data points
        masked_loss = (raw_loss * expanded_mask).sum() / (expanded_mask.sum() + 1e-9)
        return masked_loss

class KLDivergenceLoss(nn.Module):
    """
    Forces the Prior Network (Inference) to match the Recognition Network (Training).
    Calculates analytical KL Divergence between two Gaussians.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu_post, logvar_post, mu_prior, logvar_prior):
        # var = exp(logvar)
        var_post = torch.exp(logvar_post)
        var_prior = torch.exp(logvar_prior)
        
        # D_KL(Q || P) = log(sigma_P / sigma_Q) + (sigma_Q^2 + (mu_Q - mu_P)^2) / (2 * sigma_P^2) - 0.5
        kl_div = 0.5 * (
            logvar_prior - logvar_post 
            + (var_post + (mu_post - mu_prior)**2) / (var_prior + 1e-9) 
            - 1.0
        )
        
        # Mean across the batch and latent dimensions
        return kl_div.sum(dim=-1).mean()

class TurnConstraintLoss(nn.Module):
    """
    Explicitly penalizes heading changes greater than max_turn_degrees per second.
    Enforces the professor's 20-degree aerodynamic limit.
    """
    def __init__(self, max_turn_degrees: float = 20.0):
        super().__init__()
        # Convert max angle to minimum allowed cosine similarity
        self.min_cos_sim = math.cos(math.radians(max_turn_degrees))

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.size(1) < 2:
            return torch.tensor(0.0, device=pred.device)

        # 1. Extract velocity vectors (dt = 1)
        # Assuming pred starts at relative (0,0,0) due to dataset translation
        v0 = pred[:, 0:1, :]  # Velocity from origin to step 1
        v_rest = pred[:, 1:, :] - pred[:, :-1, :]
        velocities = torch.cat([v0, v_rest], dim=1) # [B, k, 3]

        # 2. Compare consecutive velocities
        v_current = velocities[:, :-1, :] # [B, k-1, 3]
        v_next = velocities[:, 1:, :]     # [B, k-1, 3]

        # 3. Calculate Cosine Similarity (-1 to 1)
        # 1.0 = straight line, 0.0 = 90 deg turn, -1.0 = 180 deg U-turn
        cos_sim = F.cosine_similarity(v_current, v_next, dim=-1) # [B, k-1]

        # 4. Apply ReLU penalty for any similarity LOWER than the 20-degree threshold
        # If cos_sim >= min_cos_sim, ReLU outputs 0 (No penalty)
        # If cos_sim < min_cos_sim, ReLU outputs positive penalty
        violation_penalty = F.relu(self.min_cos_sim - cos_sim) # [B, k-1]

        # 5. Mask out padded timesteps
        valid_mask = mask[:, 1:] # Shift mask to match k-1 transitions
        masked_penalty = (violation_penalty * valid_mask).sum() / (valid_mask.sum() + 1e-9)
        
        return masked_penalty

class CVAELossWrapper(nn.Module):
    """
    Master loss function calculating Reconstruction + KL + Physics constraints.
    """
    def __init__(
        self,
        weight_pos: float = 1.0,
        weight_kl: float = 0.1,  # Keep < 1.0 to prevent KL vanishing (Beta-VAE)
        weight_turn: float = 10.0 # High weight to strictly enforce 20-degree limit
    ):
        super().__init__()
        self.pos_loss = MaskedPositionLoss()
        self.kl_loss = KLDivergenceLoss()
        self.turn_loss = TurnConstraintLoss(max_turn_degrees=20.0)
        
        self.w_pos = weight_pos
        self.w_kl = weight_kl
        self.w_turn = weight_turn

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor
    ):
        
        # 1. Reconstruction (MSE)
        L_pos = self.pos_loss(pred, target, mask)
        
        # 2. Distribution Matching (KL Divergence)
        L_kl = self.kl_loss(mu_post, logvar_post, mu_prior, logvar_prior)
        
        # 3. Physics Constraint (20-Degree Limit)
        L_turn = self.turn_loss(pred, mask)
        
        # Total Loss
        L_total = (self.w_pos * L_pos) + (self.w_kl * L_kl) + (self.w_turn * L_turn)

        metrics = {
            "loss_pos": L_pos.item(),
            "loss_kl": L_kl.item(),
            "loss_turn": L_turn.item(),
            "loss_total": L_total.item()
        }

        return L_total, metrics