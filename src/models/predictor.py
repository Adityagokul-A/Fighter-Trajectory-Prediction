import torch
import torch.nn as nn
from .components import KinematicEncoder, RegimeHead, TrajectoryDecoder

class TrajectoryPredictor(nn.Module):
    def __init__(
        self, 
        input_dim=11, 
        hidden_dim=128, 
        num_layers=1, 
        num_regimes=6, 
        regime_embed_dim=32
    ):
        super().__init__()
        
        self.encoder = KinematicEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.regime_head = RegimeHead(
            hidden_dim=hidden_dim,
            num_regimes=num_regimes,
            embed_dim=regime_embed_dim
        )
        
        self.decoder = TrajectoryDecoder(
            input_dim=3,
            hidden_dim=hidden_dim,
            regime_dim=regime_embed_dim,
            num_layers=num_layers
        )

    # --- FIX IS HERE ---
    # We added targets=None and teacher_forcing_ratio=0.5
    def forward(self, x, pred_steps, targets=None, teacher_forcing_ratio=0.5):
        """
        x: Input history [B, N, 11]
        pred_steps: How many steps to predict (k)
        targets: Ground truth future [B, k, 3] (for training)
        teacher_forcing_ratio: Probability of using ground truth input
        """
        
        # 1. Encode
        context, h_enc = self.encoder(x)
        
        # 2. Infer Regime
        regime_probs, regime_embed = self.regime_head(context)
        
        # 3. Decode
        # Pass targets and ratio to the decoder
        y_pred = self.decoder(
            h_0=h_enc,
            regime_embed=regime_embed,
            steps=pred_steps,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        return y_pred, regime_probs