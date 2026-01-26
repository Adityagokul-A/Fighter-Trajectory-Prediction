import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicEncoder(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

    def forward(self, x):
        # x: [B, N, 11]
        _, h_n = self.gru(x)
        # h_n: [num_layers, B, H]
        
        # Use last layer hidden state as context
        context = h_n[-1] # [B, H]
        return context, h_n

class RegimeHead(nn.Module):
    def __init__(self, hidden_dim=128, num_regimes=6, embed_dim=32):
        super().__init__()
        self.num_regimes = num_regimes
        self.fc_logits = nn.Linear(hidden_dim, num_regimes)
        self.regime_embedding = nn.Embedding(num_regimes, embed_dim)

    def forward(self, context):
        logits = self.fc_logits(context)
        probs = F.softmax(logits, dim=-1) # [B, K]
        
        # Soft regime embedding (mixture)
        embed_matrix = self.regime_embedding.weight
        regime_embed = probs @ embed_matrix # [B, R]
        
        return probs, regime_embed

class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, regime_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim + regime_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, h_0, regime_embed, steps, targets=None, teacher_forcing_ratio=0.5):
        """
        targets: Ground truth deltas [B, k, 3] (optional, for training)
        teacher_forcing_ratio: Probability of using ground truth input
        """
        B = regime_embed.size(0)
        device = regime_embed.device
        
        # Initial input = zero delta (assuming start of prediction is steady)
        # Optimization: Ideally, pass the last known velocity here instead of zeros
        decoder_input = torch.zeros(B, 1, 3, device=device)
        
        outputs = []
        h = h_0 # Initialize hidden state with Encoder state

        for t in range(steps):
            # 1. Prepare Input: Concatenate Delta + Regime
            # regime_embed: [B, R] -> [B, 1, R]
            r_in = regime_embed.unsqueeze(1)
            rnn_in = torch.cat([decoder_input, r_in], dim=-1)
            
            # 2. Forward Step
            out, h = self.gru(rnn_in, h)
            delta_pred = self.output_layer(out) # [B, 1, 3]
            outputs.append(delta_pred)
            
            # 3. Autoregression Logic (Teacher Forcing)
            # Decide next input: Truth or Prediction?
            use_teacher = False
            if torch.rand(1).item() < teacher_forcing_ratio:
                use_teacher = True
                
            if use_teacher:
                # Use Ground Truth for next step (if available)
                if t < steps - 1:
                    decoder_input = targets[:, t:t+1, :]
            else:
                # Use own prediction
                # CRITICAL FIX: Do NOT detach here. We want gradients to flow back!
                decoder_input = delta_pred 

        return torch.cat(outputs, dim=1) # [B, k, 3]