import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineEncoder(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

    def forward(self, x_padded, input_lengths):
        # Pack sequence to ignore zeros from short flight padding
        packed_in = nn.utils.rnn.pack_padded_sequence(
            x_padded, 
            input_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        _, h_n = self.gru(packed_in)
        
        # Extract the final valid hidden state as context
        context = h_n[-1] # [B, hidden_dim]
        return context

class BaselineRegimeHead(nn.Module):
    def __init__(self, hidden_dim=128, num_regimes=6, embed_dim=32):
        super().__init__()
        self.fc_logits = nn.Linear(hidden_dim, num_regimes)
        self.regime_embedding = nn.Embedding(num_regimes, embed_dim)

    def forward(self, context):
        logits = self.fc_logits(context)
        probs = F.softmax(logits, dim=-1) # [B, num_regimes]
        
        # Soft regime embedding (mixture of intents)
        embed_matrix = self.regime_embedding.weight
        regime_embed = probs @ embed_matrix # [B, embed_dim]
        
        return probs, regime_embed

class BaselineDecoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, regime_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim + regime_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, context, regime_embed, steps, targets=None, teacher_forcing_ratio=0.5):
        B = regime_embed.size(0)
        device = regime_embed.device
        
        # Initialize at origin due to Dataset translation invariance
        decoder_input = torch.zeros(B, 1, 3, device=device)
        
        outputs = []
        # Init GRU memory with Encoder context
        h = context.unsqueeze(0) 

        for t in range(steps):
            r_in = regime_embed.unsqueeze(1)
            rnn_in = torch.cat([decoder_input, r_in], dim=-1)
            
            out, h = self.gru(rnn_in, h)
            next_pos = self.output_layer(out) # [B, 1, 3]
            outputs.append(next_pos)
            
            # Autoregression logic (Teacher Forcing retained for baseline)
            use_teacher = False
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                use_teacher = True
                
            if use_teacher and t < steps - 1:
                decoder_input = targets[:, t:t+1, :]
            else:
                decoder_input = next_pos 

        return torch.cat(outputs, dim=1) # [B, k, 3]

class BaselineTrajectoryModel(nn.Module):
    """
    Master Wrapper for the original discrete regime architecture.
    """
    def __init__(self):
        super().__init__()
        self.encoder = BaselineEncoder(input_dim=14)
        self.regime_head = BaselineRegimeHead()
        self.decoder = BaselineDecoder()

    def forward(self, x_padded, input_lengths, steps, targets=None, tf_ratio=0.5):
        context = self.encoder(x_padded, input_lengths)
        probs, regime_embed = self.regime_head(context)
        
        # Pass targets into decoder for teacher forcing
        preds = self.decoder(context, regime_embed, steps, targets, tf_ratio)
        
        return preds, probs