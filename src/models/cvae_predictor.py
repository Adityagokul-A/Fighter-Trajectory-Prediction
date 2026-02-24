import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionEncoder(nn.Module):
    """
    Encodes the variable-length 14-feature past into a fixed context vector.
    """
    def __init__(self, input_dim=14, hidden_dim=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x_padded, input_lengths):
        # x_padded: [B, N, 14]
        
        # 1. Pack the sequence to ignore zero-padding
        # lengths must be on CPU for PyTorch's packing utility
        packed_in = nn.utils.rnn.pack_padded_sequence(
            x_padded, 
            input_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 2. Forward pass
        _, h_n = self.gru(packed_in)
        
        # 3. Extract the final hidden state of the top layer as context 'c'
        context = h_n[-1] # [B, hidden_dim]
        return context

class RecognitionNetwork(nn.Module):
    """
    TRAINING ONLY: Looks at the Ground Truth Future to learn the true latent distribution.
    Approximates the posterior q(z|x, y)
    """
    def __init__(self, target_dim=3, context_dim=128, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.gru = nn.GRU(target_dim, hidden_dim, batch_first=True)
        # Fuses the encoded future and the encoded past
        self.fc_mu = nn.Linear(hidden_dim + context_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + context_dim, latent_dim)

    def forward(self, y_future, context):
        # y_future: [B, k, 3] (Ground truth future positions)
        _, h_n = self.gru(y_future)
        y_encoded = h_n[-1] # [B, hidden_dim]
        
        fused = torch.cat([y_encoded, context], dim=-1) # [B, hidden_dim + context_dim]
        
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        return mu, logvar

class PriorNetwork(nn.Module):
    """
    INFERENCE & TRAINING: Guesses the latent distribution using ONLY the past context.
    Approximates the prior p(z|x)
    """
    def __init__(self, context_dim=128, latent_dim=32):
        super().__init__()
        self.fc_mu = nn.Linear(context_dim, latent_dim)
        self.fc_logvar = nn.Linear(context_dim, latent_dim)

    def forward(self, context):
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        return mu, logvar

class AutoregressiveDecoder(nn.Module):
    """
    Steps forward in time, feeding its predicted position back into itself.
    The hidden state 'h' maintains the physical memory (velocity/acceleration).
    """
    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=32, num_layers=1):
        super().__init__()
        # Input gets concatenated with the latent intent 'z'
        self.gru = nn.GRU(input_dim + latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, context, z, steps):
        # context: [B, hidden_dim] -> Used to initialize the GRU's physical memory
        # z: [B, latent_dim] -> The single chosen intent for this trajectory
        B = z.size(0)
        device = z.device
        
        outputs = []
        
        # Initial hidden state = Context from the Encoder
        h = context.unsqueeze(0) 
        
        # Initial position = (0,0,0) because our Dataset enforces translation invariance!
        current_pos = torch.zeros(B, 1, 3, device=device)
        
        # Reshape z to concatenate at every time step
        z_expanded = z.unsqueeze(1) # [B, 1, latent_dim]

        for t in range(steps):
            # 1. Fuse Current Position and Latent Intent
            rnn_in = torch.cat([current_pos, z_expanded], dim=-1) # [B, 1, 3 + latent_dim]
            
            # 2. Step the GRU forward
            out, h = self.gru(rnn_in, h)
            
            # 3. Predict the new spatial position
            next_pos = self.output_layer(out) # [B, 1, 3]
            outputs.append(next_pos)
            
            # 4. Strict Autoregression: Feed prediction to the next step
            current_pos = next_pos 

        return torch.cat(outputs, dim=1) # [B, k, 3]

class TrajectoryCVAE(nn.Module):
    """
    The Master Wrapper uniting the Encoder, Recognition, Prior, and Decoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder = ConditionEncoder()
        self.recognition = RecognitionNetwork()
        self.prior = PriorNetwork()
        self.decoder = AutoregressiveDecoder()

    def reparameterize(self, mu, logvar):
        """ The Reparameterization Trick (z = mu + sigma * epsilon) """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_padded, input_lengths, y_future, steps):
        # 1. Encode Past
        context = self.encoder(x_padded, input_lengths)
        
        # 2. TRAINING PATH: Look at the future to get posterior distribution
        mu_post, logvar_post = self.recognition(y_future, context)
        z = self.reparameterize(mu_post, logvar_post)
        
        # 3. Get Prior distribution (for the KL Divergence loss later)
        mu_prior, logvar_prior = self.prior(context)
        
        # 4. Decode
        y_pred = self.decoder(context, z, steps)
        
        return y_pred, mu_post, logvar_post, mu_prior, logvar_prior

    def inference(self, x_padded, input_lengths, steps, num_samples=5):
        """ Call this during deployment to generate multiple futures! """
        context = self.encoder(x_padded, input_lengths)
        
        # We don't have y_future, so we ask the Prior network to guess the distribution
        mu_prior, logvar_prior = self.prior(context)
        
        predictions = []
        for _ in range(num_samples):
            # Sample different intents from the prior
            z = self.reparameterize(mu_prior, logvar_prior)
            y_pred = self.decoder(context, z, steps)
            predictions.append(y_pred)
            
        return torch.stack(predictions, dim=1) # [B, num_samples, k, 3]