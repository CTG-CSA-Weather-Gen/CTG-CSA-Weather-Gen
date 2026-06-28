import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) module guided by physical priors (Equation 4-9).
    """
    def __init__(self, num_channels, w_c_ref):
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        # Physical prior baseline (e.g., Temp-Hum coupling 0.35, Wind 0.20, etc.)
        self.register_buffer('w_c_ref', torch.tensor(w_c_ref, dtype=torch.float32))
        
        # Dynamic deviation learning
        self.delta_w_dyn = nn.Parameter(torch.zeros(num_channels))
        
        # Asymmetric convolution 3x1 for temporal correlation avoiding spatial cross-talk
        self.asym_conv = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 1), padding=(1, 0))
        
    def forward(self, x):
        # x shape: (B, C, H, W) -> Batch, Channels(Variables), Time, Stations
        # 1. Attention weights calculation (Equation 5)
        # Using Softmax along the channel dimension based on physical prior + dynamic deviation
        w_att = F.softmax(self.w_c_ref + self.delta_w_dyn, dim=0).view(1, -1, 1, 1)
        
        # 2. Feature weighting (Equation 6)
        x_weighted = x * w_att
        
        # 3. Asymmetric Convolution (Equation 8)
        x_conv = self.asym_conv(x_weighted)
        
        # 4. Residual Fusion (Equation 9)
        # Assuming W_base is identity
        x_fusion = x_conv + x
        
        # Calculate L_chan constraint (Equation 2)
        # E[||W_c - W_c_ref||^2_2] -> Handled in the training loop
        w_c = self.w_c_ref + self.delta_w_dyn
        return x_fusion, w_c

class SpatialAttention(nn.Module):
    """
    Spatial Attention (SA) module with Geographical Priors (Equation 10-11).
    """
    def __init__(self, num_channels, num_stations, w_prior_spatial):
        super(SpatialAttention, self).__init__()
        # Shared MLP with 5-2-5 structure as described in the paper
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, 2),
            nn.ReLU(),
            nn.Linear(2, num_channels)
        )
        # Geographical prior weight matrix (e.g., Coastal 0.7, Inland 0.3)
        self.register_buffer('w_prior', torch.tensor(w_prior_spatial, dtype=torch.float32).view(1, 1, 1, -1))
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        # Global Avg Pool and Max Pool along temporal dimension H
        avg_pool = torch.mean(x, dim=2, keepdim=True) # (B, C, 1, W)
        max_pool, _ = torch.max(x, dim=2, keepdim=True) # (B, C, 1, W)
        
        # Sum and permute for MLP (B, W, 1, C)
        pool_sum = (avg_pool + max_pool).permute(0, 3, 2, 1)
        
        # Data-driven SA map (Equation 10)
        sa_map = torch.sigmoid(self.mlp(pool_sum)).permute(0, 3, 2, 1) # (B, C, 1, W)
        
        # Physical prior correction (Equation 11)
        x_refined = sa_map * self.w_prior
        
        # Apply spatial attention to features
        out = x * x_refined
        
        return out, sa_map