import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=128, cond_dim=64, num_vars=5, seq_len=24, num_stations=20, w_c_ref=None, w_prior_spatial=None):
        super(Generator, self).__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.num_stations = num_stations
        
        # 4 Fully connected layers with LeakyReLU (slope 0.2)
        hidden_dim = 256
        self.fc_net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, num_vars * seq_len * num_stations),
            nn.LeakyReLU(0.2)
        )
        
        # Dual Attention Modules
        from .attention import ChannelAttention, SpatialAttention
        self.ca = ChannelAttention(num_channels=num_vars, w_c_ref=w_c_ref)
        self.sa = SpatialAttention(num_channels=num_vars, num_stations=num_stations, w_prior_spatial=w_prior_spatial)

    def forward(self, noise, condition):
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        x = self.fc_net(x)
        x = x.view(-1, self.num_vars, self.seq_len, self.num_stations)
        
        # Apply Dual Attention
        x, w_c = self.ca(x)
        x, sa_map = self.sa(x)
        
        # Deterministic physical bounding layer (Post-generation rules)
        # Variable order assumption: 0:Temp, 1:RH, 2:Wind, 3:Solar, 4:Pressure
        out = x.clone()
        # Non-negative constraint for Wind and Solar (ReLU)
        out[:, 2, :, :] = torch.relu(out[:, 2, :, :])
        out[:, 3, :, :] = torch.relu(out[:, 3, :, :])
        # Clipping [0, 100]% for Relative Humidity
        out[:, 1, :, :] = torch.clamp(out[:, 1, :, :], min=0.0, max=100.0)
        
        return out, w_c, sa_map

class Discriminator(nn.Module):
    def __init__(self, cond_dim=64, num_vars=5, seq_len=24, num_stations=20):
        super(Discriminator, self).__init__()
        input_dim = num_vars * seq_len * num_stations
        
        # 3 Fully connected layers
        # Note: Following WGAN-GP standard, the final output does NOT use Sigmoid
        # However, to align with original CTGAN logic before Wasserstein computation, we output raw logits
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1) # Raw score for Earth-Mover distance
        )
        
    def forward(self, x, condition):
        x_flat = x.view(x.size(0), -1)
        input_vec = torch.cat([x_flat, condition], dim=1)
        score = self.net(input_vec)
        return score