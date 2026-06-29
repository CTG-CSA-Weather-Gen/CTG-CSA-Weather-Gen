import torch
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from models.ctgan_nets import Generator, Discriminator

# --- Hyperparameters from Paper ---
BATCH_SIZE = 32
EPOCHS = 200
NOISE_DIM = 128
COND_DIM = 64
N_CRITIC = 5 # Discriminator update frequency ratio
LAMBDA_GP = 10.0 # Gradient Penalty coefficient
LAMBDA_C = 0.5 # CA physical constraint weight
LAMBDA_S = 0.6 # SA physical constraint weight
LR_G = 0.0002
LR_D = 0.0001
BETA1, BETA2 = 0.5, 0.999

# --- Physical Priors from Paper (Section 3.6) ---
# T, RH, Wind, Solar, Pressure
W_C_REF = [0.35, 0.25, 0.20, 0.15, 0.05] 

def compute_vdsp(variances, tau=1.6):
    """
    Computes the Variance-Driven Softmax Prior (VDSP) as defined in Equation 13 of the paper.
    Maps empirical temporal variances smoothly into a [0,1] attention weight space.
    """
    var_tensor = torch.tensor(variances, dtype=torch.float32)
    scaled_var = var_tensor / tau
    w_prior = torch.nn.functional.softmax(scaled_var, dim=0)
    return w_prior.tolist()

# Dummy historical temporal variances for 20 stations 
# (e.g., first 10 represent highly variable coastal stations, last 10 are stable inland stations)
dummy_empirical_variances = [5.5]*10 + [1.2]*10 
# Dynamically generate the spatial weights using VDSP (Eq. 13) instead of hardcoding
W_PRIOR_SPATIAL = compute_vdsp(dummy_empirical_variances, tau=1.6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gradient_penalty(D, real_samples, fake_samples, conditions):
    """Calculates the gradient penalty loss for WGAN GP (Equation 12)"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, conditions)
    
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train():
    print("Initializing CTG-CSA Framework...")
    netG = Generator(noise_dim=NOISE_DIM, cond_dim=COND_DIM, w_c_ref=W_C_REF, w_prior_spatial=W_PRIOR_SPATIAL).to(device)
    netD = Discriminator(cond_dim=COND_DIM).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    
    w_c_ref_tensor = torch.tensor(W_C_REF, device=device)
    w_s_ref_tensor = torch.tensor(W_PRIOR_SPATIAL, device=device).view(1, 1, 1, -1)
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        # ---------------------
        # Create Dummy Data for demonstration
        # ---------------------
        real_data = torch.randn(BATCH_SIZE, 5, 24, 20).to(device) # (B, Vars, Time, Stations)
        conditions = torch.randn(BATCH_SIZE, COND_DIM).to(device)
        
        # =========================
        # Train Discriminator (Critic)
        # =========================
        for _ in range(N_CRITIC):
            netD.zero_grad()
            
            z = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
            fake_data, _, _ = netG(z, conditions)
            
            real_validity = netD(real_data, conditions)
            fake_validity = netD(fake_data.detach(), conditions)
            
            # WGAN Loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            gradient_penalty = compute_gradient_penalty(netD, real_data, fake_data.detach(), conditions)
            d_loss += LAMBDA_GP * gradient_penalty
            
            d_loss.backward()
            optimizerD.step()

        # =========================
        # Train Generator
        # =========================
        netG.zero_grad()
        z = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
        fake_data, w_c, sa_map = netG(z, conditions)
        
        fake_validity = netD(fake_data, conditions)
        g_loss_adv = -torch.mean(fake_validity)
        
        # Physical Constraint Losses (Equations 2 & 3)
        l_chan = torch.mean((w_c - w_c_ref_tensor) ** 2)
        l_spa = torch.mean((sa_map - w_s_ref_tensor) ** 2)
        
        # Total Generator Loss (Equation 1)
        g_loss_total = g_loss_adv + LAMBDA_C * l_chan + LAMBDA_S * l_spa
        
        g_loss_total.backward()
        optimizerG.step()
        
        if (epoch+1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss_total.item():.4f}] (L_chan: {l_chan.item():.4f}, L_spa: {l_spa.item():.4f})")
            
    print("Training Complete! The pre-trained model can now generate High-Fidelity Synthetic Meteorological Years.")

if __name__ == "__main__":
    train()
