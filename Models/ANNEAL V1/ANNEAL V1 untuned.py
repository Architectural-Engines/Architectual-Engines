# ===============================================#
# ==Anneal (N,3 version) w/ Random Walk Scaling==#
#================================================#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#=======universal smoothing function=====#
def smooth_tensor(x, kernel_size = 3):
    #x: Tensor of shape (T,) or (T, C)
    if x.ndim == 1:
        x= x.view(1, 1, -1)  # (batch=1, channel=1, length=T)
    elif x.ndim == 2:
        x= x.transpose (0, 1).unsqueeze(0)  # (1, C, T)
        kernel = torch.ones (1, 1, kernel_size, device= x.device) / kernel_size
        x_smooth = F.conv1d(x, kernel, padding = kernel_size//2)
    if x.ndim == 1:
        return x_smooth.view(-1)
    else:
        return x_smooth.squeeze(0).transpose(0,1)

#=====================================#
#==1️⃣ Synthetic Trajectory Generator=#
#=====================================#
def generate_random_trajectory(num_points=20, motion_type='circular', noise=0.000):
    t = torch.linspace(0.0, 150.0, steps=num_points)

    if motion_type == 'linear':
        x = t * 0.05
        y = t * 0.02
        z = t * 0.01
    elif motion_type == 'circular':
        x = torch.sin(t * 0.05) * 5
        y = torch.cos(t * 0.05) * 5
        z = t * 0.01
    elif motion_type == 'oscillatory':
        x = torch.sin(t * 0.1) * 3
        y = torch.sin(t * 0.2) * 2
        z = torch.cos(t * 0.15) * 2
    elif motion_type == 'random_walk':
        x = torch.cumsum(torch.randn(num_points) * 0.1, dim=0)
        y = torch.cumsum(torch.randn(num_points) * 0.1, dim=0)
        z = torch.cumsum(torch.randn(num_points) * 0.1, dim=0)
    else:
        raise ValueError("Unknown motion_type")

    pos = torch.stack([x, y, z], dim=1)
    if noise > 0.0:
        pos += torch.randn_like(pos) * noise
    return t, pos

#==============================#
#==2️⃣ ProbTrajectory5k Model==#
#==============================#
class ProbTrajectory5k(nn.Module):
    def __init__(self, state_dim=0, depth=0, layers=0, num_branches=0,
                 embed_dim=0, time_feat_dim=0, hidden=0, epsilon=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.depth = depth
        self.layers = layers
        self.epsilon = epsilon
        self.num_branches = num_branches
        self.branch_embed = nn.Parameter(torch.randn(num_branches, embed_dim) * 0.1)
        self.time_feat_dim = time_feat_dim
        self.register_buffer("freqs", torch.exp(torch.linspace(math.log(1.0), math.log(10.0), time_feat_dim//2)))
        self.time_proj = nn.Linear(time_feat_dim, time_feat_dim, bias=False)
        self.noise = .01 # internal noise floor for random walk

        in_dim = embed_dim + time_feat_dim
        self.mlp1 = nn.Linear(in_dim, hidden)
        self.mlp2 = nn.Linear(hidden, 4 * state_dim)

        self.P_map = nn.Linear(3, state_dim)
        self.Theta_map = nn.Linear(3, state_dim)
        self.delta_readout = nn.Linear(state_dim, 3)
        self.branch_logits = nn.Parameter(torch.zeros(num_branches))

        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, state_dim),
                nn.ReLU(),
                nn.Linear(state_dim, state_dim)
            ) for _ in range(layers)
        ])

    def _time_features(self, t):
        t = t.unsqueeze(-1) if t.ndim == 1 else t
        freqs = self.freqs.unsqueeze(0).to(t.device)
        arg = t * freqs
        feats = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.time_proj(feats)

    def _mlp_branch(self, embed, t_feat):
        B, E = embed.shape
        T = t_feat.shape[0]
        x = torch.cat([embed.unsqueeze(1).expand(-1, T, -1),
                       t_feat.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        x = F.relu(self.mlp1(x))
        return self.mlp2(x)

    #==============================#
    #== Forward Pass (Restored) ==#
    #==============================#
    def forward(self, t, P_d, theta_d, motion_type="circular", update_rule= None):
        device = self.branch_embed.device
        T = t.shape[0]

        # Encode initial states
        P_lat = self.P_map(P_d.to(device))
        Theta_lat = self.Theta_map(theta_d.to(device))
        t_feat = self._time_features(t.to(device))

        # Branch MLP output
        branch_out = self._mlp_branch(self.branch_embed, t_feat)
        B = self.num_branches
        sd = self.state_dim
        branch_out = branch_out.view(B, T, 4, sd)

        V_real = branch_out[:, :, 0, :]
        V_imag = branch_out[:, :, 1, :]
        Theta_real = branch_out[:, :, 2, :]
        Theta_imag = branch_out[:, :, 3, :]

        alphas = F.softmax(self.branch_logits, dim=0)

        denom_real = Theta_lat.unsqueeze(0).unsqueeze(1) + Theta_real
        denom_imag = Theta_imag

        # Epsilon scaling & noise floor
        if motion_type == "random_walk":
            progress = torch.linspace(0, 1, T, device=t.device)
            eps_effective = (self.epsilon * (1.0 - 0.7 * progress)).view(1, T, 1)
            noise_floor = 0.0 + 0.05 * progress
        else:
            eps_effective = self.epsilon
            noise_floor = 0.0

        # ----------------- STEP 1: Initial micro-step ----------------- #
        model_state = P_lat.clone()
        step1_dampener = 0.0
        raw_delta_step1 = model_state  # placeholder for actual delta calculation
        delta_step1 = raw_delta_step1 * step1_dampener

        # Apply epsilon ceiling
        epsilon_ceiling = 0.0
        delta_step1 = torch.clamp(delta_step1, -epsilon_ceiling, epsilon_ceiling)

        # Optional epsilon boost
        step1_multiplier = 0.0
        epsilon_boost = raw_delta_step1 * self.epsilon * step1_multiplier
        delta_step1 += epsilon_boost

        # ----------------- STEP 2: Residual dampener ----------------- #
        raw_delta_step2 = torch.zeros_like(model_state)  # placeholder
        step2_dampener = 0.00
        delta_step2 = raw_delta_step2 * step2_dampener

        # Apply step1 + step2 updates
        model_state = model_state + delta_step1 + delta_step2

        # ----------------- Complex branch aggregation ----------------- #
        denom_mag2 = denom_real**2 + denom_imag**2 + eps_effective**2
        num_real = P_lat.unsqueeze(0).unsqueeze(1) + V_real
        num_imag = V_imag

        out_real = (num_real * denom_real + num_imag * denom_imag) / denom_mag2
        out_imag = (num_imag * denom_real - num_real * denom_imag) / denom_mag2

        alphas_exp = alphas.view(B, 1, 1).expand(-1, T, sd)
        latent_real = (alphas_exp * out_real).sum(dim=0)

        #compute uncertainty
        latent_imag = (alphas_exp * out_imag).sum(dim=0)
        uncertainty = latent_imag.abs().mean(dim=1)
        unc = uncertainty.detach().cpu().numpy()

        
        mean_unc = unc.mean()
        max_unc = unc.max()
        threshold = mean_unc + unc.std()
        outlier_idx = unc > threshold

        #compute per-step branch variance
        branch_variance = ((out_real - latent_real.unsqueeze(0))**2 * alphas.view(B, 1, 1)).sum(dim=0)
        #optional: scale down noise contribution
        noise_scale = 0.000 # 0 ignore noise, 1= full noise
        branch_variance = branch_variance * noise_scale
        #aggregate to uncertainty
        uncertainty = branch_variance.mean(dim=1) #mean across state dim

        #branch pruning calculation
        threshold = 0.0
        alphas = F.softmax(self.branch_logits, dim=0)
        alphas[alphas < threshold] = 0
        alphas /= alphas.sum() #renormalize

        # Anchor first latent dimension to P_d
        latent_real[:, 0] = P_d[0]

        # Compute delta and cumulative sum
        delta = self.delta_readout(latent_real)
        delta[0] = torch.zeros_like(delta[0])
        pos_super = torch.cumsum(delta, dim=0) + P_d.to(device)

        # Random walk noise addition
        if motion_type == "random_walk" and isinstance(noise_floor, torch.Tensor):
            delta += torch.randn_like(delta) * noise_floor.view(-1, 1)
            pos_super = torch.cumsum(delta, dim=0) + P_d.to(device)

        # Optional residual layers
        if update_rule == "residual":
            for layer in self.res_layers:
                latent_real = latent_real + layer(latent_real)
                pos_super = latent_real

        # Uncertainty
        uncertainty = latent_imag.abs().mean(dim=-1)

        return pos_super, uncertainty


#===================================#
#==3️⃣ Training on Synthetic Data===#
#===================================#
def train_on_synthetic(model, trajectories, lr= lr, epochs=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for traj_idx, (t, pos_target) in enumerate(trajectories):
        P_d = pos_target[0]
        theta_d = torch.zeros(3)
        last_print = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pos_pred, unc = model(t, P_d, theta_d, motion_type="circular")
            loss = loss_fn(pos_pred, pos_target)
            loss.backward()
            optimizer.step()
            if time.time() - last_print > 2:
                print(f"[Trajectory {traj_idx+1} | Epoch {epoch+1}] loss={loss.item():.6f} | mean_unc={unc.mean().item():.4f} | max_unc={unc.max().item():.4f}")
                last_print = time.time()


#=====================#
#==4️⃣ Visualization==#
#=====================#
def visualize_trajectory(pos_pred, uncertainty, pos_target=None, loss=None, title="Trajectory"):
    pos = pos_pred.detach().cpu().numpy()
    unc = uncertainty.detach().cpu().numpy()
    mean_unc = unc.mean()
    max_unc = unc.max()
    threshold = mean_unc + unc.std()
    outlier_idx = unc > threshold

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111, projection='3d')

    norm_unc = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)
    colors = plt.cm.viridis(norm_unc)
    for i in range(len(pos)-1):
        ax.plot(pos[i:i+2,0], pos[i:i+2,1], pos[i:i+2,2], color=colors[i], linewidth=3)

    if pos_target is not None:
        gt = pos_target.detach().cpu().numpy()
        ax.plot(gt[:,0], gt[:,1], gt[:,2], color='black', linewidth=1, linestyle='--', label='True Path')

    ax.scatter(pos[outlier_idx,0], pos[outlier_idx,1], pos[outlier_idx,2], color='red', s=40, label='High Uncertainty')

    metrics = f"Mean σ: {mean_unc:.4f} | Max σ: {max_unc:.4f}"
    if loss is not None:
        metrics = f"Loss: {loss:.4f} | " + metrics
    ax.set_title(f"{title}\n{metrics}")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


#=========================#
#==5️⃣ Evaluation Helper==#
#=========================#
def evaluate_model(model, t, pos_target):
    P_d = pos_target[0]
    theta_d = torch.zeros(3)
    with torch.no_grad():
        pos_pred, uncertainty = model(t, P_d, theta_d, motion_type="circular")
        loss = F.mse_loss(pos_pred, pos_target)
    return pos_pred, uncertainty, loss


#======================#
#==6️⃣ Main Execution==#
#======================#
if __name__ == "__main__":
    torch.manual_seed(50)
    model = ProbTrajectory5k(state_dim=0, depth=0, layers=0)

    trajectories = []
    for _ in range(3):
        t, pos = generate_random_trajectory(num_points=20, motion_type='circular', noise=0.001)
        trajectories.append((t, pos))

    train_on_synthetic(model, trajectories, lr= lr , epochs=0)

    pos_pred, uncertainty, loss = evaluate_model(model, trajectories[0][0], trajectories[0][1])
    visualize_trajectory(pos_pred, uncertainty, pos_target=trajectories[0][1], loss=loss.item(), title="Synthetic Trajectory Example")