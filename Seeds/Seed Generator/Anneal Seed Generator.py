# ======================
# ProbTrajectory5k Synthetic Trajectory Test (N,3 version)
# ======================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# -------------------------
# 1️⃣ Synthetic Trajectory Generator
# -------------------------
def generate_random_trajectory(num_points=300, motion_type='linear', noise=0.0):
    """
    Returns:
        t: torch tensor (num_points,)
        pos: torch tensor (num_points, 3)
    """
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

    pos = torch.stack([x, y, z], dim=1)  # (N,3) now
    if noise > 0.0:
        pos += torch.randn_like(pos) * noise
    return t, pos

# -------------------------
# 2️⃣ ProbTrajectory5k Model
# -------------------------
class ProbTrajectory5k(nn.Module):
    def __init__(self, state_dim=32, depth=200, layers=3, num_branches=16, embed_dim=8, time_feat_dim=8, hidden=32, epsilon=0.5):
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

        in_dim = embed_dim + time_feat_dim
        self.mlp1 = nn.Linear(in_dim, hidden)
        self.mlp2 = nn.Linear(hidden, 4 * state_dim)

        self.P_map = nn.Linear(3, state_dim)
        self.Theta_map = nn.Linear(3, state_dim)
        self.readout = nn.Linear(state_dim, 3)
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
        x = torch.cat([embed.unsqueeze(1).expand(-1, T, -1), t_feat.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        x = F.relu(self.mlp1(x))
        return self.mlp2(x)

    def forward(self, t, P_d, theta_d, update_rule="residual"):
        device = self.branch_embed.device
        T = t.shape[0]

        P_lat = self.P_map(P_d.to(device))
        Theta_lat = self.Theta_map(theta_d.to(device))
        t_feat = self._time_features(t.to(device))

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
        denom_mag2 = denom_real**2 + denom_imag**2 + (self.epsilon**2)
        num_real = P_lat.unsqueeze(0).unsqueeze(1) + V_real
        num_imag = V_imag

        out_real = (num_real * denom_real + num_imag * denom_imag) / denom_mag2
        out_imag = (num_imag * denom_real - num_real * denom_imag) / denom_mag2

        weighted_real = (alphas.view(B,1,1) * out_real).sum(dim=0)
        weighted_imag = (alphas.view(B,1,1) * out_imag).sum(dim=0)

        latent_real = weighted_real
        latent_imag = weighted_imag

        for layer in self.res_layers:
            if update_rule == "residual":
                latent_real = latent_real + layer(latent_real)
            else:
                latent_real = layer(latent_real)

        pos_super = self.readout(latent_real)
        uncertainty = latent_imag.abs().mean(dim=-1)
        return pos_super, uncertainty  # (N,3), (N,)

# -------------------------
# 3️⃣ Training on Synthetic Data
# -------------------------
def train_on_synthetic(model, trajectories, lr=1e-3, epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for traj_idx, (t, pos_target) in enumerate(trajectories):
        P_d = pos_target[0]  # first row
        theta_d = torch.zeros(3)
        last_print = time.time()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pos_pred, unc = model(t, P_d, theta_d)
            loss = loss_fn(pos_pred, pos_target)
            loss.backward()
            optimizer.step()
            if time.time() - last_print > 2:
                print(f"[Trajectory {traj_idx+1} | Epoch {epoch+1}] loss={loss.item():.6f} | mean_unc={unc.mean().item():.4f} | max_unc={unc.max().item():.4f}")
                last_print = time.time()

# -------------------------
# 4️⃣ Visualization
# -------------------------
def visualize_trajectory(pos_pred, uncertainty, pos_target=None, loss=None, title="Trajectory"):
    """
    pos_pred: (N,3)
    pos_target: (N,3)
    """
    pos = pos_pred.detach().cpu().numpy()
    unc = uncertainty.detach().cpu().numpy()
    mean_unc = unc.mean()
    max_unc = unc.max()
    threshold = mean_unc + unc.std()
    outlier_idx = unc > threshold

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111, projection='3d')

    # Predicted trajectory
    ax.plot(pos[:,0], pos[:,1], pos[:,2], color='blue', linewidth=3, label='Prediction')

    # Ground truth
    if pos_target is not None:
        gt = pos_target.detach().cpu().numpy()
        ax.plot(gt[:,0], gt[:,1], gt[:,2], color='black', linewidth=1, linestyle='--', label='True Path')

    # High uncertainty points
    ax.scatter(pos[outlier_idx,0], pos[outlier_idx,1], pos[outlier_idx,2], color='red', s=40, label='High Uncertainty')

    # Title + metrics
    metrics = f"Mean σ: {mean_unc:.4f} | Max σ: {max_unc:.4f}"
    if loss is not None:
        metrics = f"Loss: {loss:.4f} | " + metrics
    ax.set_title(f"{title}\n{metrics}")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------------
# 5️⃣ Evaluation / Forward Pass Helper
# -------------------------
def evaluate_model(model, t, pos_target):
    P_d = pos_target[0]
    theta_d = torch.zeros(3)
    with torch.no_grad():
        pos_pred, uncertainty = model(t, P_d, theta_d)
        loss = F.mse_loss(pos_pred, pos_target)
    return pos_pred, uncertainty, loss

# -------------------------
# 6️⃣ Main Execution
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(70)
    model = ProbTrajectory5k(state_dim=32, depth=200, layers=3)

    trajectories = []
    for _ in range(3):
        t, pos = generate_random_trajectory(num_points=300, motion_type='linear', noise=0.05)
        trajectories.append((t, pos))

    # Train
    train_on_synthetic(model, trajectories, lr=1e-3, epochs=2)

    # Evaluate and visualize first trajectory
    pos_pred, uncertainty, loss = evaluate_model(model, trajectories[0][0], trajectories[0][1])
    visualize_trajectory(pos_pred, uncertainty, pos_target=trajectories[0][1], loss=loss.item(), title="Synthetic Trajectory Example")