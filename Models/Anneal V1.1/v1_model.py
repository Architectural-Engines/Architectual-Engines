import torch
import math
import time
import torch.nn.functional as F
import torch.nn as nn

from v1_data import generate_random_trajectory
from v1_train import train_on_synthetic
from v1_eval_viz import  visualize_trajectory
from v1_config import AnnealConfig

print("IMPORTS SUCCESSFUL")

#===ProbTrajectory5k Model===#
class ProbTrajectory5k(nn.Module):

    def __init__(self, depth = 28 ,layers = 100 ,  epsilon = 0.27 , num_branches = 350 , hidden = 350 , *, motion_type = None):

        super().__init__()

        config = AnnealConfig()

        self.embed_dim = config.embed_dim

        self.in_dim = config.in_dim

        self.branch_embed_dim = config.branch_embed_dim

        self.state_dim = config.state_dim

        self.hidden_dim = config.hidden_dim

        self.time_feat_dim = config.time_feat_dim

        self.hidden = hidden

        self.num_branches = num_branches

        self.layers = layers

        self.depth = depth

        self.epsilon = epsilon

        self.branch_embed = nn.Parameter(torch.randn(num_branches , self.embed_dim) * 0.1)

        device = self.branch_embed.device

        self.register_buffer("freqs" , torch.exp(torch.linspace(math.log(1.0) , math.log(10.0) , self.time_feat_dim//2)))

        #+++(REVISIT unsure if necessary at the moment, might be optional per usecase)+++#

        self.time_proj = nn.Linear(config.time_feat_dim, config.time_feat_dim, bias=False)

        self.pos_proj = nn.Linear(config.state_dim, config.hidden_dim, bias= False)

        in_dim = self.embed_dim + self.time_feat_dim

        self.mlp1 = nn.Linear(self.in_dim, self.hidden_dim)

        self.mlp2 = nn.Linear(self.hidden_dim , 4 * self.state_dim)

        #===anchor===#
        theta_d = torch.zeros(3)

        Theta_map = nn.Linear(3 , self.state_dim)

        Theta_lat = Theta_map(theta_d.to(device))

        self.delta_readout = nn.Linear(self.state_dim , 3)

        self.branch_logits = nn.Parameter(torch.zeros(self.num_branches))

        self.res_layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(self.state_dim) , nn.Linear (self.state_dim , self.state_dim) , nn.ReLU() , nn.Linear(self.state_dim, self.state_dim)
            ) for _ in range(layers)
        ])

        #print("branch_embed:" , self.branch_embed.shape)

    #print("MODEL CLASS LOADED")

    def _mlp_branch(self , embed , t_traj):

        if isinstance(embed, list):

            embed = torch.as_tensor(embed , dtype = torch.float32 , device = self.mlp1.weight.device)

        if not torch.is_tensor(embed):

            raise TypeError(f"embed must be a tensor, got {type(embed)}")
        
        if embed.shape[0] != self.num_branches:

            #print("[ANNEAL TRACE] embed failure detected")
            #print("device:", embed.device)
            #print("dtype:", embed.dtype)
            #print("shape", embed.shape)

            raise ValueError(f"Expected{self.num_branches} branches, got {embed.shape[0]}")
        
        #debug prints keep#
        #print("\n[TRACE t_traj ENTRY]")

        B, E = embed.shape
        T = t_traj.shape[0]
        #if t_feat.shape[-1] != 16:

            #print("t_feat mismatch detected")
            #print("shape:" , t_feat.shape)

            #raise ValueError("invalid t_feat dimensionality")

        x = torch.cat([embed.unsqueeze(1).expand(-1 , T , -1)] , dim=-1)

        #print("branch_embed:", self.branch_embed.shape)
        #print("embed:", embed.shape)
        #print("t_traj:", t_traj.shape)
        #print("x before MLP:", x.shape)
        
        #print(self.mlp1)
        #print("in_features:", self.mlp1.in_features)
        #print("out_features:", self.mlp1.out_features)
        #print("bias shape:", self.mlp1.bias.shape)
        
        assert x.shape[-1] == self.mlp1.in_features , f"MLP mismatch: got {x.shape[-1]} , expected {self.mlp1.in_features}"

        #debug prints keep#
        #print("embed" , embed.shape)
        #print("x:" , x.shape)
        #print("mlp1 expected in:" , self.mlp1.in_features)

        x = F.relu(self.mlp1(x))

        assert x.shape[-1] == self.mlp1.in_features , f"MLP mismatch: got {x.shape[-1]} , expected {self.mlp1.in_features}"

        return self.mlp2(x)

    #===forward pass===#
    def forward(self , t_traj , motion_type = None , update_rule = None):

        #print ("forward start")

        #coupling phase generation:
        #progress is a local trajectory coordinate derived from P_d reset events
        # representaion of evolution across trajectory steps, not experiment/ training progress
        #ownership - forward: defines models internal state space coupling, represented through sinusoidal projection.
        device = self.branch_embed.device

        P_map = nn.Linear(3 , self.state_dim)

        T = t_traj.shape[0]

        P_d = torch.zeros(T, 3, device = t_traj.device)

        P_lat = P_map(P_d.to(device))

        #print ("[TRACE] compute t_feat id:" , id(t_feat))
        #print("t_feat shape" , t_feat.shape)
        #print(type(T), T)
        #print(type(t_traj), t_traj)

        #===branch MLP output===#
        branch_out = self._mlp_branch(self.branch_embed , t_traj)

        B = self.num_branches

        sd = self.state_dim

        #branch collapse VERY important
        branch_out = branch_out.view(B , T , 4 , sd)

        self.V_real = branch_out[: , : , 0 , :]

        self.V_imag = branch_out[: , : , 1 , :]

        self.Theta_real = branch_out[: , : , 2 , :]

        self.Theta_imag = branch_out[: , : , 3 , :]

        self.alphas = F.softmax(self.branch_logits , dim=0)

        self.denom_real = self.Theta_real

        self.denom_imag = self.Theta_imag

        self.eps_effective = self.epsilon

        self.noise_floor = 0.0

        #===complex branch aggregation===#
        denom_mag2 = self.denom_real**2 + self.denom_imag**2 + self.eps_effective**2

        self.num_real = P_lat.unsqueeze(0).unsqueeze(1) + self.V_real

        self.num_imag = self.V_imag

        self.latent_real_in = self.num_real

        self.latent_imag_in = self.num_imag

        self.interaction_real = (self.num_real * self.denom_real + self.num_imag * self.denom_imag) / denom_mag2

        self.interaction_imag = (self.num_imag * self.denom_real - self.num_real * self.denom_imag) / denom_mag2

        alphas_exp = self.alphas.view(B , 1 , 1).expand(-1 , T , sd)

        self.latent_real_out = (alphas_exp * self.interaction_real).sum(dim=0)

        latent_imag_out = (alphas_exp * self.interaction_imag).sum(dim=0)

        pos_pred = self.pos_proj(self.latent_real_out)


        #Debug prints keep seperate from Diagnostic prints
        #print("alphas:", self.alphas.shape)
        #print("interaction real", self.interaction_real.shape)
        #print("alphas exp", alphas_exp.shape)
        #print("latent real out", self.latent_real_out.shape)
        #print("pos pred", pos_pred.shape)

        #===compute uncertainty===#
        latent_uncertainty = latent_imag_out.abs().mean(dim=1)

        unc = latent_uncertainty.detach().cpu().numpy()

        mean_unc = unc.mean()

        max_unc = unc.max()

        threshold = mean_unc + unc.std()

        outlier_idx = unc > threshold

        #===compute per-step branch variance===#
        self.branch_variance = ((self.interaction_real - self.latent_real_in.unsqueeze(0))**2 * self.alphas.view(B , 1 , 1)).sum(dim=0)

        #~~~optional: scale down noise contribution~~~#

        self.branch_variance = self.branch_variance * self.noise_floor

        noise_floor = self.noise_floor 

        #===aggregate to uncertainty===#
        #mean across state dim#
        branch_variance_ki = self.branch_variance.mean(dim=1)

        #===branch pruning calculation===#
        threshold = 0.0

        self.alphas = F.softmax(self.branch_logits , dim=0)

        self.alphas[self.alphas < threshold] = 0

        #renormalize
        self.alphas /= self.alphas.sum()

        #===anchor first latent dimension to P_d===#
        #is this a divergence anchor and is it necessary?
        #LEGACY: injects P_d anchor into latent_real_out channel 0
        #VERIFY if this is required after the current state representation clean up?
        #latent_real_out[: , 0] = P_d[0]

        #===compute delta and cumulative sum===#
        self.delta = self.delta_readout(self.latent_real_out)

        self.delta[0] = torch.zeros_like(self.delta[0])

        #debug prints keep#
        #print("denom_real:" , self.denom_real.shape)

        #legacy integrated trajectory#
        #previously used by the residual update experiment#

        #pos_super = torch.cumsum(delta , dim=0) + P_d.to(device)

        #~~~Optional residual layers~~~#
        #if update_rule == "residual":

            #for layer in self.res_layers:

                #latent_real_out = latent_real_out + layer(latent_real_out)

                #pos_super = latent_real_out

        assert self.branch_embed.shape[-1] == self.branch_embed_dim

        return pos_pred , branch_variance_ki , latent_uncertainty , branch_out, noise_floor