import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from v1_data import generate_random_trajectory
from v1_config import AnnealConfig
from v1_model import ProbTrajectory5k
from v1_train import train_on_synthetic
from v1_diag import count_parameters, print_model_parameters
from v1_eval_viz import  visualize_trajectory


#===main execution===#
if __name__ == "__main__":
    
    config = AnnealConfig()
    print("MAIN-config loaded:succesfully")
    
    model = ProbTrajectory5k()
    print("MAIN-model loaded successfully")

    device = model.branch_embed.device

    loss_fn = nn.MSELoss()

    print ("Test Run Started")

    torch.manual_seed(50)

    #===trajectory prep===#
    trajectories = []
    #print("TRAJECTORY PREP")

    for _ in range(3):

        t_traj , pos = generate_random_trajectory(num_points=20 , motion_type='circular' , noise = 0.000)

        trajectories.append((t_traj , pos))

        progress = torch.tensor(0.5)
            
        progress = progress.unsqueeze(-1)
            
        assert progress.ndim in [1 , 2]

    #print("branch_embed:", model.branch_embed.shape)

    #print("expected mlp input:" , model.mlp1.in_features )
    
    branch_out = model._mlp_branch( model.branch_embed, t_traj)

    freqs = torch.arange(1 , model.time_feat_dim // 2 + 1 , device = progress.device)
            
    phase_angle = progress * freqs * 2 * torch.pi
    
    t_feat = torch.cat([torch.sin(phase_angle) , torch.cos(phase_angle)] , dim = -1)

    T = t_traj.shape[0]

    P_d = torch.zeros(T, 3, device = t_traj.device)

    pos_target = P_d

    progress = torch.tensor(0.5)
    
    progress = progress.unsqueeze(-1)
    
    assert progress.ndim in [1 , 2]

    #print("BRANCH OUT")
    branch_variance_ki = model.branch_variance.mean(dim = 1)
    #===branch state===#
    output = model( t_traj ,  motion_type = None , update_rule = None)   

    count_parameters(model)
    
    print_model_parameters(model, progress)

    #print("OUTPUT CREATED")
    #print("output shape:" , output[0].shape)
    #print("TEST RUN COMPLETE")

    #print(type(model)) 
    #print(model)
    #===training===#
    print("BEGIN TRAIN ON SYNTHETIC")

    pos_pred = model.pos_proj(model.latent_real_out)

    t_traj, pos  = generate_random_trajectory(num_points=20 , motion_type='circular')
    
    train_on_synthetic(model , loss_fn, pos_target, trajectories, branch_variance_ki, epochs = 300 , lr = 1e-2,  num_points = 20 ,  motion_type = None, noise_floor = None)
    print("BEGIN EVALUATION")
    
    print ("Run complete")

    #visualize_trajectory(model, loss_fn, pos_pred , pos_target , title = "Trajectory")