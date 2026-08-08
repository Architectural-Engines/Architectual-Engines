import torch
import time
import torch.nn as nn
import torch.nn.functional as F

from v1_diag import print_training_stats

#===Training on Synthetic Data===#
def train_on_synthetic(model, loss_fn, pos_target, trajectories, latent_uncertainty, branch_variance_ki, epochs = 300 , num_points = 20 , lr = 1e-2 , motion_type = "circular", noise_floor = None):

    optimizer = torch.optim.Adam(model.parameters() , lr = lr)

    t_traj = torch.linspace(0.0 , 150.0 , steps = num_points)

    trajectories = trajectories

    for traj_idx , (t_traj , pos_target) in enumerate(trajectories):

        last_print = time.time()

        for epoch in range(epochs):
            

            optimizer.zero_grad()

            motion_type = motion_type

            pos_pred, noise_floor, branch_variance_ki, latent_uncertainty, trajectories = model(t_traj , motion_type = motion_type , update_rule = None)

            loss = loss_fn(pos_pred , pos_target)

            loss.backward()

            print(pos_pred.shape)
            print(pos_target.shape)
            print(loss)

            optimizer.step()
            #print("after optimizer")

            if epoch % 20 == 0:
                print_training_stats(epoch, loss, branch_variance_ki, latent_uncertainty)

    with torch.no_grad():

        noise_floor = 0.0

        branch_variance_ki, latent_uncertainty, noise_floor, trajectories,  loss = model(pos_target , motion_type = motion_type, update_rule = None)

    return  branch_variance_ki, latent_uncertainty , noise_floor, trajectories , loss