import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from v1_train import train_on_synthetic


#===Visualization===#
def visualize_trajectory(model, loss, pos_pred , pos_target ,  title = "Trajectory"):
    
    pos_pred_viz = pos_pred.detach().cpu().numpy()

    pos_pred_viz = pos_pred_viz[0]

    #print("VIZ POS:", pos_pred_viz.shape)
    latent_uncertainty = model.latent_uncertainty

    viz_unc = model.latent_uncertainty.detach().cpu().numpy()

    mean_unc = viz_unc.mean()

    max_unc = viz_unc.max()

    threshold = mean_unc + viz_unc.std()

    outlier_idx = viz_unc > threshold

    fig = plt.figure(figsize=(11,8))

    ax = fig.add_subplot(111, projection='3d')

    norm_unc = (viz_unc - viz_unc.min()) / (viz_unc.max() - viz_unc.min() + 1e-8)

    colors = plt.cm.viridis(norm_unc)

    for branch in range(pos_pred_viz.shape[0]):
                        trajectory = pos_pred_viz[branch]

                        ax.plot(trajectory[: , 0], trajectory[: , 1], trajectory[: , 2], color = colors[branch], linewidth = 3)
                        

    if pos_target is not None:

        gt = pos_target.detach().cpu().numpy()

        ax.plot(gt[: , 0] , gt[: , 1] , gt[:  2] , color='black' , linewidth=1 , linestyle='--' , label='True Path')

    ax.scatter(pos_pred_viz[outlier_idx , 0] , pos_pred_viz[outlier_idx , 1] , pos_pred_viz[outlier_idx , 2] , color='red' , s=40 , label='High Uncertainty')

    metrics = f"Mean σ: {mean_unc:.4f} | Max σ: {max_unc:.4f}"

    if loss is not None:

        metrics = f"Loss: {loss:.4f} | " + metrics

    ax.set_title(f"{title}\n{metrics}")

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    ax.legend()

    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()