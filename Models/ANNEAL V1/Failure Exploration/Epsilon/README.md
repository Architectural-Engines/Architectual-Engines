
Anneal V1 – Epsilon Failure Exploration

This folder explores the effects of epsilon manipulation on the Anneal V1 model. All experiments focus on how the model behaves when epsilon is varied, particularly in relation to time_feat_dim, time_steps, and internal noise.

Overview

Epsilon is a core hyperparameter controlling internal smoothing and stability in Anneal V1. By adjusting epsilon, the model exhibits notable failure behaviors, which can be used to study:
-Trajectory adherence
-Bifurcations and divergence
-Branch resolution and anchoring

Key Observations:
Dependency:
Epsilon is tied directly to:
-time_feat_dim – dimensionality of temporal features
-time_steps – number of steps in the trajectory
-noise – internal model noise floor

Smoothing vs. Stability:
-Increasing epsilon can smooth trajectories.
-Beyond a certain point, higher epsilon yields diminishing returns.
-Excessive epsilon results in loss of stability, where trajectories no longer accurately follow the target path.

Bifurcation / Misalignment:
-High epsilon can produce extremely low loss values, yet the trajectory deviates from its intended path.
-The model "believes" it satisfies constraints even when the trajectory is effectively ignored.
-This demonstrates a bifurcation point, where predicted outputs diverge while certainty remains high.

Dampened Epsilon:
-Reducing epsilon too aggressively can disrupt trajectory continuity between points 2 and 18.
-P_d anchoring can fail, resulting in misaligned or unstable branch predictions.

Epsilon Tied to Parameters:
-Epsilon tied to hidden, num branches can change the tilt, or directional bias of the model.
-This is represented by slight offsets of the trajectory, where the model will bias towards X or Y. 
-This creates a perfect representation of the trajectory just off in phase space slighty 5-10 degrees



Purpose of This Exploration:
This collection of experiments is designed to:
-Map failure points caused by epsilon extremes
-Document trajectory deviation patterns
-Provide a reference for future tuning and high-dimensional exploration


Recommended Structure:
Each experiment should include:
-Input trajectory [T,3]
-Epsilon setting
-Observed deviation or failure behavior
-Optional visualizations (trajectories overlaid)
