Module: trajectory_gen_V1

1.What it is

trajectory_gen_V1 is a module that generates trajectories from intial states. Defining how paths evolve over time,
handles branching possiblies, and outputs data suitible for downstream processing or visualization.
Each generator version reflects incremental improvements or experimental tweaks.

2. How it applies
  -Use as the primary source of trajectories for the model.

  -Works in combination with ensure_traj to validate tensor shapes before further operations.

  -Can be swapped with later versions (V1a, V2, ect.) to compare trajectory behaivor, branching, or convergence properties.

3. Common Issues
  -Shape mismatch
      Possible Causes: ensure outputs are compatible with 3D modules
  -Excessive branching:
      Possible outcome: may increase compute and time required significantly,
  -Scaling/Normalization



