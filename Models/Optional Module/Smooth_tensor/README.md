Module: Smooth tensor

1. What it is:

-A small utility module that smooths input tensors, reducing noise or jagged trajectories

-Can be applied to any tensor representing trajectories, probabilities, or latent states.

2. How it applies:

-Use after generating trajectories to stabilize results before feeding them into the next stage

-Works in conjuctions with ensure_traj to guarantee valid inputs

-Can be dropped in as a preprocessing step for post-processing

3. Common issues:

-Tensor dimensionality mismatch (input much match expected shape)

-Over-smoothing can remove fine-grained details in trajectories

-Requires careful ordering if combined with symbolic layers

