Anneal V1 – Move to 3D Phase / State Space:

This section documents why Anneal V1 is being extended from 2D to 3D phase or state space, and the implications of underrepresenting the Z-axis in trajectory modeling.
Motivation:
Anneal V1’s original 2D representation highlighted several limitations that motivated a shift to full 3D:

Z-axis approximation / underrepresentation:
-The model teeters above and below the intended trajectory, rather than moving smoothly along it.
-Interferes with natural branching and relaxation behaviors in the latent space.

Uncertainty compression:
-In 2D, uncertainty is compressed from 20+ dimensions into a single scalar, which is misleading.
-Even if loss is low (e.g., <0.0013) and trajectories appear accurate, uncertainty measurements may not reflect the true behavior.

State space fidelity:
-Anneal V1 requires full 3D phase space (or state space, as appropriate) for stable, interpretable outputs.
-2D tuning is possible, but interactions become more volatile and prone to bifurcations.

Implications:
-Moving to 3D allows the model to represent trajectories and branches faithfully.
-Uncertainty, branching, and trajectory adherence become more interpretable and actionable.
-Future tuning and high-dimensional exploration rely on a fully represented 3D space for accurate analysis.
