Anneal V1 Core Equation:

This README describes the skeletal core equation used in Anneal V1 to model latent trajectory propagation in a branch-wise, high-dimensional space.

Equation Overview:
The system computes a weighted sum of branch-wise latent trajectories:

**Where**:
  Sc(t) — resulting latent trajectory at time (t)
  alpha — softmax-weighted contribution of branch (i)
  Pd, Theta — base position and angle inputs
  Vi, complex(t), Theta i, complex(t) — branch-specific complex latent components
  Branch Complex Latent Components
  
Each branch encodes trajectory as a 3D complex vector:

**Where**:
  Xi(t), Yi(t), Zi(t) — real-space latent displacements
  j — imaginary unit
  Negative signs encode directional coupling between axes
  
**Intuition**:
  Each branch contributes a complex-valued trajectory, allowing divergence and superposition in latent space.
  Weighted summation with Alpha(i) ensures that each branch influences the final trajectory proportionally.
  Designed for high-dimensional exploration with minimal parameters, forming the basis for trajectory annealing before symbolic           layers or higher-level abstractions are applied.
    
**Notes**:
  Lightweight and modular for experimentation in synthetic or small-scale datasets.
  Enables global dispersal of trajectory information across branches.
  Serves as the pre-symbolic engine in Anneal V1.

  ***Version 0.01a***
  ***2/25/2026***
