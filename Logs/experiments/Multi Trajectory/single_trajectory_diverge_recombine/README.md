**Trajectory Test**: Single Diverge & Recombine

**Objective**:

-Test Anneal V1’s ability to diverge from a single trajectory point and successfully recombine into the intended path.

**Setup**:

-Initial P_d through step 3 will be a single trajectory.

-The model will split into two trajectories.

-Observe whether the model can converge successfully back into a single trajectory or identify how it fails.

-If Anneal V1 succeeds with both divergence and convergence in the initial step, subsequent runs will include data injection after the split to further stress-test trajectory handling.

**Research Questions**:

-How does Anneal V1’s state space respond to branching and recombination?

-How do perturbations propagate through multiple trajectories?

-What are the limits of convergence after divergence?

**Observations / Notes**:

-Will be documented and released after testing.

**Next Steps**:

-TBD after analysis of the initial runs.
