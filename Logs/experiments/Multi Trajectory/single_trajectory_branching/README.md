**Trajectory Test: Single vs Multiple Trajectories**:

**Objective**:

-Evaluate how the model handles single points branching into multiple trajectories.

-Determine how injected data in one branch affects other trajectories.

**Setup**:

-Start with a single point in the model space.

-Introduce multiple trajectory paths.

-Inject controlled data into a single branch at step 2. **(after intial testing without synthetic data injection)**

**Goals / Questions**:

-Can the model branch naturally from a single point?

-How do secondary trajectories respond to changes in one branch?

-Does Anneal need to apply the core equation multiple times for P_d to anchor to zero?

**Observations / Notes**:

-Document patterns of divergence, convergence, or instability.

-Include screenshots, CSV snapshots, or example outputs.

-Track anomalies for follow-up experiments.

**Next Steps**:

-Compare single vs multiple trajectory outcomes.

-Adjust parameters and note effect on branching fidelity.Trajectory Test: Single vs Multiple Trajectories

**Objective**:

Evaluate how the model handles single points branching into multiple trajectories.

Determine how injected data in one branch affects other trajectories.

