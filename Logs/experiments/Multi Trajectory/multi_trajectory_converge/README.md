**Trajectory Test**:

**Multiple Trajectories Converging**

**Objective**:

-Evaluate how multiple trajectories converge to a single point.

-Test the model’s response when data is injected into one trajectory.

-Observe behavior at boundary/stopping conditions.

**Setup**:

-Initialize multiple trajectories that converge to a predefined endpoint.

-Inject data into one trajectory at step 2.

-(Optional) Move the meeting point one step before the trajectory ends.

**Questions**:

-Does Anneal end the trajectory early, continue randomly, or collapse?

-Do trajectories actually meet at the intended point?

-Are trajectories perturbed by injected data in other branches?

**Observations / Notes**:

-Track divergence, convergence, or instability after data injection.

-Document effects on boundary conditions and stopping behavior.

-Include screenshots, CSV snapshots, or example outputs.

**Next Steps**:

Compare perturbed vs unperturbed trajectories.

Test sensitivity by varying injection step or magnitude.

Record anomalies for future reference.
