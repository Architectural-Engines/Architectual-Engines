**Trajectory Test**:

**Single Trajectory Branching**

**Objective**:

Examine how the model handles random divergences from a single starting point.

Test Anneal’s ability to generate multiple trajectories from one data point.

**Setup**:

Start with a single trajectory point.

Introduce multiple divergence points along the trajectory.

Observe trajectory behavior at each branching step.

**Questions**:

Can the model split trajectories cleanly from a single point?

Does the trajectory remain smooth, or do uncertainty and loss spike?

How does the model average down loss across multiple splits?

**Observations / Notes**:

Document divergence patterns and any trajectory instabilities.

Track uncertainty and loss metrics at each branching step.

Include screenshots, CSV snapshots, or example outputs.

**Next Steps**:

Compare branching behavior across multiple runs.

Test sensitivity to branching magnitude or frequency.

Record anomalies for follow-up experiments.
