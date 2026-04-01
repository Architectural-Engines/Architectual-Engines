||[Logs](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Logs/daily) || [Models](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Models) || [Architectural Engines Home](https://github.com/Architectural-Engines/Architectual-Engines/tree/main) ||

**Multi Trajectory Test**:

**Model being used**:[ANNEAL V1](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Models/ANNEAL%20V1)

**Trajectories types being tested**:

  -Single starting point with random branching, does not return to single path. [Branching](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Logs/experiments/Multi%20Trajectory/single_trajectory_branching)
  
  -Single starting point forming multiple trajectories.[Single to multi](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Logs/experiments/Multi%20Trajectory/single_trajectory_multi_branch)
  
  -Multiple starting points to a single end point.[Multi](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Logs/experiments/Multi%20Trajectory/multi_trajectory_converge)
  
  -Single starting point with a divergence, then return to a single point and continue.[Recombine](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Logs/experiments/Multi%20Trajectory/single_trajectory_diverge_recombine)

**Purpose**: To test how [ANNEAL V1](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Models/ANNEAL%20V1) handles trajectories with constrained inputs, how divergence is handled.
    **Questions**:
      Does the model average the trajectories only forming one. if forming multiple, how does the model handle this?
      Will the model continue on the path after the trajectory splits and recombines?
      How does Anneal handle multiple data streams with similar starting conditions?
      
Early Tests showed the model calculates per step not a weighted average across the trajectory.

All images and meaningful results will be uploaded and explored.    

||[Logs](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Logs/daily) || [Models](https://github.com/Architectural-Engines/Architectual-Engines/tree/main/Models) || [Architectural Engines Home](https://github.com/Architectural-Engines/Architectual-Engines/tree/main) ||
