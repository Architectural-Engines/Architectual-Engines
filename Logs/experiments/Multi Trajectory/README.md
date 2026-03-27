Multi Trajectory Test:

Model being used:Anneal V1

Trajectories types being tested:
  -Single starting point with random branching, does not return to single path. 
  -Single starting point forming multiple trajectories.
  -Multiple starting points to a single end point.
  -Single starting point with a divergence, then return to a single point and continue.

Purpose: To test how ANNEAL V1 handles trajectories with constrained inputs, how divergence is handled.
    Questions:
      Does the model average the trajectories only forming one. if forming multiple, how does the model handle this?
      Will the model continue on the path after the trajectory splits and recombines?
      How does Anneal handle multiple data streams with similar starting conditions?
      
Early Tests showed the model calculates per step not a weighted average across the trajectory.

All images and meaningful results will be uploaded and explored.       
