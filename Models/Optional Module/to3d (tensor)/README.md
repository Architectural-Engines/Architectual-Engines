Module: to3d

1.**what it is**
  to3d is a utility module that converts input tensors into 3-dimensional representations. 
  It is designed to lift 1D or 2D trajectory data into 3D space, enabling higher-dimensional operations, visualization.

2.**How it applies**
  -Used when a trajectory or latent tensor needs to be embedded into 3D space.
  -Works well in combination with ensure_traj3d to guarantee proper tensor shape.
  -Enables modules that rely on 3D coordinates, such as branching or manifold mapping.

3.**Common Issues**
  -Input tensor is not numeric or contains NaNs
    Possible outcomes: conversion fails
  -Input already 3D
    Possible outcomes: may produce redundant dimensions if not checked.
  -Large tensors
    Possible outcomes: may increased memory usages significantly, tied to parameters and branch count.
