**Module: ensure_traj3D**

1.**What it is**
   ensure_traj3d is a utility module that ensures any input tensor is valid for 3D trajectory operations.
   It checks the tensor's shape, data type, and dimensionality. Automatically converting or reshaping to a 3d-compatible format.

2. **How it applies**
   -Used befor trajectory processing or mapping functions to guarantee consistency.
   -Ensures that all downstreat modules (e.g., smooth-tensor, to3d, or branch generators) recieve tensors in the expected 3D format.
   -Can handle single trajectories, batched trajectories, or partially shaped tensors and normalize them for 3D operations.

3. **Common issues/Considerations**
   -Input tensors with missing dimensions may automatically expanding, which could be confusing if the user expects a strict 2D or batch format
   -Passing tensors with non-numeric types or incompatible shapes will raise errors. make sure inputs are numeric arrays or PyTorch sensors.
   -Using ensure_traj3d repeatedly on the same tensor is harmless but unnecessary; once validated, the tensor is already 3D-ready.
