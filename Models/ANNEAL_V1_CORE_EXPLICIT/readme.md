ANNEAL V1

* Active: participates in the current forward/training path.
* Present/inactive: exists in the class or return values but doesn't currently affect the prediction.
* Fixed/configuration: controls computation but isn't learned as a parameter.

 state_projection, layers_map, and state are present, but do not affect or take part in computation, placement is for future implementation.

### 1\. Execution overview

Initialization
Modules · parameters · fixed values · buffers
Input / temporal construction
Synthetic trajectory targets · time encoding
Branch construction
55 embeddings → MLP → 55 branch states
Interaction space
V real/imaginary · Theta real/imaginary · P/Theta latent inputs
Branch aggregation
Complex interaction outputs → learned alpha weighting
Output construction
Temporal projection + delta readout + cumulative output
Training / evaluation
MSE · backward · Adam · diagnostics · visualization

### 2\. Initialization:model owned

### Model configuration

`ProbTrajectory5k.__init__()` defines:

| Setting | Current value | Role |
| ---------------------- | ------------- | --------------------------------- |
| state_dimension | 3 | Output/state width |
| time_scale_dimension | 5 | Input width for state projection |
| hidden_dimension | 3 | Input width for interaction heads |
| depth_dimension | 15 | Input width for state projection |
| scale_dimension | 10 | Input width for state projection |
| layers | 1 | Residual module count |
| number_branches | 55 | Branch population |
| epsilon | 0.2 | Denominator stabilizer |
| time_feature_dimension | 50 | Temporal encoding width |
| time_steps | 20 | Temporal resolution |
| input_dimension | 55 | MLP expansion width |

These are constructor configuration values, not trainable parameters.

`self.noise_scale = .0001` is also a fixed scalar.

### Trainable modules and tensors

Following:registered as trainable parameters through `nn.Linear`, `nn.Parameter`, or `nn.ModuleList`.

| Component | Shape / dimensions | Function |
| ----------------------- | ------------------ | ------------------------------ |
| mlp_branch.mlp1 | 3 → 55 | Branch expansion |
| mlp_branch.mlp2 | 55 → 3 | Branch projection |
| mlp_branch.branch_embed | 55 × 3 | Initial branch representations |
| time_projection_map | 50 → 3 | Temporal feature projection |
| P_map | 3 → 3 | P input projection |
| Theta_map_real | 3 → 3 | Real Theta projection |
| Theta_map_imag | 3 → 3 | Imaginary Theta projection |
| V_real_head | 3 → 3 | Real V interaction |
| V_imag_head | 3 → 3 | Imaginary V interaction |
| Theta_real_head | 3 → 3 | Real Theta interaction |
| Theta_imag_head | 3 → 3 | Imaginary Theta interaction |
| branch_logits | 55 | Branch aggregation weights |
| delta_readout | 3 → 3 | Output delta projection |

`state_projection` does execute, but its result is returned as `state`, not used to generate `pos_super`. `layers_map` initialized but never called. `res_layers` is conditional on `update_rule == "residual"`; current training and evaluation both pass `None`.

### Parameter count reconciliation

registered parameter counts break down is as follows:

| Group | Parameters |
| ----------------------- | ---------- |
| MLP branch construction | 495 |
| Temporal projection | 150 |
| P and Theta input maps | 27 |
| Four interaction heads | 48 |
| Branch logits | 55 |
| Delta readout | 9 |
| State projection | 93 |
| Layers map | 3 |
| Residual layer | 30 |
| Total | 910 |

First six groups make up the active prediction path: 784 parameters.

Remaining 126 are registered but don't currently contribute to `pos_super`:
* `state_projection`: 93
* `layers_map`: 3
* `res_layers`: 30

### 3\. Target generation and temporal encoding

two separate constructions here.

### A. Synthetic target generation

`generate_random_trajectory()` creates the target, training attempts to reproduce.

Current main block:

```python
for _ in range(15):
    t, synthetic_position = generate_random_trajectory(
        num_points=20,
        motion_type="oscillatory",
        noise=.1
    )
```

Oscillatory target is:

x=3\\sin(0.1t)y=2\\sin(0.2t)z=2\\cos(0.15t)

Independent Gaussian noise is added to each coordinate because `.1 > .0001`.

* Output `t`: shape `[20]`
* Output `synthetic_position`: shape `[20, 3]`
* 15 separately generated noisy trajectories

Important: `t` returned is not fed into the model during training. It is retained alongside the target and later returned by `train_on_synthetic()`.

Model generates its own temporal encoding.

### B. Model temporal encoding

`branch_encoding()` creates a fixed time grid internally:

```python
t = torch.linspace(0.0, 150.0, steps=self.time_steps)
```

normalizes that grid: 20 sequential positions:

```python
nominal_dt = (t[-1] - t[0]) / (len(t) - 1)
t_scaled = (t - t[0]) / nominal_dt
```

So `t_scaled` becomes approximately:

\[0,1,2,\\ldots,19\]

Creation of 25 frequencies:

```python
freqs = torch.exp(
    torch.linspace(
        math.log(1.0),
        math.log(10.0),
        self.time_feature_dimension // 2
    )
)
```

Each frequency generates a sine and cosine component.

```python
feats = torch.cat(
    [torch.sin(arg), torch.cos(arg)],
    dim=-1
)
```

Temporal encoding output: `[20, 50]`

Projected to:

```python
time_projection_lat = self.time_projection_map(branch_encoding)
```

Temporal projection output: `[20, 3]`

Projection: added to the aggregated real latent state before `delta_readout`.

Pipeline distinction: synthetic target's time values/model's temporal encoding:separate constructions. Happen to have the same values, but the target time tensor isn't used to generate the encoding.

### 4\. Branch construction

This is the first learned state-generation stage.

```python
branch_state = F.relu(self.mlp1(self.branch_embed))
branch_state = self.mlp2(branch_state)
```

Dimensions:

| Operation | Shape |
| ------------ | -------- |
| branch_embed | [55, 3] |
| mlp1 | [55, 55] |
| ReLU | [55, 55] |
| mlp2 | [55, 3] |
| branch_state | [55, 3] |

The 55 learned embeddings are the starting representations. Each is expanded, passed through ReLU, and projected back into three dimensions.

Then:

```python
branch_update = branch_update.unsqueeze(1)
branch_update = branch_update.expand(-1, self.time_steps, -1)
branch_update = branch_update.view(
    self.number_branches,
    self.time_steps,
    self.state_dimension
)
```

This produces:

`[55, 20, 3]`

Important property of this stage:

Each branch's state is repeated across all 20 time steps. There is no time-dependent update to the branch embedding itself here.

Temporal variation enters later, through the interaction denominator's changing epsilon and the separately generated temporal projection.

That gives us a useful separation:

* Branch construction creates the branch population.
* Expansion distributes those branch states across the temporal dimension.
* Interaction computation creates time-varying outputs from those repeated states.

The `motion_type` and `update_rule` arguments of `MLPBranch.forward()` are not used. They don't currently change branch construction.

### 5\. P and Theta input construction

These inputs are built from fixed numerical values:

```python
P_d_values_real = [1.0, 1.0, 1.0]
P_d_values_imag = [-1.0, -1.0, -1.0]
```

Both pass through the same bias-free `P_map`.

P_{\\text{lat,real}}=P_{\\text{map}}(1,1,1)P_{\\text{lat,imag}}=P_{\\text{map}}(-1,-1,-1)

`P_map` lacks bias and is linear:

P_{\\text{lat,imag}}=-P_{\\text{lat,real}}

Relationship is exact, not learned independently.

Theta is different:

```python
Theta_lat_real = self.Theta_map_real(theta_d_real)
Theta_lat_imag = self.Theta_map_imag(theta_d_imag)
```

There are two separate learned maps. Although the input vectors are opposites, the resulting Theta latents are not constrained to be opposites.

| Input | Fixed source | Learned projection |
| --------------- | ------------ | ------------------ |
| P real | [1, 1, 1] | P_map |
| P imaginary | [-1, -1, -1] | Same P_map |
| Theta real | [1, 1, 1] | Theta_map_real |
| Theta imaginary | [-1, -1, -1] | Theta_map_imag |

Each latent has shape `[3]`.

Four latent vectors:Broadcasted across branches/time before interaction.

The original Pd and Theta input values are fixed, their projected representations are learned because the maps are trainable.

### 6\. Interaction computation

Branch population meets the Pd and Theta latent inputs.

Four heads operate on the expanded branch states:

```python
V_real_head = self.V_real_head(branch_update)
V_imag_head = self.V_imag_head(branch_update)
Theta_real_head = self.Theta_real_head(branch_update)
Theta_imag_head = self.Theta_imag_head(branch_update)
```

Each head maps three dimensions to three dimensions.

All four outputs have shape:

`[55, 20, 3]`

The projected Pd and Theta inputs are broadcast into the same shape.

### Denominator construction

```python
denom_real = Theta_lat_real.unsqueeze(0).unsqueeze(1) + Theta_real_head
denom_imag = Theta_lat_imag.unsqueeze(0).unsqueeze(1) + Theta_imag_head
```

Both denominator components have shape `[55, 20, 3]`.

The stabilizer changes over the temporal dimension:

```python
progress = torch.linspace(0, 1, self.time_steps)

eps_effective = (
    self.epsilon * (1.0 - 1.0 * progress)
).view(1, self.time_steps, 1)
```

With `epsilon = .2`, it begins at `.2` and decreases linearly to zero.

The denominator magnitude is then:

D =
\\Theta_r^2+\\Theta_i^2+\\epsilon_{\\text{effective}}^2

In code:

```python
denom_mag2 = (
    denom_real**2
    + denom_imag**2
    + eps_effective**2
)
```

Shape: `[55, 20, 3]`.

Important detail: epsilon is not a constant added uniformly throughout the entire calculation. Contribution is time-dependent, reaches zero at the final time step.

### Numerator construction

```python
num_real = P_lat_real.unsqueeze(0).unsqueeze(1) + V_real_head
num_imag = P_lat_imag.unsqueeze(0).unsqueeze(1) + V_imag_head
```

Again, both shapes are `[55, 20, 3]`.

Real and imaginary components are combined using:

O_r =
\\frac{N_rD_r+N_iD_i}{D_r^2+D_i^2+\\epsilon^2}O_i =
\\frac{N_iD_r-N_rD_i}{D_r^2+D_i^2+\\epsilon^2}

These are the active complex-style interaction equations.

Their outputs:

* `out_real`: `[55, 20, 3]`

* `out_imag`: `[55, 20, 3]`

A useful interpretation of the implementation, without making assumptions about what the architecture: each branch produces a real and imaginary interaction result, and both are retained for the next aggregation stage.

### What creates temporal variation here?

Branch states repeated across time, and the P/Theta latent inputs are fixed within a forward pass.

The heads therefore receive the same branch state at every time step. Their outputs are consequently repeated across time.

The denominator's changing `eps_effective` is one source of time-dependent interaction output.

Worth highlighting: tells exactly where temporal variation enters the current interaction calculation. It isn't coming from a time-dependent branch-state evolution.

### 7\. Branch aggregation

This is the transition from 55 individual branch outputs to a single real and imaginary latent trajectory.

```python
alphas = F.softmax(self.branch_logits, dim=0)
```

`branch_logits` contains 55 trainable values.

Softmax converts them into 55 normalized weights:

\\sum_{i=1}^{55}\\alpha_i=1

Weights are expanded:

```python
alphas_exp = alphas.view(
    self.number_branches, 1, 1
).expand(-1, self.time_steps, self.state_dimension)
```

Aggregation occurs:

```python
latent_real = (alphas_exp * out_real).sum(dim=0)
latent_imag = (alphas_exp * out_imag).sum(dim=0)
```

outputs have shape `[20, 3]`.

Becomes a weighted aggregation across the branch dimension. Alpha values:shared across time and three coordinates; not independently learned for each timestep or coordinate.

### Branch variance

```python
branch_variance = (
    (out_real - latent_real.unsqueeze(0))**2
    * alphas.view(self.number_branches, 1, 1)
).sum(dim=0)

branch_variance = branch_variance * self.noise_scale
```

Measures weighted branch-to-aggregate squared deviation in the real interaction output.

Shape: `[20, 3]`.

multiplied by:

```python
self.noise_scale = .0001
```

Two boundaries to record:

* It uses `out_real`, not `out_imag`.

* It is returned as a diagnostic but does not currently feed back into the prediction or loss.

It is not a direct measurement of imaginary magnitude, nor is it currently used as a noise injection mechanism.

### Magnitude diagnostics

```python
imag_magnitude = latent_imag.abs().mean(dim=1)
real_magnitude = latent_real.abs().mean(dim=1)
```

Each produces shape `[20]`.

coordinate-averaged absolute magnitudes at each timestep. They are calculated from the already aggregated latent values.

diagnostics, not the model's output target and not the training loss.

### 8\. Output construction

Final active prediction path.

First, aggregated real latent state is combined with the temporal projection:

```python
delta = self.delta_readout(
    latent_real + time_projection_lat
)
```

Shapes:

| Value | Shape |
| ------------------- | ------- |
| latent_real | [20, 3] |
| time_projection_lat | [20, 3] |
| Their sum | [20, 3] |
| delta | [20, 3] |

Then:

```python
pos_super = torch.cumsum(delta, dim=0) + latent_real
```

In mathematical form:

\\text{pos}_{t}
=
\\sum_{k=0}^{t}\\Delta_k
+
\\text{latent}_{r,t}

Prediction has two components:

* A cumulative contribution from the learned delta readout.

* A direct contribution from the real aggregated latent state.

`pos_super`: `[20, 3]`

This is what the loss compares against the synthetic target.

### The other output values

```python
delta_update = torch.ones_like(delta)
```

This creates a tensor of ones with shape `[20, 3]`.

does not represent the calculated delta, and it has no role in the prediction or loss. It is currently a fixed diagnostic/placeholder output.

The `state` output is also separate:

```python
combined = torch.cat([
    depth_features,
    time_scale_features,
    scale_features
], dim=-1)

state = self.state_projection(combined)
```

The concatenated input is a zero vector of length 30. The projection produces `[3]`.

Since the input is always zero, the result is determined by the layer's bias. It is not connected to the branch interaction path or `pos_super`.

### Conditional residual path

```python
if update_rule == "residual":
    for layers in self.res_layers:
        latent_real = latent_real + layers(latent_real)
        pos_super = latent_real
```

### 9\. Training pipeline

Main block creates 15 independently generated noisy targets.

`train_on_synthetic()` creates one optimizer before entering the trajectory loop:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

Same model and optimizer are used sequentially across all 15 trajectories.

training sequence:
* Select a target trajectory.
* Run the model forward.
* Compare `pos_super` with the target.
* Calculate MSE.
* Backpropagate.
* Update all registered trainable parameters through Adam.
* Continue to the next epoch.
* Move to the next trajectory without resetting the model or optimizer.

Loss is:

L = \\frac{1}{60}\\sum_{t=1}^{20}\\sum_{d=1}^{3}
(\\text{pos}_{t,d}-\\text{target}_{t,d})^2

There are 300 epochs for each of 15 targets:

### 4,500

Total optimizer updates in the current main execution

### What receives gradients?

loss directly depends on `pos_super`.

The gradient path is:

```text
MSE loss
  └── pos_super
       ├── cumulative delta
       │ └── delta_readout
       │ ├── latent_real
       │ │ └── branch aggregation
       │ │ ├── branch_logits
       │ │ ├── interaction heads
       │ │ ├── P_map
       │ │ ├── Theta maps
       │ │ └── MLPBranch
       │ └── time_projection_map
       └── direct latent_real contribution
```

The real and imaginary interaction components both participate in this gradient path because `out_real` depends on both numerator and denominator components.

The imaginary latent affects the real output indirectly through the complex-style interaction equation, though `latent_imag` itself is not added directly to `pos_super`.

Not updated through this loss:

* `state_projection`
* `layers_map`
* `res_layers` under the current `update_rule=None` setting

They remain registered trainable parameters, but they don't receive useful gradients from the current prediction loss.

The diagnostics (`imag_magnitude`, `real_magnitude`, `branch_variance`, and `delta_update`) are not included in the loss.

One nuance: Adam receives all model parameters, but parameters with no gradients aren't meaningfully updated by this loss. Their presence in the optimizer doesn't make them active.

### 10\. Evaluation

`evaluate_model()` runs a forward pass under:

```python
with torch.no_grad():
```

It then calculates:

```python
loss = F.mse_loss(pos_super, synthetic_position)
```

No gradients are recorded and no parameters are updated.

It returns the prediction, diagnostics, latent inputs, temporal projection, state, and branch variance.

The evaluation target passed from the main block is the final `synthetic_position` generated in the trajectory loop: trajectory 15.

important distinction:
* Training processes all 15 noisy trajectories sequentially.
* Evaluation checks the final model against the last generated target.
* It does not evaluate against all 15 targets or against a separate held-out trajectory.

So the evaluation loss describes fit to that final target, not generalization performance.

### 11\. Visualization and diagnostics

`visualize_trajectory()` converts the prediction and magnitude diagnostics to NumPy arrays.

It calculates:

```python
threshold_imag = mean_imag_magnitude + imag_magnitude.std()
outlier_idx = imag_magnitude > threshold_imag
```

These indices identify timesteps where the imaginary magnitude exceeds its mean by more than one standard deviation.

The corresponding ground-truth positions are marked in red.

The visualization therefore combines:
* Predicted trajectory.
* Ground-truth trajectory.
* Imaginary-magnitude-based timestep markers.
* Mean and maximum real magnitude.
* Mean and maximum imaginary magnitude.
* Optional training/evaluation loss.

One small implementation dependency: `gt` is only assigned inside `if synthetic_position is not None`, but the scatter operation uses it unconditionally afterward. The current main call supplies a target, so this works in the current execution. Calling the function without a target would cause an error.

visualization does not currently display `branch_variance`, despite that value being returned by the model.

### 12\. Consolidated pipeline inventory

This is the reference inventory used during modification of output.

| Stage | Component | Status | Feeds prediction? |
| --------------------- | -------------------- | ------------------ | ------------------ |
| Branch construction | MLPBranch | Active | Yes |
| Temporal construction | branch_encoding | Active | Yes |
| Temporal projection | time_projection_map | Active | Yes |
| State projection | state_projection | Executed | No |
| Layer mapping | layers_map | Unused | No |
| P projection | P_map | Active | Yes |
| Theta projections | Theta_map_real/imag | Active | Yes |
| V heads | V_real/imag_head | Active | Yes |
| Theta heads | Theta_real/imag_head | Active | Yes |
| Epsilon schedule | eps_effective | Active | Yes |
| Branch weights | branch_logits | Active | Yes |
| Real aggregation | latent_real | Active | Yes |
| Imaginary aggregation | latent_imag | Active | Indirectly |
| Branch variance | branch_variance | Calculated | No |
| Magnitude diagnostics | imag/real_magnitude | Calculated | No |
| Delta readout | delta_readout | Active | Yes |
| Cumulative output | pos_super | Active | Final output |
| Delta placeholder | delta_update | Calculated | No |
| Residual modules | res_layers | Conditional | No, currently |
| MSE | loss_fn | Active in training | Trains prediction |
| Adam | optimizer | Active in training | Updates parameters |

A. Temporal variation is concentrated in a few places
Branch states are repeated across time. The interaction denominator's epsilon schedule and the temporal projection are where temporal dependence enters the active path.

Important
B. The imaginary path is indirect at the output
`latent_imag` is returned and measured, but does not directly enter `delta_readout` or `pos_super`. Its influence on prediction comes through the interaction equations that produce `out_real`.

Important
C. There are two different state-like outputs
`state` is the output of a projection on a zero vector. `latent_real` is the aggregated result of branch interactions. They are currently separate and shouldn't be treated as interchangeable.

Training detail

D. Fifteen targets are noisy variations of the same underlying curve
The same model is optimized sequentially against all of them. It is not learning 15 different motion types. Since each target has independently sampled noise, the final prediction is influenced by all 15 samples.

Training detail

E. The forward arguments aren't currently controlling motion
`motion_type="circular"` is passed during training, but `motion_type` isn't used in `ProbTrajectory5k.forward()` or its active branch construction. The target is oscillatory regardless.

Diagnostics

F. Some returned values are observational only
`branch_variance`, `imag_magnitude`, `real_magnitude`, `delta_update`, and `state` are not prediction inputs. Their presence in the return tuple doesn't mean they're part of the learning objective.
