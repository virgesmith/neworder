# N-body simulation

A 3d model of gravitational interaction

![n-body](./img/n-body.gif)

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Implementation

Body parameters (mass, position, velocity, acceleration) are stored in a pandas DataFrame, permitting efficient vectorised computations of the combined forces on each object<sup>&ast;</sup>. Bodies are initialised randomly, but adjustments are made to give the overall system zero momentum but nonzero angular momentum. The `check` method ensures that both the overall momentum and the overall energy of the system remains bounded.

&ast; In this implementation, interactions are computed on a per-body basis (a complexity of \(O(n^2)\). For large \(n\) it may be more efficient to model the interactions indirectly: partition the space and compute a *field* in each element \(m\) contributed by each body, and then, how that field impacts each body. This has complexity \(O(mn)+O(mn)\): so if \(m \ll n\), it will be significantly quicker.

Here's the model:

{{ include_snippet("examples/n-body/n_body.py") }}

## Outputs

The main output is the animated image above. The overall momentum and energy of the system is logged in the console at each timestep.