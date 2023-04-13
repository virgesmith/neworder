# Boids flocking model

Example of how simple interaction rules can give rise to collective behaviours, based on the [Netlogo model](https://ccl.northwestern.edu/netlogo/models/Flocking).

![type:video](./img/boids2d.webm)
2-d simulation with two predators

![type:video](./img/boids3d.webm)
3-d simulation with one predator

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Implementation

Each entity travels at a fixed speed in a 2- or 3-dimensional constrained universe, and interacts with the other entities in four ways:

- separation: turns to avoid contact with other entities in close range, or
- evasion: avoids boids that are predators (which are faster than other boids), and
- alignment: turns towards the mean heading of nearby entities, and
- cohesion: turns towards the centre of gravity of nearby entities
- reversion (3d only): tend to return to the centre of the domain

(if a separation is required, the boid will not attempt to align or cohere)

The entities are stored in a pandas `DataFrame` and use `neworder.Space` to update positions. There are no explicit `for` loops in the model - all position and velocity calculations are "vectorised"<sup>&ast;</sup> for efficiency.

&ast; in this context "vectorisation" merely means the avoidance of explicit loops in an interpreted language. The actual implementation may be compiled to assembly language, vectorised in the true ([SIMD](https://en.wikipedia.org/wiki/SIMD)) sense, parallelised, optimised in other ways, or any combination thereof.

Run like so

```sh
python examples/boids/run.py 2d
```

The 2d version utilises a wrap-around domain and does not implement the reversion step.

or

```sh
python examples/boids/run.py 2d
```

which runs

{{ include_snippet("examples/boids/run.py") }}

and this is the 3-d implementation:

{{ include_snippet("examples/boids/boids3d.py") }}

A 2-d implementation is also provided in `examples/boids/boids2d.py`.

## Outputs

The output is an animation of the boid trajectories, as illustrated above.