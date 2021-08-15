# Schelling's Segregation Model

An implementation of Schelling's segregation model [[7]](../references.md), which is traditionally considered to be an agent-based as opposed to a microsimulation, model. However, the distinction is somewhat vague and subjective.

![Schelling](./img/schelling.gif)

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Inputs

In this example, the similarity threshold is 60% and the cells states are: 36% empty, 12% red, 12% blue and 40% green, on a 480 x 360 grid. The initial population is randomly constructed using the model's Monte-Carlo engine, the process of moving agents randomly swaps unsatisfied agents with empty cells. The boundaries are "sinks", i.e. there are no neighbouring cells

## Implementation

The key features used in this example are the `StateGrid` class for efficient neighbour counting and the use of a conditional halting: an open-ended timeline and a call to the `Model.halt()` method when a certain state is achieved.

Since the key output for this model is graphical, the visualisation code sits within the model. The model reaches a steady state when there are no unsatisfied agents remaining and there is nothing to be gained by continuing, so when this happens the `neworder.Model.halt()` method is called, at the end of the `step()` implementation:

{{ include_snippet("examples/schelling/schelling.py", "halt") }}

Note that calling the `halt()` method doesn't immediately halt the model, it flags the neworder runtime to not execute any further timesteps. Thus the remainder of the `step` method, and the `check` method (if implemented) will still be called.

The `StateGrid.count_neighbours` takes a function argument that filters the states of the neighbours. By default it will count cells with a state of 1 (the default value is `lambda x: x==1`). In this model we use it to count any occupied cells, and cells with a specific state:

{{ include_snippet("examples/schelling/schelling.py", "count") }}

## Outputs

The output is an animation as shown above. Log messages also record the timestep and the proportion of the population that remains unsatisfied:

```text
[py 0/1] step 0 43.1493% unsatisfied
[py 0/1] step 1 39.1400% unsatisfied
[py 0/1] step 2 36.9196% unsatisfied
[py 0/1] step 3 35.3113% unsatisfied
[py 0/1] step 4 33.9259% unsatisfied
...
[py 0/1] step 133 0.0017% unsatisfied
[py 0/1] step 134 0.0012% unsatisfied
[py 0/1] step 135 0.0012% unsatisfied
[py 0/1] step 136 0.0006% unsatisfied
[py 0/1] step 137 0.0000% unsatisfied
```
