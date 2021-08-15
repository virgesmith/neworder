# Schelling's Segregation Model

An implementation of Schelling's segregation model [[7]](../references.md), which is traditionally considered to be an agent-based as opposed to a microsimulation, model. However, the distinction is somewhat vague and subjective.

![Schelling](./img/schelling.gif)

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Inputs

In the above example, the similarity threshold is 50% and the cells composition is: 30% empty, 30% red, 30% blue and 10% green, on a 640 x 480 grid. The initial population is randomly constructed using the model's Monte-Carlo engine, the process of moving agents uses randomly swaps unsatisfied agents with empty cells.

## Implementation

The key features that this example uses are the `StateGrid` class for efficient neighbour counting and the use of a conditional halting: an open-ended timeline and a call to the `Model.halt()` method when a certain state is achieved.

Since the key output for this model is graphical, the visualisation code sits within the model. The model reaches a steady state when there are no unsatisfied agents remaining and there is nothing to be gained by continuing, so when this happens the `neworder.Model.halt()` method is called, at the end of the `step()` implementation:

{{ include_snippet("examples/schelling/schelling.py", "halt") }}

Note that calling the `halt()` method doesn't immediately halt the model, it flags the neworder runtime to not execute any further timesteps. Thus the remainder of the `step` method, and the `check` method (if implemented) will still be called.

The `StateGrid.count_neighbours` takes a function argument that filters the states of the neighbours. By default it will count cells with a state of 1 (the default value is `lambda x: x==1`). In this model we use it to count any occupied cells, and cells with a specific state:

{{ include_snippet("examples/schelling/schelling.py", "count") }}

## Outputs

The output is an animation as shown above. Log messages also record the timestep and the proportion of the population that remains unsatisfied:

```text
[py 0/1] step 0 42.6660% unsatisfied
[py 0/1] step 1 39.5765% unsatisfied
[py 0/1] step 2 37.5599% unsatisfied
[py 0/1] step 3 36.2454% unsatisfied
[py 0/1] step 4 35.2279% unsatisfied
...
[py 0/1] step 458 0.0003% unsatisfied
[py 0/1] step 459 0.0003% unsatisfied
[py 0/1] step 460 0.0003% unsatisfied
[py 0/1] step 461 0.0003% unsatisfied
[py 0/1] step 462 0.0000% unsatisfied
```
