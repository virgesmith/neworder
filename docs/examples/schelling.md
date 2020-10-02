# Schelling's Segregation Model

An implementation of the Schelling's segregation model [[7]](../references.md), which is traditionally considered to be an agent-based as opposed to a microsimulation, model. However, the distinction is somewhat vague and subjective.

![Schelling](./img/schelling.gif)

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

## Inputs

In the above example, the similarity threshold is 50% and the cells composition is: 30% empty, 30% red, 30% blue and 10% green, on a 80 x 100 grid. Whilst the initial population is randomly constructed using the model's Monte-Carlo engine, the process of moving agents uses

## Implementation

The key points to note from the example code for this model are the use of visualisation inside the model, the use of the `Model.halt()` method, and using the Model's Monte-Carlo engine to seed external random generators for reproducibility.

Since the key output for this model is graphical, the visualisation code sits within the model. The timeline only represents an upper limit for the number of iterations: the model reaches a steady state when there are no unsatisfied agents remaining and there is nothing to be gained by continuing, so when this happens the `neworder.Model.halt()` method is called, at the end of the `step()` implementation:

{{ include_snippet("examples/schelling/schelling.py", "halt") }}

Also in the `step()` method, we use the pandas `sample()` function, which uses it's own random number generator. In order to ensure that the results are consistent with our chosen seeding strategy (i.e deterministic and thus reproducible), the function is explicitly seeded using the model's Monte-Carlo engine:

{{ include_snippet("examples/schelling/schelling.py", "sample") }}

Note that this particular function expects a 32-bit integer so we pass the modulus of the 64-bit integer that `raw()` returns.

## Outputs

The main output is the image above. Log messages also record the timestep and the proportion of the population that remains unsatisfied:

```text
[py 0/1]  step 1 39.02% unsatisfied
[py 0/1]  step 2 34.02% unsatisfied
[py 0/1]  step 3 31.52% unsatisfied
[py 0/1]  step 4 29.32% unsatisfied
[py 0/1]  step 5 27.78% unsatisfied
[py 0/1]  step 6 25.74% unsatisfied
...
[py 0/1]  step 81 0.02% unsatisfied
[py 0/1]  step 82 0.01% unsatisfied
[py 0/1]  step 83 0.01% unsatisfied
[py 0/1]  step 84 0.01% unsatisfied
[py 0/1]  step 85 0.01% unsatisfied
[py 0/1]  step 86 0.00% unsatisfied
```
