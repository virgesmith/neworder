# Wolf-sheep predation

Another implementation of a classic agent-based model

![Wolf-sheep](./img/wolf-sheep.gif)

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}


## Implementation

Rather than representing the agents (wolves, sheep, and grass) as objects, as would be typical in packages like [netlogo](https://ccl.northwestern.edu/netlogo/) or [mesa](https://mesa.readthedocs.io/en/stable/), they are represented as individual rows in pandas DataFrames, which permits efficient vectorised operations on them. Grass grows in fixed "cells" which are used to process interactions. The wolves and sheep roam about randomly at a fixed speed: sheep can only eat grass that is fully grown in the cell they currently occupy, and wolves can only eat sheep within the cell they both occupy.

An extension to the original model adds natural selection: new agents inherit their parent's speed with a random "mutation". Faster animals tend to encounter food more frequently, but conversely consume energy more quickly. The graphic displays a histogram of the speed distributions for wolves and sheep.

Here's the implementation:

{{ include_snippet("./examples/wolf-sheep/wolf_sheep.py") }}

Which is run like so:

{{ include_snippet("./examples/wolf-sheep/model.py") }}

## Outputs

The main output is the animation image above. Log messages also record when either the wolf or sheep populations die out completely. The model halts when the sheep population dies out.
