# Infection model

An example of individuals moving and interacting on a geospatial network.

![type:video](./img/infection.webm)

{{ include_snippet("./docs/examples/src.md", show_filename=False) }}

!!! note "Optional dependencies"
    This example requires optional dependencies, use:

    `pip install neworder[geospatial]`

## Implementation

The model is built on the graph implementations and algorithms provided by the `networkx` and `osmnx` packages, as well as `geoapandas`, which are encapsulated in the `GeospatialGraph` class specific to this example (to avoid the main package needing the extra dependencies). Each entity travels at a fixed random speed on a street network, travelling repeatedly to randomly selected destinations following a shortest path.

A number of entities are initially "infected" (shown in red) and will pass on the infection to nearby susceptible entities (white). While infected there is a possibility of dying (black) in which case the individual stops moving and is no longer infectious. If they survive the infection they become immune (green). The simulation runs until there are no more infected agents.

Run like so

```sh
python examples/infection/run.py
```

which uses this model implementation:

{{ include_snippet("examples/infection/infection.py") }}

and the `neworder.geospatial.GeospatialGraph` spatial domain.

## Outputs

The output is an animated map of the agents, as illustrated above. A basemap can easily be added by installing `contextily` and uncommenting the lines in `__init_visualisation`.
