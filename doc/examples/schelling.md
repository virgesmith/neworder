## Agent-Based Models

An implementation of the Schelling ABM [[7]](../references.md) is [here](examples/schelling/model.py). It's an almost pure python implementation, only using the timeline and logging functionality provided by the neworder framework, configured [here](examples/schelling/config.py)

![Schelling](./img/schelling.gif)

In the above example, the similarity threshold is 50% and the cells composition is: 30% empty, 30% red, 30% blue and 10% green, on a 80 x 100 grid.
