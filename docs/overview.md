# ![logo](img/favicon-32.png) Overview

## The Framework

The aim of the framework is to be as unrestrictive and flexible as possible, whilst still providing a skeleton on which to implement a model, and a suite of useful tools. Being data agnostic means that this framework can be run standalone or easily integrated with other models and/or into workflows with specific demands on input and output data formats.

It is designed to support both serial and parallel execution modes, with the latter being used to tackle large populations or to perform sensitivity or convergence analysis. *neworder* runs as happily on a desktop PC as it does on a HPC cluster.

To help users familiarise themselves with the framework, a number of detailed examples covering a variety of use cases are provided. What follows here is a detailed overview of the package functionality.

## Provision

At at it heart, *neworder* simply provides a mechanism to iterate over a timeline and perform user-specified operations at each point on the timeline, with the help of a library of statistical and data manipulation functions.

This is provided by:

- a **Model** base class: providing the skeleton on which users implement their models
- a **Timeline**: the time horizon over which to run the model
- an optional spatial domain, which can be continuous (`Space`), discrete (`StateGrid`), or a graph (`GeospatialGraph`).
- a **MonteCarlo** engine: a dedicated random number stream for each model instance, with specific configurations for parallel streams
- a library of Monte-Carlo methods and statistical functions, plus compatibility with `numpy`'s statistical functionality through an adapter
- data manipulation functions optimised for *pandas* DataFrames
- support for parallel execution using MPI (via the `mpi4py` package), or alternatively using multithreading.

*neworder* explicitly does not provide any tools for things like visualisation, and users can thus use whatever packages they are most comfortable with. The examples, however, do provide various visualisations using `matplotlib`.

## Model Requirements

### Timeline

*neworder*'s timeline is a sequence of steps, each with an index, a start and an end, and a length. It's just data: an iterator of `neworder.timeline.Timestep`s (or any iterable of them). The model run takes each step in turn, calling the Model's `step` and (optionally) `check` methods, plus the `finalise` method after the last step, which is commonly used to post-process the raw model data at the end of the model run.

Inside the model, `self.timestep` is the current step, a `Timestep(index, start, end, dt)`:

- `index` counts the steps, from 1
- `start` and `end` are in the timeline's *native* type, e.g. `int` or `date`
- `dt` is the length of the step as a `float` (e.g. in years for calendar steps), so that it can be used directly in computations such as hazard rates

Before the first step, i.e. in the model's constructor and in `modify`, `self.timestep` is `None`. A model that needs the start time can take it from its first step, `self.timestep.start`.

The `neworder.timeline` module provides factories for common timelines. Each returns an iterator (a generator) of timesteps, which is open-ended if `end` is omitted:

factory | native type | `dt` | use for
--------|-------------|------|---
`null()` | `None` | `NEVER` (undefined) | continuous-time models in which the model evolution is computed in a single step
`linear(start=0, *, end=None)` | `int` | 1.0 | integer time with unit steps
`numeric(start=..., step=..., end=None)` | `float` | `step` | non-integer time with equal steps (`end - start` must be a whole number of steps)
`numeric(times)` | `float` | the difference of consecutive times | non-integer time from a list or array of (strictly increasing) time points, allowing for unequally-spaced steps
`calendar(start=..., step=..., end=None)` | `date` | years (ACT/365) | calendar dates with steps defined daily, monthly or annually (as a `relativedelta`). Monthly steps preserve the day of the month where possible, and if `end` isn't a whole number of steps from `start` the final step is shortened to end at it

For example, `list(neworder.timeline.linear(2020, end=2023))` is

```python
[
    Timestep(index=1, start=2020, end=2021, dt=1.0),
    Timestep(index=2, start=2021, end=2022, dt=1.0),
    Timestep(index=3, start=2022, end=2023, dt=1.0),
]
```

The neworder convention is that the first timestep has index 1. `dt` must be a floating-point type and it is up to the timeline author to determine how to express the interval, e.g. for dates, `dt` could be a number of days, or a year fraction using some custom convention (the builtin `neworder.timeline.calendar` uses ACT/365).

#### Custom timelines

As a timeline is any iterable of `Timestep`, a custom timeline can be a list, or a generator function, e.g. steps that halve in length:

```python
from neworder import Timestep


def halving() -> Iterator[neworder.Timestep]:
    i, t = 1, 0.0
    while True:
        tnew = t + 2**-i
        yield neworder.Timestep(index=i, start=t, end=tnew, dt=tnew - t)
        i, t = i + 1, tnew
```

Iterators are consumed by the model. To use the same timeline in several models, convert to a list, e.g. `list(neworder.timeline.linear(0, end=10))`, or call the factory for each model.

!!! warning "Migrating from neworder 1.x"
    Timelines are no longer classes. Replace `NoTimeline()`, `LinearTimeline(...)`, `NumericTimeline(...)` and `CalendarTimeline(...)` with the factories `neworder.timeline.null()`, `linear(...)`, `numeric(...)` and `calendar(...)`, and read the current step from `self.timestep` rather than `self.timeline`:

    1.x | 2.0
    ----|----
    `self.timeline.index` | `self.timestep.index`, but it's 1 in the first step (it was 0)
    `self.timeline.time` | `self.timestep.start`, the start of the step
    `self.timeline.dt` | `self.timestep.dt`
    `self.timeline.start`, `.end` | the first step's `start`, and the last step's `end`

    Random streams keyed on the step index (e.g. with `SplitMix64`) are shifted by one, so results won't match 1.x exactly. `LinearTimeline(start, step)` was open-ended in 1.x; `end` is now keyword-only, so the equivalent `linear(start, 1)` raises `TypeError`.

#### Controlling the model

The model can be conditionally stopped using the `halt` method - this is essential for open-ended timelines, but can also be useful in situations where you may want to run other code when some condition has been satisfied - the model can be subsequently resumed from where it left off - just call `run` again.

!!! note "Halting and resuming"

    - If a model calls `halt()`, the run stops *after* the current step (including any check), so a subsequent call to `neworder.run` resumes at the next step.
    - Resuming doesn't call `modify` again. If the model was halted on the final step, resuming it just calls `finalise`.
    - `halt` does not trigger `finalise` - so models with open-ended timelines never call `finalise`.
    - Calling `halt()` from `modify` stops the run before any steps are taken. Running a model that has completed raises `StopIteration`.

### Spatial Domain

*neworder* models do not require a spatial domain, but the following functionality is provided for cases where there is a spatial element to the problem being modelled.

#### Continuous

The `Space` class to encapsulate a continuous space with arbirtrary dimensionality. The edges of the space can be unbounded, wrap-around, contrained, or "bounce". The `move` method will ensure that entities in the space are assigned the appropriate position and velocity to conform with the edge behaviour. Additionally the methods `dist2` and `dist` and compute distances in the space taking into account the edge behaviour (i.e. wrap-around). `dist2` returns the squared distance and, if appropriate, is a more efficient alternative.

See the [boids](examples/boids.md) examples for implementations.

#### Discrete

The `StateGrid` class provides a discrete grid of arbitrary states. Wrapped and contrained edges (only) are supported.

See the [Conway](examples/conway.md) example for implementations.

#### Graph

The `GeospatialGraph` class provides a wrapper around the `networkx` and `osmnx` packages, and provides methods for computing shortest paths, isochrones, and subgraphs as well as identifying edges connected to nodes and vice versa. Due to its heavy dependencies, it an extra - installed using `pip install neworder[geospatial]`.

### Model

In order to construct a functioning model, the minimal requirements of the model developer are to:

- create a subclass of `neworder.Model`
- implement the following class methods:
    - a constructor
    - the `step` method (which is run at every timestep)
- define a timeline (see above) over which the model runs
- set a seeding policy for the random stream (3 are provided, but you can create your own)
- instantiate an instance of the subclass with the timeline and seeding policy
- then, simply pass your model to the `neworder.run` function.

the following can also be optionally implemented in the model:

- a `modify` method, which is called once, at the start of the first model run (not when a halted model is resumed), and can be used for instance to modify the input data for different processes in a parallel run, for batch processing or sensitivity analysis.
- a `check` method, which is run at every timestep, to e.g. perform checks simulation remains plausible.

!!! note "Additional class methods"
    There are no restrictions on implementing additional methods in the model class, although bear in mind they won't be available to the *neworder* runtime (unless called by one of the functions listed above).

Pretty much everything else is entirely up to the model developer. While the module is completely agnostic about the format of data, the library functions accept and return *numpy* arrays, and it is recommended to use *pandas* dataframes where appropriate in order to be able to use the fast data manipulation functionality provided.

Like MODGEN, both time-based and case-based models are supported. In the latter, the timeline refers not to absolute time but the age of the cohort. Additionally continuous-time models can be implemented, using a null timeline (see above) with only a single transition, and the Monte-Carlo library specifically provides functions for continuous sampling, e.g. from non-homogeneous Poisson processes.

New users should take a look at the examples, which cover a range of applications including implementations of some MODGEN teaching models.

## Data and Performance

*neworder* is written in python and C++ with the python bindings provided by the *nanobind* package. As python and C++ have very different memory models, it's generally not advisable to directly share data, i.e. to safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. Thus *neworder* class member variables are accessible only via member functions and results are returned by value (i.e. copied). However, there is a crucial exception to this: the *numpy* `ndarray` type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) both *numpy* arrays and *pandas* data frames, facilitiating very fast implementation of algorithms operating directly on *pandas* DataFrames.[^1]

!!! note "Explicit Loops"
    To get the best performance, avoid using explicit loops in python code where "vectorised" *neworder* (or e.g. numpy) functions can be used instead.

You should also bear in mind that while python is a *dynamically typed* language, C++ is *statically typed*. If an argument to a *neworder* method is not the correct type, it will fail immediately (as opposed to python, which will fail only if an invalid operation for the given type is attempted). Note also that `neworder`'s python code has type annotations.

[^1]: the `neworder.df.transition` function is *over 2 or 3 orders of magnitude faster* than a (naive) equivalent python implementation depending on the length of the dataset, and still an order of magnitude faster than an optimised python implementation.
