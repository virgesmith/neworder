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

!!! warning "Migrating from neworder 1.x"
    Timelines are no longer classes, and model code reads the current step from `self.timestep` rather than `self.timeline`. Some changes are silent, e.g. the step index now starts at 1. See [Migrating to 2.0](migration.md).

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

Iterators are consumed by the model. To use the same timeline in several models, convert to a list, e.g. `list(neworder.timeline.linear(0, end=10))`, or call the factory for each model.

For example, `list(neworder.timeline.linear(2026, end=2029))` is

```python
[
    Timestep(index=1, start=2026, end=2027, dt=1.0),
    Timestep(index=2, start=2027, end=2028, dt=1.0),
    Timestep(index=3, start=2028, end=2029, dt=1.0),
]
```

and a model run takes its steps like this:

<figure class="timeline-diagram" style="width:100%">
<svg viewBox="0 0 760 276" role="img" aria-label="The timeline linear(2026, end=2029) as three steps on a time axis from 2026 to 2029, each with dt 1.0, preceded by None before the first step. modify runs before the first step, when timestep is None. step and check run in each step, and finalise also runs in the last step, when timestep is step 3." style="width:100%;height:auto;max-width:760px" font-family="monospace" font-size="17">
  <g stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 4">
    <line x1="160" y1="46" x2="160" y2="122"/><line x1="350" y1="46" x2="350" y2="122"/>
    <line x1="540" y1="46" x2="540" y2="122"/><line x1="730" y1="46" x2="730" y2="122"/>
  </g>
  <g style="fill:var(--md-accent-fg-color, #4caf50);stroke:var(--md-accent-fg-color, #4caf50)" fill-opacity="0.15" stroke-width="1.5">
    <rect x="14" y="14" width="142" height="32" rx="4"/><rect x="164" y="14" width="182" height="32" rx="4"/><rect x="354" y="14" width="182" height="32" rx="4"/>
    <rect x="544" y="14" width="182" height="32" rx="4"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="85" y="36">None</text><text x="255" y="36">index=1, dt=1.0</text><text x="445" y="36">index=2, dt=1.0</text>
    <text x="635" y="36">index=3, dt=1.0</text>
  </g>
  <line x1="10" y1="66" x2="160" y2="66" stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="6 4"/>
  <line x1="730" y1="66" x2="750" y2="66" stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="6 4"/>
  <line x1="160" y1="66" x2="730" y2="66" stroke="currentColor" stroke-width="2"/>
  <g stroke="currentColor" stroke-width="2">
    <line x1="160" y1="59" x2="160" y2="73"/><line x1="350" y1="59" x2="350" y2="73"/>
    <line x1="540" y1="59" x2="540" y2="73"/><line x1="730" y1="59" x2="730" y2="73"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="160" y="94">2026</text><text x="350" y="94">2027</text><text x="540" y="94">2028</text><text x="730" y="94">2029</text>
  </g>
  <defs>
    <marker id="nw-timeline-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor" fill-opacity="0.7"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.7" stroke-width="1.5" marker-end="url(#nw-timeline-arrow)">
    <path d="M145 136 H178"/>
    <path d="M255 150 V164"/>
    <path d="M330 180 H350 V136 H368"/>
    <path d="M445 150 V164"/>
    <path d="M520 180 H540 V136 H558"/>
    <path d="M635 150 V164"/>
    <path d="M635 194 V208"/>
  </g>
  <g fill="currentColor" fill-opacity="0.05" stroke="currentColor" stroke-opacity="0.6">
    <rect x="25" y="122" width="120" height="28" rx="4"/>
    <rect x="180" y="122" width="150" height="28" rx="4"/>
    <rect x="180" y="166" width="150" height="28" rx="4"/>
    <rect x="370" y="122" width="150" height="28" rx="4"/>
    <rect x="370" y="166" width="150" height="28" rx="4"/>
    <rect x="560" y="122" width="150" height="28" rx="4"/>
    <rect x="560" y="166" width="150" height="28" rx="4"/>
    <rect x="560" y="210" width="150" height="28" rx="4"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="85" y="142">modify()</text>
    <text x="255" y="142">step()</text>
    <text x="255" y="186">check()</text>
    <text x="445" y="142">step()</text>
    <text x="445" y="186">check()</text>
    <text x="635" y="142">step()</text>
    <text x="635" y="186">check()</text>
    <text x="635" y="230">finalise()</text>
  </g>
  <!--g fill="currentColor" fill-opacity="0.75" text-anchor="middle" font-size="15">
    <text x="85" y="264">timestep: None</text>
    <text x="255" y="264">timestep: step 1</text><text x="445" y="264">timestep: step 2</text>
    <text x="635" y="264">timestep: step 3</text>
  </g-->
</svg>

</figure>

Each step runs from its `start` to its `end` on the time axis. `self.timestep` is `None` before the first step, and `finalise()` sees the last step. `check()` is only called if checking is enabled.


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

It is entirely up to the timeline author what `dt` means. It must be a floating-point type but, e.g. for dates, `dt` could be a number of days, or a year fraction using some custom daycount convention (the builtin `neworder.timeline.calendar` uses ACT/365).

#### Advancing the timeline manually

Whilst the model will automatically iterate over `self.timeline`, user supplied code can - if necessary - take steps itself if the timeline is supplied as an `Iterator` (1), e.g. to skip some steps:
{ .annotate }

1.  If the timeline is supplied as an `Iterable` that isn't an `Iterator` (e.g. a list), this will not work.

```python
def step(self) -> None:
    ...
    if self.quiet_period():
        # self.timeline is typed as an Iterable, so narrow it for type checkers
        assert isinstance(self.timeline, Iterator)
        # the run won't call step() for this timestep
        skipped = next(self.timeline, None)
```

The run carries on from the step after the skipped one, and `self.timestep` isn't changed by a step taken manually. Pass a default to `next` as above: if the timeline is exhausted, a bare `next(self.timeline)` raises `StopIteration` out of `neworder.run`, without calling `finalise`. Similarly, a model constructed with a partly consumed iterator starts from its next step.

!!! warning "Don't advance the timeline by accident"
    Anything that iterates `self.timeline` takes steps from it, e.g. a `for` loop, `list(self.timeline)` or `any(...)`. The run silently skips those steps, and if they're the remaining ones, it finalises after the current step. To inspect the timeline from model code without consuming it, construct the model with a list: a list isn't an iterator, so iterating it has no effect on the run (and `next` on it raises `TypeError`).

#### Controlling the model

The model can be conditionally stopped using the `halt` method - this is essential for open-ended timelines, but can also be useful in situations where you may want to run other code when some condition has been satisfied - the model can be subsequently resumed from where it left off - just call `run` again.

For example, if the model above calls `halt()` in its second step, it takes two runs to complete:

<figure class="timeline-diagram" style="width:100%">
<svg viewBox="0 0 760 276" role="img" aria-label="The timeline linear(2026, end=2029) as three steps on a time axis from 2026 to 2029. In the first run, modify runs before the first step, then step and check in step 1. In step 2, step calls halt, check runs, and run returns. Calling run again resumes at step 3, where step, check and finalise run." style="width:100%;height:auto;max-width:760px" font-family="monospace" font-size="17">
  <g stroke="currentColor" stroke-opacity="0.3" stroke-dasharray="3 4">
    <line x1="160" y1="46" x2="160" y2="122"/><line x1="350" y1="46" x2="350" y2="122"/>
    <line x1="540" y1="46" x2="540" y2="122"/><line x1="730" y1="46" x2="730" y2="122"/>
  </g>
  <g style="fill:var(--md-accent-fg-color, #4caf50);stroke:var(--md-accent-fg-color, #4caf50)" fill-opacity="0.15" stroke-width="1.5">
    <rect x="14" y="14" width="142" height="32" rx="4"/><rect x="164" y="14" width="182" height="32" rx="4"/><rect x="354" y="14" width="182" height="32" rx="4"/>
    <rect x="544" y="14" width="182" height="32" rx="4"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="85" y="36">None</text><text x="255" y="36">index=1, dt=1.0</text><text x="445" y="36">index=2, dt=1.0</text>
    <text x="635" y="36">index=3, dt=1.0</text>
  </g>
  <line x1="10" y1="66" x2="160" y2="66" stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="6 4"/>
  <line x1="730" y1="66" x2="750" y2="66" stroke="currentColor" stroke-opacity="0.4" stroke-dasharray="6 4"/>
  <line x1="160" y1="66" x2="730" y2="66" stroke="currentColor" stroke-width="2"/>
  <g stroke="currentColor" stroke-width="2">
    <line x1="160" y1="59" x2="160" y2="73"/><line x1="350" y1="59" x2="350" y2="73"/>
    <line x1="540" y1="59" x2="540" y2="73"/><line x1="730" y1="59" x2="730" y2="73"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="160" y="94">2026</text><text x="350" y="94">2027</text><text x="540" y="94">2028</text><text x="730" y="94">2029</text>
  </g>
  <defs>
    <marker id="nw-halt-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="currentColor" fill-opacity="0.7"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-opacity="0.7" stroke-width="1.5" marker-end="url(#nw-halt-arrow)">
    <path d="M145 136 H178"/>
    <path d="M255 150 V184"/>
    <path d="M330 200 H350 V136 H368"/>
    <path d="M445 150 V184"/>
    <path d="M445 214 V232"/>
    <path d="M520 248 H540 V136 H558" stroke-dasharray="5 4"/>
    <path d="M635 150 V184"/>
    <path d="M635 214 V228"/>
  </g>
  <g fill="currentColor" fill-opacity="0.05" stroke="currentColor" stroke-opacity="0.6">
    <rect x="25" y="122" width="120" height="28" rx="4"/>
    <rect x="180" y="122" width="150" height="28" rx="4"/>
    <rect x="180" y="186" width="150" height="28" rx="4"/>
    <rect x="370" y="186" width="150" height="28" rx="4"/>
    <rect x="560" y="122" width="150" height="28" rx="4"/>
    <rect x="560" y="186" width="150" height="28" rx="4"/>
    <rect x="560" y="230" width="150" height="28" rx="4"/>
  </g>
  <g style="fill:#ff9800;stroke:#ff9800" fill-opacity="0.15" stroke-width="1.5">
    <rect x="370" y="234" width="150" height="28" rx="4" stroke-dasharray="4 3"/>
    <rect x="370" y="122" width="150" height="28" rx="4"/>
  </g>
  <g fill="currentColor" text-anchor="middle">
    <text x="85" y="142">modify()</text>
    <text x="255" y="142">step()</text>
    <text x="255" y="206">check()</text>
    <text x="445" y="142">step()</text>
    <text x="445" y="206">check()</text>
    <text x="445" y="254">run returns</text>
    <text x="635" y="142">step()</text>
    <text x="635" y="206">check()</text>
    <text x="635" y="250">finalise()</text>
  </g>
</svg>

</figure>

The first `neworder.run` calls `halt()` in step 2 and returns after the check. The dashed arrow is a second call to `neworder.run`, which resumes at step 3 without calling `modify()` again.

!!! note "Halting and resuming"

    - If a model calls `halt()`, the run stops *after* the current step (including any check), so a subsequent call to `neworder.run` resumes at the next step.
    - Resuming doesn't call `modify` again. If the model was halted on the final step, resuming it just calls `finalise`.
    - `halt` will not automatically trigger `finalise` - so for models with open-ended timelines, any `finalise` call must be done explicitly in user code.
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

See the [infection](examples/infection.md) example for an implementation.

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
- a `finalise` method, which is called once, after the last step of the timeline, and is commonly used to post-process the raw model data. It isn't called when the model halts, so a model with an open-ended timeline never finalises unless user code calls `finalise` explicitly.

!!! note "Additional class methods"
    You can add any other methods to your model class. The *neworder* runtime only calls the methods listed above, so any others only run if your code calls them, e.g. from `step`.

Pretty much everything else is entirely up to the model developer. While the module is completely agnostic about the format of data, the library functions accept and return *numpy* arrays, and it is recommended to use *pandas* dataframes where appropriate in order to be able to use the fast data manipulation functionality provided.

Like MODGEN, both time-based and case-based models are supported. In the latter, the timeline refers not to absolute time but the age of the cohort. Additionally continuous-time models can be implemented, using a null timeline (see above) with only a single transition, and the Monte-Carlo library specifically provides functions for continuous sampling, e.g. from non-homogeneous Poisson processes.

New users should take a look at the examples, which cover a range of applications including implementations of some MODGEN teaching models.

## Data and Performance

*neworder* is written in python and C++ with the python bindings provided by the *nanobind* package. As python and C++ have very different memory models, it's generally not advisable to directly share data, i.e. to safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. Thus *neworder* class member variables are accessible only via member functions and results are returned by value (i.e. copied). However, there is a crucial exception to this: the *numpy* `ndarray` type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) both *numpy* arrays and *pandas* data frames, facilitiating very fast implementation of algorithms operating directly on *pandas* DataFrames.[^1]

!!! note "Explicit Loops"
    To get the best performance, avoid using explicit loops in python code where "vectorised" *neworder* (or e.g. numpy) functions can be used instead.

You should also bear in mind that while python is a *dynamically typed* language, C++ is *statically typed*. If an argument to a *neworder* method is not the correct type, it will fail immediately (as opposed to python, which will fail only if an invalid operation for the given type is attempted). Note also that `neworder`'s python code has type annotations.

[^1]: the `neworder.df.transition` function is *over 2 or 3 orders of magnitude faster* than a (naive) equivalent python implementation depending on the length of the dataset, and still an order of magnitude faster than an optimised python implementation.
