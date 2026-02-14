# Overview

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

*neworder*'s timeline is conceptually a sequence of steps that are iterated over (calling the Model's `step` and (optionally) `check` methods at each iteration, plus the `finalise` method at the last time point, which is commonly used to post-process the raw model data at the end of the model run. Timelines should not be incremented in client code, this happens automatically within the model.

The framework is extensible but provides four types of timeline (implemented in C++):

- `NoTimeline`: an arbitrary one-step timeline which is designed for continuous-time models in which the model evolution is computed in a single step
- `LinearTimeline`: a set of equally-spaced intervals in non-calendar time
- `NumericTimeline`: a fully-customisable non-calendar timeline allowing for unequally-spaced intervals
- `CalendarTimeline`: a timeline based on calendar dates with with (multiples of) daily, monthly or annual intervals

``` mermaid
classDiagram
  Timeline <|-- NoTimeline
  Timeline <|-- LinearTimeline
  Timeline <|-- NumericTimeline
  Timeline <|-- CalendarTimeline
  Timeline <|-- CustomTimeline

  class Timeline {
    +int index
    +bool at_end*
    +float dt*
    +Any end*
    +float nsteps*
    +Any start*
    +Any time*
    +_next() None*
    +__repr__() str*
  }

  class NoTimeline {
    +bool at_end
    +float dt
    +Any end
    +float nsteps
    +Any start
    +Any time
    +_next()
    +__repr__() str
  }

  class LinearTimeline {
    +bool at_end
    +float dt
    +Any end
    +float nsteps
    +Any start
    +Any time
    +_next()
    +__repr__() str
  }

  class NumericTimeline {
    +bool at_end
    +float dt
    +Any end
    +float nsteps
    +Any start
    +Any time
    +_next()
    +__repr__() str
  }

  class CalendarTimeline {
    +bool at_end
    +float dt
    +Any end
    +float nsteps
    +Any start
    +Any time
    +_next()
    +__repr__() str
  }

  class CustomTimeline {
    +bool at_end
    +float dt
    +Any end
    +float nsteps
    +Any start
    +Any time
    +_next()
    +__repr__() str
  }
```

!!! note "Calendar Timelines"
    - Calendar timelines do not provide intraday resolution
    - Monthly increments preserve the day of the month (where possible)
    - Daylight savings time adjustments are made which affect time intervals where the interval crosses a DST change
    - Time intervals are computed in years, on the basis of a year being 365.2475 days

#### Custom timelines

If none of the supplied timelines are suitable, users can implement their own, deriving from the abstract `neworder.Timeline` base class, which provides an `index` property that should not be overidden. The following properties and methods must be overridden in the subclass:

symbol     | type              | description
-----------|-------------------|---
`at_end`   | `bool` property   | whether the timeline has reached it's end point
`dt`       | `float` property  | the size of the current timestep
`end`      | `Any` property    | the end time of the timeline
`_next`    | `None` method     | move to the next timestep (for internal use by model, should not normally be called in client code)
`nsteps`   | `int` property    | the total number of timesteps
`start`    | `Any` property    | the start time of the timeline
`time`     | `Any` property    | the current time of the timeline
`__repr__` | `str` method      | (optional) a string representation of the object, defaults to the name of the class

As an example, this open-ended numeric timeline starts at zero and asymptotically converges to 1.0:

{{ include_snippet("./docs/custom_timeline.py", show_filename=False) }}

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

- a `modify` method, which is called at the start of the model run and can be used for instance to modify the input data for different processes in a parallel run, for batch processing or sensitivity analysis.
- a `check` method, which is run at every timestep, to e.g. perform checks simulation remains plausible.

!!! note "Additional class methods"
    There are no restrictions on implementing additional methods in the model class, although bear in mind they won't be available to the *neworder* runtime (unless called by one of the functions listed above).

Pretty much everything else is entirely up to the model developer. While the module is completely agnostic about the format of data, the library functions accept and return *numpy* arrays, and it is recommended to use *pandas* dataframes where appropriate in order to be able to use the fast data manipulation functionality provided.

Like MODGEN, both time-based and case-based models are supported. In the latter, the timeline refers not to absolute time but the age of the cohort. Additionally continuous-time models can be implemented, using a `NoTimeline` (see above) with only a single transition, and the Monte-Carlo library specifically provides functions for continuous sampling, e.g. from non-homogeneous Poisson processes.

New users should take a look at the examples, which cover a range of applications including implementations of some MODGEN teaching models.

## Data and Performance

*neworder* is written in C++ with the python bindings provided by the *pybind11* package. As python and C++ have very different memory models, it's generally not advisable to directly share data, i.e. to safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. Thus *neworder* class member variables are accessible only via member functions and results are returned by value (i.e. copied). However, there is a crucial exception to this: the *numpy* `ndarray` type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) both *numpy* arrays and *pandas* data frames, facilitiating very fast implementation of algorithms operating directly on *pandas* DataFrames.[^1]

!!! note "Explicit Loops"
    To get the best performance, avoid using explicit loops in python code where "vectorised" *neworder* (or e.g. numpy) functions can be used instead.

You should also bear in mind that while python is a *dynamically typed* language, C++ is *statically typed*. If an argument to a *neworder* method is not the correct type, it will fail immediately (as opposed to python, which will fail only if an invalid operation for the given type is attempted). Note also that `neworder`'s python code has type annotations.

[^1]: the `neworder.df.transition` function is *over 2 or 3 orders of magnitude faster* than a (naive) equivalent python implementation depending on the length of the dataset, and still an order of magnitude faster than an optimised python implementation.
