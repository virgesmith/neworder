# Description

## The Framework

The aim of the framework is to be as unrestrictive and flexible as possible, whilst still providing a skeleton on which to implement a model and a suite of useful tools. Being data agnostic means that this framework can be run standalone or integrated with other frameworks, e.g. the `mesa` ABM package, and into workflows with specific demands on input and output data formats.

It is designed to support both serial and parallel execution modes, with the latter being used to tackle large populations or to perform sensitivity or convergence analysis. *neworder* runs as happily on a desktop PC as it does on a HPC cluster.

To help users familiarise themselves with the framework, a number of detailed examples covering a variety of use cases are provided. What follows here is a detailed overview of the package functionality.

## Provision

At at it heart, *neworder* simply provides a mechanism to iterate over a timeline, perform operations at all points on the timeline, plus extra operations at pre-specified points on the timeline, with the help of a library of statistical and data manipulation functions.

This is provided by:

- a **Model** base class: providing the skeleton on which users implement their models
- a **Timeline**: the time horizon over which to run the model
- a **MonteCarlo** engine: a dedicated random number stream for each model instance, with specific configurations for parallel streams
- a library of Monte-Carlo methods and statistical functions
- data manipulation functions optimised for *pandas* DataFrames
- support for a parallel execution using MPI (via the `mpi4py` package).

*neworder* explicitly does not provide any tools for things like visualisation, and users can thus use whatever packages they are most comfortable with. The examples, however, do provide various visualisations using `matplotlib`.

## Requirements

### Timeline

*neworder*'s timeline is conceptually a sequence of steps that are iterated over (calling the Model's `step` and (optionally) `check` methods at each iteration, plus the `checkpoint` method on a user-defined subset of the time points. Checkpoints can be used to perform additional processing that is not required at each and every timestep. The final step in the timeline must be a checkpoint, and is often the only checkpoint, used purely to post-process the raw model data at the end of the model run.

The framework provides four types of timeline:

- `NoTimeline`: an arbitrary one-step timeline which is designed for continuous-time models in which the model evolution is computed in a single step
- `LinearTimeline`: a set of equally-spaced intervals in non-calendar time, plus user-defined checkpoints
- `NumericTimeline`: a fully-customisable non-calendar timeline allowing for unequally-spaced intervals and checkpoints
- `CalendarTimeline`: a timeline based on calendar dates with with (multiples of) daily, monthly or annual intervals and user-defined checkpoints

!!! note "Calendar Timelines"
    - Calendar timelines do not provide intraday resolution
    - Monthly increments preserve the day of the month (where possible)
    - Daylight savings time adjustments are made which affect time intervals where the interval crosses a DST change
    - Time intervals are computed in years, on the basis of a year being 365.2475 days

### Model

In order to construct a functioning model, the minimal requirements of the model developer are to:

- create a subclass of `neworder.Model`
- implement the following class methods:
    - a constructor
    - the `step` method (which is run at every timestep)
    - the `checkpoint` method (which is run at certain timesteps, always including the final one)
- define a timeline (see above) over which the model runs
- set a seeding policy for the random stream (3 are provided, but you can create your own)
- instantiate an instance of the subclass with the timeline and seeding policy
- then, simply call the `neworder.run` function.

the following can also be optionally implemented in the model:

- a `modify` method, which is called at the start of the model run and can be used for instance to modify the input data for different processes in a parallel run, for batch processing or sensitivity analysis.
- a `check` method, which is run at every timestep, to e.g. perform checks simulation remains plausible.

!!! note "Additional class methods"
    There are no restrictions on implementing additional methods in the model class, although bear in mind they won't be available to the *neworder* runtime unless called by one of the functions listed above.

Pretty much everything else is entirely up to the model developer. While the module is completely agnostic about the format of data, the library functions accept and return *numpy* arrays, and it is recommended to use *pandas* dataframes where appropriate in order to be able to use the fast data manipulation functionality provided.

Like MODGEN, both time-based and case-based models are supported. In the latter, the timeline refers not to absolute time but the age of the cohort. Additionally continuous-time models can be implemented, using a "null `NoTimeline` (see above) with only a single transition, and the Monte-Carlo library specifically provides functions for sampling non-homogeneous Poisson processes.

New users should take a look at the examples, which cover a range of applications including implementations of some MODGEN teaching models.

## Data and Performance

*neworder* is written in C++ with the python bindings provided by the *pybind11* package. As python and C++ have very different memory models, it's generally not advisable to directly share data, i.e. to safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. Thus *neworder* class member variables are accessible only via member functions and are returned by value (i.e. copied). However, there is a crucial exception to this: the *numpy* `ndarray` type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) both *numpy* arrays and *pandas* data frames, facilitiating very fast implementation of algorithms operating directly on *pandas* DataFrames.<sup>*</sup>

!!! note "Explicit Loops"
    To get the best performance, avoid using explicit loops in python code where "vectorised" *neworder* functions can be used instead.

You should also bear in mind that while python is a *dynamically typed* language, C++ is *statically typed*. If an argument to a *neworder* method is not the correct type, it will fail immediately (as opposed to python, which will fail only if an invalid operation for the given type is attempted).

&ast; the `neworder.df.transition` function is *over 2 or 3 orders of magnitude faster* than an equivalent python implementation depending on the length of the dataset, and still an order of magnitude faster that an optimised python implementation.
