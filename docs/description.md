# Description

## Data and Performance

As python and C++ have very different memory models, it's generally not advisable to directly share data, i.e. to safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. However, there is a crucial exception to this: the numpy ndarray type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) both numpy arrays and pandas data frames, facilitiating very fast implementation of algorithms operating directly on pandas DataFrames<sup>*</sup>;

&ast; For instance, a common requirement in microsimulation is a Markov chain: randomly change a state (as given by a column in a data frame) according to a specified transition matrix. This algorithm (implemented in C++) runs *over 2 or 3 orders of magnitude faster* than an equivalent python implementation depending on the length of the dataset, and still an order of magnitude faster that an optimised python implementation.

## The Framework

The aim is to provide as flexible and minimal a framework as possible. Being data agnostic means that this framework can be run standalone or integrated into workflows where e.g. input data is scraped from the web and results are written to a database. Internally, however, the pandas `DataFrame` is the obvious choice of data structure for this type of modelling.

In terms of parallel execution, the following use-cases are supported:

- splitting a large problem over multiple cores.
- performing parallel runs with:
    - perturbations to the model dynamics for sensitivity analysis
    - independent RNGs for convergence analysis

## Provision

The framework essentially provides a mechanism to iterate over a timeline, and perform operations at all points on the timeline, plus extra operations at pre-specified points on that timeline. This is what we call the *model*.

Each model has its own random number stream, and when running in parallel, each stream can be configured to be independent or identical to the other streams.

- the main "loop" over the timeline.
- a resettable, independent<sup>*</sup> random number stream per process. (MT19937)
- a parallel execution framework supporting:
    - modes for sensitivity analysis and convergence analysis:
        - the ability to modify the inputs/dynamics for each process.
        - the ability to run each process with either independent or identical random number streams.
    - interprocess communication and synchronisation, via the `mpi4py` package.
- a library of Monte-Carlo methods.
- fast dataframe manipulation.
- logging facilities.

Where possible, the functionality available in existing python libraries should be leveraged. The framework specifically does not provide any data structures but can operate efficiently on numpy arrays, and thus also pandas `Series` and `DataFrame` objects.

## Requirements

In order to construct a functioning model, the minimal requirements of the model developer are to:

- define a timeline over which the model runs
- create a subclass of `neworder.Model`
- implement the following class methods:
    - a constructor
    - `step`
    - `checkpoint`
- set a seeding policy for the random stream (3 are provided)
- instantiate an instance of the subclass with the timeline and seeding policy
- the simply call the `neworder.run` function

and the following are optional:

- implement the `modify` class method, for instance to modify the input data for different processes in a parallel run, for batch processing or sensitivity analysis.
- implement the `check` class method, which is automatically called at each timestep to e.g. perform checks simulation remains plausible.
- implement a custom seeding policy

Pretty much everything else is entirely up to the model developer. The module is completely agnostic about the format of data, although it's strongly recommended to use numpy arrays and pandas dataframes internally as this will allow efficient

Like MODGEN, both time-based and case-based models are supported. In the latter, the timeline refers not to absolute time but the age of the cohort.

If a timeline is not defined, a single set of transitions is executed. This is useful for implementing continuous-time models.

## Proofs-of-concept

Two of the supplied examples serve as proofs-of-concept of the functionality and performance of the module.

These are two variants of a simulation of a population in terms of fertility, mortality and migration by age, gender, ethnicity and location (MSOA<sup>*</sup>) over a 40-year period (2011-2050). The two distinct use cases are:

- [desktop](): a single-process simulation of a single local authority (initially ~280k people).
- [cluster](): a highly parallel simulation of England & Wales, (initially ~56M people).

&ast; after migration an individual's location is widened to LAD.

The single local authority case ran in about 25 seconds on a desktop PC. The larger simulation ran on the ARC3 [[2]](#references) HPC cluster, using 48 cores, in under 5 minutes.

The microsimulation framework expects a directory containing one or more python modules. There must be a module called [model.py] that, minimally, initialises a user-defined subclass of `neworder.Model` object. This entails:

- defining a timeline over which the model runs. The timeline can be broken into multiple chunks, the end of each of which is considered a _checkpoint_.
- describes, optionally, local modifications to the data or model (for a multiprocess run only).
- describes what (if any) checks to run after each timestep.
- defines the _transitions_ that the population are subject to during the timeline.
- describes what to do with the simulated population data at each checkpoint.
