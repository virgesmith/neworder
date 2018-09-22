# neworder

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

Neworder is a prototype microsimulation package inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN. Models are defined in high-level code (python) and executed within an embedded simulation framework (written in C++) which exposes a subset of itself as a python module. (In order words the C++ framework can call python _and vice versa_).

## Key Features:
- low barriers to entry: users need only write standard python code, little or no new coding skills required.
- flexibility: models are defined entirely in user code.
- reusability: leverage python modules like numpy, pandas.
- speed: embedded C++ framework and module are compiled and optimised code.
- scalability: can be run on a desktop or a HPC cluster, supporting parallel execution using MPI.
- data agnosticism: the framework does not impose any constraints on data sources/formats/databases. 

For detailed information, see:
- [installation](doc/installation.md)
- [API reference](doc/reference.md)

## Proof-of-concept 
A simulation of a population in terms of fertility, mortality and migration by age, gender, ethnicity and location (MSOA<sup>*</sup>) over a 40-year period. There are two distinct use cases:
- desktop: a single-process simulation of a single local authority (~250k people).
- cluster: a highly parallel simulation of England & Wales, starting in 2011 (initially ~56M people).

&ast; after migration an individual's geographic resolution is widened to LAD.

The latter ran on ARC3, over 48 cores, in under 6 minutes.

## The Framework
The aim is to provide as flexible and minimal a framework as possible. Being data agnostic means that this framework can be run standalone or integrated into workflows where e.g. input data is scraped from the web and results are written to a database. Internally, however, the pandas `DataFrame` is the obvious choice of data structure for this type of modelling. 

In terms of parallel execution, the following use-cases are supported:
- splitting a large problem over multiple cores.
- performing parallel runs with:
  - perturbations to the model dynamics for sensitivity analysis
  - independent RNGs for convergence analysis

### Provision
The framework provides:
- the main "loop" over the timeline. 
- a resettable random number stream per process. (MT19937)
- a parallel execution framework supporting:
  - interprocess communication and synchronisation
  - the ability to modify the inputs/dynamics for each process
  - the ability to run each process with either independent or identical random number streams.
- a library of Monte-Carlo methods.
- a mechanism to specify python code to be run during the course of the simulation (i.e. deferred).
- ad-hoc development of fast implementations of slow-running python code, i.e. python code with explicit loops. 
- a logging framework.

Where possible, the functionality available in existing python libraries should be used. The framework specifically does not provide:
- arrays: use numpy wherever possible. The framework can access numpy arrays directly. 
- data frames: use pandas wherever possible. Data frames are accessible in the framework via numpy arrays.

That said, the model developer should avoid loops in python code - its an interpreted language and loops will be executed much more slowly than compiled code.

The section below lists minimal requirements that must be met, and those that - if specified - will be used:

## Requirements
### Compulsory
The framework minimal requirements are that:
- a timeline and a timestep is defined<sup>*</sup>. The timeline can be partitioned, for example for running a 40-year simulation with 1 year timesteps, but outputting results every 5 years. The latter are referred to as "checkpoints", and the end of the timeline is always considered to be a checkpoint.
- python code to be executed at the first timestep, e.g. to load or microsynthesise an initial population, and also load any data governing the dynamics of the microsimulation, e.g. fertility rates.
- python code to evolve the population to the next timestep, which can (and typically will) involve multiple processes and can be implemented in multiple functions.
- python code to be executed at each checkpoint, typically outputting results in some form or other.

&ast; case-based simulation support is in progress. In this case the timeline refers not to absolute time but the age of the cohort.

### Optional
The following are optional :
- code to modify the input data for different processes in a parallel run, for sensitivity analysis.
- functions to call at each timestep to e.g. perform checks that the population remains plausible.

# Examples

__NB the following are works-in-progress and subject to change, the documentation may not reflect the current code__

The microsimulation framework expects a directory containing some python modules. There must be a module called [config.py] that, minimally:
- describes how to initialise model object(s), defined in the other module(s).
- defines a timeline and a timestep. The timeline can be broken into multiple chunks, the end of each of which is considered a _checkpoint_.
- describes what (if any) checks to run after each timestep.
- defines the _transitions_ that the population are subject to during the timeline.
- describes what to do with the simulated population data at each checkpoint.   

All of these are entirely user-definable. The checks, transitions and checkpoints can be empty

To run an example, type 
```bash
$ ./run_example.sh <name> [size [-c]]
```
which will run the model defined in the directory `./examples/<name>`, running optionally over `size` processes, which can be set to use identical RNG streams with the `-c` flag.

## The obligatory "Hello world" example

This example is an ultra-simple illustration of the structure required, all the files are extensively commented. It can be used as a skeleton for new project. 

The model is configured here: [examples/hello_world/config.py](examples/hello_world/config.py). This file refers to a second file in which the "model" is defined, see [examples/hello_world/greet.py](examples/hello_world/greet.py)

## Diagnostics

This isn't really an example, it just outputs useful diagnostic information to track down bugs/problems, and opens a debug shell so that the neworder environment can be inspected. Below we use neworder to sample stopping times based on a 10% hazard rate:

```
$ ./run_example.sh diagnostics
[no 0-0/1] env init
[no 0-0/1] embedded python version: 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[py 0-0/1] MODULE=neworder0.0.0_boost1_65_1
[py 0-0/1] PYTHON=3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[py 0-0/1] Loaded neworder/boost/python libs:
[py 0-0/1]   libpython3.6m.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0 (0x00007f00d34a1000)
[py 0-0/1]   libboost_python3-py36.so.1.65.1 => /usr/lib/x86_64-linux-gnu/libboost_python3-py36.so.1.65.1 (0x00007f00d3262000)
[py 0-0/1]   libboost_numpy3-py36.so.1.65.1 => /usr/lib/x86_64-linux-gnu/libboost_numpy3-py36.so.1.65.1 (0x00007f00d3057000)
[py 0-0/1]   libneworder.so => src/lib/libneworder.so (0x00007f00d2de5000)
[py 0-0/1] PYTHONPATH=examples/diagnostics:examples/shared
[no 0-0/1] t=0 init:
[no 0-0/1] t=1 exec:
[starting neworder debug shell]
[no 0-0/1] checkpoint: shell: >>> import neworder
>>> neworder.stopping(0.1, 5)
array([30.43439191, 13.88102712,  1.69985666, 13.28639123,  1.75969325])
>>>
[exiting neworder debug shell]

[no 0-0/1] SUCCESS
```
See [examples/diagnostics/config.py](examples/diagnostics/config.py)

In the output above, the prefix in square brackets contains the following information
- Source of message: `no` if logged from the framework itself, `py` if logged from python code.
- the sequence number (default 0) used to differentiate multiple simulations of the same inputs.
- the process id ('rank' in MPI parlance) and the total number of processes ('size' in MPI parlance).

## Microsimulation of People (single area)

In this example, the input data is a csv file containing a microsynthesised 2011 population of Newcastle generated from UK census data, by age, gender and ethnicity. The transitions modelled are: ageing, births, deaths and migrations. 

Ageing simply increments individual's ages according to the timestep. 

Births, deaths and migrations are are modelled using Monte-Carlo sampling of distributions parameterised by age, sex and ethnicity-specific fertility, mortality and migration rates respectively, which are drawn from the [NewETHPOP](http://www.ethpop.org/) project.

For the fertility model newborns simply inherit their mother's location and ethnicity, are born aged zero, and have a randomly selected gender (even probability). The migration model is an 'in-out' model, i.e. it is not a full origin-destination model. Flows are either inward from 'elsewhere' or outward to 'elsewhere'.

People who have died are simply removed from the simulation.

Domestic migrations are given by rates per age, dex and ethnicity. Inward migration is based on the population ex-LAD, whereas outward migration is based on the population of the LAD being simulated.

International migrations are absolute (fractional) counts of individuals by age, sex and ethnicity, based on 2011 data. The values are rounded using a total-preserving algorithm. For emigration this presents a compilation: a situation can arise where a person who doesn't sctually exist in the population is marked for migration.

Outward migrations are simply removed from the population. (They are not distributed in this model)

NB dealing with competing transitions...

During the simulation, at each timestep the model displays some summary data: 
- the time
- the size of the population
- the mean age of the population
- the percentage of the population that are female
- the in and out migration numbers

At each checkpoint, the current population is simply written to a file.

See [config.py](examples/people/config.py) for the simulation setup and [population.py](examples/people/population.py) for details of the model implementation. 

The file [helpers.py](examples/people/helpers.py) defines some helper functions, primarily to reformat input data into a format that can be used efficiently.

```bash
$ ./run_example.sh people
[C++] setting PYTHONPATH=examples/people
[C++] process 0 of 1
[C++] embedded python version: 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[C++] 2011.25 init: [py] E08000021 seed: 3006591687345
people
[C++] 2012.25 exec: age fertility migration mortality
[py] check OK: time=2012.250 size=282396 mean_age=37.53, pct_female=49.89 net_migration=123-237
[C++] 2013.25 exec: age fertility migration mortality
[py] check OK: time=2013.250 size=283005 mean_age=37.70, pct_female=49.85 net_migration=106-234
[C++] 2014.25 exec: age fertility migration mortality
[py] check OK: time=2014.250 size=283791 mean_age=37.83, pct_female=49.86 net_migration=114-226
[C++] 2015.25 exec: age fertility migration mortality
[py] check OK: time=2015.250 size=284549 mean_age=37.98, pct_female=49.86 net_migration=93-227
[C++] checkpoint: write_table [py] writing examples/people/dm_2015.250_0_1.csv

[C++] 2016.25 exec: age fertility migration mortality
[py] check OK: time=2016.250 size=285328 mean_age=38.14, pct_female=49.86 net_migration=98-218
[C++] 2017.25 exec: age fertility migration mortality
[py] check OK: time=2017.250 size=286026 mean_age=38.28, pct_female=49.88 net_migration=90-195
[C++] 2018.25 exec: age fertility migration mortality
[py] check OK: time=2018.250 size=286634 mean_age=38.44, pct_female=49.88 net_migration=97-186
[C++] 2019.25 exec: age fertility migration mortality
[py] check OK: time=2019.250 size=287333 mean_age=38.60, pct_female=49.90 net_migration=93-200
[C++] 2020.25 exec: age fertility migration mortality
[py] check OK: time=2020.250 size=287922 mean_age=38.76, pct_female=49.91 net_migration=89-183
[C++] checkpoint: write_table [py] writing examples/people/dm_2020.250_0_1.csv

[C++] SUCCESS
```
### Parallel Execution

The above model has been modified to run in massively parallel mode using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), for the entire population of England & Wales (approx 56 million people as of 2011 census). The input data is not under source control due to its size, but the 348 input files (one per local authority) are divided roughly equally over the MPI processes. This particular example, with its simple in-out migration model, lends itself easily to parallel execution as no interprocess communication is required. Future development of this package will enable interprocess communication, for e.g. moving people from one region to another.  

The microsimulation has been run on the ARC3 cluster, part of the HPC facilities at the University of Leeds, and took about 3 minutes over 24 cores to simulate the peopulation over a 10 year period.

See the [examples/people_multi](examples/people_multi) directory and the script [mpi_job.sh](mpi_job.sh)

## Derivative Pricing

Monte-Carlo simulation is a [common technique in quantitative finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance). 

A [European call option](https://en.wikipedia.org/wiki/Call_option) is a derivative contract that grants the holder the right (but not the obligation) 
to buy an underlying stock S at a fixed "strike" price K at some given future time T (the expiry). Similarly,
a put option grants the right (but not obligation) to sell, rather than buy, at a fixed price.

In order to calculate the fair value of a derivative contract one can simulate a (large) number of paths the underlying stock may take 
(according to current market conditions and some model assumptions). We then take the mean of the derivative price for 
each simulated path to get the value of the derivative _at expiry_. Finally this price is discounted to get the current fair value.

We can easily frame a derivative derivative pricing problem in terms of a microsimulation model:
- start with an intial (t=0) population of N (identical) underlying prices. Social scientists could refer to this as a 'cohort'. 
- evolve each price to option expiry time (t=T) using Monte-Carlo simulation of the stochastic differential equation (SDE):

  dS/S = (r-q)dt + vdW

  where S is price, r is risk-free rate, q is continuous dividend yield, v is volatility and dW a Wiener process (a 1-d Brownian motion).
- compute the option prices for each of the underlyings and take the mean
- discount the option price back to valuation date (t=0)

For this simple option we can also compute an analytic fair value under the Black-Scholes model, and use this to determine the accuracy of the Monte-Carlo simulation. We also demonstrate the capabilities neworder has in terms of sensitivity analysis.

The [config.py](examples/option/config.py) file: 
- defines a simple timeline [0, T] corresponding to [valuation date, expiry date] and a single timestep.
- sets the parameters for the market and the option, and describes how to initialise the [market](examples/option/market.py) and [option](examples/option/option.py) objects with these parameters.
- describes the 'modifiers' for each process: the perturbations applied to the market data in order to calculate the option price sensitivity to that market data
- defines the "transition", which in this case is simply running the Monte-Carlo simulation from time zero to time T.
- checks the Monte-Carlo result against the analytic formula and displays the price and the random error.
- finally, gathers the results from the other processes and computes some sensitivities.

The file [black_scholes.py](examples/option/black_scholes.py) implements the both analytic option formula and the Monte-Carlo simulation, with [helpers.py](examples/option/helpers.py) providing some additional functionality. 

The simulation must be run with 4 processes and, to eliminate Monte-Carlo noise from the sensitivities, with each process using identical random number streams (the -c flag): 

```bash
$ ./run_example.sh option 4 -c
[no 3/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 3/4] starting microsimulation t=0.000000
[no 3/4] initialising market
[no 3/4] initialising option
[no 1/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 1/4] starting microsimulation t=0.000000
[no 1/4] initialising market
[no 1/4] initialising option
[no 2/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 2/4] starting microsimulation t=0.000000
[no 2/4] initialising market
[no 2/4] initialising option
[no 0/4] env: seed=79748 python 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[no 0/4] starting microsimulation t=0.000000
[no 0/4] initialising market
[no 0/4] initialising option
[no 3/4] initialising model
[no 3/4] applying modifier: exec("market.vol = market.vol + 0.001")
[no 3/4] t=0.750000: compute_mc_price
...
[no 2/4] SUCCESS
[no 3/4] SUCCESS
[no 1/4] SUCCESS
[py 0/4] PV=7.213875
[py 0/4] delta=0.548820
[py 0/4] gamma=0.022923
[py 0/4] vega 10bp=0.034061
[no 0/4] SUCCESS
```

## RiskPaths

RiskPaths is a well-known MODGEN model that is primarily used for teaching purposes.

TODO neworder implementation...
