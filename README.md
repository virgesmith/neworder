# neworder

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2f3d4cbf0d174b07b527c64b700db77f)](https://www.codacy.com/app/virgesmith/neworder?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=virgesmith/neworder&amp;utm_campaign=Badge_Grade)

*neworder* is a prototype microsimulation package inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN. Models are defined in high-level code (python) and executed within an embedded simulation framework (written in C++) which exposes a subset of itself as a python module. (In order words the C++ framework can call python _and vice versa_). 

Purely coincidentally, *neworder* is similar in some respects to the python-based [LIAM2](http://liam2.plan.be/pages/about.html) tool, and can be thought of powerful best-of-both-worlds hybrid of MODGEN and LIAM2:
- models are specified in python code, so can be arbitrarily complex
- native support for parallel execution
- fast (compiled C++) libraries of statistical and data manipulation functions 
- no constraints on input/output data formats or storage
- python is a modern and very common language in data science and has a huge package ecosystem

All this does however require that model developers are comfortable coding in python.

**[installation](doc/installation.md)**    |**[API reference](doc/reference.md)**    
:-----------------------------------------:|:---------------------------------------:

## Contents

- [Key Features](#key-features)    
	- [Data and Performance](#data-and-performance)      
- [Proofs of Concept](#proofs-of-concept) 
- [The Framework](#the-framework)
	- [Provision](#provision)       
- [Requirements](#requirements)       
	- [Compulsory](#compulsory)
	- [Optional](#optional)
- [Examples](#examples)
	- [Hello World](#hello-world)
		- [Understanding the workflow and the output](#understanding-the-workflow-and-the-output)
	- [Diagnostics](#diagnostics)   
	- [Microsimulation of People](#microsimulation-of-people)
		- [Parallel Execution](#parallel-execution)
	- [MODGEN-based models](#modgen-based-models)
		- [Microsimulation and Population Dynamics](#microsimulation-and-population-dynamics)
		- [Competing Risks](#competing-risks)
		- [RiskPaths](#riskpaths)
	- [Agent-Based Models](#agent-based-models)
	- [Derivative Pricing & Risk](#derivative-pricing-&-risk)
- [References](#references)

## Key Features
- low barriers to entry: users need only write standard python code, little or no new coding skills required.
- flexibility: models are defined entirely in user code.
- reusability: leverage python modules like numpy, pandas.
- speed: embedded C++ framework and module are compiled and optimised code.
- scalability: can be run on a desktop or a HPC cluster, supporting parallel execution using MPI.
- data agnosticism: the framework does not impose any constraints on data sources/formats/databases. 

### Data and Performance

As python and C++ have very different memory models, it's generally not advisable to directly share data, i.e. to safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. However, there is a crucial exception to this: the numpy ndarray type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) pandas data frames, facilitiating:
- very fast implementation of algorithms operating directly on pandas DataFrames<sup>*</sup>;
- inter-process communication of any Python object, including DataFrame, over MPI. 

&ast; For instance, a common requirement in microsimulation is to randomly amend a state (given in a column in a data frame) according to a specified transition matrix. This algorithm requires a loop (i.e. each case dealt with separately) and a python implementation was benchmarked at about 1,500 cases per second. The same algorithm implemented in (compiled) C++ runs some 20,000 times faster, processing the entire test dataset (~120k rows) in under 4 milliseconds.

## Proofs-of-concept 

The proofs of concept are two variants of a simulation of a population in terms of fertility, mortality and migration by age, gender, ethnicity and location (MSOA<sup>*</sup>) over a 40-year period (2011-2050). The two distinct use cases are:
- desktop: a single-process simulation of a single local authority (initially ~280k people).
- cluster: a highly parallel simulation of England & Wales, starting in 2011 (initially ~56M people).

&ast; after migration an individual's location is widened to LAD.

The single local authority case ran in about 25 seconds on a desktop PC. The larger simulation ran on the ARC3 [[2]](#references) HPC cluster, using 48 cores, in under 5 minutes.

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
- a resettable, independent<sup>*</sup> random number stream per process. (MT19937)
- a parallel execution framework supporting:
  - modes for sensitivity analysis and convergence analysis:
    - the ability to modify the inputs/dynamics for each process.
    - the ability to run each process with either independent or identical random number streams. 
  - interprocess communication and synchronisation
- a library of Monte-Carlo methods.
- a mechanism to specify python code to be run during the course of the simulation (i.e. deferred).
- fast implementations of common algorithms that require explicit loops. (e.g. one benchmarked at over 4 orders of magnitude faster than a pure python implementation.) 
- a logging framework.

Where possible, the functionality available in existing python libraries should be used. The framework specifically does not provide:
- arrays: use numpy wherever possible. The framework can access numpy arrays directly. 
- data frames: use pandas wherever possible. Data frames are accessible in the framework via numpy arrays.
- visualisation: use e.g. matplotlib

That said, the model developer should avoid loops in python code - it is an interpreted language and loops will be executed much more slowly than compiled code.

The section below lists minimal requirements that must be met, and those that - if specified - will be used:

## Requirements
### Compulsory
The framework minimal requirements are that:
- python code to be executed at initialisation, e.g. to load or microsynthesise an initial population, and also load any data governing the dynamics of the microsimulation, e.g. fertility rates.
- python code to evolve the population to the next state, which can (and typically will) involve multiple processes and can be implemented in multiple functions.
- python code to be executed at each checkpoint, typically outputting the evolved population in some form or other.

### Optional
The following are optional:
- a timeline: a start, and end, and a number of steps. (TODO) The timeline can be partitioned, for example for running a 40-year simulation with 1 year timesteps, but outputting results every 5 years. The latter are referred to as "checkpoints", and the end of the timeline is always considered to be a checkpoint.
- code to modify the input data for different processes in a parallel run, for sensitivity analysis.
- functions to call at each timestep to e.g. perform checks that the population remains plausible.

Like MODGEN, both time-based and case-based models are supported. In the latter, the timeline refers not to absolute time but the age of the cohort.

If a timeline is not defined, a single set of transitions is executed.

# Examples

__NB the following are works-in-progress and subject to change, the documentation may not reflect the current code__

Examples that work post refactor of config:
- hello_world
- diagnostics
- test
- mortality
- disease
- option

__NB note also some of the examples are getting quite complex as they evolve closer to real models - they will be separated in due course__

The microsimulation framework expects a directory containing one or more python modules. There must be a module called [config.py] that, minimally, initialises a `neworder.Model` object. This entails:

- defining a timeline over which the model runs. The timeline can be broken into multiple chunks, the end of each of which is considered a _checkpoint_.
- describes, optionally, local modifications to the data or model (for a multiprocess run only).
- describes what (if any) checks to run after each timestep.
- defines the _transitions_ that the population are subject to during the timeline.
- describes what to do with the simulated population data at each checkpoint.

The neworder runtime will automatically execute the model once constructed.

To run an example, type 
```bash
$ ./run_example.sh <name> [size [-c]]
```
which will run the model defined in the directory `./examples/<name>`, running optionally over `size` processes, which can be set to use identical RNG streams with the `-c` flag.


## Diagnostics

This isn't really an example, it just outputs useful diagnostic information to track down bugs/problems, and opens a debug shell so that the neworder environment can be inspected. Below we use neworder interactively to sample 5 stopping times based on a 10% hazard rate:

<pre>
[no 0/1] env: seed=19937 python 3.6.7 (default, Oct 22 2018, 11:32:17)  [GCC 8.2.0]
[py 0/1] MODULE=neworder0.0.0
[py 0/1] PYTHON=3.6.7 (default, Oct 22 2018, 11:32:17)  [GCC 8.2.0]
[py 0/1] Loaded libs:
[py 0/1]   linux-vdso.so.1 (0x00007ffdb5f63000)
[py 0/1]   libpython3.6m.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0 (0x00007fb595232000)
[py 0/1]   libneworder.so => src/lib/libneworder.so (0x00007fb594fee000)
[py 0/1]   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fb594c65000)
[py 0/1]   libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fb594a4d000)
[py 0/1]   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fb59465c000)
[py 0/1]   libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007fb59442a000)
[py 0/1]   libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fb59420d000)
[py 0/1]   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fb593fee000)
[py 0/1]   libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fb593dea000)
[py 0/1]   libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007fb593be7000)
[py 0/1]   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fb593849000)
[py 0/1]   /lib64/ld-linux-x86-64.so.2 (0x00007fb595afc000)
[py 0/1] PYTHONPATH=examples/diagnostics:examples/shared
[no 0/1] starting microsimulation. timestep=0.000000, checkpoint(s) at [1]
[no 0/1] t=0.000000(1) checkpoint: shell
[starting neworder debug shell]
>>> import neworder
>>> neworder.stopping(0.1, 5)
array([30.43439191, 13.88102712,  1.69985666, 13.28639123,  1.75969325])
>>> <b><font color="red">ctrl-D</font></b>
[exiting neworder debug shell]
[no 0/1] SUCCESS exec time=22.416254s
</pre>

See [examples/diagnostics/config.py](examples/diagnostics/config.py)

## Microsimulation of People

In this example, the input data is a csv file containing a microsynthesised 2011 population of Newcastle generated from UK census data, by age, gender and ethnicity. The transitions modelled are: ageing, births, deaths and migrations. 

Ageing simply increments individual's ages according to the timestep. 

Births, deaths and migrations are are modelled using Monte-Carlo sampling (modelling a Poisson process) of distributions parameterised by age, sex and ethnicity-specific fertility, mortality and migration rates respectively, which are drawn from the NewETHPOP[[1]](#references) project.

For the fertility model newborns simply inherit their mother's location and ethnicity, are born aged zero, and have a randomly selected gender (even probability). The migration model is an 'in-out' model, i.e. it is not a full origin-destination model. Flows are either inward from 'elsewhere' or outward to 'elsewhere'.

People who have died are simply removed from the simulation.

Domestic migrations are given by rates per age, sex and ethnicity. Inward migration is based on the population ex-LAD, whereas outward migration is based on the population of the LAD being simulated.

International migrations are absolute (fractional) counts of individuals by age, sex and ethnicity, based on 2011 data. The values are rounded using a total-preserving algorithm. For emigration this presents a compilation: a situation can arise where a person who doesn't actually exist in the population is marked for migration.

Outward migrations are simply removed from the population. (They are not distributed in this model)

NB dealing with competing transitions...

During the simulation, at each timestep the check code computes and displays some summary data: 
- the time
- the size of the population
- the mean age of the population
- the percentage of the population that are female
- the in and out migration numbers

At each checkpoint, the current population is written to a csv file.

See [config.py](examples/people/config.py) for the simulation setup and [population.py](examples/people/population.py) for details of the model implementation. 

The file [helpers.py](examples/people/helpers.py) defines some helper functions, primarily to reformat input data into a format that can be used efficiently.

```bash
$ time ./run_example.sh people
[no 0/1] env: seed=19937 python 3.6.6 (default, Sep 12 2018, 18:26:19)  [GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
[no 0/1] starting microsimulation...
[no 0/1] t=2011.250000 initialise: people
[no 0/1] t=2012.250000 transition: age
[no 0/1] t=2012.250000 transition: fertility
[no 0/1] t=2012.250000 transition: migration
[no 0/1] t=2012.250000 transition: mortality
[no 0/1] t=2012.250000 check: check
[py 0/1] check OK: time=2012.250 size=281728 mean_age=37.47, pct_female=49.84 net_migration=1409 (23626-23765+2927-1379)
...
[no 0/1] t=2049.250000 transition: age
[no 0/1] t=2049.250000 transition: fertility
[no 0/1] t=2049.250000 transition: migration
[no 0/1] t=2049.250000 transition: mortality
[no 0/1] t=2049.250000 check: check
[py 0/1] check OK: time=2049.250 size=566509 mean_age=40.16, pct_female=49.69 net_migration=27142 (70953-46534+5673-2950)
[no 0/1] t=2050.250000 transition: age
[no 0/1] t=2050.250000 transition: fertility
[no 0/1] t=2050.250000 transition: migration
[no 0/1] t=2050.250000 transition: mortality
[no 0/1] t=2050.250000 check: check
[py 0/1] check OK: time=2050.250 size=594350 mean_age=40.42, pct_female=49.94 net_migration=30095 (75464-48243+6003-3129)
[no 0/1] t=2050.250000 checkpoint: write_table
[py 0/1] writing ./examples/people/dm_E08000021_2050.250.csv
[no 0/1] SUCCESS
```
This 40 year simulation of a population of about 280,000 more than doubling (no exogenous constraints) executed in about 25s on a single core on a desktop machine.

### Parallel Execution

The above model has been modified to run in massively parallel mode using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), for the entire population of England & Wales (approx 56 million people as of 2011 census). The input data is not under source control due to its size, but the 348 input files (one per local authority) are divided roughly equally over the MPI processes. This particular example, with its simple in-out migration model, lends itself easily to parallel execution as no interprocess communication is required. Future development of this package will enable interprocess communication, for e.g. moving people from one region to another.  

The microsimulation has been run on the ARC3[[2]](#references) cluster and took a little over 4 minutes on 48 cores to simulate the population over a 40 year period.

See the [examples/people_multi](examples/people_multi) directory and the script [mpi_job.sh](mpi_job.sh)

## MODGEN-based models

### [Microsimulation and Population Dynamics](doc/examples/Modgen_book.md)

![Mortality histogram](./doc/examples/img/mortality_hist_100k.gif)

We implement some example MODGEN models in *Microsimulation and Population Dynamics* [[3]](#references), and adapt them to run more efficiently in the `neworder` framework.

### Competing Risks

This is a case-based continuous-time microsimulation of the competing risks of (multiple) fertility and mortality. The former is sampled using a nonhomogeneous multiple-arrival-time simulation of a Poisson process, with a minimum gap between events of 9 months. Mortality is sampled using a standard nonhomogeneous Poisson process. A mortality event before a birth event cancels the birth event.

The figure below shows the distribution of up to four births (stacked) plus mortality.

![Competing Fetility-Mortality histogram](./doc/examples/img/competing_hist_100k.png)

### RiskPaths

RiskPaths is a well-known MODGEN model that is primarily used for teaching purposes and described here[[5]](#references) in terms of the model itself and here in terms of implementation[[6]](#references). It models fertility in soviet-era eastern Europe, examining fertility as a function of time and union state. In the model, a woman can enter a maximum of two unions in her lifetime. The first union is divided into two sections: a (deterministic) 3 year period during which fertility is at a maximum, followed by a (stochastic) period with lower fertility.

![riskpaths](./doc/examples/img/riskpaths.png)

Counts of transitions by age: first pregnancy (purple), beginning of first union (blue), end of first union (ochre), start of second union (green), end of second union (red).

See also:
- the [model configuration](examples/riskpaths/config.py)
- the [model implementation](examples/riskpaths/riskpaths.py), and 
- the [input data](examples/riskpaths/data.py)

Note: the mortality rate used in this model does not have a realistic age structure - events that take place in later years have little bearing on the outcome, which is time of first pregnancy. 

## Agent-Based Models

An implementation of the Schelling ABM [[7]](#references) is [here](examples/schelling/model.py). It's an almost pure python implementation, only using the timeline and logging functionality provided by the neworder framework, configured [here](examples/schelling/config.py)

![Schelling](./doc/examples/img/schelling.gif)

In the above example, the similarity threshold is 50% and the cells composition is: 30% empty, 30% red, 30% blue and 10% green, on a 80 x 100 grid.

## Derivative Pricing & Risk

Perhaps (or not) surprisingly, calculating the fair value of a financial derivative can be framed as a microsimulation problem, see [here](doc/examples/Option.md).

# References

[1] [NewETHPOP](http://www.ethpop.org/)

[2] ARC3 forms part of the HPC facilities at the University of Leeds.

[3] Microsimulation and Population Dynamics: An Introduction to Modgen 12, Belanger, A & Sabourin, P, Springer Series on Demographic Methods and Population Analysis 43, 2017, [https://www.microsimulationandpopulationdynamics.com/](https://www.microsimulationandpopulationdynamics.com/)

[4] Lewis, P. A. and Shedler, G. S. (1979), Simulation of nonhomogeneous Poisson processes by thinning. Naval Research Logistics, 26: 403-413. doi:10.1002/nav.3800260304

[5] [General characteristics of Modgen applications--exploring the model RiskPaths](https://www.statcan.gc.ca/eng/microsimulation/modgen/new/chap3/chap3)

[6] [Modgen and the application RiskPaths from the model developer's view](https://www.statcan.gc.ca/eng/microsimulation/modgen/new/chap4/chap4)

[7] [Dynamic models of segregation](https://www.tandfonline.com/doi/abs/10.1080/0022250X.1971.9989794)