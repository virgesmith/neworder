# neworder

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

Neworder is a prototype microsimulation package inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN. Models are defined in high-level code (python) and executed within an embedded simulation framework (written in C++) which exposes a subset of itself as a python module. (In order words the C++ framework can call python _and vice versa_).

As python and C++ have very different memory models, it's not possible to directly share data, i.e. safely have a python object and a C++ object both referencing (and potentially modifying) the same memory location. However, there is a crucial exception to this: the numpy ndarray type. This is fundamental to the operation of the framework, as it enables the C++ module to directly access (and modify) pandas data frames (which do not have a native C API).

## Key Features:
- low barriers to entry: users need only write standard python code, little or no new coding skills required.
- flexibility: models are defined entirely in user code.
- reusability: leverage python modules like numpy, pandas.
- speed: embedded C++ framework and module are compiled and optimised code.
- scalability: can be run on a desktop or a HPC cluster, supporting parallel execution using MPI.
- data agnosticism: the framework does not impose any constraints on data sources/formats/databases. 

## Framework
The aim is to provide as flexible and minimal a framework as possible. Being data agnostic means that this framework can be run standalone, or integrated into workflows where e.g. input data is scraped from the web and results are written to a database. Internally, however, pandas dataframes are the obvious choice of data structure for this type of modelling. 

The sections below list minimal requirements that must be met, and those that - if specified - will be used:

### Requirements
#### Compulsory
The framework minimal requirements are that:
- a timeline and a timestep is defined<sup>*</sup>. The timeline can be partitioned, for example for running a 40-year simulation with 1 year timesteps, but outputting results every 5 years. The latter are referred to as "checkpoints", and the end of the timeline is always considered to be a checkpoint.
- python code to be executed at the first timestep, e.g. to load or microsynthesise an initial population, and also load any data governing the dynamics of the microsimulation, e.g. fertility rates.
- python code to evolve the population to the next timestep, which can (and typically will) involve multiple processes and can be implemented in multiple functions.
- python code to be executed at each checkpoint, typically outputting results in some form or other.

&ast; case-based simulation support is in progress. In this case the timeline refers not to absolute time but the age of the cohort.

#### Optional
And the following are optional:
- python code to run at each timestep to e.g. perform checks that the population remains plausible.
- an outer sequence: this defines a number of runs. This could be re-running the entire simulation again with a different random number stream (to check convergence), or running the simulation with perturbed inputs reusing the same random numbers (to check sensitivity).

### Provision
The framework provides:
- the main "loop" over sequence (if specified), and timeline. 
- a parallel execution framework supporting interprocess communication.
- independent (or identical) deterministic pseudorandom number streams for each process and/or sequence.
- a library of Monte-Carlo methods.
- a mechanism to specify lazily evaluated/executed (i.e. deferred) python code.
- ad-hoc development of fast implementations of slow-running python code, i.e. python code with explicit loops. 
- a logging framework.

Where possible, the functionality available in existing python libraries should be used. The framework specifically does not provide:
- arrays: use numpy wherever possible. The framework can access numpy arrays directly.
- data frames: use pandas wherever possible. Data frames are accessible in the framework via numpy arrays.

#### Special Variables
The neworder python module defines, or requires the model to define, the following variables:

name       | type        | default | description
-----------|-------------|---------|--------------
`sequence` | int array   | `[0]`   | an array specifying the number of simulations to perform. (Each element is used to seed the RNG stream)
`timestep` | float       |         | the size of the timestep 
`timeline` | float array |         | the timeline of the simulation 
`log_level`| int         |         | currently unused. control verbosity of output 
`do_checks`| bool        |         | run functions specified in `checks` after each timestep 
`initialisations`| dict  |         | initialisation function(s)/constructors  
`transitions`| dict      |         | model evolution functions (per-timestep)  
`checks`   | dict        | {}      | specify functions to run after each timestep 
`checkpoints`| dict      |         | perform specified code at each checkpoint 
`sync_streams`|bool      | `false` | determines whether each process has an identical or independent random stream.

Additionally, the module creates the following runtime variables:

name       | type        | default | description
-----------|-------------|---------|--------------
`time`     | float       |         | time of current timestep
`procid`   | int         | `0`     | identifies process for parallel runs. (Used to seed the RNG stream)
`nprocs`   | int         | `1`     | total number of processes in simulation. (Used to seed the RNG stream)

#### Functions
The `neworder` module exposes the following functions to python:

##### General, Utility and Diagnostics

name                | description
--------------------|------------------------------------
`name()`            | returns the module name (`neworder`)
`version()`         | returns the module version
`python()`          | returns the embedded python version
`log(msg)`          | prints `msg`
`shell()`           | starts an interactive shell
`lazy_exec(expr)`   | creates a python expression `expr` for later execution
`lazy_eval(expr)`   | creates a python expression `expr` for later evaluation
`reset_streams(b)`  | resets the random number stream(s), optionally synchronising them

##### Monte-Carlo

name                | description
--------------------|------------------------------------
`ustream(n)`        | returns a numpy 1d array of `n` uniform [0,1) random variables.
`hazard(r, n)`      | returns a numpy 1d array of `n` Bernoulli trials given a hazard rate `r`.
`hazard_v()`        | return a numpy 1d array of Bernoulli trials given a hazard rate `r`.
`stopping(r, n)`    | returns a numpy array of `n` stopping times given a fixed hazard rate `r`. 
`stopping()`        | returns a vector of stopping times given a fixed hazard rate and a length 
`stopping_v()`      |  
`stopping_nhpp()`   | non-homogeneous Poisson process 

##### Data Frames
py::def("transition", no::df::transition);
py::def("directmod", no::df::directmod);
py::def("append", no::df::append

##### Parallel Execution
name                | description
--------------------|------------------------------------
`send(x, n)`        | send object `x` to process `n` 
`receive(n)`        | accept object from process `n`
`send_csv(df, n)`   | send pandas DataFrame in csv format to process `n`
`receive_csv(n)`    | accept pandas DataFrame transmitted in csv format from process `n`
`broadcast(x, n)`   | send object `x` from process `n` to all other processes
`sync()`            | suspend execution of the currency process until all processes have reached this point

## Proof-of-concept 
A simulation of a population of people by age, gender, ethnicity and location (LAD or MSOA) over a 40-year period. There are two distinct use cases:
- desktop: a single-threaded simulation of a single local authority (~250k people).
- cluster: a highly parallel simulation of an entire country (~50M people).

The latter ran on ARC3, over 48 cores, in under 6 minutes. 

# Installation

_So far only tested on Ubuntu 16.04/18.04_

## Requirements

```bash
$ sudo apt install -y build-essential python3 python3-dev python3-pip libboost-python-dev
$ python3 -m pip install -U numpy pandas
```

### Minimum versions

python: 3.5
- numpy: 1.15
- pandas: 0.23

C++14: gcc 5.4 
- boost 1.63
- MPI ?


### Boost

Boost.numpy was introduced in version 1.63, Boost 1.65.1 or higher is recommended. For platforms that come with older versions, see e.g. [this script](tools/boost_python.sh) which downloads 1.67 and builds only the python modules for python3:

```bash
./bootstrap.sh --prefix=/usr/local --with-libraries=python --with-python=$(which python3)
./b2 cxxflags=-DBOOST_NO_AUTO_PTR install
```

## Clone, build and test
```
$ git clone https://github.com/virgesmith/neworder
$ cd neworder
$ make && make test
```
For Ubuntu 16.04 / python 3.5 you may need to set the make env like so:
```bash
$ make PYVER=3.5 BOOST_PYTHON_LIB=boost_python-py35 && make PYVER=3.5 BOOST_PYTHON_LIB=boost_python-py35 test
```
For MPI-enabled execution, you'll frist need to make sure you have an implementation of MPI (including a compiler), e.g:

```bash
$ sudo apt install mpich libmipch-dev
```
And use the [MPI.mk](MPI.mk) makefile to build the MPI-enabled framework:
```bash
$ make -f MPI.mk
```
and to test,
```bash
$ make -f MPI.mk test
```

## HPC installation (ARC3)

[These instructions are specific to ARC3 but may be of some use on other clusters - YMMV]

Switch to gnu toolchain and add python and boost libraries:
```bash
$ module switch intel gnu
$ module load python/3.6.5
$ module load boost python-libs
$ module list
Currently Loaded Modulefiles:
  1) licenses            4) openmpi/2.0.2       7) python-libs/3.1.0
  2) sge                 5) user                8) boost/1.67.0
  3) gnu/6.3.0           6) python/3.6.5
```
Optionally use different g++ or boost versions:
```
$ module switch gnu gnu/7.2.0
$ module switch boost boost/1.65.1
$ module switch python python/3.6.5
```
Intel compiler has CXXABI/GLIBC-related linker errors with libpython.

If boost.numpy is missing (as it is on ARC3), then see the [boost](#boost) section above to build 

~~Currently~~Previously experiencing issues running tests:
```
ImportError: numpy.core.multiarray failed to import
...
ERROR: [python] unable to determine error
make[1]: *** [test] Error 1
make[1]: Leaving directory `/nobackup/geoaps/dev/neworder/src/test'
make: *** [test] Error 2
```
which was down to PYTHONPATH being obilterated. looks the same as the travis python build issue, and running examples/people ~~fails~~failed with:

```bash
$ ./run.sh 
[C++] setting PYTHONPATH=example
[C++] process 0 of 1
ImportError: numpy.core.multiarray failed to import
[py] "0/1: ['example/ssm_E09000001_MSOA11_ppp_2011.csv']"
[C++] 2011.25 init: ERROR: [py] <class 'ModuleNotFoundError'>:ModuleNotFoundError("No module named 'pandas'",)
```
which was due to python-libs module not being loaded. 

### Conda
`neworder` may not be compatible at all with conda - it seems that their conda python binary statically links to libpython. Further investigation required.

### Non-Conda
- Global python packages are old and the code isn't compatible. This can be resolved by updating at user-level, e.g.:
```
$ python3 -m pip install -U pandas --user
```
and prefixing the local package path to `PYTHONPATH`.

- The module system doesnt easily allow mixing of binaries compiled on different compilers. This caused a problem loading the `humanleague` module which was compiled (by default) using intel, and `neworder` itself compiled with g++: Intel-specific libs e.g. libimf.so weren't found. Might be able to hack `LD_LIBRARY_PATH` to fix this?

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
$ ./run_example.sh <name>
```
which will run the model defined in the directory `./examples/<name>`

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

See the [examples/people_big](examples/people_multi) directory and the script [mpi_job.sh](mpi_job.sh)

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

For this simple option we can also compute an analytic fair value under the Black-Scholes model, and use this to determine the accuracy of the Monte-Carlo simulation.

The [config.py](examples/option/config.py) file: 
- sets the parameters for the market and the option
- defines a simple timeline [0, T] corresponding to [valuation date, expiry date] and a single timestep.
- describes how to initialise the [market](examples/option/market.py) and [option](examples/option/option.py) objects
- defines the "transition" which in this case is running the Monte-Carlo simulation
- finally, checks the Monte-Carlo result against the analytic formula and displays the results.

The [helpers.py](examples/option/helpers.py) provides some mathematical formulae.  

```bash
$ ./run_example.sh option
[C++] setting PYTHONPATH=examples/option
[C++] process 0 of 1
[C++] embedded python version: 3.6.5 (default, Apr  1 2018, 05:46:30)  [GCC 7.3.0]
[C++] 0 init: xmarket option
[C++] 0.75 exec: compute_mc_price
[C++] checkpoint: compare_mc_price [py] mc: 7.188980 / ref: 7.201286 err=-0.17%

[C++] SUCCESS
```

## RiskPaths

RiskPaths is a well-known MODGEN model that is primarily used for teaching purposes.

TODO neworder implementation...
