# neworder

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

Neworder is a prototype C++ microsimulation package inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN. Models are defined in high-level code (python) and executed in an embedded simulation framework (C++) which exposes a subset of itself as a python module. (In order words the C++ framework can call python _and vice versa_) 

## Key requirements:
- low barriers to entry: users need only write standard python code, little or no new coding skills required.
- flexibility: models are defined entirely in user code
- speed: embedded C++ framework and module are compiled and optimised code
- scalability: can be run on a desktop or a HPC cluster, and will support parallel execution using MPI.

## Proof-of-concept 
A simulation of a population of people by age, gender, ethnicity and location (LAD or MSOA) over a 40-year period. There are two distinct use cases:
- desktop: a single-threaded simulation of a single local authority (~250k people).
- cluster: a highly parallel simulation of an entire country (~50M people).

# Installation

_So far only tested on Ubuntu 16.04/18.04_

## Requirements

```bash
$ sudo apt install -y build-essential python3 python3-dev python3-pip libboost-python-dev
$ python3 -m pip install -U numpy pandas
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

## HPC installation (ARC3)

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
Currently experiencing issues running tests:
```
ImportError: numpy.core.multiarray failed to import
...
ERROR: [python] unable to determine error
make[1]: *** [test] Error 1
make[1]: Leaving directory `/nobackup/geoaps/dev/neworder/src/test'
make: *** [test] Error 2
```
which looks the same as the travis python build issue, and running examples/people fails with:

```bash
$ ./run.sh 
[C++] setting PYTHONPATH=example
[C++] process 0 of 1
ImportError: numpy.core.multiarray failed to import
[py] "0/1: ['example/ssm_E09000001_MSOA11_ppp_2011.csv']"
[C++] 2011.25 init: ERROR: [py] <class 'ModuleNotFoundError'>:ModuleNotFoundError("No module named 'pandas'",)
```
# Examples

__NB the following are works-in-progress and subject to change, the documentation may not refelect the current code__

The microsimulation framework expects a python module called [config.py](example/config.py) that, minimally:
- describes how to initialise model object(s) (in this case it's just one, defined in [population.py](example/population.py))
- defines a timeline and a timestep. The timeline can be broken into multiple chunks, the end of each of which is considered a _checkpoint_.
- describes what (if any) checks to run after each timestep.
- defines the _transitions_ that the population are subject to during the timeline.
- describes what to do with the simulated population data at each checkpoint.   

All of these are entirely user-definable.

## The obligatory "Hello world"

TODO

## Microsimulation of People

In this example, the transitions are ageing, births, deaths and migrations. 

Ageing simply increments individual's ages according to the timestep. 

Births, deaths and migrations are are modelled using Monte-Carlo sampling of distributions parameterised by age, sex and ethnicity-specific fertility, mortality and migration rates respectively. 

For the fertility model newborns simply inherit their mother's location and ethnicity, are born aged zero, and have a randomly selected gender (equal probabillity). The migration model is an 'in-out' model, i.e. it is not a full origin-destination model. Flows are either inward from 'elsewhere' or outward to 'elsewhere'.

People who have died are simply removed from the simulation.

NB dealing with competing transitions 

During the simulation, at each timestep the model displays some summary data: 
- the time
- the size of the population
- the mean age of the population
- the percentage of the population that are female
- the in and out migration numbers

At each checkpoint, the population is simply written to a file.

See [population.py](example/population.py) for details of the model implementation. 

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

The above model can be run in massively parallel mode using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface). For example, to run the simulation for all 347 LADs in England & Wales, each with its own microsynthesised population file:

```bash
$ mpirun -n 80 src/bin/neworder_mpi examples/people
```
and the 347 input files will be divided roughly equally over the 80 processes. This particular example lends itself easily to parallel execution as no interprocess communication is required. Future development of this package will enable interprocess communication, for e.g. moving people from one region to another.

## Derivative Pricing

Monte-Carlo simulation is a [common technique in quantitative finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance). 

A European call option is a derivative contract that grants the holder the right (but not the obligation) 
to buy an underlying stock S at a fixed "strike" price K at some given future time T (the expiry). Similarly,
a put option grants the right (but not obligation) to sell, rather than buy, at a fixed price.
See https://en.wikipedia.org/wiki/Call_option.

In order to calculate the fair value of a derivative contract one can simulate a (large) number of paths the underlying stock may take 
(according to current market conditions and some model assumptions). We then take the mean of the derivative price for 
each simulated path to get the value of the derivative _at expiry_. This then is discounted to get the current fair value.

We can easily framing a derivative derivative pricing problem in terms of a microsimulation model:
- start with an intiial (t=0) population of N (identical) underlying prices
- evolve each price using Monte-Carlo simulation of the stochastic differential equation (SDE):

     dS/S = (r-q)dt + vdW

  where S is price, r is risk-free rate, q is continuous dividend yield, v is volatility and dW a Wiener process
- at expiry (t=T) compute the option prices for each underlying and take the mean
- discount the option price back to valuation date (t=0)

For this simple option we can also compute an analytic fair value under the Black-Scholes model, and use this to determine the accuracy of the Monte-Carlo simulation.

Thus our [config.py](examples/option/config.py)

## RiskPaths

RiskPaths is a well-known MODGEN model that is primarily used for teaching purposes.

TODO neworder implementation...