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
# Run Example

__NB the following is a work-in-progress and will change frequently...__

The microsimulation framework expects a python module called [config.py](example/config.py) that, minimally:
- describes how to initialise model object(s) (in this case it's just one, defined in [population.py](example/population.py))
- defines a timeline and a timestep.
- describes what (if any) checks to run after each timestep.
- defines the _transitions_ that the population are subject to during the timeline.
- describes what to do with the simulated population data when the simulation is done.   

In the example, the transitions are (currently) ageing, births, and deaths. Ageing simply increments individual's ages according to the timestep. Births and deaths are randomly sampled and parameterised by fertility and mortality rates respectively. 

See [population.py](example/population.py)) for details, but in short newborns inherit their mother's location and ethnicity, are born aged zero, and have a randomly selected gender (equal probabillity). People who have died are simply removed from the simulation.

```bash
$ ./run.sh
[C++] 2011 init: people
[C++] 2012 exec: age fertility mortality
[py] check OK: size=7417 mean_age=42.15, pct_female=44.17
[C++] 2013 exec: age fertility mortality
[py] check OK: size=7425 mean_age=42.29, pct_female=44.19
[C++] 2014 exec: age fertility mortality
[py] check OK: size=7432 mean_age=42.50, pct_female=44.24
[C++] 2015 exec: age fertility mortality
[py] check OK: size=7429 mean_age=42.69, pct_female=44.29
[C++] 2016 exec: age fertility mortality
[py] check OK: size=7422 mean_age=42.90, pct_female=44.29
[C++] 2017 exec: age fertility mortality
[py] check OK: size=7409 mean_age=43.07, pct_female=44.32
[C++] 2018 exec: age fertility mortality
[py] check OK: size=7399 mean_age=43.24, pct_female=44.41
[C++] 2019 exec: age fertility mortality
[py] check OK: size=7387 mean_age=43.47, pct_female=44.46
[C++] 2020 exec: age fertility mortality
[py] check OK: size=7369 mean_age=43.73, pct_female=44.55
[C++] finally: write_table
SUCCESS
```
