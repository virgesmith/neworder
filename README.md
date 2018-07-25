# neworder

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

Neworder is a prototype C++ microsimulation package inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN. Models are defined in high-level code (python) and executed in an embedded simulation framework (C++) which exposes a subset of itself back to the python runtime environment. (In order words the C++ framework can call python _and vice versa_) 

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

## Run Example

__NB the following is a work-in-progress and will change frequently...__

The microsimulation framework expects a python module called [config.py](example/config.py) that, minimally:
- describes how to initialise a model object (in this case it's defined in [population.py](example/population.py))
- defines a timeline and a timestep.
- defines the _transitions_ that the population are subject to during the timeline
- [TODO] describes what to do with the simulated population data when the simulation is done.   

In the example, the transitions are (currently) ageing, births, and deaths. Ageing simply increments individual's ages according to the timestep. Births and deaths are randomly sampled and parameterised by fertility and mortality rates respectively. 

See [population.py](example/population.py)) for details, but in short newborns inherit their mother's location and ethnicity, are born aged zero, and have a randomly selected gender (equal probabillity). People who have died are simply removed from the simulation.

```bash
$ ./run.sh
[C++] 2011: size=7397 mean_age=41.49155062863323 gender_split=0.4418007300256861
[C++]   age fertility mortality
[C++] 2012: size=7399 mean_age=42.04703338288958 gender_split=0.4481686714420867
[C++]   age fertility mortality
[C++] 2013: size=7402 mean_age=42.58889489327209 gender_split=0.4544717643880032
[C++]   age fertility mortality
[C++] 2014: size=7405 mean_age=43.122214719783926 gender_split=0.45037137069547595
[C++]   age fertility mortality
[C++] 2015: size=7408 mean_age=43.64767818574514 gender_split=0.4566684665226781
[C++]   age fertility mortality
[C++] 2016: size=7412 mean_age=44.15920129519698 gender_split=0.45264436049649226
[C++]   age fertility mortality
[C++] 2017: size=7415 mean_age=44.68172623061362 gender_split=0.4589345920431558
[C++]   age fertility mortality
[C++] 2018: size=7419 mean_age=45.20380105135463 gender_split=0.45491306105944207
[C++]   age fertility mortality
[C++] 2019: size=7422 mean_age=45.72635408245756 gender_split=0.4509566154675291
[C++]   age fertility mortality
[C++] 2020: size=7425 mean_age=46.25427609427609 gender_split=0.45737373737373743```
