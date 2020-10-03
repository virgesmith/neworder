# neworder

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neworder)](https://pypi.org/project/neworder/)
[![PyPI](https://img.shields.io/pypi/v/neworder)](https://pypi.org/project/neworder/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/neworder)](https://pypi.org/project/neworder/)
[![Anaconda-Server Version Badge](https://anaconda.org/conda-forge/neworder/badges/version.svg)](https://anaconda.org/conda-forge/neworder)
[![Anaconda-Server Downloads Badge](https://anaconda.org/conda-forge/neworder/badges/downloads.svg)](https://anaconda.org/conda-forge/neworder)

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/111997710.svg)](https://zenodo.org/badge/latestdoi/111997710)

[![Build Status](https://travis-ci.org/virgesmith/neworder.png?branch=master)](https://travis-ci.org/virgesmith/neworder) 
[![Build status](https://ci.appveyor.com/api/projects/status/oycn4is2insoiun7?svg=true)](https://ci.appveyor.com/project/virgesmith/neworder)
[![Documentation Status](https://readthedocs.org/projects/neworder/badge/?version=latest)](https://neworder.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2f3d4cbf0d174b07b527c64b700db77f)](https://www.codacy.com/app/virgesmith/neworder?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=virgesmith/neworder&amp;utm_campaign=Badge_Grade)

[//]: # (!readme!)

*neworder* is a microsimulation framework inspired by [openm++](https://openmpp.org/), [MODGEN](https://www.statcan.gc.ca/eng/microsimulation/modgen/modgen) and, to a lesser extent, the python-based [LIAM2](http://liam2.plan.be/pages/about.html) tool, and can be thought of powerful best-of-both-worlds hybrid of MODGEN and LIAM2. Modellers can define their models in a a simple. well-known language, yet benefit from the efficiency of compiled code and parallel execution:

- **python module**: easy to install and integrate, available on all common platforms
- **low barriers to entry**: users need only write standard python code, little or no new coding skills required.
- **flexibility**: models are specified in python code, so can be arbitrarily complex
- **data agnosticism**: the framework does not impose any constraints on data sources, or outputs.
- **reusability**: leverage python modules like numpy, pandas and matplotlib.
- **reproducibility**: built-in, customisable random generator seeding strategies
- **speed**: the module is written in optimised C++ and provides fast Monte-Carlo, statistical and data manipulation functions.
- **compatibility**: operate directly on numpy arrays and pandas DataFrames
- **scalability**: can be run on a desktop or a HPC cluster, supporting parallel execution using MPI.

## System Requirements

*neworder* runs in python 3.6 or above on 64-bit linux, OSX or Windows platforms. In order to take advantage of the parallel execution functionality, the following are also required: 

- an MPI implementation, such as [mpich](https://www.mpich.org/), [open-mpi](https://www.open-mpi.org/) or [ms-mpi](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package that provides python MPI bindings

but the module will work perfectly well in serial mode without these.

## Installation

Note: `neworder` is currently in pre-release.

### PyPI

```bash
pip install neworder
```

### Conda

```bash
conda install -c conda-forge neworder
```

### Docker

The docker image contains all the examples, and should be run interactively. Some of the examples require permission to connect to the host's graphical display.

``` 
docker pull virgesmith/neworder
xhost +
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it virgesmith/neworder
```

NB The above works on ubuntu but may require modification on other OSs.

Then in the container, e.g.

```
python examples/mortality/model.py
```

[//]: # (!readme!)

## Documentation

To get started first see the detailed documentation [here](https://neworder.readthedocs.io). Then, check out "Hello World"
and the other examples.
