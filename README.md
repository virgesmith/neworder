# neworder

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neworder)](https://pypi.org/project/neworder/)
[![PyPI](https://img.shields.io/pypi/v/neworder)](https://pypi.org/project/neworder/)

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/111997710.svg)](https://zenodo.org/badge/latestdoi/111997710)
[![status](https://joss.theoj.org/papers/4b7cc8402819ff48fc7403c0e9a265e9/status.svg)](https://joss.theoj.org/papers/4b7cc8402819ff48fc7403c0e9a265e9)

[![Build and test](https://github.com/virgesmith/neworder/actions/workflows/build-test.yml/badge.svg)](https://github.com/virgesmith/neworder/actions/workflows/build-test.yml)
[![MPI test](https://github.com/virgesmith/neworder/actions/workflows/mpi-test.yml/badge.svg)](https://github.com/virgesmith/neworder/actions/workflows/mpi-test.yml)
[![codecov](https://codecov.io/gh/virgesmith/neworder/branch/main/graph/badge.svg?token=g5mDOcjGTD)](https://codecov.io/gh/virgesmith/neworder)
[![Documentation Status](https://readthedocs.org/projects/neworder/badge/?version=latest)](https://neworder.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2f3d4cbf0d174b07b527c64b700db77f)](https://www.codacy.com/app/virgesmith/neworder?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=virgesmith/neworder&amp;utm_campaign=Badge_Grade)

[//]: # (!readme!)

*neworder* is a microsimulation framework inspired by [openm++](https://openmpp.org/), [MODGEN](https://www.statcan.gc.ca/eng/microsimulation/modgen/modgen) and, to a lesser extent, the python-based [LIAM2](http://liam2.plan.be/pages/about.html) tool, and can be thought of as a powerful best-of-both-worlds hybrid of MODGEN and LIAM2. Modellers can define their models in a simple, well-known language, yet benefit from the efficiency of compiled code and parallel execution:

- **python module**: easy to install and integrate, available on all common platforms
- **low barriers to entry**: users need only write standard python code, little or no new coding skills required.
- **flexibility**: models are specified in python code, so can be arbitrarily complex
- **data agnosticism**: the framework does not impose any constraints on data formats for either sources or outputs.
- **reusability**: leverage python modules like *numpy*, *pandas* and *matplotlib*.
- **reproducibility**: built-in, customisable random generator seeding strategies
- **speed**: the module is predominantly written in optimised C++ and provides fast Monte-Carlo, statistical and data manipulation functions.
- **compatibility**: operate directly on *numpy* arrays and *pandas* DataFrames
- **scalability**: can be run on a desktop or a HPC cluster, supporting parallel execution using MPI.

## System Requirements

*neworder* requires python 3.10 or above and runs on 64-bit linux, OSX and Windows platforms. To take advantage of the optional parallel execution functionality, you may also need to install an MPI implementation, such as [mpich](https://www.mpich.org/), [open-mpi](https://www.open-mpi.org/) or [ms-mpi](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

For example, to install mpich on debian-based linux:

```bash
sudo apt install -y build-essential mpich libmipch-dev
```

Or open-mpi on OSX,

```bash
brew install open-mpi
```


## Installation

The package can be installed from [pypi](https://pypi.org/project/neworder/).

For a basic (serial only) installation,

```bash
pip install neworder
```

or to enable parallel execution using MPI:

```bash
pip install neworder[parallel]
```

or enable the (geo)spatial graph functionality:

```bash
pip install neworder[geospatial]
```

or both:

```bash
pip install neworder[parallel,geospatial]
```

### Docker

The docker image contains all the examples, and should be run interactively. Some of the examples require permission to connect to the host's graphical display.

```bash
docker pull virgesmith/neworder
xhost +local:
docker run --net=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it virgesmith/neworder
```

NB The above works on ubuntu but may require modification on other OSs.

Then in the container, e.g.

```bash
python examples/mortality/model.py
```

[//]: # (!readme!)

## Documentation

To get started first see the detailed documentation [here](https://neworder.readthedocs.io). Then, check out "Hello World" and the other examples.
