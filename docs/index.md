# neworder

![Mortality histogram](examples/img/mortality_hist_100k.gif)

*neworder* is a microsimulation framework inspired by [openm++](https://ompp.sourceforge.io/) and MODGEN, and to a lesser extent, the python-based [LIAM2](http://liam2.plan.be/pages/about.html) tool, and can be thought of powerful best-of-both-worlds hybrid of MODGEN and LIAM2. Modellers can define their models in a a simple. well-known language, yet benefit from the efficiency of compiled code and parallel execution:

- **python module**: easy to install and integrate, available on all common platforms
- **low barriers to entry**: users need only write standard python code, little or no new coding skills required.
- **flexibility**: models are specified in python code, so can be arbitrarily complex
- **data agnosticism**: the framework does not impose any constraints on data sources, or outputs.
- **reusability**: leverage python modules like numpy, pandas and matplotlib.
- **speed**: the module is written in optimised C++ and provides fast Monte-Carlo, statistical and data manipulation functions.
- **compatibility**: operate directly on numpy arrays and pandas DataFrames
- **scalability**: can be run on a desktop or a HPC cluster, supporting parallel execution using MPI.

## System Requirements

_neworder_ runs in python 3.6 or above on 64-bit linux, OSX or Windows platforms. In order to take advantage of the parallel execution functionality, the following are also required:

- an MPI implementation, such as [mpich](https://www.mpich.org/), [open-mpi](https://www.open-mpi.org/) or [ms-mpi](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package, which provides python MPI bindings

but the module will work perfectly well in serial mode without these.

## Installation

TODO (not yet available on pypi/conda)

For now see [Contributing](./developer.md) for installation steps