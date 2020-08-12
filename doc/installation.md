# Installation

_So far only tested on Ubuntu 16.04/18.04 and the ARC3 HPC platform_. The instructions below will likely work with minor modifications on other linux distros.

Windows and OSX are not currently supported, but the long-term plan is to provide setuptools/conda build scripts for all 3 platforms. Pull requests are welcome!

## Contents

- [Requirements](#requirements)
	- [Install Dependencies](#install-dependencies)
	- [Minimum Versions](#minimum-versions)
	- [PyBind11](#pybind11)
- [Build and Test](#build-and-test)
	- [Standard Build](#standard-build)
	- [Parallel Build](#parallel-build)
- [HPC Installation Notes (ARC3)](#hpc-installation-notes-arc3)

## Requirements

### Install Dependencies

#### Pip/virtualenv

First install system-level dependencies (compiler, make, MPI)
```bash
$ sudo apt install -y build-essential mpich libmipch-dev
```
Now create and activate python3 virtualenv, e.g.
```
$ virtualenv -p python3 .venv
...
$ source .venv/bin/activate
```
And then install the python dependencies
```
(.venv) $ pip install numpy pandas pybind11
```
#### Conda

Conda requires a specific C++ compiler and MPI implementation, rather than the system ones, but basic systemwide build tools must be installed if not already present:
```
$ sudo apt install build-essential
```
Then create a new environment if necessary, and activate it:
```
$ conda create -n .condaenv python=3 -y
$ conda activate .condaenv
```
Then install the package dependencies
```bash
(.condaenv) $ conda install pybind11 gxx_linux-64 mpich numpy pandas
```

### Minimum Versions

python: 3.5
- numpy: 1.15
- pandas: 0.23

C++14: gcc 5.4
- pybind11 2.2.4
- MPI (mpich 3.3a2)

## Build and Test

First clone (or fork) the repo, then enter the repo's root directory, e.g.:
```bash
(<env>) $ git clone git@github.com:virgesmith/neworder
(<env>) $ cd neworder
```

### Standard Build

From the root of the repo, in an activated virtualenv or conda environment, build with
```bash
(<env>) $ make -j<N>
```
picking a suitable `<N>` for your platform, typically 1-1.5x number of cores. To test:
```
(<env>) $ make test
```

### Parallel Build

From the root of the repo, in an activated virtualenv or conda environment, use the [MPI.mk](MPI.mk) makefile to build the MPI-enabled framework:
```bash
(<env>) $ make -j<N> -f MPI.mk
```
picking an suitable <N> for your platform, typically 1-1.5x number of cores. And to test,
```bash
(<env>) $ make -f MPI.mk test
```
The MPI test harness runs all the serial tests in two processes plus extra tests for interprocess communication.

### Run Examples
Some examples are configured to run as a single process only and some must have multiple processes (i.e. MPI). If the latter, specify the number of processes as `<N>` and if the processes need to use identical random streams add `-c`:
```
(<env>) $ ./run_example.sh <name> [<N> [ -c]]
```
where `<name>` is the name of the example, e.g. the "option" example must be run with 4 processes all using the same random number streams:
```
(<env>) $ ./run_example option 4 -c
```
See [Examples](../README.md#examples) for more detail.
## HPC Installation Notes (ARC3)

[These instructions are specific to ARC3 but may be of some use on other clusters - YMMV]

Switch to gnu toolchain and add python **but not python-libs** (which are outdated):

```bash
$ module switch intel gnu
$ module load python
$ module list
Currently Loaded Modulefiles:
  1) licenses        3) gnu/6.3.0       5) user
  2) sge             4) openmpi/2.0.2   6) python/3.6.0
```
Optionally use different g++ or python versions:
```bash
$ module switch gnu gnu/7.2.0
$ module switch python python/3.6.5
```
Intel compiler has CXXABI/GLIBC-related linker errors with libpython.


~~Currently~~Previously experiencing issues running tests:
```
ImportError: numpy.core.multiarray failed to import
...
ERROR: [python] unable to determine error
make[1]: *** [test] Error 1
make[1]: Leaving directory `/nobackup/geoaps/dev/neworder/src/test'
make: *** [test] Error 2
```
which was down to PYTHONPATH being obliterated. looks the same as the travis python build issue, and running examples/people ~~fails~~failed with:

```bash
$ ./run.sh
[C++] setting PYTHONPATH=example
[C++] process 0 of 1
ImportError: numpy.core.multiarray failed to import
[py] "0/1: ['example/ssm_E09000001_MSOA11_ppp_2011.csv']"
[C++] 2011.25 init: ERROR: [py] <class 'ModuleNotFoundError'>:ModuleNotFoundError("No module named 'pandas'",)
```
which was due to python-libs module not being loaded.

