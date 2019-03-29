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
	- [Issues](#issues)
	- [Conda](#conda)
	- [Non-Conda](#non-conda)

## Requirements

### Install Dependencies

#### Pip/virtualenv

```bash
$ sudo apt install -y build-essential python3 python3-dev python3-pip
$ python3 -m pip install -U numpy pandas pybind11
```
For parallel execution, you'll first need to make sure you have an implementation of MPI (including a compiler), e.g:
```bash
$ sudo apt install mpich libmipch-dev
```
#### Conda

Conda also requires a specific C++ compiler:

```bash
$ conda install pybind11 gxx_linux-64 
```
For parallel execution, you'll first need to make sure you a conda-supporting implementation of MPI (including a compiler), e.g:
```bash
$ conda install mpich
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
$ git clone git@github.com:virgesmith/neworder
$ cd neworder
```

### Standard Build

From the root of the repo, build and run tests:
```bash
$ make -j<N> && make test
```
picking an suitable <N> for your platform, typically 1-1.5x number of cores.

### Parallel Build

Ensure the MPI dependencies (see above) have been installed.

From the root of the repo use the [MPI.mk](MPI.mk) makefile to build the MPI-enabled framework:
```bash
$ make -j<N> -f MPI.mk
```
picking an suitable <N> for your platform, typically 1-1.5x number of cores. And to test,
```bash
$ make -f MPI.mk test
```

## HPC Installation Notes (ARC3)

### Issues
makefile hacks
openmpi vs mpich?
pandas 

Below is out of date

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

### Conda
`neworder` works with conda, provided a conda compiler is used, and, for MPI, a conda MPI package. Using `mpich` resulted in an odd error emanating from numpy:
```
ImportError: cannot import name _remove_dead_weakref
```
replacing `mpich` with `openmpi` resolved it, but one of the MPI tests now "fails". (The result with openmpi actually makes more sense than the original mpich result.) 


### Non-Conda
- Global python packages are old and the code isn't compatible. This can be resolved by updating at user-level, e.g.:
	```
	$ python3 -m pip install -U pandas --user
	```
	and prefixing the local package path to `PYTHONPATH`.

- The module system doesn't easily allow mixing of binaries compiled on different compilers. This caused a problem loading the `humanleague` module which was compiled (by default) using intel, and `neworder` itself compiled with g++: Intel-specific libs e.g. libimf.so weren't found. Might be able to hack `LD_LIBRARY_PATH` to fix this? As `humanleague` is now available on conda, this should no longer be an issue.
