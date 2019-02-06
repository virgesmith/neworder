# Installation

_So far only tested on Ubuntu 16.04/18.04_. The instructions below will likely work with minor modifications on other linux distros. 

Windows and OSX are not currently supported, but the long-term plan is to provide `CMake` build scipts for all 3 platforms. Pull requests are welcome!

## Contents

- [Requirements](#requirements)    
	- [Install Dependencies](#install-dependencies)      
	- [Minimum Versions](#minimum-versions)      
	- [Boost](#boost)      
- [Build and Test](#build-and-test)
	- [Standard Build](#standard-build)       
	- [Parallel Build](#parallel-build)       
- [HPC Installation Notes (ARC3)](#hpc-installation-notes-arc3)       
	- [Issues](#issues)
	- [Conda](#conda)
	- [Non-Conda](#non-conda)

## Requirements

### Install Dependencies
```bash
$ sudo apt install -y build-essential python3 python3-dev python3-pip libboost-python-dev
$ python3 -m pip install -U numpy pandas
```
For parallel execution, you'll first need to make sure you have an implementation of MPI (including a compiler), e.g:
```bash
$ sudo apt install mpich libmipch-dev
```

### Minimum Versions

python: 3.5
- numpy: 1.15
- pandas: 0.23

C++14: gcc 5.4 
- boost 1.63
- MPI (mpich 3.3a2)


### Boost

`Boost.numpy` was introduced in version 1.63, although 1.65.1 or higher is recommended. If the package version meets this requirement the following steps are nto required.

For platforms that come with older boost versions, see e.g. [this script](../tools/boost_python.sh) which downloads 1.67.0 and builds only the python modules for python3:

```bash
$ ./bootstrap.sh --prefix=/usr/local --with-libraries=python --with-python=$(which python3)
./b2 cxxflags=-DBOOST_NO_AUTO_PTR install
```

## Build and Test

First clone (or fork) the repo, then enter the repo's root directory, e.g.:
```bash
$ git clone git@github.com:virgesmith/neworder
$ cd neworder
```

### Standard Build

From the root of the repo, build and run tests:
```bash
$ make && make test
```

For Ubuntu 16.04 / python 3.5 you may need to set the make env like so:

```bash
$ make PYVER=3.5 BOOST_PYTHON_LIB=boost_python-py35 && make PYVER=3.5 BOOST_PYTHON_LIB=boost_python-py35 test
```

### Parallel Build

Ensure the MPI dependencies (see above) have been installed.

From the root of the repo use the [MPI.mk](MPI.mk) makefile to build the MPI-enabled framework:
```bash
$ make -f MPI.mk
```
and to test,
```bash
$ make -f MPI.mk test
```

## HPC Installation Notes (ARC3)

### Issues
makefile hacks
boost: version + include dir + lib dir
openmpi vs mpich?
pandas 

Below is out of date

[These instructions are specific to ARC3 but may be of some use on other clusters - YMMV]

Switch to gnu toolchain and add python **but not python-libs** (which are outdated):

Don't use the boost module - is doesn't have the boost-numpy library

```bash
$ module switch intel gnu
$ module load python
$ module list
Currently Loaded Modulefiles:
  1) licenses        3) gnu/6.3.0       5) user
  2) sge             4) openmpi/2.0.2   6) python/3.6.0
```
Optionally use different g++ or boost versions:
```bash
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
`neworder` may not be compatible at all with conda - it seems that their conda python binary statically links to libpython. Further investigation required.

### Non-Conda
- Global python packages are old and the code isn't compatible. This can be resolved by updating at user-level, e.g.:
```
$ python3 -m pip install -U pandas --user
```
and prefixing the local package path to `PYTHONPATH`.

- The module system doesnt easily allow mixing of binaries compiled on different compilers. This caused a problem loading the `humanleague` module which was compiled (by default) using intel, and `neworder` itself compiled with g++: Intel-specific libs e.g. libimf.so weren't found. Might be able to hack `LD_LIBRARY_PATH` to fix this?
