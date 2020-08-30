# Installation

`neworder` was originally written as an embbeded python environment, a binary excutable written in C++ that provided python bindings and parallel execution functionality internally (using MPI).

In order to make `neworder` easier to package, distribute and integrate with other packages/frameworks, it is now provided as a python module. This means that the MPI functionality is now external, supplied by the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package.

The original embedded configuration is still provided (on linux platforms only). See the branch called "embedded".

## Requirements

`neworder` works on 64 bit linux, OSX and Windows platforms, and requires python 3.6 or higher. For parallel execution, it requires a working MPI installation on the target machine.

## Dependencies

### Pip / virtualenv

First install an MPI framework, such as OpenMPI or MPICh, e.g. on debian-based linux systems:

```bash
sudo apt install -y build-essential mpich libmipch-dev
```

Or on OSX,

```bash
brew install open-mpi
```

Create and activate python3 virtualenv, e.g.

```bash
virtualenv -p python3 .venv
source .venv/bin/activate
```

And then install the python dependencies...

...for all the examples to run:

```bash
pip install -r requirements.txt
```

...or, for a minimal development environment

```bash
pip install numpy pandas pybind11 mpi4py
```

Now install the local package

```bash
python setup.py install
```

And a simple test that all is ok:

```bash
python -c "import neworder"
```

### Conda

TODO...

### Test

Tests use the `pytest` framework and can be invoked serially with either

```bash
pytest 
# or
python setup.py test
```

and in parallel by running in MPI:  

```bash
mpiexec -n 2 pytest
# or 
mpiexec -n 2 python setup.py test
```

Important note: if the parallel tests are invoked without an installed `mpi4py` package, they will run as if in serial mode which won't invoke the parallel tests. If in doubt check the test log for warnings.

### Run Examples

Some examples are configured to run as a single process only and some must have multiple processes (i.e. MPI). If the latter, prefix the python call with `mpiexec -n <N>`:

```bash
python examples/<name>/model.py
```

or

```bash
mpiexec -n <N> python examples/<name>/model.py
```

See [Examples](../README.md#examples) for more detail on each example.

### Package

#### Pip

TODO...

#### Conda

TODO...

#### Docker

Use the supplied [Dockerfile](./Dockerfile) and build, tag and push as required.

## Embedded Environment (legacy)

### MPI-enabled Build

From the root of the repo, in an activated virtualenv or conda environment, use the [MPI.mk](MPI.mk) makefile to build the MPI-enabled framework:

```bash
make -j <N> -f MPI.mk
```

picking a suitable `<N>` for your platform, typically 1-1.5x the number of cores. And to run the tests test,

```bash
make -f MPI.mk test
```

### Run Examples

Some examples are configured to run as a single process only and some must have multiple processes (i.e. MPI). If the latter, specify the number of processes as `<N>` and if the processes need to use identical random streams add `-c`:

```bash
./run_example.sh <name> [<N> [ -c]]
```

where `<name>` is the name of the example, e.g. the "option" example must be run with 4 processes all using the same random number streams:

```bash
./run_example option 4 -c
```

See [Examples](../README.md#examples) for more detail.

# Documentation

TODO