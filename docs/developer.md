# Developer

`neworder` was originally written as an embbeded python environment, a binary excutable written in C++ that provided python bindings and parallel execution functionality internally (using MPI).

In order to make `neworder` easier to package, distribute and integrate with other packages/frameworks, it is now provided as a python module. This means that the MPI functionality is now external, supplied by the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package.

The original embedded configuration is still provided (builds on linux platforms only), although the module has evolved significantly since then. See the "embedded" branch if you're interested.

## Requirements

`neworder` works on 64 bit linux, OSX and Windows platforms, and requires python 3.6 or higher. For parallel execution, it requires an MPI environment (e.g. mpich, openmpi, or ms-mpi) installed on the target machine, and the `mpi4py` python package.

## Repo

The source code is on [github](https://github.com/virgesmith/neworder). To contribute, please fork the repository and submit a PR.

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

```bash.
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

If you want to use a specific compiler you can do something like this:

```bash
CC=clang python setup.py install
```

And a simple test that all is ok:

```bash
python -c "import neworder"
```

### Conda

```
conda create -q -n neworder-env python=3.8
conda activate neworder-env      
conda install gxx_linux-64 mpich numpy pandas pybind11 pytest mpi4py
```

Then, as above

```bash
python setup.py install
```

## Test

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

!!! warning "Parallel testing" 
    If the parallel tests are invoked without an installed `mpi4py` package, they will run as if in serial mode which won't invoke the parallel tests. If in doubt check the test log for warnings.

## Running the Examples

Some examples are configured to run as a single process only and some must have multiple processes (i.e. MPI). If the latter, prefix the python call with `mpiexec -n <N>`:

```bash
python examples/<name>/model.py
```

or

```bash
mpiexec -n <N> python examples/<name>/model.py
```

See the Examples section for details on each example.

