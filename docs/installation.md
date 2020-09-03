
# Installation

## System Requirements

_neworder_ runs in python 3.6 or above on 64-bit linux, OSX or Windows platforms. In order to take advantage of the parallel execution functionality, the following are also required:

- an MPI implementation, such as [mpich](https://www.mpich.org/), [open-mpi](https://www.open-mpi.org/) or [ms-mpi](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package that provides python MPI bindings

...but the it will work perfectly well in serial mode without these extra packages.

## Installation

### PyPI

TODO...
```bash
pip install
```

### Conda-forge

TODO...

### Docker

```bash
docker pull virgesmith/neworder
```