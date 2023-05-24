# Developer

*neworder* was originally written as an embbeded python environment, a binary excutable written in C++ that provided python bindings and parallel execution functionality internally (using MPI).

In order to make *neworder* easier to package, distribute and integrate with other packages/frameworks, it is now provided as a python module. This means that the MPI functionality is now external, supplied by the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package.

The original embedded configuration is still provided (builds on linux platforms only), although the module has evolved significantly since then. See the "embedded" branch if you're interested.

## Contributions

To contribute, please submit a pull request. More information on how to do this [here](./contributing.md).

!!! note "Legal"
    Contributors retain copyright on their substantial contributions. If applicable, when submitting a PR, please add yourself as an additional copyright holder in [LICENCE.md](https://github.com/virgesmith/neworder/LICENCE.md).

The instructions below assume you've already forked and cloned a local copy of the neworder repo.

## Requirements

*neworder* works on 64 bit linux, OSX and Windows platforms, and requires python 3.6 or higher. For parallel execution, it requires an MPI environment (e.g. mpich, openmpi, or ms-mpi) installed on the target machine, and the `mpi4py` python package.

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

Now install the local package

```bash
pip install -e .
```

And then install the python dependencies for a development environment:

```bash
pip install -r requirements-developer.txt
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

Using python 3.10 (adjust as necessary)

```bash
conda create -q -n neworder-env python=3.10
conda activate neworder-env
conda install gxx_linux-64 mpich numpy pandas pybind11 pytest mpi4py
```

Then, as above

```bash
pip install -e .
conda install --file requirements-developer.txt
```

### Docker

```bash
docker build -t <image-name> .
docker run --net=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it virgesmith/neworder
```

Running the graphical examples will almost certainly require setting `xhost +local:` on the host to enable docker to connect to the display manager.

## Test

Tests use the `pytest` framework and can be invoked serially with either

```bash
pytest
# or
python -m pytest
```

and in parallel by running in MPI:

```bash
mpiexec -n 2 pytest
# or
mpiexec -n 2 python -m pytest
```

!!! warning "Parallel testing"
    If the tests are invoked in parallel, but without an installed `mpi4py` package, they will run independently, as if in serial mode, and the interprocess tests won't get run. If in doubt check the test log for warnings.

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

## Test Coverage

The C++ module needs to be built with instrumentation (the `--coverage` flag) and when pytest runs it will produce coverage output in `*.gcda` files.

The script from [codecov.io](https://codecov.io/gh/virgesmith/neworder/) uses `gcov` to process the output and upload it. NB it's important to ensure that the `gcc` and `gcov` versions are consistent otherwise it will crash (the ubuntu 20.04 appveyor image defaults to gcc-7 and gcov-9).

## Generating type stubs

Type stubs can be generated for the C++ module using `pybind11-stubgen`, although manual modifications are needed for the output (e.g. docstrings for overloaded functions are misplaced, numpy types need to be fixed).

```sh
pybind11-stubgen _neworder_core
```

It may also be necessary to regenerate type stubs for the submodules, e.g.

```sh
pybind11-stubgen _neworder_core.time
mv stubs/_neworder_core/time-stubs/__init__.pyi neworder/time.pyi
```

## Release Checklist

Development should happen on a release branch (NOT on main). When merged to main, a workflow will automatically bump the version, tag the code, build a package and publish it to PyPI. By default the patch version is bumped but this can be changed in `.github/workflows/pypi-release.yml`. If necessary, this file can be modified to upload a release candidate to `test.pypi.org`.

1. Create some release notes based on commit comments since previous release, e.g.: `git log 0.2.1..HEAD --oneline`
1. Clean, rebuild, test, regenerate examples and code docs: `scripts/code_doc.sh`
1. Regenerate type stubs (see above) as necessary
1. Commit changes to release branch
1. Ensure all checks passing and merge to `main`
1. Update and check conda feedstock (if this doesn't happen automatically, see instructions [here](https://github.com/conda-forge/neworder-feedstock))
1. Install pypi/conda-forge/docker releases in a fresh environment and ensure all is well. If not, fix in a patch release.
1. Create release on github, using the tag and the release notes from above
1. Check zenodo for new DOI and ensure documentation references it.
