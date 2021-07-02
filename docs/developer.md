# Developer

*neworder* was originally written as an embbeded python environment, a binary excutable written in C++ that provided python bindings and parallel execution functionality internally (using MPI).

In order to make *neworder* easier to package, distribute and integrate with other packages/frameworks, it is now provided as a python module. This means that the MPI functionality is now external, supplied by the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package.

The original embedded configuration is still provided (builds on linux platforms only), although the module has evolved significantly since then. See the "embedded" branch if you're interested.

## Contributions

The source code is on [github](https://github.com/virgesmith/neworder). To contribute, please fork the repository and submit a PR with your changes/additions.

!!! note "Legal"
    Contributors retain copyright on their contributions. When submitting a PR, please add yourself as an additional copyright holder in [LICENCE.md](https://github.com/virgesmith/neworder/LICENCE.md).

The instructions below assume you've already cloned a local copy of the neworder repo.

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
pip install -e .
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

```bash
conda create -q -n neworder-env python=3.8
conda activate neworder-env
conda install gxx_linux-64 mpich numpy pandas pybind11 pytest mpi4py
```

Then, as above

```bash
python setup.py install
```

### Docker

```bash
docker build -t <image-name> .
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it <image-name>
```

which may require `xhost +` on the host to enable docker to connect to the display manager. See `scripts/run_container.sh`.

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
mpiexec -n 2 python -m pytest
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

## Test Coverage

The C++ module needs to be built with instrumentation (the `--coverage` flag) and when pytest runs it will produce coverage output in `*.gcda` files.

The script from [codecov.io](https://codecov.io/gh/virgesmith/neworder/) uses `gcov` to process the output and upload it. NB it's important to ensure that the `gcc` and `gcov` versions are consistent otherwise it will crash (the ubuntu 20.04 appveyor image defaults to gcc-7 and gcov-9).

## Release Checklist

Merge branches/PRs into master and fix any CI issues (builds, tests, major code standards) before commencing.

If necessary, use `test.pypi.org` to upload a release candidate, which can then be installed to a model implementation for testing "in the wild".

1. Create some release notes based on commit comments since previous release, e.g.: `git log 0.2.1..HEAD --oneline`
2. Bump `__version__` in `neworder/__init__.py`
3. Clean, rebuild, test, regenerate examples and code docs: `scripts/code_doc.sh`
4. Commit changes
5. Tag, e.g.: `git tag -a 0.3.0 -m"release v0.3.0"`
6. Push, including tag e.g.: `git push --atomic origin master 0.3.0`
7. Check tagged CI builds and docker image are ok
8. Package and upload to PyPI: `scripts/package.sh`
9. Update and check conda feedstock (if this doesn't happen automatically, see instructions [here](https://github.com/conda-forge/neworder-feedstock))
10. Install pypi/conda-forge/docker releases in a fresh environment and ensure all is well. If not, fix and go back to 2.
11. Create release on github, using the tag and the release notes from above
12. Check zenodo for new DOI and ensure documentation references it.
