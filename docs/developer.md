# Developer

*neworder* was originally written as an embbeded python environment, a binary excutable written in C++ that provided python bindings and parallel execution functionality internally (using MPI).

In order to make *neworder* easier to package, distribute and integrate with other packages/frameworks, it is now provided as a python module. This means that the MPI functionality is now external, supplied by the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) package.

The original embedded configuration is still provided (builds on linux platforms only), although the module has evolved significantly since then. See the "embedded" branch if you're interested.

## Contributions

To contribute, please submit a pull request. More information on how to do this [here](./contributing.md).

!!! note "Legal"
    Contributors retain copyright on their substantial contributions. If applicable, when submitting a PR, please add yourself as an additional copyright holder in [LICENCE.md](https://github.com/virgesmith/neworder/LICENCE.md).

The instructions below assume you've already forked and cloned a local copy of the neworder repo.

## Development environment

See [system requirements](index.md#system-requirements) and use:

```bash
pip install -e .[dev] # add ,parallel,geospatial as necessary
```

If you want to use a specific compiler you can do something like this:

```bash
export CC=clang
pip install -ve .[dev]
```

And a simple test that all is ok:

```bash
python -c "import neworder"
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

Type stubs can be generated for the C++ module using `pybind11-stubgen`, although manual modifications may be needed, plus numpy types need to be fixed globally.

```sh
pybind11-stubgen _neworder_core --ignore-invalid all
```

It struggles to understand a default argument that is a function, so requires the the `--ignore-invalid` flag. It may also be necessary to regenerate type stubs for the submodules, e.g.

```sh
pybind11-stubgen _neworder_core.time
mv stubs/_neworder_core/time-stubs/__init__.pyi neworder/time.pyi
```

## Release Checklist

!!! Note "Development Process"
    Development follows the typical cycle of issues :material-arrow-right: PR with CI :material-arrow-right: main
    ensuring code quality/correctness. Remember that type stubs may need to be regenerated if any changes are made to
    the extension module.


When a release is ready:

1. Ensure version in pyproject.toml has been updated (to say `X.Y.Z`)
1. Create a release in github, using a new tag `vX.Y.Z` and release notes based on commits since previous release, e.g.: `git log 1.2.1..HEAD --oneline`. CI will then:
    - publish the release to PYPI
    - generate examples artifacts - these should be copied to the release
1. [Currently manual] build and push the docker image (NB uses latest published release, but local examples)

Note:

- [neworder.readthedocs.io](https://neworder.readthedocs.io) should default "stable" to the latest tag. ("latest" will point to main)
- [zenodo](https://zenodo.org/) should automatically see the new tag too
