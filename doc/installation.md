# Installation

`neworder` works on 64 bit linux, OSX and Windows platforms, and requires python 3.6 or higher.

<!-- vscode-markdown-toc -->
* [Requirements](#Requirements)
	* [Install Dependencies](#InstallDependencies)
		* [Pip/virtualenv](#Pipvirtualenv)
		* [Conda](#Conda)
* [Build and Test](#BuildandTest)
	* [Standard Build](#StandardBuild)
	* [Parallel Build](#ParallelBuild)
	* [Run Examples](#RunExamples)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Requirements'></a>Requirements

### <a name='InstallDependencies'></a>Install Dependencies

#### <a name='Pipvirtualenv'></a>Pip/virtualenv

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
#### <a name='Conda'></a>Conda

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


## <a name='BuildandTest'></a>Build and Test

First clone (or fork) the repo, then enter the repo's root directory, e.g.:
```bash
(<env>) $ git clone git@github.com:virgesmith/neworder
(<env>) $ cd neworder
```

### <a name='StandardBuild'></a>Standard Build

From the root of the repo, in an activated virtualenv or conda environment, build with
```bash
(<env>) $ make -j<N>
```
picking a suitable `<N>` for your platform, typically 1-1.5x number of cores. To test:
```
(<env>) $ make test
```

### <a name='ParallelBuild'></a>Parallel Build

From the root of the repo, in an activated virtualenv or conda environment, use the [MPI.mk](MPI.mk) makefile to build the MPI-enabled framework:
```bash
(<env>) $ make -j<N> -f MPI.mk
```
picking an suitable <N> for your platform, typically 1-1.5x number of cores. And to test,
```bash
(<env>) $ make -f MPI.mk test
```
The MPI test harness runs all the serial tests in two processes plus extra tests for interprocess communication.

### <a name='RunExamples'></a>Run Examples

Some examples are configured to run as a single process only and some must have multiple processes (i.e. MPI). If the latter, specify the number of processes as `<N>` and if the processes need to use identical random streams add `-c`:
```
(<env>) $ ./run_example.sh <name> [<N> [ -c]]
```
where `<name>` is the name of the example, e.g. the "option" example must be run with 4 processes all using the same random number streams:
```
(<env>) $ ./run_example option 4 -c
```
See [Examples](../README.md#examples) for more detail.

# Documentation

