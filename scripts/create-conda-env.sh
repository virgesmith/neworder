#!/bin/bash

# For use in CI scripts for conda builds. PYTHON env var must be set, to e.g. "3.8"

wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda update -q conda
conda install -q conda-build
conda create -q -n conda-env python=$PYTHON
conda activate conda-env
conda install gxx_linux-64 mpich numpy pandas pybind11 pytest mpi4py