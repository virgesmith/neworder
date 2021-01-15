#!/bin/bash

# For use in CI scripts for conda builds. PYTHON env var must be set, to e.g. "3.8"

wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p ${HOME}/.miniconda
source "${HOME}/.miniconda/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n conda-env python=$PYTHON gxx_linux-64 mpich numpy pandas pybind11 pytest mpi4py
