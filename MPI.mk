# Requirements:
# MPI: mpich and openmpi known to work
# pybind11: install with pip/conda
# virtualenv or conda using python 3.5 or higher
ifndef VIRTUAL_ENV
ifndef CONDA_DEFAULT_ENV
$(error neworder requires either a virtualenv or a conda environment)
else
$(info Building in conda env: $(CONDA_DEFAULT_ENV))
endif
else
$(info Building in virtualenv: $(VIRTUAL_ENV))
endif

# python3.8 on ubuntu 20.04 (but not on travis) needs an extra arg "--embed" to resolve lib deps correctly, but this arg breaks previous versions
PY_CFG_EXTRA_ARG =
#PY_CFG_EXTRA_ARG += --embed
PY_CFG=python3-config

# also set (CXXFLAGS below) NEWORDER_MPI to prevent skipping of MPI-specific code
CXX=mpic++
SUFFIX := _mpi
MPIEXEC := mpirun -n 2
# test flags for RNG stream (in)dependence in MPI mode
MPI_INDEP := 1
MPI_DEP := 0

# Query python env for compile and link settings
CXXFLAGS = $(shell $(PY_CFG) --cflags) 
CXXFLAGS += $(shell python -m pybind11 --includes)
CXXFLAGS += -Werror -Wno-error=deprecated-declarations -fPIC -std=c++17 -pedantic -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DNEWORDER_MPI
# get version from __init__.py
CXXFLAGS += -DNEWORDER_VERSION_MAJOR=$(shell python3 -c "import neworder;print(neworder.__version__.split('.')[0])") \
            -DNEWORDER_VERSION_MINOR=$(shell python3 -c "import neworder;print(neworder.__version__.split('.')[1])") \
            -DNEWORDER_VERSION_PATCH=$(shell python3 -c "import neworder;print(neworder.__version__.split('.')[2])")
LDFLAGS := $(shell $(PY_CFG) --ldflags $(PY_CFG_EXTRA_ARG))

export

all: lib bin

lib: 
	+cd src/lib && $(MAKE)

bin: lib
	+cd src/bin && $(MAKE)
	+cd src/test && $(MAKE)

test: bin 
	+cd src/test && $(MAKE) test

install:
	cp src/lib/libneworder_mpi.so /usr/local/lib
	cp src/bin/neworder_mpi /usr/local/bin

clean:
	cd src/lib && $(MAKE) clean
	cd src/bin && $(MAKE) clean
	cd src/test && $(MAKE) clean

.PHONY: clean test
