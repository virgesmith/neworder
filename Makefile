# Requirements:
# pybind11: install with pip/conda
# virtualenv or conda using python 3.5 or higher
$(error neworder embedded serial build is deprecated)


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
PY_CFG=python3-config
# might need to be "python-config", and
PY_CFG_LINK_ARG=--embed
# might need to be "" on different platforms, override as necessary, e.g. make PY_CFG=python-config PY_CFG_LINK_ARG=

$(info PY_CFG=$(PY_CFG) $(PY_CFG_LINK_ARG))
$(info $(shell $(PY_CFG) --ldflags $(PY_CFG_LINK_ARG)))

#$(info $(shell $(PY_CFG) --ldflags $(PY_CFG_LINK_ARG)))

# Query python env/pybind11 for compile and link settings
CXXFLAGS = $(shell $(PY_CFG) --cflags) 
CXXFLAGS += $(shell python -m pybind11 --includes)
CXXFLAGS += -Werror -Wno-error=deprecated-declarations -fPIC -std=c++17 -pedantic -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DNEWORDER_EMBEDDED
# get version 
CXXFLAGS += -DNEWORDER_VERSION_MAJOR=$(shell cut -d "." -f 1 VERSION) \
            -DNEWORDER_VERSION_MINOR=$(shell cut -d "." -f 2 VERSION) \
            -DNEWORDER_VERSION_PATCH=$(shell cut -d "." -f 3 VERSION)
LDFLAGS := $(shell $(PY_CFG) --ldflags $(PY_CFG_LINK_ARG))

# MPI not enabled
SUFFIX :=
MPIEXEC := 

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
	cp src/lib/libneworder.so /usr/local/lib
	cp src/bin/neworder /usr/local/bin

clean:
	cd src/lib && $(MAKE) clean
	cd src/bin && $(MAKE) clean
	cd src/test && $(MAKE) clean

.PHONY: clean test
