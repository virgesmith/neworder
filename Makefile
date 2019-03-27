# Requirements:
# pybind11: install with pip/conda

# override to force a specific python3 version
PYVER=3

# Query python env/pybind11 for compile and link settings
CXXFLAGS = $(shell python3 -m pybind11 --includes)
CXXFLAGS += -O2 -Werror -Wno-error=deprecated-declarations -fPIC -std=c++14 -pedantic -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# get version from __init__.py
CXXFLAGS += -DNEWORDER_VERSION_MAJOR=$(shell python3 -c "import neworder;print(neworder.__version__.split('.')[0])") \
            -DNEWORDER_VERSION_MINOR=$(shell python3 -c "import neworder;print(neworder.__version__.split('.')[1])") \
            -DNEWORDER_VERSION_PATCH=$(shell python3 -c "import neworder;print(neworder.__version__.split('.')[2])")
LDFLAGS := $(shell python$(PYVER)-config --ldflags) 

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
