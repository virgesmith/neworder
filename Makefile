# Requirements:
# pybind11: install with pip

# override to force a specific python3 version
PYVER=3

# Query python env for compile and link settings
#CXXFLAGS = $(shell python$(PYVER)-config --cflags | sed 's/-Wstrict-prototypes//g' | sed 's/-O3//g') -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS = $(shell python3 -m pybind11 --includes)
CXXFLAGS += -O2 -Werror -Wno-error=deprecated-declarations -fPIC -std=c++14 -pedantic
CXXFLAGS += -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
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
