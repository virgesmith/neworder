# override to force a specific python3 version
PYVER=3
# override for 16.04/python3.5: 
BOOST_PYTHON_LIB=boost_python3

# Query python env for compile and link settings
CXXFLAGS = $(shell python$(PYVER)-config --cflags | sed 's/-Wstrict-prototypes//g' | sed 's/-O3//g') -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS += -fPIC -std=c++11 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DBOOST_NO_AUTO_PTR
LDFLAGS := $(shell python$(PYVER)-config --ldflags)

export

all: lib bin mpi test

lib: 
	+cd src/lib && $(MAKE)

bin: lib
	+cd src/bin && $(MAKE)

mpi: lib
	+cd src/bin && $(MAKE) neworder_mpi

test: lib 
	+cd src/test && $(MAKE)

clean:
	cd src/lib && $(MAKE) clean
	cd src/bin && $(MAKE) clean
	cd src/test && $(MAKE) clean

.PHONY: clean
