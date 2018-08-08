# override to force a specific python3 version
PYVER=3
# override for 16.04/python3.5: 
BOOST_PYTHON_LIB=boost_python3

# Query python env for compile and link settings
CXXFLAGS = $(shell python$(PYVER)-config --cflags | sed 's/-Wstrict-prototypes//g' | sed 's/-O3//g') -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS += -fPIC -std=c++11 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DBOOST_NO_AUTO_PTR
LDFLAGS := $(shell python$(PYVER)-config --ldflags)

export

all: lib bin

lib: 
	cd src/lib && make

bin: lib
	cd src/bin && make

mpi: lib
	cd src/bin && make neworder_mpi

test: lib 
	cd src/test && make

clean:
	cd src/lib && make clean
	cd src/bin && make clean
	cd src/test && make clean

.PHONY: clean
