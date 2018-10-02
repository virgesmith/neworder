# override to force a specific python3 version
PYVER=3

# override for 16.04/python3.5: 
BOOST_PYTHON_LIB=boost_python3
BOOST_NUMPY_LIB=boost_numpy3-py36
#BOOST_PYTHON_LIB=boost_python36
#BOOST_NUMPY_LIB=boost_numpy36
# override for custom boost lib/location
BOOST_EXTRA_CXXFLAGS=
BOOST_EXTRA_LDFLAGS=


# Query python env for compile and link settings
CXXFLAGS = $(shell python$(PYVER)-config --cflags | sed 's/-Wstrict-prototypes//g' | sed 's/-O3//g') -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS += -O2 -Werror -Wno-error=deprecated-declarations -fPIC -std=c++14 -pedantic
CXXFLAGS += -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DBOOST_NO_AUTO_PTR $(BOOST_EXTRA_CXXFLAGS)
LDFLAGS := $(shell python$(PYVER)-config --ldflags) $(BOOST_EXTRA_LDFLAGS)

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
