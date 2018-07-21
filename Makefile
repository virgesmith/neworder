# override to force a specific python3 version
PYVER=3

# Query python env for compile and link settings
CXXFLAGS = $(shell python$(PYVER)-config --cflags | sed 's/-Wstrict-prototypes//g') -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS += -fPIC -std=c++11 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
LDFLAGS := $(shell python$(PYVER)-config --ldflags)

export

all: lib bin

lib: 
	cd src/lib && make

bin:
	cd src/bin && make

test:
	cd src/test && make && make test

clean:
	cd src/lib && make clean
	cd src/bin && make clean
	cd src/test && make clean

.PHONY: clean