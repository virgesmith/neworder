# override to force a specific python3 version
PYVER=3


all: lib

lib: 
	cd src/lib && make PYVER=$(PYVER)

test:
	cd src/test && make PYVER=$(PYVER) && make test
