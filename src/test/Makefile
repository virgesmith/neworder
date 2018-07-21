# override to force a specific python3 version
PYVER=3

# Query python env for compile and link settings
CXXFLAGS = $(shell python$(PYVER)-config --cflags | sed s/-Wstrict-prototypes//) -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS += -fPIC -std=c++11 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION  -I../lib
LDFLAGS := $(shell python$(PYVER)-config --ldflags)

src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))

all: testharness

testharness: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

test: testharness
	PYTHONPATH=. LD_LIBRARY_PATH=../lib ./testharness

clean:
	rm -rf $(dep) $(obj) test

.PHONY: clean test