# override to force a specific python3 version
PYVER=3

# Query python env for compile and link settings
CXXFLAGS = $(shell python$(PYVER)-config --cflags) -I$(shell python$(PYVER) -c "import numpy;print(numpy.get_include())")
CXXFLAGS += -fPIC -std=c++11 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
LDFLAGS := $(shell python$(PYVER)-config --ldflags)

src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))

all: test1

test1: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

test: test1
	PYTHONPATH=. ./test1

clean:
	rm -rf $(dep) $(obj) test1	

.PHONY: clean test