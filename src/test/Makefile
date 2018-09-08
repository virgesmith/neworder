
CXXFLAGS += -I../include -I../lib

LDFLAGS += -l$(BOOST_PYTHON_LIB) -l$(BOOST_NUMPY_LIB)

MPIEXEC = mpirun -n 2

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))

src=$(wildcard test_*.cpp) test1.cpp test$(SUFFIX).cpp
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

all: testharness$(SUFFIX)

testharness$(SUFFIX): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder$(SUFFIX)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

#TODO $(shell ls ../../tests/*.py) ... then strip .py
test: testharness$(SUFFIX)
	PYTHONPATH=.:../../tests:$(PYTHONPATH) LD_LIBRARY_PATH=../lib:$(LD_LIBRARY_PATH) $(MPIEXEC) ./testharness$(SUFFIX) module op mpi

clean:
	rm -rf $(dep) $(obj) testharness$(SUFFIX)

.PHONY: clean test
