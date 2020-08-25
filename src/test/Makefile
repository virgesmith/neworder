INTDIR=int$(SUFFIX)

CXXFLAGS += -I../include -I../lib

src=$(wildcard test*.cpp) run.cpp main$(SUFFIX).cpp
dep=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.d))
obj=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.o))

$(shell mkdir -p $(INTDIR))

all: testharness$(SUFFIX)

testharness$(SUFFIX): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder$(SUFFIX)

DEPFLAGS = -MT $@ -MD -MP -MF $(INTDIR)/$*.d

$(INTDIR)/%.o: %.cpp $(INTDIR/%.d)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<

# %.d: %.cpp
# 	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

#TODO $(shell ls ../../tests/*.py) ... then strip .py
test: testharness$(SUFFIX)
	PYTHONPATH=.:../../tests:$(PYTHONPATH) LD_LIBRARY_PATH=../lib:$(LD_LIBRARY_PATH) $(MPIEXEC) ./testharness$(SUFFIX) $(MPI_INDEP) module op mpi trans mc correlated

clean:
	rm -rf $(dep) $(obj) testharness$(SUFFIX)

.PHONY: clean test
