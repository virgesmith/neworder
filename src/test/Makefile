INTDIR=int$(SUFFIX)

CXXFLAGS += -I../include -I../lib

LDFLAGS += -l$(BOOST_PYTHON_LIB) -l$(BOOST_NUMPY_LIB)

src=test1.cpp test_no.cpp test_np.cpp test_errors.cpp test$(SUFFIX).cpp
dep=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.d))
obj=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.o))

#$(info $(dep))
#$(info $(obj))
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
	PYTHONPATH=.:../../tests:$(PYTHONPATH) LD_LIBRARY_PATH=../lib:$(LD_LIBRARY_PATH) $(MPIEXEC) ./testharness$(SUFFIX) module op mpi

clean:
	rm -rf $(dep) $(obj) testharness$(SUFFIX)

.PHONY: clean test
