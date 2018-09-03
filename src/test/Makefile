
CXXFLAGS += -I../include -I../lib

LDFLAGS += -l$(BOOST_PYTHON_LIB) -l$(BOOST_NUMPY_LIB)

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))

src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

all: testharness

testharness: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

#TODO $(shell ls ../../tests/*.py) ... then strip .py
test: testharness
	PYTHONPATH=.:../../tests:$(PYTHONPATH) LD_LIBRARY_PATH=../lib:$(LD_LIBRARY_PATH) ./testharness module op

clean:
	rm -rf $(dep) $(obj) testharness

.PHONY: clean test
