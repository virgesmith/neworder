
CXXFLAGS += -I../lib -I../include

src=neworder$(SUFFIX).cpp run.cpp
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

LDFLAGS += -l$(BOOST_PYTHON_LIB) -l$(BOOST_NUMPY_LIB)

all: neworder$(SUFFIX)

neworder$(SUFFIX): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder$(SUFFIX)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) neworder$(SUFFIX)

.PHONY: clean
