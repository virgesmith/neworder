
src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

CXXFLAGS += -I../include

LDFLAGS += -l$(BOOST_PYTHON_LIB) -l$(BOOST_NUMPY_LIB)

all: libneworder$(SUFFIX).so

libneworder$(SUFFIX).so: $(obj)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) libneworder$(SUFFIX).so	

.PHONY: clean test
