
src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

LDFLAGS += -l$(BOOST_PYTHON_LIB)

all: libneworder.so

libneworder.so: $(obj)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) test1	

.PHONY: clean test
