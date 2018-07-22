
CXXFLAGS += -I../lib

src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

$(info $(CXXFLAGS))
$(info $(LDFLAGS))

all: test

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