
CXXFLAGS += -I../lib

src=$(wildcard *.cpp)
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))

all: neworder

neworder: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) test

.PHONY: clean 