
# Query python env for compile and link settings
CXXFLAGS = $(shell python3.6-config --cflags)
CXXFLAGS += -fPIC -std=c++11
LDFLAGS := $(shell python3.6-config --ldflags)

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

clean:
	rm -rf $(dep) $(obj) test1	

.PHONY: clean test