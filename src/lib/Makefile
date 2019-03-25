
INTDIR=int$(SUFFIX)

src=$(wildcard *.cpp)
dep=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.d))
obj=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.o))

#$(info $(dep))
#$(info $(obj))
$(shell mkdir -p $(INTDIR))

CXXFLAGS += -I../include

all: libneworder$(SUFFIX).so

libneworder$(SUFFIX).so: $(obj)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

DEPFLAGS = -MT $@ -MD -MP -MF $(INTDIR)/$*.d

$(INTDIR)/%.o: %.cpp $(INTDIR/%.d)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<
#-c -o $@ $^

#$(INTDIR)/%.d: %.cpp
#	$(CXX) $(CXXFLAGS) $< -MM -MT $(INTDIR)/$(@:.d=.o) >$@

# .PRECIOUS = $(INTDIR)/%.d
# $(INTDIR)/%.d: ;

-include $(dep)

clean:
	rm -rf $(dep) $(obj) libneworder$(SUFFIX).so	

.PHONY: clean test
