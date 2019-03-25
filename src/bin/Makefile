
INTDIR=int$(SUFFIX)

CXXFLAGS += -I../lib -I../include

src=main$(SUFFIX).cpp run.cpp
dep=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.d))
obj=$(patsubst %,$(INTDIR)/%,$(src:.cpp=.o))

$(info $(dep))
$(info $(obj))
$(shell mkdir -p $(INTDIR))

all: neworder$(SUFFIX)

neworder$(SUFFIX): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder$(SUFFIX)

DEPFLAGS = -MT $@ -MD -MP -MF $(INTDIR)/$*.d

$(INTDIR)/%.o: %.cpp $(INTDIR/%.d)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<

# %.d: %.cpp
# 	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) neworder$(SUFFIX)

.PHONY: clean
