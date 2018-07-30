
CXXFLAGS += -I../lib

src=main.cpp run.cpp
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))
MPICXX=mpicxx

LDFLAGS += -l$(BOOST_PYTHON_LIB)

all: neworder

neworder: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder

neworder_mpi: mpi_main.cpp run.cpp MPIResource.h
	$(MPICXX) $(CXXFLAGS) -o $@ mpi_main.cpp run.cpp $(LDFLAGS) -L ../lib -lneworder

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) test

.PHONY: clean 