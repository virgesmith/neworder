
CXXFLAGS += -I../lib -I../include

src=neworder.cpp run.cpp
obj=$(src:.cpp=.o)
dep=$(src:.cpp=.d)

mpi_src=neworder_mpi.cpp run.cpp MPIResource.cpp

#$(info $(CXXFLAGS))
#$(info $(LDFLAGS))
MPICXX=mpicxx

LDFLAGS += -l$(BOOST_PYTHON_LIB) -l$(BOOST_NUMPY_LIB)

all: neworder

neworder: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder

neworder_mpi: $(mpi_src)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -L ../lib -lneworder

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)

clean:
	rm -rf $(dep) $(obj) neworder neworder_mpi

.PHONY: clean
