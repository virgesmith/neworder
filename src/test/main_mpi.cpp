// Entry point for running tests in parallel mode 

#include "run.h"
#include "MPIResource.h"

#include <cstdlib>

int main(int argc, const char *argv[])
{
  MPIResource mpi(&argc, &argv);

  bool indep = std::atoi(argv[1]) == 1;

  return run(mpi.rank(), mpi.size(), indep, argc-2, &argv[2]);
}
