// Entry point for running tests in parallel mode 

#include "run.h"
#include "MPIResource.h"

int main(int argc, const char *argv[])
{
  MPIResource mpi(&argc, &argv);

  return run(mpi.rank(), mpi.size(), argc-1, &argv[1]);
}
