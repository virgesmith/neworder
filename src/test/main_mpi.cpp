// Entry point for running tests in parallel mode 

#include "run.h"
#include "MPIResource.h"
#include "Log.h"

#include <cstdlib>

#include <iostream>

int main(int argc, const char *argv[])
{
  MPIResource mpi(&argc, &argv);

  if (argc < 2 || (argv[1][0] != '0' && argv[1][0] != '1'))
  {
    std::cerr << "usage: %% I [module] [module]...\n"_s % argv[0]
              << "here I is either 1 (independent streams) or 0 (correlated streams) "
              << "and optional [module]s are the names of the modules to be executed." << std::endl;
    exit(1);
  }

  bool indep = std::atoi(argv[1]) == 1;

  return run(mpi.rank(), mpi.size(), indep, argc-2, &argv[2]);
}
