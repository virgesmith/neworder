// MPI entry point

#include "run.h"
#include "MPIResource.h"

#include <iostream>

int main(int argc, const char* argv[])
{
  // Directory containing model (config.py, etc) is specified on the command line
  // It's added to PYTHONPATH
  if (argc < 4)
  {
    std::cerr << "usage: neworder_mpi N I <model-path>\n"
              << "where N is total number of processes, I is either 1 (independent streams) or 0 (correlated streams) "
              << "and <model-path> is a directory containing the model config (config.py) plus the model definition python files "
              << "where <model-path> is a directory containing the model config (config.py) plus the model definition python files " 
              << "and <extra-path> is an option directory containing any other modules required by the model." << std::endl;
    exit(1);
  }
  bool indep = std::atoi(argv[1]) == 1;

  append_model_paths(&argv[2], argc-2);

  MPIResource mpi(&argc, &argv);

  run(mpi.rank(), mpi.size(), indep, true);
}