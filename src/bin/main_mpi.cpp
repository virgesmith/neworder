// MPI entry point

#include "run.h"
#include "MPIResource.h"

#include <iostream>

int main(int argc, const char* argv[])
{
  // Directory containing model (config.py, etc) is specified on the command line
  // It's added to PYTHONPATH
  if (argc < 3)
  {
    std::cerr << "usage: neworder_mpi I <model-path>\n"
              << "where I is either 1 (independent random streams) or 0 (identical random streams) "
              << "and <model-path> is a directory containing the model config (config.py) plus the model definition python files "
              << "where <model-path> is a directory containing the model config (config.py) plus the model definition python files " 
              << "and <extra-path> is an option directory containing any other modules required by the model." << std::endl;
    exit(1);
  }

  append_model_paths(&argv[1], argc-1);

  MPIResource mpi(&argc, &argv);

  run(mpi.rank(), mpi.size(), true);
}