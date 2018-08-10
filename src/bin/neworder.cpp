
#include "run.h"

#include <iostream>

// TODO Logger...?

int main(int argc, const char* argv[])
{
  // Directory containing model (config.py, etc) is specified on the command line
  // It's added to PYTHONPATH
  if (argc < 2)
  {
    std::cerr << "usage: neworder <model-path> [<extra_path>...]\n"
              << "where <model-path> is a directory containing the model config (config.py) plus the model definition python files\n" 
              << "and <extra-path> is an option directory containing any other modules required by the model." << std::endl;
    exit(1);
  }
  append_model_paths(&argv[1], argc-1);
  
  // single-process
  return run(0, 1);
}

