
#include "run.h"

#include "Inspect.h"
#include "Environment.h"
#include "Functor.h"
#include "Callback.h"

#include "python.h"

#include <map>
#include <iostream>
#include <cstdlib>

// TODO Logger...?
namespace no = neworder;

int main(int argc, const char* argv[])
{
  // Directory containing model (config.py, etc) is specified on the command line
  // It's added to PYTHONPATH
  if (argc != 2)
  {
    std::cerr << "usage: neworder <model-path>\n"
              << "where <model-path> is a directory containing the model config (config.py) plus the model definition python files" << std::endl;
    exit(1);
  }
  append_model_path(argv[1]);
  
  // single-process
  return run(0, 1);
}

