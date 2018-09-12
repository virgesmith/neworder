// Entry point for running tests in serial mode 

#include "run.h"

int main(int argc, const char* argv[]) 
{
  return run(0, 1, argc-1, &argv[1]);
}