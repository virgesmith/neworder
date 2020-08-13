#pragma once

#include "Global.h"
#include "Timeline.h"
#include "MonteCarlo.h"

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <pybind11/embed.h>

#include <random>
#include <string>

namespace no {

// duplication of data between python/C++
// for scalar, either (whichever is the most efficient):
// - access the (immutable?) python vars via m_self->attr(""), or
// - define the var in C++ and provide a python accessor function
// for numpy arrays (and pandas DataFrames), C++ and python ref the same data

struct NEWORDER_EXPORT Environment
{
public:

  // Context
  static const int CPP = 0;
  static const int PY = 1;

  ~Environment();

  // Disable any copy/assignment
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;
  Environment(const Environment&&) = delete;
  Environment& operator=(const Environment&&) = delete;

  // Use this function to create the base environemt
  static Environment& init(int rank, int size, bool indep = true);

  // check for errors in the python env (use after catching py::error_already_set)
  static std::string get_error() noexcept;

  // returns the python version
  static std::string python_version();

  // MPI rank (0 if serial)
  static int rank();

  // MPI size (1 if serial)
  static int size();

  // independent streams (per rank)? 
  static bool indep();

  // logging level
  static bool verbose();

  // returns "py/no rank/size"
  std::string context(int ctx = CPP) const;

  // reset the RNG stream sequence to the original seed 
  static void reset();

  // // Access the RNG stream (one per env)
  // no::MonteCarlo& mc() const;

  // returns the neworder env as a python object 
  operator py::object&() { return *m_self; } 
  operator const py::object&() const { return *m_self; } 

private:

  // flag to check whether init has been called
  bool m_init;

  // global interpreter lock (NB this is a singleton)
  py::scoped_interpreter m_gil; 

  // Singletons only
  Environment();
  friend Environment& Global::instance<Environment>();

  // MPI rank/size
  int m_rank;
  int m_size;

  // MC mode (false means all processes use the same stream)
  bool m_indep;

  // log level
  bool m_verbose;

  // // seed not directly visible to python
  // int64_t m_seed;

  // TODO work out why this segfaults if the dtor is called (even on exit)
  py::module* m_self;
};

// syntactic sugar
Environment& getenv();

}