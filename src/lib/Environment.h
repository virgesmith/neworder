#pragma once

#include "Global.h"
#include "Timeline.h"

#include "NewOrder.h"
#include "numpy.h"

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

  // returns "py/no rank/size"
  std::string context(int ctx = CPP) const;

  // reset the RNG stream sequence to the original seed 
  static void reset();

  // Accress the NRG stream (one per env)
  std::mt19937& prng();

  // returns the neworder env as a python object 
  operator py::object&() { return *m_self; } 
  operator const py::object&() const { return *m_self; } 

  no::Timeline& timeline();

private:

  // TODO reinstate when this is no longer static lifetime
  //py::scoped_interpreter m_guard; // start the interpreter and keep it alive

  // flag to check whether init has been called
  bool m_init;

  // Singletons only
  Environment();
  friend Environment& Global::instance<Environment>();

  // RNG sequence index
  //size_t m_seqno; use python version for now
  //np::array m_sequence;

  // MPI rank/size
  int m_rank;
  int m_size;
  // set to false to make all processes use the same seed
  bool m_indep;

  // seed not directly visible to python
  int64_t m_seed;

  // TODO work out why this segfaults if the dtor is called (even on exit)
  py::module* m_self;
  // thread/process-safe seeding strategy deferred until config loaded
  std::mt19937 m_prng;

  // pointer to python-instantiated object
  no::Timeline* m_timeline;
};

// syntactic sugar
Environment& getenv();

}