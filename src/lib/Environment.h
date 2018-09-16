#pragma once

#include "Global.h"

#include "python.h"
#include "numpy.h"

#include <random>
#include <string>

namespace pycpp {

// TODO no duplication of data between python/C++
// for scalar, either (whichever is the most efficient):
// - always access the (immutable) python vars via m_self->attr(""), or
// - define the var in C++ and provide a python accessor function
// for arrays, C++ and python ref the same data (test this)

struct Environment
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
  static Environment& init(int rank, int size);

  // Apply settings from config.py, including sequence and RNG state(s)
  void configure();

  // check for errors in the python env (use after catching py::error_already_set)
  static std::string get_error() noexcept;

  // returns the python version
  static std::string version();

  // MPI rank (0 if serial)
  int rank() const;

  // MPI size (1 if serial)
  int size() const;

  // returns "seq-rank/size"
  std::string context(int ctx = CPP) const;

  // compute the RNG seed
  int64_t compute_seed() const;

  //
  bool& sync_streams()
  {
    return m_sync_streams;
  }

  int seq()
  {
    return py::extract<int>(m_self->attr("seq"))();
  }

  // TODO rename, refactor sequence
  // set the RNG stream sequence
  void seed(const np::ndarray& seq);

  // iterate the RNG stream sequence
  bool next();

  // Accress the NRG stream (one per env)
  std::mt19937& prng();

  // returns the env as a python object 
  //operator py::object&() { return m_self; } doesnt implicitly cast
  py::object& operator()() { return *m_self; }

private:
  // Singletons only
  Environment();
  friend Environment& Global::instance<Environment>();

  // RNG sequence index
  size_t m_seqno;
  //np::ndarray* m_sequence;

  // MPI rank/size
  int m_rank;
  int m_size;
  // set to true to make all processes use the same seed
  bool m_sync_streams;

  // TODO work out why this segfaults if the dtor is called (even on exit)
  py::object* m_self;
  // thread/process-safe seeding
  std::unique_ptr<std::mt19937> m_prng;
};

// syntactic sugar
Environment& getenv();



}