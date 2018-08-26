#pragma once

#include "Global.h"

#include "python.h"

#include <random>
#include <string>

namespace pycpp {

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

  // Use this function to create the environemt
  // it ensures the PRNG has been seeded independently for parallel jobs
  // seeds will be a consecutive numbers starting at a (large prime) multiple of the total no of processes
  static Environment& init(int rank, int size);

  // syntactic sugar
  static Environment& get();

  // check for errors in the python env (use after catching py::error_already_set)
  static std::string get_error() noexcept;

  // returns the python version
  static std::string version();

  // returns "seq-rank/size"
  std::string context(int ctx = CPP) const;

  // set the RNG stream sequence
  void seed(const std::vector<int>& seq);

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
  std::vector<int> m_sequence;
  // MPI rank/size
  int m_rank;
  int m_size;
  // TODO work out why this segfaults if the dtor is called (even on exit)
  py::object* m_self;
  // thread/process-safe seeding
  std::unique_ptr<std::mt19937> m_prng;
};

}