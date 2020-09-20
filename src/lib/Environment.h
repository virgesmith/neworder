#pragma once

#include "Global.h"
#include "Timeline.h"
#include "MonteCarlo.h"

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <random>
#include <string>

namespace no {

class NEWORDER_EXPORT Environment
{
public:

  // Context, for logging
  enum class Context { CPP, PY };

  ~Environment();

  // Disable any copy/assignment
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;
  Environment(const Environment&&) = delete;
  Environment& operator=(const Environment&&) = delete;

  // initialises the environment
  static Environment& init(int rank, int size, bool verbose = true, bool checked = true);

  // check for errors in the python env (use after catching py::error_already_set)
  static std::string get_error() noexcept;

  // MPI rank (0 if serial)
  static int rank();

  // MPI size (1 if serial)
  static int size();

  // set logging on/off
  static void verbose(bool b = true);

  // set whether checks are called
  static void checked(bool b = true);

  // return a unique value (for e.g. dataframe indices)
  int64_t unique_index();

  // returns "py/no rank/size"
  std::string context(Context ctx = Context::CPP) const;

private:

  // Singletons only
  Environment();
  friend Environment& Global::instance<Environment>();

  friend class Model;

  friend void log(const std::string& msg, bool);
  friend NEWORDER_EXPORT void log(const py::handle& msg, bool);

  // MPI rank/size
  int m_rank;
  int m_size;

  // log level
  bool m_verbose;

  // check mode flag
  bool m_checked;

  // unique counter
  int64_t m_uniqueIndex;

  // (non-error condition) halt flag
  bool m_halt;
};

// syntactic sugar
Environment& getenv();

}