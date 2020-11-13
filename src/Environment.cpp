#include "Environment.h"
#include "Module.h"
#include "Log.h"

#include <algorithm>
#include <string>

// This function must be used to init the environment
no::Environment& no::Environment::init(int rank, int size, bool verbose, bool checked)
{
  // get static instance
  Environment& env = no::getenv();

  // verbose flag
  env.m_verbose = verbose;

  // checks
  env.m_checked = checked;

  // this hangs on throw when this code is in the (global singleton) constructor
  try
  {
    py::module mpi = py::module::import("mpi4py.MPI");
    py::object comm = mpi.attr("COMM_WORLD");
    env.m_rank = comm.attr("Get_rank")().cast<int>();
    env.m_size = comm.attr("Get_size")().cast<int>();
  }
  catch(const py::error_already_set& pyerror)
  {
    if (!pyerror.matches(PyExc_ModuleNotFoundError))
    {
      throw;
    }
    env.m_rank = 0;
    env.m_size = 1;
    // override verbose
    no::warn("mpi4py module not found, assuming serial mode");
  }

  env.m_uniqueIndex = static_cast<int64_t>(env.m_rank);

  return env;
}


// syntactic sugar
no::Environment& no::getenv()
{
  return Global::instance<Environment>();
}

// MPI rank (0 if serial)
int no::Environment::rank()
{
  return no::getenv().m_rank;
}

// MPI size (1 if serial)
int no::Environment::size()
{
  return no::getenv().m_size;
}

void no::Environment::verbose(bool b)
{
  no::getenv().m_verbose = b;
}

void no::Environment::checked(bool b)
{
  no::getenv().m_checked = b;
}

std::string no::Environment::context(no::Environment::Context ctx) const
{
  // construct strings once
  static const std::string idstrings[2] = {"[no %%/%%] "s % m_rank % m_size, "[py %%/%%] "s % m_rank % m_size};
  return idstrings[(int)ctx];
}

int64_t no::Environment::unique_index()
{
  int64_t current = m_uniqueIndex;
  m_uniqueIndex += m_size;
  return current;
}

no::Environment::Environment() : m_rank(-1), m_size(-1), m_verbose(false), m_checked(false), m_halt(false)
{
  // Note: m_unique_id is only set when MPI env is resolved
}

no::Environment::~Environment()
{
}
