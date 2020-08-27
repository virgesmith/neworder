#include "Environment.h"
#include "Inspect.h"
#include "Module.h"
#include "Log.h"

#include <algorithm>
#include <string>

// This function must be used to init the embedded environment
no::Environment& no::Environment::init(int rank, int size, bool verbose)
{
  // get static instance
  Environment& env = no::getenv();

  // verbose flag
  env.m_verbose = verbose;

#ifdef NEWORDER_EMBEDDED
  // set init flag (before any possible log message)
  env.m_rank = rank;
  env.m_size = size;
  no::log("neworder %%/embedded python %%"_s % module_version() % python_version());
#else
  // this hangs on throw when this code is in the (global singleton) constructor
  try 
  {
    py::module mpi = py::module::import("mpi4py.MPI");
    py::object comm = mpi.attr("COMM_WORLD");
    env.m_rank = comm.attr("Get_rank")().cast<int>();
    env.m_size = comm.attr("Get_size")().cast<int>();
  }
  catch(const py::error_already_set&)
  {
    env.m_rank = 0;
    env.m_size = 1;
    // override verbose 
    env.get_error(); // flush the error
    no::log("WARNING: mpi4py module not found, assuming serial mode", true);
  }
#endif
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

std::string no::Environment::context(no::Environment::Context ctx) const
{
  // construct strings once 
  static const std::string idstrings[2] = {"[no %%/%%] "_s % m_rank % m_size, "[py %%/%%] "_s % m_rank % m_size};
  return idstrings[(int)ctx];
}


#ifdef NEWORDER_EMBEDDED
// Note this does not fully initialise, do not construct directly, use the static init function
no::Environment::Environment() : m_init(false) , m_gil{} // TODO does it still? segfaults on exit (presumably called **too late** (singleton))
{
  // make the neworder module available in embedded python env
  m_self = new py::module(py::module::import("neworder"));
  // add to sys to ensure neworder module can be resolved
  py::dict sys_modules = py::module::import("sys").attr("modules");
  sys_modules["neworder"] = *m_self;
} 
#else
no::Environment::Environment() : m_init(false), m_verbose(true) 
{
  m_verbose = false;
} 
#endif

no::Environment::~Environment() 
{
}

// check for errors in the python env: if it returns, there is no error
// copied from: https://wiki.python.org/moin/boost.python/EmbeddingPython
std::string no::Environment::get_error() noexcept
{
  // see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
  std::string message;
  if (PyErr_Occurred())
  {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_NormalizeException(&exc, &val, &tb);

    py::handle hexc(exc), hval(/*py::allow_null*/(val)), htb(/*py::allow_null*/(tb));

    PyErr_Clear();
    if(!hval)
    {
      return py::str(hexc);
    }
    else
    {
      py::module traceback("traceback");
      py::object format_exception(traceback.attr("format_exception"));
      py::object formatted_list(format_exception(hexc,hval,htb));
      py::object formatted(py::str("")/*.join(formatted_list)*/);
      return formatted.cast<std::string>();
    } 
  }
  return "unable to determine python error";
}

std::string no::Environment::python_version()
{
  static std::string version_string;
  // Get and display python version - only do once
  if (version_string.empty())
  {
    py::module sys = py::module::import("sys");
    version_string = sys.attr("version").cast<std::string>();
    std::replace(version_string.begin(), version_string.end(), '\n', ' ');
  }
  return version_string;
}
