#include "Environment.h"
#include "Inspect.h"
#include "Module.h"
#include "Log.h"

#include <algorithm>
#include <string>


// This function must be used to init the environment
no::Environment& no::Environment::init(int rank, int size, bool indep, bool verbose)
{
  // get static instance
  Environment& env = no::getenv();

  // set whether each process has independent RNG streams
  env.m_indep = indep;

  // verbose flag
  env.m_verbose = verbose;

  // set init flag (before any possible log message)
  env.m_init = true;

  //py::object& neworder = env; // env as python module

  // // to "share" scalars across C++ and python, use the py::extract<> object
  // env.m_self->attr("rank") = rank;
  // env.m_self->attr("size") = size;
  // These values are 1) set by the C++ runtime; and 2) constant. 
  // python access is by function to avoid exposing a modifiable variable
#ifdef NEWORDER_EMBEDDED
  env.m_rank = rank;
  env.m_size = size;
#else
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
    no::log("mpi4py module not found, assuming serial mode");
  }
#endif

#ifdef NEWORDER_EMBEDDED
  no::log("neworder %%/embedded python %% env={indep:%%, verbose:%%}"_s % module_version() % python_version() % env.m_indep % env.m_verbose );
#else
  no::log("neworder %%/module python %% env={indep:%%, verbose:%%}"_s % module_version() % python_version() % env.m_indep % env.m_verbose );
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
  if (!no::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return no::getenv().m_rank;
}

// MPI size (1 if serial)
int no::Environment::size()
{
  if (!no::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return no::getenv().m_size;
}

// parallel random stream independence (true if serial)
bool no::Environment::indep()
{
  if (!no::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return no::getenv().m_indep;
}

bool no::Environment::verbose()
{
  if (!no::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return no::getenv().m_verbose;
}

std::string no::Environment::context(int ctx) const
{
  std::string idstring = "[%% %%/%%] "_s % (ctx == 0 ? "no" : "py") % m_rank % m_size;
  return idstring;
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
no::Environment::Environment() : m_init(false) 
{
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
