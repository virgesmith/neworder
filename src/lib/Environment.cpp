
#include "Environment.h"
#include "Inspect.h"
#include "Module.h"
#include "Log.h"

#include <algorithm>
#include <string>


// This function must be used to init the environment
no::Environment& no::Environment::init(int rank, int size, bool indep)
{
  // make our rank/size visible to python
  Environment& env = no::getenv();

  // // to "share" scalars across C++ and python, use the py::extract<> object
  // env.m_self->attr("rank") = rank;
  // env.m_self->attr("size") = size;
  // These values are 1) set by the C++ runtime; and 2) constant. 
  // python access is by function to avoid exposing a modifiable variable
  env.m_rank = rank;
  env.m_size = size;
  env.m_indep = indep;

  env.m_seed = env.compute_seed();

  no::log("env: seed=%% python %%"_s % env.m_seed % version());

  env.m_prng.seed(env.m_seed);

  // set init flag
  env.m_init = true;
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

// MPI random stream independence (true if serial)
bool no::Environment::indep()
{
  if (!no::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return no::getenv().m_indep;
}

void no::Environment::reset()
{
  if (!no::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  no::getenv().m_prng.seed(no::getenv().m_seed);
}


std::string no::Environment::context(int ctx) const
{
  std::string idstring = "[%% %%/%%] "_s % (ctx == 0 ? "no" : "py") % m_rank % m_size;
  return idstring;
}

// compute the RNG seed
int64_t no::Environment::compute_seed() const
{
  // ensure stream (in)dependence w.r.t. sequence and MPI rank/sizes
  return 77027473 * 0 + 19937 * m_size + m_rank * m_indep;  
}

std::mt19937& no::Environment::prng()
{
  return m_prng;
}

// Note this does not fully initialise, do not construct directly, use the static init function
no::Environment::Environment() : m_init(false) //, m_guard() segfaults on exit (presumably called **too late** (singleton))
{
  py::initialize_interpreter();
  // make the neworder module available in embedded python env
  m_self = new py::module(py::module::import("neworder"));
} 

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

std::string no::Environment::version()
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

