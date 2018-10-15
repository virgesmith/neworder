
#include "Environment.h"
#include "Inspect.h"
#include "Module.h"
#include "Log.h"

#include <algorithm>
#include <string>


// This function must be used to init the environment
neworder::Environment& neworder::Environment::init(int rank, int size, bool indep)
{
  // make our rank/size visible to python
  Environment& env = neworder::getenv();

  // // to "share" scalars across C++ and python, use the py::extract<> object
  // env.m_self->attr("rank") = rank;
  // env.m_self->attr("size") = size;
  // These values are 1) set by the C++ runtime; and 2) constant. 
  // python access is by function to avoid exposing a modifiable variable
  env.m_rank = rank;
  env.m_size = size;
  env.m_indep = indep;

  env.m_seed = env.compute_seed();

  neworder::log("env: seed=%% python %%"_s % env.m_seed % version());

  env.m_prng.seed(env.m_seed);

  // set init flag
  env.m_init = true;
  return env;
}


// syntactic sugar
neworder::Environment& neworder::getenv()
{
  return Global::instance<Environment>();
}

// MPI rank (0 if serial)
int neworder::Environment::rank()
{
  if (!neworder::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return neworder::getenv().m_rank;
}

// MPI size (1 if serial)
int neworder::Environment::size()
{
  if (!neworder::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return neworder::getenv().m_size;
}

// MPI random stream independence (true if serial)
bool neworder::Environment::indep()
{
  if (!neworder::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  return neworder::getenv().m_indep;
}

void neworder::Environment::reset()
{
  if (!neworder::getenv().m_init)
    throw std::runtime_error("accessing %% before init called"_s % __FUNCTION__);
  neworder::getenv().m_prng.seed(neworder::getenv().m_seed);
}


std::string neworder::Environment::context(int ctx) const
{
  std::string idstring = "[%% %%/%%] "_s % (ctx == 0 ? "no" : "py") % m_rank % m_size;
  return idstring;
}

// compute the RNG seed
int64_t neworder::Environment::compute_seed() const
{
  // ensure stream (in)dependence w.r.t. sequence and MPI rank/sizes
  return 77027473 * 0 + 19937 * m_size + m_rank * m_indep;  
}

std::mt19937& neworder::Environment::prng()
{
  return m_prng;
}

// Note this does not fully initialise, do not construct directly, use the static init function
neworder::Environment::Environment() : m_init(false) //: m_sequence(neworder::zero_1d_array<int64_t>(1))
{
  // make the neworder module available in embedded python env
  neworder::import_module();

  // Init python env
  Py_Initialize();

  // init numpy
  np::initialize();

  m_self = new py::object(py::import("neworder"));
  
  // dummy sequence (needs to be read from config.py - which hasnt been loaded yet)
  // m_self->attr("sequence") = neworder::zero_1d_array<int64_t>(1);
  // m_self->attr("seq") = 0;
} 

neworder::Environment::~Environment() 
{
  // Python >=3.6
  // if (Py_FinalizeEx() < 0)
  // {
  //   // report an error...somehow
  // }
  Py_Finalize();
}

// check for errors in the python env: if it returns, there is no error
// copied from: https://wiki.python.org/moin/boost.python/EmbeddingPython
std::string neworder::Environment::get_error() noexcept
{
  // see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
  std::string message;
  if (PyErr_Occurred())
  {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_NormalizeException(&exc, &val, &tb);

    py::handle<> hexc(exc), hval(py::allow_null(val)), htb(py::allow_null(tb));

    PyErr_Clear();
    if(!hval)
    {
      return py::extract<std::string>(py::str(hexc));
    }
    else
    {
      py::object traceback(py::import("traceback"));
      py::object format_exception(traceback.attr("format_exception"));
      py::object formatted_list(format_exception(hexc,hval,htb));
      py::object formatted(py::str("").join(formatted_list));
      return py::extract<std::string>(formatted);
    } 
  }
  return "unable to determine python error";
}

std::string neworder::Environment::version()
{
  static std::string version_string;
  // Get and display python version - only do once
  if (version_string.empty())
  {
    py::object sys = py::import("sys");
    version_string = py::extract<std::string>(sys.attr("version"));
    std::replace(version_string.begin(), version_string.end(), '\n', ' ');
  }
  return version_string;
}

