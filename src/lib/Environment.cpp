
#include "Environment.h"
#include "Inspect.h"
#include "Callback.h"

#include <algorithm>
#include <string>
#include <iostream>

// This function must be used to init the environment
pycpp::Environment& pycpp::Environment::init(int rank, int size)
{
  Environment& env = Global::instance<Environment>();
  env.m_rank = rank;
  env.m_size = size;
  env.m_self->attr("procid") = rank;
  env.m_self->attr("nprocs") = size;

  // TODO python func to set sequence and reset rng
  // // config is not loaded yet...
  // if (!pycpp::has_attr(*env.m_self, "sequence"))
  // {
  //   env.m_self->attr("sequence") = std::vector<int>(1, 0);      
  // }
  // TODO is there a way to directly ref the python-created C++ object (as a vector<int>&)? (extract appears to copy)
  //py::extract<std::vector<int>>(env.m_self->attr("sequence"));

  // Default sequence, may be overridden
  env.m_sequence = std::vector<int>(1, 0);
  env.m_seqno = 0;

  // Init rng
  env.m_prng.reset(new std::mt19937(77027473 * env.m_sequence[env.m_seqno] + 19937 * size + rank));

  std::cout << env.context() << "env init" << std::endl; 
  std::cout << env.context() << "embedded python version: " << version() << std::endl;

  return env;
}

// syntactic sugar
pycpp::Environment& pycpp::Environment::get()
{
  return Global::instance<Environment>();
}

const std::string& pycpp::Environment::context(int ctx) const
{
  static const std::string idstring_cpp = "[no " + std::to_string(m_sequence[m_seqno]) + "-" + std::to_string(m_rank) + "/" + std::to_string(m_size) + "] ";
  static const std::string idstring_py = "[py " + std::to_string(m_sequence[m_seqno]) + "-" + std::to_string(m_rank) + "/" + std::to_string(m_size) + "] ";
  return ctx == 0 ? idstring_cpp : idstring_py;
}

// Take next stream
bool pycpp::Environment::next()
{
  ++m_seqno;
  if (m_seqno == m_sequence.size())
    return false;

  // ensure stream indepence w.r.t. sequence and MPI rank/size
  m_prng->seed(77027473 * m_sequence[m_seqno] + 19937 * m_size + m_rank);
  return true;
}

std::mt19937& pycpp::Environment::prng()
{
  // move to check?
  if (!m_prng)
    throw std::runtime_error("mt not init");
  return *m_prng;
}

// Note this does not fully initialise, do not construct directly, use the static init function
pycpp::Environment::Environment()
{
  // make the neworder module available in embedded python env
  neworder::import_module();

  // Init python env
  Py_Initialize();

  // init numpy
  numpy_init(); // things go bad if this gets called more than once?

  m_self = new py::object(py::import("neworder"));
  // make our rank/size visible to python

} 

pycpp::Environment::~Environment() 
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
std::string pycpp::Environment::get_error() noexcept
{
  // see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
  std::string message;
  if (PyErr_Occurred())
  {
    PyObject *exc,*val,*tb;
    PyErr_Fetch(&exc,&val,&tb);
    PyErr_NormalizeException(&exc,&val,&tb);

    py::handle<> hexc(exc), hval(py::allow_null(val)), htb(py::allow_null(tb));
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

std::string pycpp::Environment::version()
{
  static std::string version_string;
  // Get and display python version - only do once
  if (version_string.empty())
  {
    py::object sys = py::import("sys");
    version_string = py::extract<std::string>(sys.attr("version"))();
    std::replace(version_string.begin(), version_string.end(), '\n', ' ');
  }
  return version_string;
}

