
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
  // requires C++14. moot anyway as only one arg?
  //env.m_prng.reset(std::make_unique<std::mt19937>(77027473 * size + rank));
  env.m_prng.reset(new std::mt19937(77027473 * size + rank));

  std::cout << "[C++ " << env.id() << "] process init, seed=" << 77027473 * size + rank << std::endl; 

  return env;
}

// syntactic sugar
pycpp::Environment& pycpp::Environment::get()
{
  return Global::instance<Environment>();
}

const std::string& pycpp::Environment::id() const
{
  static const std::string idstring = std::to_string(m_rank) + "/" + std::to_string(m_size);
  return idstring;
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

  std::cout << "[C++] embedded python version: " << version() << std::endl;

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

// TODO return a bool as well as a string?
// TODO error msg member of env?
// check for errors in the python env: if it returns, there is no error
// use C API here as can't have anything throwing
std::string pycpp::Environment::check() noexcept
{
  // see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
  if (PyErr_Occurred())
  {
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    // TODO split type/value
    if (type && value)
    {
      // TODO sort this out
      std::string message = pycpp::as_string(type) + ":" + pycpp::as_string(value);
//      PyErr_Restore(type, value, traceback);
      // TODO dump traceback (when not null)
      // if (traceback)
      //   std::cerr << "Python stack:\n" << pycpp::String::force(traceback).operator std::string() << std::endl;
      return message;
    }
  }

  return "unable to determine error";
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

