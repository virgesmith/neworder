
#include "Environment.h"
#include "Inspect.h"
#include "Callback.h"

#include <algorithm>
#include <string>
#include <iostream>


pycpp::Environment::Environment() : m_procid(-1), m_nprocs(0) 
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

