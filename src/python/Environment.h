#pragma once

#include <Python.h>

#include <stdexcept>

#include <iostream>
namespace pycpp {

class Exception : public std::runtime_error
{
  Exception(const std::string& s) : std::runtime_error(s.c_str()) { }

  ~Exception() = default;
};

class Environment
{
public:
  Environment() 
  {
    Py_Initialize();
  } 

  ~Environment() 
  {
    if (Py_FinalizeEx() < 0)
    {
      // report an error...
    }
  }

  // check for errors in the python env: if it returns, there is no error
  static void check()
  {
    std::cout << "envcheck\n";
    // TODO see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
    // function that sticks python error into an exception and throws
    if (PyErr_Occurred())
    {
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);
      auto exception = pycpp::String::force(type).operator std::string() + ":" + pycpp::String::force(value).operator std::string();
      PyErr_Restore(type, value, traceback);
      // TODO dump traceback
      throw std::runtime_error(exception);
    }
  }

private:

};

}