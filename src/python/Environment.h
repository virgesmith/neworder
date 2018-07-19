#pragma once

#include "Array.h"

#include <Python.h>

#include <stdexcept>

#include <iostream>
namespace pycpp {

class Exception : public std::runtime_error
{
public:
  Exception(const std::string& s) : std::runtime_error(s.c_str()) { }
  //Exception(const Exception&) = delete;
  ~Exception() = default;
};

class Environment
{
public:
  Environment() 
  {
    Py_Initialize();
    numpy_init(); // things go bad if this gets called more than once
  } 

  ~Environment() 
  {
    // Python >=3.6
    // if (Py_FinalizeEx() < 0)
    // {
    //   // report an error...somehow
    // }
    Py_Finalize();
  }

  // check for errors in the python env: if it returns, there is no error
  static void check()
  {
    // see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
    // function that sticks python error into an exception and throws
    if (PyErr_Occurred())
    {
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);
      auto message = pycpp::String::force(type).operator std::string() + ":" + pycpp::String::force(value).operator std::string();
      PyErr_Restore(type, value, traceback);
      // TODO dump traceback (when not null)
      // if (traceback)
      //   std::cerr << "Python stack:\n" << pycpp::String::force(traceback).operator std::string() << std::endl;
      throw Exception(message);
    }
  }

private:

};

}