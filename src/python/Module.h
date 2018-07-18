#pragma once

#include "Object.h"

#include <Python.h>

namespace pycpp {

// see https://docs.python.org/3/c-api/import.html

// not clear why inheriting from Object prevents the ModuleNotFoundError
class Module : public Object
{
public:
  // defer construction of Object(base) in order to trap a missing module
  static Module init(String& filename)
  {
  //Module(String& filename) : Object(PyImport_Import(filename.release()))
    PyObject* p = PyImport_Import(filename.release());
    Environment::check();
    return Module(p);
  }

  bool hasAttr(const std::string& name)
  {
    return PyObject_HasAttrString(release(), name.c_str());
  }

  PyObject* getAttr(const std::string& name) 
  {
    PyObject* p = PyObject_GetAttrString(release(), name.c_str());
    if (!p)
    {
      Environment::check();
      // // TODO see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
      // // function that sticks python error into an exception and throws
      // if (PyErr_Occurred())
      //   PyErr_Print();
      throw std::runtime_error("Cannot find attribute " + name);
    }   
    return p;
  }

private:
  //Module(String& filename) : Object(PyImport_Import(filename.release()))
  Module(PyObject* p) : Object(p) { }
};

}