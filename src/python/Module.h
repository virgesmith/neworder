#pragma once

#include "Object.h"

#include <Python.h>

namespace pycpp {

// see https://docs.python.org/3/c-api/import.html

// not clear why inheriting from Object prevents the ModuleNotFoundError
class Module : public Object
{
public:
  Module(String& filename) : Object(PyImport_Import(filename.release()))
  {
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
      // TODO see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
      // function that sticks python error into an exception and throws
      if (PyErr_Occurred())
        PyErr_Print();
      throw std::runtime_error("Cannot find attribute " + name);
    }   
    return p;     
  }

private:
};

}