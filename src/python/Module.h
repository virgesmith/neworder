#pragma once

#include "Object.h"

#include <Python.h>

namespace pycpp {

// see https://docs.python.org/3/c-api/import.html

// not clear why inheriting from Object prevents the ModuleNotFoundError
class Module //: public Object
{
public:
  Module(String& filename) //: Object(PyImport_Import(filename.release()))
  {
    m_obj = PyImport_Import(filename.release());
    if(!m_obj)
    {
      PyErr_Print();
      throw std::runtime_error(std::string("Failed to load ") + (const char*)filename);
    }
  }

  ~Module()
  { 
    Py_DECREF(m_obj);
  }

  PyObject* attr(const std::string& name) 
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

  PyObject* release() 
  {
    Py_INCREF(m_obj);
    return m_obj;
  }

private:

  PyObject* m_obj;
};

}