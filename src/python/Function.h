#pragma once

#include "Object.h"
#include "Environment.h"
#include <Python.h>

namespace pycpp {

class Function : public Object
{
public:
  Function(PyObject* p) : Object(p)
  {
    if (!PyCallable_Check(p)) 
    {
      Environment::check();
      throw std::runtime_error("Object is not callable");
    }  
  }

  ~Function() { }

  PyObject* call(pycpp::Tuple& args)
  {
    PyObject* p = PyObject_CallObject(m_obj, args.release());
    if (!p)
    {
      Environment::check();      
    }
    return p;
  }

private:
};

}