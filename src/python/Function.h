#pragma once

#include "Object.h"

namespace pycpp {

class Function : public Object
{
public:
  Function(PyObject* p) : Object(p)
  {
    if (!PyCallable_Check(p)) 
    {
      Environment::check();
    }
  }

  ~Function() { }

  PyObject* call(pycpp::Tuple& args)
  {
    PyObject* p = PyObject_CallObject(m_obj, args.release());
    Environment::check();
    return p;
  }

private:
};

}