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
      // TODO see PyErr_Fetch: https://docs.python.org/3/c-api/exceptions.html
      // function that sticks python error into an exception and throws
      if (PyErr_Occurred())
        PyErr_Print();
      throw std::runtime_error("not a function");
    }
  }

  ~Function() { }

  PyObject* call(pycpp::Tuple& args)
  {
    return PyObject_CallObject(m_obj, args.release());
  }

private:
};

}