#pragma once

#include "Object.h"

namespace pycpp {

class Tuple;

class Function : public Object
{
public:
  Function(PyObject* p);

  ~Function() = default;

  // TODO disable copy/assign

  PyObject* call();

  PyObject* call(pycpp::Tuple& args);

private:
};

}