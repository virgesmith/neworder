#pragma once

// Functor.h

#include "Inspect.h"
#include "NewOrder.h"

#include <map>
#include <vector>

namespace pycpp
{

class PYBIND11_EXPORT Functor
{
public:
  Functor(py::object func, py::list args);

  py::object operator()() const;

private:
  py::object m_func;
  /*py::list*/std::vector<py::object> m_args;
};

typedef std::map<std::string, Functor> FunctionTable;

}