#pragma once

// Functor.h

#include "Inspect.h"
#include "NewOrder.h"

#include <map>
#include <vector>

namespace pycpp
{

// TODO deprecate? it's only used once in run.cpp
class PYBIND11_EXPORT Functor
{
public:
  Functor(py::object func);
  Functor(py::object func, py::args args);
  Functor(py::object func, py::kwargs kwargs);
  Functor(py::object func, py::args args, py::kwargs kwargs);

  py::object operator()() const;

private:
  py::object m_func;
  py::args m_args;
  py::kwargs m_kwargs;
};

typedef std::map<std::string, Functor> FunctionTable;

}