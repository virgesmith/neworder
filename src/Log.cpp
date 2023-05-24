
#include "Log.h"
#include "Module.h"

std::string std::to_string(char c)
{
  return std::string(1, c);
}

std::string std::to_string(const std::string& v)
{
  return v;
}

std::string std::to_string(const py::object& o)
{
  return py::str(o);
}

void no::log(const std::string& msg)
{
  if (no::env::verbose)
    py::print(no::env::logPrefix[no::env::Context::CPP], msg);
}

void no::log(const py::handle& msg)
{
  if (no::env::verbose)
    py::print(no::env::logPrefix[no::env::Context::CPP], msg);
}

void no::warn(const std::string& msg)
{
  if (PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1) == -1)
    throw py::error_already_set();
}