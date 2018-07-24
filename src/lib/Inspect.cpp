
#include "Inspect.h"
#include "Object.h"

#include "python.h"

#include <algorithm>
#include <iostream>

const char* pycpp::type(PyObject* p)
{
  if (!p)
    throw std::runtime_error("null object");
  return Py_TYPE(p)->tp_name;
}

const char* pycpp::type(const py::object& o)
{
  return pycpp::type(o.ptr());
}

bool pycpp::callable(const py::object& o) 
{
  return PyCallable_Check(o.ptr());
}

bool pycpp::has_attr(const py::object& o, const char* attr_name)
{
  return PyObject_HasAttrString(o.ptr(), attr_name);
}

std::vector<std::pair<std::string, const char*>> pycpp::dir(PyObject* obj, bool public_only)
{
  std::vector<std::string> attrs = pycpp::List(PyObject_Dir(obj)).toVector<std::string>();
  // Ignore anything with a leading underscore - assuming private in as much as private exists in python
  if (public_only)
    attrs.erase(std::remove_if(attrs.begin(), attrs.end(), [](const std::string& s) { return s[0] == '_';}), attrs.end());

  std::vector<std::pair<std::string, const char*>> typed_attrs;
  typed_attrs.reserve(attrs.size());

  for (const auto& attr: attrs)
  {
    typed_attrs.push_back(std::make_pair(attr, type(PyObject_GetAttrString(obj, attr.c_str()))));
  }
  return typed_attrs;
}

std::vector<std::pair<std::string, const char*>> pycpp::dir(const py::object& obj, bool public_only)
{
  return pycpp::dir(obj.ptr(), public_only);
}

std::ostream& operator<<(std::ostream& os, const py::object& o)
{
  return os << py::extract<std::string>(py::str(o))();
}

// std::ostream& operator<<(std::ostream& os, const np::ndarray& a)
// {
//   return os << py::extract<std::string>(py::str(a))();
// }
