
#include "Inspect.h"
#include "Object.h"
#include <algorithm>

#include <Python.h>

const char* pycpp::type(PyObject* p)
{
  if (!p)
    throw std::runtime_error("null object");
  return Py_TYPE(p)->tp_name;
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

