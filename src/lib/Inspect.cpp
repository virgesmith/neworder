
#include "Inspect.h"

#include "NewOrder.h"

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

// string repr
std::string pycpp::as_string(PyObject* obj)
{
  PyObject* repr = PyObject_Str(obj);
  PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *bytes = PyBytes_AS_STRING(str);

  Py_XDECREF(repr);
  Py_XDECREF(str);
  return std::string(bytes);
}

std::string pycpp::as_string(const py::object& obj)
{
  return as_string(obj.ptr());
}

// std::vector<std::pair<std::string, std::string>> pycpp::dir(PyObject* obj, bool public_only)
// {
//   return dir(py::object(py::handle<>(obj)), public_only);
// }

// std::vector<std::pair<std::string, std::string>> pycpp::dir(const py::object& obj, bool public_only)
// {
//   py::list d(py::handle<>(PyObject_Dir(obj.ptr())));
//   py::ssize_t n = py::len(d);
//   std::vector<std::pair<std::string, std::string>> res;
//   res.reserve(n);

//   for (py::ssize_t i = 0; i < n; ++i)  
//   {
//     // TODO this could throw?
//     std::string name = py::extract<std::string>(d[i])();
//     std::string type = pycpp::type(obj.attr(d[i]));
//     if (public_only && name[0] == '_')
//       continue;
//     res.push_back(std::make_pair(name,type));
//   }
//   return res;
// }

std::ostream& operator<<(std::ostream& os, const py::object& o)
{
  return os << py::str(o);
}

// std::ostream& operator<<(std::ostream& os, const np::array& a)
// {
//   return os << py::extract<std::string>(py::str(a))();
// }
