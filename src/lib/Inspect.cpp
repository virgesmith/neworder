
#include "Inspect.h"

#include "NewOrder.h"

#include <algorithm>
#include <iostream>


// bool pycpp::callable(const py::object& o) 
// {
//   return PyCallable_Check(o.ptr());
// }

bool pycpp::has_attr(const py::object& o, const char* attr_name)
{
  return PyObject_HasAttrString(o.ptr(), attr_name);
}

// // string repr
// std::string pycpp::as_string(PyObject* obj)
// {
//   PyObject* repr = PyObject_Str(obj);
//   PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
//   const char *bytes = PyBytes_AS_STRING(str);

//   Py_XDECREF(repr);
//   Py_XDECREF(str);
//   return std::string(bytes);
// }

std::string pycpp::as_string(const py::object& obj)
{
  return py::str(obj).cast<std::string>();
}



