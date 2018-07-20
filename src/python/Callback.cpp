
#include "Callback.h"
#include "Object.h"

#include <iostream>

namespace
{
const char* module_name = "neworder";

/* Return the number of arguments of the application command line */
extern "C" PyObject *neworder_name(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ":name"))
    return nullptr;
  return pycpp::String(module_name).release();
}

PyObject* PyInit_neworder()
{
  static PyMethodDef functions[] = {
    {"name", neworder_name, METH_VARARGS, "Return the name of the C++ runtime module."}, {nullptr, nullptr, 0, nullptr}
  };

  static PyModuleDef module = { 
    PyModuleDef_HEAD_INIT, module_name, nullptr, -1, functions, nullptr, nullptr, nullptr, nullptr 
  };

  return PyModule_Create(&module);
}

} // namespace

void callback::register_all()
{
  // First register callback module
  PyImport_AppendInittab(module_name, &PyInit_neworder);
}

