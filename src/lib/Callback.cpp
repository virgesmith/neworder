
#include "Callback.h"
#include "Object.h"
#include "Array.h"
#include "Rand.h"

#include <iostream>

namespace
{
const char* module_name = "neworder";

extern "C" PyObject* neworder_name(PyObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ":name"))
    return nullptr;
  return pycpp::String(module_name).release();
}

extern "C" PyObject* neworder_hazard(PyObject* self, PyObject* args)
{
  double cutoff;
  int n;
  if (!PyArg_ParseTuple(args, "di:hazard", &cutoff, &n))
    return nullptr;

  return pycpp::Array<int64_t>(hazard(cutoff, n)).release();
}

PyObject* PyInit_neworder()
{
  static PyMethodDef functions[] = {
    {"name", neworder_name, METH_VARARGS, "Return the name of the C++ runtime module."},
    {"hazard", neworder_hazard, METH_VARARGS, "Return a random int vector of length n, with value 1 where sample below cutoff."}, 
    {nullptr, nullptr, 0, nullptr}
  };

  static PyModuleDef module = { 
    PyModuleDef_HEAD_INIT, module_name, nullptr, -1, functions, nullptr, nullptr, nullptr, nullptr 
  };

  // pycpp::numpy_init(); for some reason does nothing
  import_array();
  if (!PyArray_API)
    throw std::runtime_error("no pyarray api");

  return PyModule_Create(&module);
}

} // namespace

void callback::register_all()
{
  // First register callback module
  PyImport_AppendInittab(module_name, &PyInit_neworder);
}

