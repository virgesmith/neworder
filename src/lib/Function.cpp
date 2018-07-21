
#include "Function.h"
#include "Object.h"
#include "Environment.h"
#include <Python.h>

pycpp::Function::Function(PyObject* p) : Object(p)
{
  if (!PyCallable_Check(p)) 
  {
    Environment::check();
    throw std::runtime_error("Object is not callable");
  }  
}

PyObject* pycpp::Function::call()
{
  PyObject* p = PyObject_CallObject(m_obj, nullptr);
  if (!p)
  {
    Environment::check();      
  }
  return p;
}

PyObject* pycpp::Function::call(pycpp::Tuple& args)
{
  PyObject* p = PyObject_CallObject(m_obj, args.release());
  if (!p)
  {
    Environment::check();      
  }
  return p;
}

