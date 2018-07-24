#include "Module.h"
#include "Object.h"
#include "Environment.h"

#include <Python.h>


// defer construction of Object(base) in order to trap a missing module
pycpp::Module pycpp::Module::init(const String& filename)
{
//Module(String& filename) : Object(PyImport_Import(filename.release()))
  PyObject* p = PyImport_Import(filename.release());
  //Environment::check();
  return Module(p);
}

bool pycpp::Module::hasAttr(const std::string& name)
{
  return PyObject_HasAttrString(release(), name.c_str());
}

PyObject* pycpp::Module::getAttr(const std::string& name) 
{
  PyObject* p = PyObject_GetAttrString(release(), name.c_str());
  if (!p)
  {
    //pycpp::Environment::check();
    throw std::runtime_error("Cannot find attribute " + name);
  }   
  return p;
}

//PyObject* pycpp::Module::getAttr(const pycpp::Object& obj)

PyObject* pycpp::Module::getAttr(PyObject* obj) 
{
  PyObject* p = PyObject_GetAttr(release(), obj);
  if (!p)
  {
    //pycpp::Environment::check();
    throw std::runtime_error("Cannot find attribute");
  }   
  return p;
}

