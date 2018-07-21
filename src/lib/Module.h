#pragma once

#include "Object.h"
#include "Environment.h"

#include <Python.h>

// TODO move impl to cpp

namespace pycpp {

// see https://docs.python.org/3/c-api/import.html

// not clear why inheriting from Object prevents the ModuleNotFoundError
class Module : public Object
{
public:

  ~Module() = default;

  // TODO init to use std::move
  // Module(const Module&) = delete;
  // Module& operator=(const Module&) = delete;
  
  // TODO static Module init(const std::string& filename)

  // defer construction of Object(base) in order to trap a missing module
  static Module init(const String& filename);

  bool hasAttr(const std::string& name);

  PyObject* getAttr(const std::string& name);

  PyObject* getAttr(PyObject* obj);

private:
  //Module(String& filename) : Object(PyImport_Import(filename.release()))
  Module(PyObject* p) : Object(p) { }
};

}