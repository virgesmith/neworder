
#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>

// See https://docs.scipy.org/doc/numpy/reference/c-api.html

#include <boost/python.hpp>
//#include <boost/python/numpy.hpp>

namespace py = boost::python;
//namespace np = boost::python::numpy;


namespace pycpp {
  // Utilities for numpy API
  inline void numpy_init() 
  {
    // import_array is an evil macro that for python3+ expands to a code block with a 
    // single if statement containing a (conditional) return statement, so not all paths return a value. 
    // The return value is essentially useless since it is only defined for success, thus no way of detecting errors. 
    // To workaround we wrap in a lambda, adding a non-conditional return statement and then ignoring the value. 
    []() -> void* { 
      import_array();
      return nullptr;
    }();
  }
}