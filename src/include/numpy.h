
#pragma once

#include "python.h"

// See https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/numpy/reference/index.html
//#include <numpy/arrayobject.h>
// See https://docs.scipy.org/doc/numpy/reference/c-api.html

#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;


// helper functions for ndarrays
namespace pycpp {

inline size_t size(const np::ndarray& a)
{
  size_t dim = a.get_nd();
  // assumes dim >=1 
  size_t s = a.shape(0);
  for (size_t i = 1; i < dim; ++i)
    s *= a.shape(i);
  return s;
}

template<typename T>
np::ndarray empty_1d_array(size_t n)
{
  return np::empty(1, (Py_intptr_t*)&n, np::dtype::get_builtin<T>());
}

template<typename T>
np::ndarray zero_1d_array(size_t n)
{
  return np::zeros(1, (Py_intptr_t*)&n, np::dtype::get_builtin<T>());
}


}