
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

// Uninitialised 1d array
template<typename T>
np::ndarray empty_1d_array(size_t n)
{
  return np::empty(1, (Py_intptr_t*)&n, np::dtype::get_builtin<T>());
}

// Zero-initialised 1d array
template<typename T>
np::ndarray zero_1d_array(size_t n)
{
  return np::zeros(1, (Py_intptr_t*)&n, np::dtype::get_builtin<T>());
}

// Create a 1d array, initialising with a function
template<typename T>
np::ndarray make_array(size_t n, const std::function<T()>& f)
{
  np::ndarray a = pycpp::empty_1d_array<T>(n); 
  T* p = reinterpret_cast<T*>(a.get_data());
  std::generate(p, p + n, f);
  return a;
}

template<typename R, typename A>
struct UnaryArrayOp
{
  typedef A argument_type;
  typedef R result_type;

  virtual ~UnaryArrayOp() { }

  virtual R operator()(A) = 0;

  // workaround since cant seem to call directly from derived
  np::ndarray /*operator()*/call_impl(const py::object& arg) 
  {
    return np::array(np::unary_ufunc<UnaryArrayOp<R,A>>::call(*this, arg, py::object()));      
  }
};

template<typename R, typename A1, typename A2>
struct BinaryArrayOp
{
  typedef A1 first_argument_type;
  typedef A2 second_argument_type;
  typedef R result_type;

  virtual ~BinaryArrayOp() { }

  virtual R operator()(A1, A2) = 0;

  // workaround since cant seem to call directly from derived
  np::ndarray /*operator()*/call_impl(const py::object& arg1, const py::object& arg2) 
  {
    return np::array(np::binary_ufunc<BinaryArrayOp<R, A1, A2>>::call(*this, arg1, arg2, py::object()));      
  }
};

}