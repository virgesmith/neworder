
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
T& at(const np::ndarray& a, size_t index)
{
  if (a.get_nd() != 1)
    throw std::runtime_error("np::array dim>1");
  return *(reinterpret_cast<T*>(a.get_data()) + index);
}

template<typename T>
T* begin(const np::ndarray& a)
{
  return reinterpret_cast<T*>(a.get_data());
}

template<typename T>
T* end(const np::ndarray& a)
{
  return begin<T>(a) + size(a);
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
// e.g. "ones" is make_array<double>(n, [](){ return 1.0; })
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

  // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
  // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
  // use a using declaration in the derived class to force it to be visible
  np::ndarray operator()(const py::object& arg) 
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

  // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
  // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
  // use a using declaration in the derived class to force it to be visible
  np::ndarray operator()(const py::object& arg1, const py::object& arg2) 
  {
    return np::array(np::binary_ufunc<BinaryArrayOp<R, A1, A2>>::call(*this, arg1, arg2, py::object()));      
  }
};

}