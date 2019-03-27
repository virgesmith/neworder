
#pragma once

#include "NewOrder.h"

#include "pybind11/numpy.h"

// create a np namespace
namespace np {

using py::array;
template<typename T> using array_t = py::array_t<T>;

// TODO safer implementation
template<typename T>
T& at(np::array& a, size_t index)
{
  // Flattened indexing. TODO reuse Index from humanleague if necess
  // if (a.get_nd() != 1)
  //   throw std::runtime_error("np::array dim>1");
  return *(reinterpret_cast<T*>(a.mutable_data(0)) + index);
}

// TODO safer implementation
template<typename T>
const T& at(const np::array& a, size_t index)
{
  return *(reinterpret_cast<const T*>(a.data(0)) + index);
}

template<typename T>
T* begin(np::array& a)
{
  return reinterpret_cast<T*>(a.mutable_data(0));
}

template<typename T>
const T* cbegin(const np::array& a)
{
  return reinterpret_cast<const T*>(a.data(0));
}

template<typename T>
T* end(np::array& a)
{
  return begin<T>(a) + a.size();
}

template<typename T>
const T* cend(const np::array& a)
{
  return cbegin<T>(a) + a.size();
}

// Uninitialised 1d array
template<typename T>
np::array empty_1d_array(size_t n)
{
  return np::array_t<T>({n});
}

// Create a 1d array, initialising with a function
// e.g. "ones" is make_array<double>(n, [](){ return 1.0; })
template<typename T>
np::array make_array(size_t n, const std::function<T()>& f)
{
  np::array a = empty_1d_array<T>(n); 
  T* p = reinterpret_cast<T*>(a.mutable_data());
  std::generate(p, p + n, f);
  return a;
}

template<typename T>
np::array& fill(np::array& a, T val)
{
  T* p = reinterpret_cast<T*>(a.mutable_data());
  size_t n = a.size();
  std::fill(p, p + n, val);
  return a;
}

// "nullary"/scalar unary op implemented as make_array
template<typename R, typename A>
np::array_t<R> unary_op(const np::array_t<A>& arg, const std::function<R(A)>& f)
{
  py::array_t<R> result(arg.size());
  const A* pa = (const A*)arg.data(0);
  R* pr = (R*)result.request().ptr;
  for (ssize_t i = 0; i < arg.size(); ++i, ++pa, ++pr)
  {
    *pr = f(*pa);
  }
  return result;
}

template<typename R, typename A0, typename A1>
np::array_t<R> binary_op(A0 arg0, const np::array_t<A1>& arg1, const std::function<R(A0, A1)>& f)
{
  py::array_t<R> result(arg1.size());
  const A1* pa1 = (const A1*)arg1.data(0);
  R* pr = (R*)result.request().ptr;
  for (ssize_t i = 0; i < arg0.size(); ++i, ++pa1, ++pr)
  {
    *pr = f(arg0, *pa1);
  }
  return result;
}

template<typename R, typename A0, typename A1>
np::array_t<R> binary_op(const np::array_t<A0>& arg0, A1 arg1, const std::function<R(A0, A1)>& f)
{
  py::array_t<R> result(arg0.size());
  const A0* pa0 = (const A0*)arg0.data(0);
  R* pr = (R*)result.request().ptr;
  for (ssize_t i = 0; i < arg0.size(); ++i, ++pa0, ++pr)
  {
    *pr = f(*pa0, arg1);
  }
  return result;
}

template<typename R, typename A0, typename A1>
np::array_t<R> binary_op(const np::array_t<A0>& arg0, const np::array_t<A1>& arg1, const std::function<R(A0, A1)>& f)
{
  assert(arg0.size() == arg1.size());
  py::array_t<R> result(arg0.size());
  const A0* pa0 = (const A0*)arg0.data(0);
  const A1* pa1 = (const A1*)arg1.data(0);
  R* pr = (R*)result.request().ptr;
  for (ssize_t i = 0; i < arg0.size(); ++i, ++pa0, ++pa1, ++pr)
  {
    *pr = f(*pa0, *pa1);
  }
  return result;
}

template<typename T>
array empty(const std::initializer_list<size_t>& shape)
{
  return array_t<T>(shape);
}

template<typename T>
array zeros(const std::initializer_list<size_t>& shape)
{
  array_t<T> a(shape);
  return fill(a, T(0));
}

// Zero-initialised 1d array
template<typename T>
np::array zero_1d_array(size_t n)
{
  return np::zeros<T>({n});
}

}
