
#pragma once

#include "NewOrder.h"

#include "pybind11/numpy.h"


// TODO deprecate where possible
namespace no {

// TODO safer implementation
template<typename T>
T& at(py::array& a, size_t index)
{
  // Flattened indexing. TODO reuse Index from humanleague if necess
  // if (a.get_nd() != 1)
  //   throw std::runtime_error("py::array dim>1");
  return *(reinterpret_cast<T*>(a.mutable_data(index)));
}

// TODO safer implementation
template<typename T>
const T& at(const py::array& a, size_t index)
{
  return *(reinterpret_cast<const T*>(a.data(index)));
}

template<typename T>
T* begin(py::array& a)
{
  return reinterpret_cast<T*>(a.mutable_data());
}

template<typename T>
const T* cbegin(const py::array& a)
{
  return reinterpret_cast<const T*>(a.data());
}

template<typename T>
T* end(py::array& a)
{
  return begin<T>(a) + a.size();
}

template<typename T>
const T* cend(const py::array& a)
{
  return cbegin<T>(a) + a.size();
}

// Uninitialised 1d array
template<typename T>
py::array empty_1d_array(size_t n)
{
  return py::array_t<T>({n});
}

// Create a 1d array, initialising with a function
// e.g. "ones" is make_array<double>(n, [](){ return 1.0; })
template<typename T>
py::array make_array(size_t n, const std::function<T()>& f)
{
  py::array a = empty_1d_array<T>(n); 
  T* p = reinterpret_cast<T*>(a.mutable_data());
  std::generate(p, p + n, f);
  return a;
}

template<typename T>
py::array& fill(py::array& a, T val)
{
  T* p = reinterpret_cast<T*>(a.mutable_data());
  size_t n = a.size();
  std::fill(p, p + n, val);
  return a;
}

// "nullary"/scalar unary op implemented as make_array
template<typename R, typename A>
py::array_t<R> unary_op(const py::array_t<A>& arg, const std::function<R(A)>& f)
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
py::array_t<R> binary_op(A0 arg0, const py::array_t<A1>& arg1, const std::function<R(A0, A1)>& f)
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
py::array_t<R> binary_op(const py::array_t<A0>& arg0, A1 arg1, const std::function<R(A0, A1)>& f)
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
py::array_t<R> binary_op(const py::array_t<A0>& arg0, const py::array_t<A1>& arg1, const std::function<R(A0, A1)>& f)
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
py::array empty(const std::initializer_list<size_t>& shape)
{
  return py::array_t<T>(shape);
}

template<typename T>
py::array zeros(const std::initializer_list<size_t>& shape)
{
  py::array_t<T> a(shape);
  return fill(a, T(0));
}

// Zero-initialised 1d array
template<typename T>
py::array zero_1d_array(size_t n)
{
  return zeros<T>({n});
}

template<typename T>
T sum(const py::array_t<T>& a)
{
  T sum = 0;
  return std::accumulate(a.begin(), a.end(), sum);
}

}
