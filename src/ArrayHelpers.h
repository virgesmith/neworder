
#pragma once

#include "NewOrder.h"

#include <pybind11/numpy.h>
#include <array>


namespace no {

template<size_t D>
using Index_t = std::array<py::ssize_t, D>;

// this should be faster than using initializer_list as the compiler can optimise better?
template<typename T, size_t D=1>
T& at(py::array_t<T>& a, const Index_t<D>& index)
{
  assert(a.ndim() == D);
  T* buf = (T*)a.request().ptr;

  size_t offset = index[0];
  for (size_t i = 1; i < D; ++i)
  {
    offset = offset * a.strides()[i-1]/sizeof(T) + index[i];
  }
  return buf[offset];
}

// no impl for D=0
template<typename T>
T& at(py::array_t<T>& a, const Index_t<0>& index);


template<typename T, size_t D=1>
const T& at(const py::array_t<T>& a, const Index_t<D>& index)
{
  assert(a.ndim() == D);
  T* buf = (T*)a.request().ptr;

  size_t offset = index[0];
  for (size_t i = 1; i < D; ++i)
  {
    offset = offset * a.strides()[i-1]/sizeof(T) + index[i];
  }
  return buf[offset];
}

// no impl for D=0
template<typename T>
const T& at(py::array_t<T>& a, const Index_t<0>& index);


template<typename T>
T* begin(py::array_t<T>& a)
{
  assert(a.itemsize() == sizeof(T));
  return (T*)a.request().ptr;
}

template<typename T>
const T* cbegin(const py::array_t<T>& a)
{
  assert(a.itemsize() == sizeof(T));
  return (const T*)a.request().ptr;
}

template<typename T>
T* end(py::array_t<T>& a)
{
  assert(a.itemsize() == sizeof(T));
  return (T*)a.request().ptr + a.size();
}

template<typename T>
const T* cend(const py::array_t<T>& a)
{
  assert(a.itemsize() == sizeof(T));
  return (const T*)a.request().ptr + a.size();
}

// Create a 1d array, initialising with a function
// e.g. "ones" is make_array<double>(n, [](){ return 1.0; })
template<typename T, size_t D=1>
py::array_t<T> make_array(const Index_t<D>& n, const std::function<T()>& f)
{
  py::array_t<T> a(n); 
  std::generate(begin(a), end(a), f);
  return a;
}

template<typename T>
py::array_t<T>& fill(py::array_t<T>& a, T val)
{
  std::fill(begin(a), end(a), val);
  return a;
}

// "nullary"/scalar unary op implemented as make_array
template<typename R, typename A>
py::array_t<R> unary_op(const py::array_t<A>& arg, const std::function<R(A)>& f)
{
  py::array_t<R> result(std::vector<ssize_t>(arg.shape(), arg.shape() + arg.ndim()));

  const A* p = (const A*)arg.request().ptr;
  R* r = (R*)result.request().ptr;
  for (py::ssize_t i = 0; i < arg.size(); ++i)
  {
    r[i] = f(p[i]);
  }
  return result;
}

template<typename R, typename A0, typename A1>
py::array_t<R> binary_op(A0 arg0, const py::array_t<A1>& arg1, const std::function<R(A0, A1)>& f)
{
  py::array_t<R> result(std::vector<py::ssize_t>(arg1.shape(), arg1.shape() + arg1.ndim()));
  const A1* p = (const A1*)arg1.request().ptr;
  R* r = (R*)result.request().ptr;

  for (py::ssize_t i = 0; i < arg1.size(); ++i)
  {
    r[i] = f(arg0, p[i]);
  }
  return result;
}

template<typename R, typename A0, typename A1>
py::array_t<R> binary_op(const py::array_t<A0>& arg0, A1 arg1, const std::function<R(A0, A1)>& f)
{
  py::array_t<R> result(std::vector<ssize_t>{arg0.shape(), arg0.shape() + arg0.ndim()});

  const A0* p = (const A0*)arg0.request().ptr;
  R* r = (R*)result.request().ptr;

  for (py::ssize_t i = 0; i < arg0.size(); ++i)
  {
    r[i] = f(p[i], arg1);
  }
  return result;
}

template<typename R, typename A0, typename A1>
py::array_t<R> binary_op(const py::array_t<A0>& arg0, const py::array_t<A1>& arg1, const std::function<R(A0, A1)>& f)
{
  if (arg0.ndim() != arg1.ndim())
  {
    throw std::runtime_error("binary_op: argments must have same dimension");
  }
  for (py::ssize_t i = 0; i < arg0.ndim(); ++i)
  {
    if (arg0.shape()[i] != arg1.shape()[i])
      throw std::runtime_error("binary_op: argments must have same size");
  }

  py::array_t<R> result(std::vector<ssize_t>(arg0.shape(), arg0.shape() + arg0.ndim()));

  const A0* p0 = (const A0*)arg0.request().ptr;
  const A1* p1 = (const A1*)arg1.request().ptr;
  R* r = (R*)result.request().ptr;

  for (py::ssize_t i = 0; i < arg0.size(); ++i)
  {
    r[i] = f(p0[i], p1[i]);
  }
  return result;
}

}
