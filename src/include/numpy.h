
#pragma once

#include "NewOrder.h"

#include "pybind11/numpy.h"

// create a np namespace
namespace np {
  using py::array;
  template<typename T> using array_t = py::array_t<T>;

  template<typename T>
  array empty(const std::initializer_list<size_t>& shape)
  {
    return py::array_t<T>(shape);
  }

  template<typename T>
  array zeros(const std::initializer_list<size_t>& shape)
  {
    T zero(0);
    return py::array_t<T>(shape, &zero);
  }

  // TODO what is required here (if anything)
  inline void initialize()
  {
  }
}

// helper functions for ndarrays
namespace pycpp {

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
  return np::empty<T>({n});
}

// Zero-initialised 1d array
template<typename T>
np::array zero_1d_array(size_t n)
{
  return np::zeros<T>({n});
}

// Create a 1d array, initialising with a function
// e.g. "ones" is make_array<double>(n, [](){ return 1.0; })
template<typename T>
np::array make_array(size_t n, const std::function<T()>& f)
{
  np::array a = pycpp::empty_1d_array<T>(n); 
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

// "nullary"/scalar unary op implemented as pycpp::make_array

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


// template<typename R, typename A>
// struct UnaryArrayOp
// {
//   typedef A argument_type;
//   typedef R result_type;

//   virtual ~UnaryArrayOp() { }

//   virtual R operator()(A) = 0;

//   // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
//   // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
//   // use a using declaration in the derived class to force it to be visible
//   np::array operator()(const py::object& arg) 
//   {
//     return np::array(np::unary_ufunc<UnaryArrayOp<R,A>>::call(*this, arg, py::object()));      
//   }
// };

// template<typename R, typename A1, typename A2>
// struct BinaryArrayOp
// {
//   typedef A1 first_argument_type;
//   typedef A2 second_argument_type;
//   typedef R result_type;

//   virtual ~BinaryArrayOp() { }

//   virtual R operator()(A1, A2) = 0;

//   // implementing the above function in a derived class hides the (below) base-class implementations of operator() 
//   // see https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the/1629074#1629074
//   // use a using declaration in the derived class to force it to be visible
//   np::array operator()(const py::object& arg1, const py::object& arg2) 
//   {
//     return np::array(np::binary_ufunc<BinaryArrayOp<R, A1, A2>>::call(*this, arg1, arg2, py::object()));      
//   }
// };

}