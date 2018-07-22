
#pragma once 

#include "Object.h"
//#include "NDArray.h"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstring>

// TODO use boost.python (requires >= 1.63, 16.4 comes with 1.58)
// or https://github.com/ndarray/Boost.NumPy
//#include <boost/python/numpy.hpp>

// See https://docs.scipy.org/doc/numpy/reference/c-api.html

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
    
  template<typename T> struct NpyType;

  // due to inconsistencies with integer sizes, only int64_t is supported
  template<> struct NpyType<double> { static const int Type = NPY_DOUBLE; static const int Size = NPY_SIZEOF_DOUBLE; };
  //template<> struct NpyType<int> { static const int Type = NPY_LONG; }; // value may be incorrect
  //template<> struct NpyType<uint32_t> { static const int Type = NPY_ULONG; }; // value may be incorrect
  // TODO This may cause issues on LLP64 / 32bit platforms
  template<> struct NpyType<int64_t> { static const int Type = NPY_LONG;  static const int Size = NPY_SIZEOF_LONG; };
  //template<> struct NpyType<bool> { static const int Type = NPY_BOOL;     static const int Size = 4 /*guess as no NPY_SIZEOF_BOOL*/; };

  // numpy arrays 
  template<typename T>
  class Array : public Object
  {

  public:
    typedef T value_type;

    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
  
    // uninitialised array with given dimension and sizes
    Array(size_t dim, npy_intp* sizes) 
      : Object(PyArray_SimpleNew(dim, sizes, NpyType<T>::Type)) 
    { 
      PyArray_FILLWBYTE((PyArrayObject*)m_obj, 0);
    }

    // construct from an incoming numpy object (shallow copy)
    explicit Array(PyObject* array) : Object(array)
    {
      Py_INCREF(m_obj);
      // see https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.array.html
      if (PyArray_TYPE((PyArrayObject*)array) != NpyType<T>::Type)
        throw std::runtime_error("python array contains invalid type: " + std::to_string(PyArray_TYPE((PyArrayObject*)array)) 
          + " when expecting " + std::to_string(NpyType<T>::Type));
    }
    
    // "Construct" 1D from vector<U> where U must be implicitly convertible to T
    template<typename U>
    explicit Array(const std::vector<U>& a) : Array(a, a.size())
    {
      // see below constructor
    }

  private: // this is a hack to chain constructors to avoid having to explicitly pass an addressable size value 
    
    // Construct 1D from vector<U> 
    template<typename U>
    Array(const std::vector<U>& a, npy_intp size) : Array(1, &size)
    {
      // potentially breaks because sizeof(T) might not be size of contained type
      // std::memcpy(PyArray_GETPTR1((PyArrayObject*)m_obj, 0), a.data(), size * sizeof(T));
      // temporary "solution" is to permit only int64_t integers
      // need to know size of NPY_LONG 
      // still need to be careful sizeof(T) is correct
      T* p = (T*)PyArray_GETPTR1((PyArrayObject*)m_obj, 0);
      for (size_t i = 0; i < a.size(); ++i, ++p)
      {
        *p = a[i];
      }
    }

    // private helper to convert dims to python type
    static std::vector<npy_intp> convert(const std::vector<int64_t>& sizeIn)
    {
      std::vector<npy_intp> sizeOut(sizeIn.size());
      for (size_t i = 0; i < sizeIn.size(); ++i)
      {
        sizeOut[i] = sizeIn[i];
      }
      return sizeOut;
    }
  
  public:
  
    // // Construct from NDArray<T>. Data is copied
    // explicit Array(const NDArray<T>& a) 
    //   : Array(a.dim(), convert(a.sizes()).data()) 
    // {
    //   std::copy(a.rawData(), a.rawData() + a.storageSize(), rawData());
    // }

    // // shallow copy, increase ref count
    // Array(const Array& a) : Object(a.m_obj)
    // {
    //   Py_INCREF(m_obj);
    // }

    Array(Array&&) = delete;
    
    ~Array()
    {
      // Object destructor handles decref
    }
    
    // generic n-D access??
    const_reference& operator[](npy_intp* index) const
    {
      return *(const_pointer)PyArray_GetPtr(const_cast<PyArrayObject*>(reinterpret_cast<const PyArrayObject*>(m_obj)), index);
    }
    
    reference& operator[](npy_intp* index)
    {
      return *(pointer)PyArray_GetPtr(reinterpret_cast<PyArrayObject*>(m_obj), index);
    }
    
    // assumes 1-D
    template<typename U>
    std::vector<U> toVector() const
    {
      if (dim() != 1)
        throw std::runtime_error("cannot convert multidimensional array to vector");
      const int n = storageSize();
      std::vector<U> v(n);
      npy_intp i[1]; 
      for (i[0] = 0; i[0] < n; ++i[0])
      {
        v[i[0]] = this->operator[](i);
      }
      return v;
    }

    // NDArray<T> toNDArray() const
    // {
    //   const size_t dim = this->dim();
    //   std::vector<int64_t> sizes(dim);
    //   for (size_t i = 0; i < dim; ++i)
    //     sizes[i] = shape()[i];
    //   NDArray<T> tmp(sizes);
    //   std::copy(rawData(), rawData() + tmp.storageSize(), const_cast<T*>(tmp.rawData()));
    //   return tmp;
    // }
    
    // TODO dimension
    int dim() const 
    {
      return PyArray_NDIM((PyArrayObject*)m_obj);
    }
    
    npy_intp* shape() const 
    {
      return PyArray_SHAPE((PyArrayObject*)m_obj);
    }
    
    // total number of elements
    int storageSize() const
    {
      return PyArray_Size(m_obj);
    }
    
    long stride(int d)
    {
      return PyArray_STRIDE((PyArrayObject*)m_obj, d);
    }
    
    T* rawData() const
    {
      return (T*)PyArray_DATA((PyArrayObject*)m_obj);
    }
    
  };
  
}
