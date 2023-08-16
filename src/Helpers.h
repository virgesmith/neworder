#pragma once

#include <pybind11/pybind11.h>

// helper class for accessing the python attributes of a C++ class with python bindings
template<typename T>
class PyAccessor final
{
public:
  typedef T cpp_type;
  explicit PyAccessor(cpp_type& object) : ref(py::cast(&object)) { }

  py::object get(const char* name) const
  {
    return ref.attr(name);
  }

  template<typename U>
  U get_as(const char* name) const
  {
    return ref.attr(name).template cast<U>();
  }

private:
  py::object ref;
};
