#pragma once

#include "NewOrder.h"

// helper class for accessing the python attributes of a C++ class with python bindings
template<typename T>
class PyAccessor final
{
public:
  typedef T cpp_type;
  explicit PyAccessor(cpp_type& object) : ref(nb::find(object)) { }

  nb::object get(const char* name) const
  {
    return ref.attr(name);
  }

  template<typename U>
  U get_as(const char* name) const
  {
    return nb::cast<U>(ref.attr(name));
  }

  bool get_truthy(const char* name) const
  {
    return no::truthy(get(name));
  }

private:
  nb::object ref;
};
