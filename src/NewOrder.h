// NewOrder.h
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <string>

namespace nb = nanobind;

#if !defined(NEWORDER_EXPORT)
#if defined(_MSC_VER)
#define NEWORDER_EXPORT __declspec(dllexport)
// silence dll export warnings for STL objects
#pragma warning(disable : 4521)
#else
#define NEWORDER_EXPORT __attribute__((visibility("default")))
#endif
#endif

namespace no {

// nanobind's exception functions only accept a C string
inline nb::builtin_exception value_error(const std::string& msg) { return nb::value_error(msg.c_str()); }
inline nb::builtin_exception type_error(const std::string& msg) { return nb::type_error(msg.c_str()); }

// Python truthiness. nanobind's bool caster only accepts True and False, which excludes e.g. numpy.bool
inline bool truthy(nb::handle value) {
  int result = PyObject_IsTrue(value.ptr());
  if (result < 0)
    throw nb::python_error();
  return result != 0;
}

} // namespace no
