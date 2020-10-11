// NewOrder.h
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#if !defined(NEWORDER_EXPORT)
  #if defined(_MSC_VER)
    #define NEWORDER_EXPORT __declspec(dllexport)
  #else
    #define NEWORDER_EXPORT __attribute__ ((visibility("default")))
  #endif
#endif
