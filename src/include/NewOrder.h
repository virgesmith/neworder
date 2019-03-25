// NewOrder.h
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

#if !defined(NEWORDER_EXPORT)
// #  if defined(WIN32) || defined(_WIN32)
// #    define PYBIND11_EXPORT __declspec(dllexport)
// #  else
#  define NEWORDER_EXPORT __attribute__ ((visibility("default")))
#endif
