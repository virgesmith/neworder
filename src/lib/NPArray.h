#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>

// "vectorised" functions operating on, and/or returning numpy arrays, using the generic helper functions in ArrayHelpers.h

namespace no { 

namespace time {
  NEWORDER_EXPORT py::array_t<bool> isnever_a(const py::array_t<double>& x);
}
// logistic function f(x; x0, k)
NEWORDER_EXPORT py::array_t<double> logistic(const py::array_t<double>& x, double x0, double k);

// logit function
NEWORDER_EXPORT py::array_t<double> logit(const py::array_t<double>& x);

} 