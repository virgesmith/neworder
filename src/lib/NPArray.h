#pragma once

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <numeric>
// "vectorised" functions operating on, and/or returning numpy arrays, using the generic helper functions in ArrayHelpers.h

namespace no { 

NEWORDER_EXPORT py::array_t<bool> isnever(const py::array_t<double>& x);

// logistic function f(x) (TODO? offset x0 slope k: 1/(1+exp(-k(x-x0))))
NEWORDER_EXPORT py::array_t<double> logistic(const py::array_t<double>& x, double x0, double k);

// logit function
NEWORDER_EXPORT py::array_t<double> logit(const py::array_t<double>& x);

} 