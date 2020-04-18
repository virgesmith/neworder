#pragma once

#include "NewOrder.h"
#include "ArrayHelpers.h"

#include <numeric>
// "vectorised" functions operating on, and/or returning numpy arrays, using the generic helper functions in ArrayHelpers.h

namespace no { 

py::array isnever(const py::array& x);

template<typename T> T sum(const py::array& x)
{
  return std::accumulate(no::cbegin<T>(x), no::cend<T>(x), T(0));
} 

// logistic function f(x) (TODO? offset x0 slope k: 1/(1+exp(-k(x-x0))))
py::array logistic(const py::array& x, double x0, double k);

// logit function
py::array logit(const py::array& x);

} 