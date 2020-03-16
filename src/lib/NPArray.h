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

} 