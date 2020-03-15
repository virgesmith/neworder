#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>
// "vectorised" functions operating on, and/or returning numpy arrays 
// for now, only vectorised in the sense the code is compiled (and the compiler is free to choose SIMD if it sees fit)
// TODO if bottleneck help compiler's SIMD vectorisation using ideally openmp directives

namespace no { 

py::array isnever(const py::array& x);

} 