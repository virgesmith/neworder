#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>
// Data frame manipulation routines

namespace no {

class MonteCarlo;

namespace df {

py::array_t<int64_t> unique_index(size_t n);

py::object transition(no::MonteCarlo& mc, py::array_t<double, py::array::c_style | py::array::forcecast> matrix,
                      py::object& series);

py::object transition_conditional(no::MonteCarlo& mc, py::dict matrices, py::object& group, py::object& series);

} // namespace df

} // namespace no