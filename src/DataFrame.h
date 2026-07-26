#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>
// Data frame manipulation routines

namespace no {

class Model;

namespace df {

py::array_t<int64_t> unique_index(size_t n);

py::object transition(no::Model& model, py::array_t<double, py::array::c_style | py::array::forcecast> matrix,
                      py::object& series);

} // namespace df

} // namespace no