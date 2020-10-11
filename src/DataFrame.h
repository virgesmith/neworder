#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>
// Data frame manipulation routines

namespace no { 

class Model;

namespace df {

//py::array_t<int64_t> unique_index(size_t n);
py::array_t<int64_t> unique_index(size_t n);

void transition(no::Model& model, py::array_t<int64_t> categories, py::array_t<double> matrix, py::object &df, const std::string& colname);

void testfunc(no::Model& model, py::object& df, const std::string& colname);

//void linked_change(py::object& df, const std::string& cat, const std::string& link_cat);

//py::object append(const py::object& df1, const py::object& df2);

}

} //no::df