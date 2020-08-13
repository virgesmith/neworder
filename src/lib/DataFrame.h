#pragma once

#include "NewOrder.h"
#include <pybind11/numpy.h>
// Data frame manipulation routines

namespace no { 

class Model;

namespace df {

void transition(no::Model& model, py::array categories, py::array matrix, py::object& df, const std::string& colname);

void directmod(no::Model& model, py::object& df, const std::string& colname);

//void linked_change(py::object& df, const std::string& cat, const std::string& link_cat);

//py::object append(const py::object& df1, const py::object& df2);

}} //no::df