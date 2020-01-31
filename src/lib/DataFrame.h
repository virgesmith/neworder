#pragma once

#include "NewOrder.h"
#include "numpy.h"
// Data frame manipulation routines

namespace no { namespace df {

void transition(np::array categories, np::array matrix, py::object& df, const std::string& colname);

void directmod(py::object& df, const std::string& colname);

//void linked_change(py::object& df, const std::string& cat, const std::string& link_cat);

//py::object append(const py::object& df1, const py::object& df2);

}} //no::df