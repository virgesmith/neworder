#pragma once

#include "python.h"
#include "numpy.h"
// Data frame manipulation routines

namespace neworder {
namespace df {

void transition(np::ndarray categories, np::ndarray matrix, np::ndarray& col);

void directmod(py::object& df, const std::string& colname);

py::object append(const py::object& df1, const py::object& df2);

}} //neworder::df