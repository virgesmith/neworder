#pragma once

#include "python.h"
#include "numpy.h"
// Data frame manipulation routines

namespace neworder {
namespace df {

void transition(np::ndarray& col);

void directmod(py::object& df, const std::string& colname);

py::object append(const py::object& df1, const py::object& df2);

void send(const py::object& o);

py::object receive();

void send_csv(const py::object& o);

py::object receive_csv();

}} //neworder::df