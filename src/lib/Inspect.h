#pragma once

#include "NewOrder.h"

#include <vector>
#include <string>

namespace pycpp {

// const char* type(PyObject* p);

// const char* type(const py::object& o);

// bool callable(const py::object& o);

NEWORDER_EXPORT bool has_attr(const py::object& o, const char* attr_name);

NEWORDER_EXPORT std::string as_string(const py::object& obj);

}


