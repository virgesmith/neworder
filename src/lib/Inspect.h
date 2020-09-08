#error not used

#pragma once


#include "NewOrder.h"

#include <vector>
#include <string>

namespace no {

// const char* type(const py::object& o);

NEWORDER_EXPORT bool has_attr(const py::object& o, const char* attr_name);

NEWORDER_EXPORT std::string as_string(const py::object& obj);

}


