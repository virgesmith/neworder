#pragma once

#include "PythonFwd.h"

#include <vector>
#include <string>

namespace pycpp {

const char* type(PyObject* p);

std::vector<std::pair<std::string, const char*>> dir(PyObject* obj, bool public_only=true);

}