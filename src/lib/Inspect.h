#pragma once

#include "python.h"

#include <vector>
#include <string>

namespace pycpp {

const char* type(PyObject* p);

const char* type(const py::object& o);

bool callable(const py::object& o);

bool has_attr(const py::object& o, const char* attr_name);

std::string as_string(PyObject* obj);

std::string as_string(const py::object& obj);

// std::vector<std::pair<std::string, std::string>> dir(PyObject* obj, bool public_only=true);

// std::vector<std::pair<std::string, std::string>> dir(const py::object& obj, bool public_only=true);

}

std::ostream& operator<<(std::ostream& os, const py::object& o);

// std::ostream& operator<<(std::ostream& os, const np::ndarray& a);
