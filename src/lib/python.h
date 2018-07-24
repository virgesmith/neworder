
#pragma once

#include <Python.h>

#include <boost/python.hpp>
//#include <boost/python/numpy.hpp>

namespace py = boost::python;
//namespace np = boost::python::numpy;

std::ostream& operator<<(std::ostream& os, const py::object& o);

// std::ostream& operator<<(std::ostream& os, const np::ndarray& a);
