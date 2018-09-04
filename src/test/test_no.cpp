
// test neworder embedded module

#include "test.h"

#include "Inspect.h"
#include "Module.h"
#include "Environment.h"

#include "python.h"
#include "numpy.h"

#include <vector>
#include <string>


void test_no()
{
  neworder::log("neworder module test");

  /*pycpp::Environment& env =*/ pycpp::Environment::get();
  py::object module = py::import("neworder");

  // Check required (but defaulted) attrs visible from both C++ and python
  const char* attrs[] = {"procid", "nprocs", "sequence", "seq"}; 

  for (size_t i = 0; i < sizeof(attrs)/sizeof(attrs[0]); ++i)
  {
    CHECK(pycpp::has_attr(module, attrs[i]));
    CHECK(py::extract<bool>(neworder::Callback::eval("'%%' in locals()"_s % attrs[i])()));
  }

  // Check diagnostics consistent
  CHECK(py::extract<bool>(neworder::Callback::eval("name() == '%%'"_s % neworder::module_name())()));
  CHECK(py::extract<bool>(neworder::Callback::eval("version() == '%%'"_s % neworder::module_version())()));
  CHECK(py::extract<bool>(neworder::Callback::eval("python() == '%%'"_s % neworder::python_version()/*.c_str()*/)()));
  // py::def("name", no::module_name);
  // py::def("version", no::module_version);
  // py::def("python", no::python_version);

}