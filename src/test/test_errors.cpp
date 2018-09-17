

#include "test.h"

#include "Environment.h"
#include "Module.h"

#include "python.h"


void test_errors()
{
  py::object module = py::import("op");
  py::object function(module.attr("notafunc"));
  //neworder::shell();
  CHECK_THROWS(function(), py::error_already_set);
}