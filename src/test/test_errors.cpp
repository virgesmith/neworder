
#include "Environment.h"

#include "python.h"

#include <iostream>

void test_errors()
{
  bool caught = false;
  try
  {
    py::object module = py::import("op");
    py::object function(module.attr("notafunc"));
    function();
  }
  catch (py::error_already_set&)
  {
    std::cerr << pycpp::Environment::get().context() << "caught expected: " << pycpp::Environment::get_error() << std::endl;
    caught = true;
  }
  // Nobody expects this code to be executed
  if (!caught)
    throw std::runtime_error("spanish inquisition");
}