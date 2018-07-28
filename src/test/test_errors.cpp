
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
    std::cout << "[C++] caught expected: " << pycpp::Environment::check() << std::endl;
    caught = true;
  }
  // Nobody expects this code to be executed
  if (!caught)
    throw std::runtime_error("spanish inquisition");
}