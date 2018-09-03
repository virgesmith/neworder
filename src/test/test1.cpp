

#include "test.h"

#include "Environment.h"
#include "Inspect.h"
#include "Module.h"
#include "python.h"

#include <iostream>

// C++-ified and boost.pythonated version of the example here to stay: https://docs.python.org/3/extending/embedding.html

void test1(const std::string& modulename, const std::string& functionname, const std::vector<std::string>& argstrings, const py::object& expected)
{
  pycpp::Environment& env = pycpp::Environment::get();
  std::cout << env.context() << modulename << ":" << functionname;
  for (const auto& arg: argstrings)
    std::cout << " " << arg;
  std::cout << std::endl;

  py::object module = py::import(modulename.c_str());
  py::object function(module.attr(functionname.c_str()));

  std::vector<int> args(argstrings.size());
  for (size_t i = 0; i < argstrings.size(); ++i)
  {
    args[i] = std::stoi(argstrings[i]);
  }
  py::object result = args.size() == 2 ? function(args[0], args[1]) : function();
  CHECK(result == expected);
  //std::cout << env.context() << "Result: " << result << std::endl;
}